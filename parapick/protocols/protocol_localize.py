# **************************************************************************
# *
# * Authors: Oier Lauzirika Zarrabeitia (olauzirika@cnb.csic.es)
# *
# * Spanish National Center for Biotechnology (CNB)
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 3 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************
import logging
import os

import numpy as np
from collections import defaultdict
from enum import Enum

import pyworkflow.protocol.params as params
from pwem.objects import SetOfCoordinates
from pwem.protocols import EMProtocol
from pyworkflow import BETA
from pyworkflow.object import Float, Integer
from pyworkflow.protocol.constants import STEPS_PARALLEL
from pyworkflow.utils import cleanPath, makePath, yellowStr
from tomo.constants import CENTER_GRAVITY
from tomo.objects import (Coordinate3D, LandmarkModel, SetOfCoordinates3D, SetOfLandmarkModels,
                          SetOfTiltSeries, SetOfTiltSeriesCoordinates, SetOfTomograms,
                          TiltSeriesCoordinate)
from tomo.protocols import ProtTomoBase

from .. import Plugin
from ..convert import geometry, star
from ..convert.naming import batches, getTiltId, parseTiltId
from ..objects import (ACQ_ORDER_FIELD, TILT_INDEX_FIELD, TS_ID_FIELD,
                       SetOfTiltSeriesMicrographs)

logger = logging.getLogger(__name__)

INIT_CHOICES = ['lifted', 'uniform']
SCORE_ATTRS = ('_score', '_cryoloScore', '_topazScore', '_xmipp_zScore')


class ParapickLocalizeOutputs(Enum):
    tiltSeriesCoordinates = SetOfTiltSeriesCoordinates
    landmarkModels = SetOfLandmarkModels
    coordinates3D = SetOfCoordinates3D


class ParapickProtLocalize(EMProtocol, ProtTomoBase):
    """ Localize particles in 3D from 2D picks on the tilt images of tilt-series, with
    ParaPick. No tomogram is reconstructed: a 3D Gaussian mixture is fitted to the picks
    of all the tilt images, using the alignment of the tilt-series.

    Input picks are a set of coordinates picked (with any single-particle picker) on
    micrographs produced by "parapick - unstack tilt-series". Outputs:

    - tilt-series coordinates (Å, relative to the centre of the volume);
    - landmark models: the picks explained by each particle, chained across tilts,
      in the frame of the raw tilt images, with residuals;
    - 3D coordinates, if tomograms are given.
    """
    _label = 'localize'
    _devStatus = BETA
    _possibleOutputs = ParapickLocalizeOutputs
    stepsExecutionMode = STEPS_PARALLEL

    INPUT_FOLDER = 'parapick_input'  # Kept to inspect or rerun ParaPick by hand
    TOMOGRAMS_STAR = 'tomograms.star'
    PICKS_STAR = 'picks.star'

    # -------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputCoordinates', params.PointerParam,
                      pointerClass=SetOfCoordinates, important=True,
                      label='2D picks',
                      help='Coordinates picked on micrographs extracted from the tilt-series '
                           '(see "parapick - unstack tilt-series"). Micrographs from other '
                           'protocols are matched to tilt images by name: '
                           '<tsId>_<index>, with the 1-based stack index of the tilt image.')
        form.addParam('inputTiltSeries', params.PointerParam,
                      pointerClass=SetOfTiltSeries, allowsNull=True,
                      label='Tilt-series',
                      help='Tilt-series that were picked. They provide the geometry (tilt '
                           'angles and alignment). If empty, they are taken from the '
                           'micrographs of the picks.')
        form.addParam('inputTomograms', params.PointerParam,
                      pointerClass=SetOfTomograms, allowsNull=True,
                      label='Tomograms (optional)',
                      help='If given, their size bounds the search volume, and the particles '
                           'are also output as 3D coordinates on them (matched by tsId).')
        form.addParam('particleDiameter', params.FloatParam, default=None,
                      important=True,
                      label='Particle diameter (Å)',
                      help='Approximate particle diameter. It sets the initial spread of the '
                           'model (half of it), which must be smaller than the distance '
                           'between neighbouring particles for crowded samples to be resolved.')
        form.addParam('thickness', params.FloatParam, default=3000.0,
                      label='Sample thickness (Å)',
                      validators=[params.Positive],
                      help='Thickness of the volume to search for particles. Ignored for the '
                           'tilt-series that have a tomogram.')
        form.addParam('boxSize', params.FloatParam, default=300.0,
                      label='Box size (Å)',
                      validators=[params.Positive],
                      help='Box size of the output 3D coordinates and landmark size.')

        form.addSection(label='Model')
        form.addParam('alpha', params.FloatParam, default=0.5,
                      label='Dirichlet concentration (alpha)',
                      validators=[params.Positive],
                      help='Values below 1 prune components supported by fewer than 1 - alpha '
                           'effective picks.')
        form.addParam('overcompleteFactor', params.FloatParam, default=4.0,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Overcomplete factor',
                      help='Initial components per pick of the most crowded tilt image.')
        form.addParam('sigma', params.FloatParam, default=0.0,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Initial sigma (Å)',
                      help='Initial standard deviation of the model. 0 uses the particle '
                           'radius. Values well above the distance between particles merge '
                           'neighbours and lose most of them.')
        form.addParam('backgroundFraction', params.FloatParam, default=0.1,
                      label='Background fraction',
                      help='Initial weight of the uniform background that absorbs false '
                           'positive picks. 0 disables it.')
        form.addParam('mergeDistance', params.FloatParam, default=-1.0,
                      label='Merge distance (Å)',
                      help='Merge particles closer than this. A value around the particle '
                           'radius works well. Negative: automatic; 0: no merging.')
        form.addParam('init', params.EnumParam, choices=INIT_CHOICES, default=0,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Initialization',
                      help='lifted: back-project the picks of the least tilted image at random '
                           'depths. uniform: random positions in the volume.')
        form.addParam('maxIterations', params.IntParam, default=500,
                      expertLevel=params.LEVEL_ADVANCED, label='Maximum iterations')
        form.addParam('tolerance', params.FloatParam, default=0.1,
                      expertLevel=params.LEVEL_ADVANCED, label='Tolerance (Å)',
                      help='Convergence threshold on the mean displacement of the particles.')
        form.addParam('annealing', params.FloatParam, default=0.9,
                      expertLevel=params.LEVEL_ADVANCED, label='Annealing rate',
                      help='Per-iteration decay of the sigma floor. 0 disables annealing.')
        form.addParam('seed', params.IntParam, default=0,
                      expertLevel=params.LEVEL_ADVANCED, label='Random seed')

        form.addSection(label='Selection')
        form.addParam('minScore', params.FloatParam, default=0.3,
                      label='Minimum score',
                      help='Keep particles supported by at least this fraction of the tilt '
                           'images. With a picker of recall R, true particles score about R. '
                           'Negative: no threshold.')
        form.addParam('minSupport', params.FloatParam, default=-1,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Minimum support',
                      help='Keep particles supported by at least this many (effective) picks. '
                           'Negative: no threshold.')
        form.addParam('topN', params.IntParam, default=-1,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Particles per tilt-series',
                      help='Keep the N best particles of each tilt-series. Negative: all.')
        form.addParam('minPickScore', params.FloatParam, default=-1,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Minimum pick score',
                      help='Ignore 2D picks whose score is below this value (only if the '
                           'picker provides scores). Negative: use all picks.')
        form.addParam('assignmentThreshold', params.FloatParam, default=0.5,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Assignment threshold',
                      validators=[params.Range(0, 1)],
                      help='Minimum probability for a pick to join the landmark chain of a '
                           'particle.')

        form.addSection(label='Compute')
        form.addHidden(params.USE_GPU, params.BooleanParam, default=True,
                       label='Use GPU?',
                       help='Run the fit on GPU (JAX). Otherwise, it runs on CPU.')
        form.addHidden(params.GPU_LIST, params.StringParam, default='0',
                       label='GPU IDs')
        form.addParam('batchSize', params.IntParam, default=16,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Tilt-series per step',
                      validators=[params.GE(1)],
                      help='Tilt-series are localized in batches of this size, one ParaPick '
                           'run per batch.')
        form.addParam('flipY', params.BooleanParam, default=False,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Flip Y of the picks?',
                      help='For pickers whose y axis points the other way.')
        form.addParam('extraParams', params.StringParam, default='',
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Additional parameters',
                      help='Extra arguments for "parapick localize".')
        form.addParallelSection(threads=1, mpi=0)

    # -------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        convertId = self._insertFunctionStep(self.convertInputStep, needsGPU=False)
        tsIds = sorted(self._getPickedTsIds())
        runIds = [self._insertFunctionStep(self.localizeStep, i, batch,
                                           prerequisites=[convertId],
                                           needsGPU=self.useGpu.get())
                  for i, batch in enumerate(batches(tsIds, self.batchSize.get()))]
        self._insertFunctionStep(self.createOutputStep, prerequisites=runIds, needsGPU=False)

    # --------------------------- STEPS functions ----------------------------
    def convertInputStep(self):
        tsSet = self._getTiltSeriesSet()
        tomograms = self._getTomogramsDict()
        picks = self._collectPicks()
        geometries = []
        nonRigid = 0
        for ts in tsSet.iterItems(orderBy='_tsId'):
            tsId = ts.getTsId()
            if tsId not in picks:
                continue
            geo, n = self._getGeometry(ts, tsSet, tomograms.get(tsId))
            geometries.append(geo)
            nonRigid += n
        if nonRigid:
            logger.warning(yellowStr('%d tilt images have non-rigid alignments (e.g. stretch); '
                                     'they were approximated by rotations.' % nonRigid))
        makePath(self._getInputPath())
        star.writeTomogramsStar(self._getInputPath(self.TOMOGRAMS_STAR), geometries)
        star.writePicksStar(self._getInputPath(self.PICKS_STAR),
                            [p for tsId in sorted(picks) for p in picks[tsId]],
                            self._getMicrographs().getSamplingRate())

    def localizeStep(self, batchIndex, tsIds):
        outDir = self._getBatchPath(batchIndex)
        cleanPath(outDir)
        args = self._getLocalizeArgs(outDir, tsIds)
        Plugin.runParapick(self, args, gpu=self.useGpu.get())

    def createOutputStep(self):
        tsPointer = self._getTiltSeriesPointer()
        tsSet = self._getTiltSeriesSet()
        tsSr = tsSet.getSamplingRate()
        micSr = self._getMicrographs().getSamplingRate()
        height = tsSet.getDim()[1]
        particles, assignments = [], []
        for name in sorted(os.listdir(self._getExtraPath())):
            batchDir = self._getExtraPath(name)
            if name.startswith('batch_') and os.path.isdir(batchDir):
                particles += star.readParticlesStar(os.path.join(batchDir, 'particles.star'))
                assignments += star.readAssignmentsStar(os.path.join(batchDir, 'assignments.star'))

        # Tilt-series coordinates: centred Å, as ParaPick
        tsCoords = self._createSet(SetOfTiltSeriesCoordinates, 'tsCoordinates%s.sqlite', '')
        tsCoords.setSetOfTiltSeries(tsPointer)
        for p in particles:
            coord = TiltSeriesCoordinate()
            coord.setTsId(p.tsId)
            coord.setPosition(p.x, p.y, p.z, 1.0)
            coord.setScore(p.score if p.score is not None else 0.0)
            coord._ppkParticleId = Integer(p.particleId)
            coord._ppkSupport = Float(p.support)
            tsCoords.append(coord)
        outputs = {ParapickLocalizeOutputs.tiltSeriesCoordinates.name: tsCoords}

        # Landmark models: one chain per particle, in the raw tilt image frame (TS pixels)
        landmarks = self._createSetOfLandmarkModels()
        landmarks.setSetOfTiltSeries(tsPointer)
        landmarks.setHasResidualInfo(True)
        byTs = defaultdict(list)
        for a in assignments:
            byTs[a.tsId].append(a)
        scale = micSr / tsSr
        makePath(self._getExtraPath('landmarks'))
        for tsId in sorted(byTs):
            fn = self._getExtraPath('landmarks', '%s.sfid' % tsId)
            cleanPath(fn)
            model = LandmarkModel(tsId=tsId, fileName=fn, size=int(round(self.boxSize.get())),
                                  applyTSTransformation=False, hasResidualInfo=True)
            for a in sorted(byTs[tsId],
                            key=lambda r: (r.particleId, r.tiltId)):
                parsed = parseTiltId(a.tiltId)
                if parsed is None:
                    continue
                x, y = a.x * scale, a.y * scale
                if self.flipY.get():
                    y = height - y
                model.addLandmark(x, y, parsed[1], a.particleId,
                                  a.residualX / tsSr, a.residualY / tsSr)
            landmarks.append(model)
        outputs[ParapickLocalizeOutputs.landmarkModels.name] = landmarks

        # 3D coordinates on the tomograms
        tomograms = self._getTomogramsDict()
        if tomograms:
            tomoSet = self.inputTomograms.get()
            tomoSr = tomoSet.getSamplingRate()
            coords3D = self._createSetOfCoordinates3D(self.inputTomograms)
            coords3D.setSamplingRate(tomoSr)
            coords3D.setBoxSize(int(round(self.boxSize.get() / tomoSr)))
            for p in particles:
                tomo = tomograms.get(p.tsId)
                if tomo is None:
                    continue
                coord = Coordinate3D()
                coord.setVolume(tomo)
                coord.setPosition(p.x / tomoSr, p.y / tomoSr, p.z / tomoSr, CENTER_GRAVITY)
                coord.setScore(p.score if p.score is not None else 0.0)
                coord._ppkParticleId = Integer(p.particleId)
                coords3D.append(coord)
            outputs[ParapickLocalizeOutputs.coordinates3D.name] = coords3D

        self._defineOutputs(**outputs)
        for out in outputs.values():
            self._defineSourceRelation(self.inputCoordinates, out)
            self._defineSourceRelation(tsPointer, out)
        if ParapickLocalizeOutputs.coordinates3D.name in outputs:
            self._defineSourceRelation(self.inputTomograms,
                                       outputs[ParapickLocalizeOutputs.coordinates3D.name])

    # --------------------------- UTILS functions ----------------------------
    def _getMicrographs(self):
        return self.inputCoordinates.get().getMicrographs()

    def _getTiltSeriesPointer(self):
        if self.inputTiltSeries.get() is not None:
            return self.inputTiltSeries
        mics = self._getMicrographs()
        if isinstance(mics, SetOfTiltSeriesMicrographs):
            return mics.getTiltSeries(asPointer=True)
        return None

    def _getTiltSeriesSet(self):
        pointer = self._getTiltSeriesPointer()
        return None if pointer is None else pointer.get()

    def _getTomogramsDict(self):
        tomoSet = self.inputTomograms.get()
        if tomoSet is None:
            return {}
        return {tomo.getTsId(): tomo.clone() for tomo in tomoSet.iterItems()}

    def _getMicrographMap(self):
        """ micId -> (tsId, tiltIndex) of the source tilt image. """
        tsSet = self._getTiltSeriesSet()
        acqOrderToIndex = {}
        result = {}
        for mic in self._getMicrographs().iterItems():
            tsIdObj = getattr(mic, TS_ID_FIELD, None)
            tsId = tsIdObj.get() if tsIdObj is not None else None
            indexObj = getattr(mic, TILT_INDEX_FIELD, None)
            index = indexObj.get() if indexObj is not None else None
            acqObj = getattr(mic, ACQ_ORDER_FIELD, None)
            if tsId and index is None and acqObj is not None and acqObj.get() is not None:
                # Micrographs from other protocols that only record the acquisition order
                if tsId not in acqOrderToIndex:
                    ts = tsSet.getTiltSeriesFromTsId(tsId)
                    acqOrderToIndex[tsId] = ({} if ts is None else
                                             {ti.getAcquisitionOrder(): ti.getIndex()
                                              for ti in ts.iterItems()})
                index = acqOrderToIndex[tsId].get(acqObj.get())
            if not tsId or index is None:
                parsed = (parseTiltId(mic.getMicName() or '')
                          or parseTiltId(os.path.basename(mic.getFileName() or '')))
                if parsed is None:
                    continue
                tsId, index = parsed
            result[mic.getObjId()] = (tsId, int(index))
        return result

    def _getPickedTsIds(self):
        micMap = self._getMicrographMap()
        coords = self.inputCoordinates.get()
        micIds = coords.getUniqueValues('_micId')
        return {micMap[micId][0] for micId in micIds if micId in micMap}

    def _collectPicks(self):
        """ tsId -> list of star.Pick, keeping only picks on enabled tilt images. """
        micMap = self._getMicrographMap()
        tsSet = self._getTiltSeriesSet()
        validTilts = {}
        for ts in tsSet.iterItems():
            validTilts[ts.getTsId()] = {ti.getIndex() for ti in ts.iterItems() if ti.isEnabled()}
        picks = defaultdict(list)
        dropped = 0
        for coord in self.inputCoordinates.get().iterItems(orderBy='_micId'):
            source = micMap.get(coord.getMicId())
            if source is None or source[1] not in validTilts.get(source[0], ()):
                dropped += 1
                continue
            tsId, index = source
            picks[tsId].append(star.Pick(tsId, getTiltId(tsId, index),
                                         coord.getX(), coord.getY(), self._getScore(coord)))
        if dropped:
            logger.warning(yellowStr('%d picks could not be matched to an enabled tilt image '
                                     'and were ignored.' % dropped))
        return picks

    @staticmethod
    def _getScore(coord):
        for attr in SCORE_ATTRS:
            value = getattr(coord, attr, None)
            if value is not None and hasattr(value, 'get') and value.get() is not None:
                return float(value.get())
        return None

    def _getGeometry(self, ts, tsSet, tomogram):
        """ Geometry of a tilt-series for ParaPick and the number of non-rigid tilts. """
        tsSr = tsSet.getSamplingRate()
        width, height = tsSet.getDim()[:2]
        if tomogram is not None:
            scale = tomogram.getSamplingRate() / tsSr
            volume = [d * scale for d in tomogram.getDim()]
        else:
            volume = None
        tilts, nonRigid = [], 0
        for ti in ts.iterItems(orderBy='_index'):
            if not ti.isEnabled():
                continue
            matrix = ti.getTransform().getMatrix() if ti.hasTransform() else None
            if matrix is not None and not geometry.isRigid(matrix):
                nonRigid += 1
            tilts.append((getTiltId(ts.getTsId(), ti.getIndex()),
                          geometry.tiltImageToRelion(matrix, ti.getTiltAngle(), tsSr,
                                                     (width, height))))
        if volume is None:
            # The volume lives in the aligned frame: rotate the image extent accordingly
            zRot = np.median([t.zRot for _, t in tilts]) if tilts else 0.0
            volume = list(geometry.alignedExtent((width, height), zRot)) + \
                [self.thickness.get() / tsSr]
        acq = tsSet.getAcquisition()
        geo = star.TiltSeriesGeometry(
            tsId=ts.getTsId(), pixelSize=tsSr, imageSize=(width, height),
            volumeSize=[geometry.evenSize(v) for v in volume], tilts=tilts,
            voltage=acq.getVoltage() or 300.0,
            sphericalAberration=acq.getSphericalAberration() or 2.7,
            amplitudeContrast=acq.getAmplitudeContrast() or 0.1)
        return geo, nonRigid

    def _getInitialSigma(self):
        sigma = self.sigma.get()
        return sigma if sigma > 0 else self.particleDiameter.get() / 2.0

    def _getInputPath(self, *paths):
        return self._getExtraPath(self.INPUT_FOLDER, *paths)

    def _getBatchPath(self, batchIndex):
        return self._getExtraPath('batch_%04d' % batchIndex)

    def _getLocalizeArgs(self, outDir, tsIds):
        args = ['localize',
                '--device', 'gpu' if self.useGpu.get() else 'cpu',
                '--tomograms', self._getInputPath(self.TOMOGRAMS_STAR),
                '--picks', self._getInputPath(self.PICKS_STAR),
                '--output', outDir,
                '--tilt-series'] + list(tsIds) + [
                '--write-assignments',
                '--assignment-threshold', self.assignmentThreshold.get(),
                '--alpha', self.alpha.get(),
                '--overcomplete-factor', self.overcompleteFactor.get(),
                '--init', INIT_CHOICES[self.init.get()],
                '--max-iterations', self.maxIterations.get(),
                '--tolerance', self.tolerance.get(),
                '--annealing', self.annealing.get(),
                '--seed', self.seed.get()]
        args += ['--sigma', self._getInitialSigma()]
        if self.backgroundFraction.get() > 0:
            args += ['--background-fraction', self.backgroundFraction.get()]
        else:
            args += ['--no-background']
        if self.mergeDistance.get() >= 0:
            args += ['--merge-distance', self.mergeDistance.get()]
        if self.minScore.get() >= 0:
            args += ['--min-score', self.minScore.get()]
        if self.minSupport.get() >= 0:
            args += ['--min-support', self.minSupport.get()]
        if self.topN.get() >= 0:
            args += ['--top-n', self.topN.get()]
        if self.minPickScore.get() >= 0:
            args += ['--min-pick-score', self.minPickScore.get()]
        if self.flipY.get():
            args += ['--flip-y']
        return ' '.join(str(a) for a in args) + ' ' + self.extraParams.get('')

    # --------------------------- INFO functions -----------------------------
    def _validate(self):
        errors = []
        diameter = self.particleDiameter.get()
        if diameter is None or diameter <= 0:
            errors.append('Please, provide the particle diameter.')
        tsSet = self._getTiltSeriesSet()
        if tsSet is None:
            errors.append('The tilt-series could not be deduced from the picks. '
                          'Please, provide them.')
            return errors
        if not self._getPickedTsIds():
            errors.append('None of the picked micrographs could be matched to a tilt image. '
                          'Micrographs must come from "parapick - unstack tilt-series" or be '
                          'named <tsId>_<index>.')
        tomograms = self.inputTomograms.get()
        if tomograms is not None and not set(tomograms.getTSIds()) & set(tsSet.getTSIds()):
            errors.append('No tomogram matches the tsId of a tilt-series.')
        return errors

    def _warnings(self):
        warnings = []
        tsSet = self._getTiltSeriesSet()
        if tsSet is not None and not tsSet.hasAlignment():
            warnings.append('The tilt-series have no alignment: the tilt axis is assumed to '
                            'be the image Y axis and the images are assumed to be aligned.')
        if self.minPickScore.get() >= 0 and self._getScore(
                self.inputCoordinates.get().getFirstItem()) is None:
            warnings.append('The picks have no score: "Minimum pick score" will fail.')
        mics = self._getMicrographs()
        if tsSet is not None and mics is not None and mics.getDim() and tsSet.getDim():
            micWidth = mics.getDim()[0] * mics.getSamplingRate()
            tsWidth = tsSet.getDim()[0] * tsSet.getSamplingRate()
            if abs(micWidth - tsWidth) > 0.02 * tsWidth:
                warnings.append('The micrographs (%.0f Å wide) do not cover the tilt images '
                                '(%.0f Å wide): were they cropped?' % (micWidth, tsWidth))
        return warnings

    def _summary(self):
        summary = []
        for output in ParapickLocalizeOutputs:
            obj = getattr(self, output.name, None)
            if obj is not None:
                summary.append('%s: %d' % (output.name, obj.getSize()))
        return summary or ['Outputs not ready yet.']

    def _methods(self):
        out = getattr(self, ParapickLocalizeOutputs.tiltSeriesCoordinates.name, None)
        if out is None:
            return []
        return ['%d particles were localized in 3D from 2D picks on the tilt images with '
                'ParaPick [deIsidroGomez2026parapick].' % out.getSize()]
