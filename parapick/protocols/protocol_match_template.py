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
import json
import logging
import os
from collections import defaultdict
from enum import Enum

import numpy as np

import pwem.objects as emobj
import pyworkflow.protocol.params as params
from pwem.protocols import EMProtocol
from pyworkflow import BETA
from pyworkflow.object import Float
from pyworkflow.protocol.constants import STEPS_PARALLEL
from pyworkflow.utils import makePath, yellowStr
from tomo.objects import SetOfCTFTomoSeries, SetOfTiltSeries, SetOfTomograms

from .. import Plugin
from ..constants import DEFAULT_VERSION, PARAPICK_LEOPARD
from ..convert import geometry, star
from ..convert.images import readSection, readVolume, writeMrc
from ..convert.naming import batches, getTiltId
from ..objects import TiltCtfLookup, setTiltImageInfo, setTiltSeriesPointer

logger = logging.getLogger(__name__)

PIXEL_SIZE_TOLERANCE = 0.01
SAMPLING_CHOICES = ['default', 'fast']
SCORE_CHOICES = ['zscore', 'mip']


class ParapickMatchTemplateOutputs(Enum):
    micrographs = emobj.SetOfMicrographs
    coordinates = emobj.SetOfCoordinates


class ParapickProtMatchTemplate(EMProtocol):
    """ Find particles on every tilt image of a set of tilt-series by 2D template matching
    with Leopard-EM, with parameters inferred for each tilt: the low-pass follows the dose
    the tilt had received before, the angular and defocus steps the resolution left, and
    the defocus range the tilt of the sample. The tilts with the highest predicted SNR
    are searched first.

    Outputs the tilt images as micrographs and the peaks found on them as coordinates,
    scored by their z-score, ready for "parapick - localize".

    The template matching is done by Leopard-EM (Lucas lab, UC Berkeley): please cite
    Giammar et al. (2026), Acta Cryst. D 82, 30-42, https://doi.org/10.1107/S2059798325009982.
    ParaPick only derives its parameters.

    Needs the optional binary: scipion3 installb parapick-leopard-<version>.
    """
    _label = 'template matching (Leopard-EM)'
    _references = ['Giammar2026', 'Grant2015', 'deIsidroGomez2026parapick']
    _devStatus = BETA
    _possibleOutputs = ParapickMatchTemplateOutputs
    stepsExecutionMode = STEPS_PARALLEL

    MIC_FOLDER = 'micrographs'
    INPUT_FOLDER = 'parapick_input'  # Kept to inspect or rerun ParaPick by hand
    TOMOGRAMS_STAR = 'tomograms.star'
    MANIFEST = 'tilts.json'
    TEMPLATE = 'template.mrc'

    # -------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputTiltSeries', params.PointerParam,
                      pointerClass=SetOfTiltSeries, important=True,
                      label='Tilt-series',
                      help='Tilt-series to search. Their dose (initial dose of each tilt '
                           'image) sets the low-pass of each tilt, and their alignment the '
                           'tilt of the sample. The images are not interpolated.')
        form.addParam('inputCtfs', params.PointerParam,
                      pointerClass=SetOfCTFTomoSeries, important=True,
                      label='CTF',
                      help='CTF estimated on the tilt-series. Tilt images without an '
                           '(enabled) CTF are not searched.')
        form.addParam('inputTemplate', params.PointerParam,
                      pointerClass='Volume', important=True,
                      label='Template',
                      help='Template volume, at the pixel size of the tilt-series (e.g. '
                           'simulated from an atomic model).')
        form.addParam('inputTomograms', params.PointerParam,
                      pointerClass=SetOfTomograms, allowsNull=True,
                      label='Tomograms (optional)',
                      help='If given, their thickness is the thickness of the sample.')
        form.addParam('particleDiameter', params.FloatParam, default=None,
                      important=True,
                      label='Particle diameter (Å)',
                      help='Diameter of the target. Sets the high-pass and the angular steps. '
                           'Peaks may be closer than it, since particles overlap in projection: '
                           'their minimum distance is one resolution element of each tilt.')
        form.addParam('thickness', params.FloatParam, default=2000.0,
                      label='Sample thickness (Å)',
                      validators=[params.Positive],
                      help='Thickness of the sample; it widens the defocus search. Ignored '
                           'for the tilt-series that have a tomogram.')
        form.addParam('symmetry', params.StringParam, default='C1',
                      label='Symmetry',
                      help='Symmetry of the template (e.g. C1, D7), to restrict the search.')
        form.addParam('energyFilter', params.BooleanParam, default=True,
                      label='Energy filter used?',
                      help='Whether the images were zero-loss filtered. Only changes the '
                           'ranking of the tilts.')

        form.addSection(label='Search')
        form.addParam('binning', params.IntParam, default=1,
                      label='Binning factor',
                      validators=[params.GE(1)],
                      help='Search the tilt images binned by this factor (Fourier cropping): '
                           'about N² faster per orientation, and the coarser pixel allows '
                           'coarser steps. The coordinates stay in the pixels of the '
                           'tilt-series. The template may be at the pixel size of the '
                           'tilt-series or of the binned images.')
        form.addParam('sampling', params.EnumParam, choices=SAMPLING_CHOICES, default=0,
                      label='Sampling',
                      help='default: angular steps capped at 4°/3° and defocus step at '
                           '500 Å. fast: steps proportional to the resolution left in each '
                           'tilt (the relative accuracy of Leopard-EM\'s defaults), uncapped. '
                           'Much faster on binned or late tilts; for locations, not poses.')
        form.addParam('maxTilts', params.IntParam, default=-1,
                      label='Tilts per tilt-series',
                      help='Search only this many tilt images of each tilt-series, those with '
                           'the highest predicted SNR (usually the first acquired). '
                           'Negative: all.')
        form.addParam('falsePositives', params.FloatParam, default=1.0,
                      label='False positives per image',
                      validators=[params.Positive],
                      help='Expected false positives per tilt image: sets the threshold of '
                           'the peaks. Localize tolerates false positives on single tilts, so '
                           'a value well above 1 keeps more true particles.')
        form.addParam('score', params.EnumParam, choices=SCORE_CHOICES, default=0,
                      label='Peak score',
                      help='zscore: Leopard-EM\'s z-score (best correlation normalised over '
                           'the orientations at each pixel), thresholded for the false '
                           'positives per image. mip: the best correlation normalised over the '
                           'image, which keeps the signal of particles that correlate at every '
                           'orientation, as on low-dose binned tilt images; needs "Peaks per '
                           'tilt image". Changing the picking and continuing only picks the '
                           'peaks again from the results.')
        form.addParam('peaksPerTilt', params.IntParam, default=-1,
                      label='Peaks per tilt image',
                      help='Keep the N best peaks of each tilt image instead of thresholding '
                           'them (a few times the particles expected per image; localize '
                           'tolerates false positives). Negative: threshold for the false '
                           'positives per image (zscore only).')
        form.addParam('orientationBatchSize', params.IntParam, default=16,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Orientation batch size',
                      validators=[params.GE(1)],
                      help='Orientations correlated at once. Lower it if the GPU runs out '
                           'of memory.')
        form.addHidden(params.GPU_LIST, params.StringParam, default='0',
                       label='GPU IDs')
        form.addParam('batchSize', params.IntParam, default=1,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Tilt-series per step',
                      validators=[params.GE(1)],
                      help='Tilt-series are searched in batches of this size, one ParaPick '
                           'run (and one GPU) per batch.')
        form.addParam('extraParams', params.StringParam, default='',
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Additional parameters',
                      help='Extra arguments for "parapick match-template".')
        form.addParallelSection(threads=1, mpi=0)

    # -------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        convertId = self._insertFunctionStep(self.convertInputStep, needsGPU=False)
        tsIds = sorted(self._getSearchedTsIds())
        # The picking is part of the arguments: changing it and continuing picks again
        runIds = [self._insertFunctionStep(self.matchStep, i, batch, self._getPicking(),
                                           prerequisites=[convertId], needsGPU=True)
                  for i, batch in enumerate(batches(tsIds, self.batchSize.get()))]
        self._insertFunctionStep(self.createOutputStep, prerequisites=runIds, needsGPU=False)

    # --------------------------- STEPS functions ----------------------------
    def convertInputStep(self):
        """ Write the tilt images as micrographs, the template, and the geometry with
        the CTF and the dose of each tilt. """
        tsSet = self.inputTiltSeries.get()
        ctfs = TiltCtfLookup(self.inputCtfs.get())
        tomograms = self._getTomogramsDict()
        template = self.inputTemplate.get()
        writeMrc(self._getInputPath(self.TEMPLATE), readVolume(template.getFileName()),
                 template.getSamplingRate())

        geometries, manifest, doses, nonRigid = [], {}, set(), 0
        for ts in tsSet.iterItems(orderBy='_tsId'):
            tsId = ts.getTsId()
            tilts = [(ti.clone(), ctfs.get(ti)) for ti in ts.iterItems(orderBy='_index')
                     if ti.isEnabled()]
            tilts = [(ti, ctf) for ti, ctf in tilts if ctf is not None]
            if not tilts:
                continue
            geo, n = self._getGeometry(tsId, tilts, tsSet, tomograms.get(tsId))
            geometries.append(geo)
            nonRigid += n
            manifest[tsId] = [ti.getIndex() for ti, _ in tilts]
            doses |= {ti.getAcquisition().getDosePerFrame() for ti, _ in tilts}
        if nonRigid:
            logger.warning(yellowStr('%d tilt images have non-rigid alignments (e.g. stretch); '
                                     'they were approximated by rotations.' % nonRigid))
        star.writeTomogramsStar(self._getInputPath(self.TOMOGRAMS_STAR), geometries)
        # The same dose for every tilt: tell ParaPick, else it infers it from the
        # pre-exposure of each tilt
        dose = doses.pop() if len(doses) == 1 else None
        with open(self._getInputPath(self.MANIFEST), 'w') as f:
            json.dump({'tilts': manifest, 'dosePerTilt': dose if dose else None}, f)

    def matchStep(self, batchIndex, tsIds, picking=None):
        Plugin.runParapick(self, self._getMatchArgs(self._getBatchPath(batchIndex), tsIds))

    def createOutputStep(self):
        manifest = self._readManifest()['tilts']
        inSet = self.inputTiltSeries.get()
        sr = inSet.getSamplingRate()

        micSet = emobj.SetOfMicrographs.create(self._getPath())
        micSet.setSamplingRate(sr)
        micSet.setAcquisition(self._getAcquisition(inSet))
        setTiltSeriesPointer(micSet, self.inputTiltSeries)
        for ts in inSet.iterItems(orderBy='_tsId'):
            indices = set(manifest.get(ts.getTsId(), []))
            for ti in ts.iterItems(orderBy='_index'):
                if ti.getIndex() not in indices:
                    continue
                tiltId = getTiltId(ts.getTsId(), ti.getIndex())
                mic = emobj.Micrograph(location=self._getMicrographFn(tiltId))
                mic.setMicName(tiltId)
                setTiltImageInfo(mic, ti)
                micSet.append(mic)
        self._defineOutputs(**{ParapickMatchTemplateOutputs.micrographs.name: micSet})
        self._defineSourceRelation(self.inputTiltSeries, micSet)

        mics = {mic.getMicName(): mic.clone() for mic in micSet.iterItems()}
        coordSet = self._createSetOfCoordinates(micSet)
        coordSet.setBoxSize(int(round(self.particleDiameter.get() / sr)))
        unmatched = 0
        for pick in self._readPicks():
            index = self._tiltIndexOfPick(pick, manifest)
            mic = None if index is None else mics.get(getTiltId(pick['rlnTomoName'], index))
            if mic is None:
                unmatched += 1
                continue
            coord = emobj.Coordinate()
            coord.setMicrograph(mic)
            coord.setX(int(round(float(pick['rlnCoordinateX']))))
            coord.setY(int(round(float(pick['rlnCoordinateY']))))
            coord._score = Float(float(pick.get('rlnAutopickFigureOfMerit', 0.0)))
            coordSet.append(coord)
        if unmatched:
            logger.warning(yellowStr('%d peaks could not be matched to a micrograph.'
                                     % unmatched))
        self._defineOutputs(**{ParapickMatchTemplateOutputs.coordinates.name: coordSet})
        self._defineSourceRelation(micSet, coordSet)
        self._defineSourceRelation(self.inputTemplate, coordSet)

    # --------------------------- UTILS functions ----------------------------
    def _getSearchedTsIds(self):
        tsIds = set(self.inputTiltSeries.get().getTSIds())
        return tsIds & set(self.inputCtfs.get().getTSIds())

    def _getTomogramsDict(self):
        tomoSet = self.inputTomograms.get()
        if tomoSet is None:
            return {}
        return {tomo.getTsId(): tomo.clone() for tomo in tomoSet.iterItems()}

    def _getGeometry(self, tsId, tilts, tsSet, tomogram):
        """ Geometry of a tilt-series, with the CTF and the dose of each tilt, and the
        number of non-rigid tilts. The tilt images are written as micrographs. """
        tsSr = tsSet.getSamplingRate()
        width, height = tsSet.getDim()[:2]
        relionTilts, nonRigid = [], 0
        columns = defaultdict(list)
        for ti, ctf in tilts:
            micFn = self._getMicrographFn(getTiltId(tsId, ti.getIndex()))
            if not os.path.exists(micFn):  # Resume
                writeMrc(micFn, readSection(ti.getFileName(), ti.getIndex()), tsSr)
            matrix = ti.getTransform().getMatrix() if ti.hasTransform() else None
            if matrix is not None and not geometry.isRigid(matrix):
                nonRigid += 1
            relionTilts.append((os.path.abspath(micFn),
                                geometry.tiltImageToRelion(matrix, ti.getTiltAngle(), tsSr,
                                                           (width, height))))
            acq = ti.getAcquisition()
            columns['rlnTomoNominalStageTiltAngle'].append(float(ti.getTiltAngle()))
            columns['rlnMicrographPreExposure'].append(float(acq.getDoseInitial() or 0.0))
            columns['rlnDefocusU'].append(float(ctf.getDefocusU()))
            columns['rlnDefocusV'].append(float(ctf.getDefocusV()))
            columns['rlnDefocusAngle'].append(float(ctf.getDefocusAngle()))
            columns['rlnPhaseShift'].append(float(ctf.getPhaseShift() or 0.0)
                                            if ctf.hasPhaseShift() else 0.0)
        if tomogram is not None:
            scale = tomogram.getSamplingRate() / tsSr
            volume = [d * scale for d in tomogram.getDim()]
        else:
            zRot = np.median([t.zRot for _, t in relionTilts])
            volume = list(geometry.alignedExtent((width, height), zRot)) + \
                [self.thickness.get() / tsSr]
        acq = tsSet.getAcquisition()
        geo = star.TiltSeriesGeometry(
            tsId=tsId, pixelSize=tsSr, imageSize=(width, height),
            volumeSize=[geometry.evenSize(v) for v in volume], tilts=relionTilts,
            voltage=acq.getVoltage() or 300.0,
            sphericalAberration=acq.getSphericalAberration() or 2.7,
            amplitudeContrast=acq.getAmplitudeContrast() or 0.1,
            tiltColumns=dict(columns))
        return geo, nonRigid

    def _getMatchArgs(self, outDir, tsIds):
        manifest = self._readManifest()
        args = ['match-template',
                '--device', 'gpu',
                '--tomograms', self._getInputPath(self.TOMOGRAMS_STAR),
                '--template', self._getInputPath(self.TEMPLATE),
                '--output', outDir,
                '--tilt-series'] + list(tsIds) + [
                '--particle-diameter', self.particleDiameter.get(),
                '--symmetry', self.symmetry.get(),
                '--false-positives', self.falsePositives.get(),
                '--orientation-batch-size', self.orientationBatchSize.get(),
                '--binning', self.binning.get(),
                '--sampling', SAMPLING_CHOICES[self.sampling.get()],
                '--score', SCORE_CHOICES[self.score.get()]]
        if self.peaksPerTilt.get() > 0:
            args += ['--peaks-per-tilt', self.peaksPerTilt.get()]
        if manifest.get('dosePerTilt'):
            args += ['--dose-per-tilt', manifest['dosePerTilt']]
        if self.maxTilts.get() > 0:
            args += ['--max-tilts', self.maxTilts.get()]
        if not self.energyFilter.get():
            args += ['--no-energy-filter']
        return ' '.join(str(a) for a in args) + ' ' + self.extraParams.get('')

    def _getPicking(self):
        return {'score': SCORE_CHOICES[self.score.get()],
                'peaksPerTilt': self.peaksPerTilt.get(),
                'falsePositives': self.falsePositives.get()}

    def _readManifest(self):
        with open(self._getInputPath(self.MANIFEST)) as f:
            return json.load(f)

    def _readPicks(self):
        picks = []
        for name in sorted(os.listdir(self._getExtraPath())):
            fn = self._getExtraPath(name, 'picks.star')
            if name.startswith('batch_') and os.path.exists(fn):
                picks += star.readStarTable(fn, 'picks')
        return picks

    @staticmethod
    def _tiltIndexOfPick(pick, manifest):
        """ Scipion index of the tilt image of a pick. ParaPick names the tilt images
        ``<tsId>_<row>.mrc``, with the 1-based row of the tilt in its geometry. """
        tsId = pick['rlnTomoName']
        stem = os.path.splitext(os.path.basename(pick['rlnMicrographName']))[0]
        indices = manifest.get(tsId, [])
        try:
            row = int(stem[len(tsId) + 1:]) - 1
        except ValueError:
            return None
        return indices[row] if stem.startswith(tsId + '_') and 0 <= row < len(indices) \
            else None

    @staticmethod
    def _getAcquisition(tsSet):
        tomoAcq = tsSet.getAcquisition()
        acq = emobj.Acquisition()
        acq.setVoltage(tomoAcq.getVoltage())
        acq.setSphericalAberration(tomoAcq.getSphericalAberration())
        acq.setAmplitudeContrast(tomoAcq.getAmplitudeContrast())
        acq.setMagnification(tomoAcq.getMagnification())
        return acq

    def _getMicrographFn(self, tiltId):
        return self._getExtraPath(self.MIC_FOLDER, tiltId + '.mrc')

    def _getInputPath(self, *paths):
        makePath(self._getExtraPath(self.INPUT_FOLDER))
        return self._getExtraPath(self.INPUT_FOLDER, *paths)

    def _getBatchPath(self, batchIndex):
        return self._getExtraPath('batch_%04d' % batchIndex)

    # --------------------------- INFO functions -----------------------------
    def _validate(self):
        errors = []
        if not Plugin.isLeopardInstalled():
            errors.append('Leopard-EM is not installed for ParaPick. Install it with: '
                          'scipion3 installb %s-%s'
                          % (PARAPICK_LEOPARD, Plugin.getActiveVersion() or DEFAULT_VERSION))
        if SCORE_CHOICES[self.score.get()] == 'mip' and self.peaksPerTilt.get() <= 0:
            errors.append('The mip score needs "Peaks per tilt image".')
        diameter = self.particleDiameter.get()
        if diameter is None or diameter <= 0:
            errors.append('Please, provide the particle diameter.')
        tsSet, ctfs = self.inputTiltSeries.get(), self.inputCtfs.get()
        if tsSet is not None and ctfs is not None and not self._getSearchedTsIds():
            errors.append('The CTF was not estimated on any of the input tilt-series '
                          '(no tilt-series id in common).')
        template = self.inputTemplate.get()
        if tsSet is not None and template is not None:
            tsSr, tmplSr = tsSet.getSamplingRate(), template.getSamplingRate()
            allowed = {tsSr, tsSr * self.binning.get()}
            if all(abs(sr - tmplSr) > PIXEL_SIZE_TOLERANCE * sr for sr in allowed):
                errors.append('The template has %.4f Å/px but the tilt-series %.4f Å/px '
                              '(%.4f Å/px binned). Rescale or simulate the template at one of '
                              'them.' % (tmplSr, tsSr, tsSr * self.binning.get()))
        return errors

    def _warnings(self):
        warnings = []
        tsSet = self.inputTiltSeries.get()
        if tsSet is not None and not tsSet.hasAlignment():
            warnings.append('The tilt-series have no alignment: the tilt axis is assumed to '
                            'be the image Y axis. It sets the direction of the defocus '
                            'gradient on each tilt image.')
        return warnings

    def _summary(self):
        coords = getattr(self, ParapickMatchTemplateOutputs.coordinates.name, None)
        mics = getattr(self, ParapickMatchTemplateOutputs.micrographs.name, None)
        if coords is None or mics is None:
            return ['Outputs not ready yet.']
        return ['%d peaks on %d micrographs (tilt images).'
                % (coords.getSize(), mics.getSize())]

    def _methods(self):
        coords = getattr(self, ParapickMatchTemplateOutputs.coordinates.name, None)
        if coords is None:
            return []
        return ['%d particles were found on the tilt images by 2D template matching with '
                'Leopard-EM [Giammar2026], with parameters derived for each tilt image from its '
                'dose [Grant2015] and geometry by ParaPick [deIsidroGomez2026parapick].'
                % coords.getSize()]
