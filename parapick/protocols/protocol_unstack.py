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
import json
import logging
import os
import traceback
from enum import Enum

import pwem.objects as emobj
import pyworkflow.protocol.params as params
from pwem.protocols import EMProtocol
from pyworkflow import BETA
from pyworkflow.protocol.constants import STEPS_PARALLEL
from pyworkflow.utils import redStr
from tomo.objects import SetOfTiltSeries

from ..convert.images import binImage, readSection, writeMrc
from ..convert.naming import batches, getTiltId
from ..objects import SetOfTiltSeriesMicrographs, TiltSeriesMicrograph

logger = logging.getLogger(__name__)


class ParapickUnstackOutputs(Enum):
    micrographs = SetOfTiltSeriesMicrographs


class ParapickProtUnstack(EMProtocol):
    """ Extract every tilt image of a set of tilt-series as a micrograph, so that the
    tilt-series can be picked with any single-particle picker. The picks can then be
    localized in 3D with "parapick - localize".

    The micrographs keep the raw (non-aligned) frame of the tilt images, and each one
    records its source tilt-series and tilt image. Disabled (excluded) tilt images are
    skipped.
    """
    _label = 'unstack tilt-series'
    _devStatus = BETA
    _possibleOutputs = ParapickUnstackOutputs
    stepsExecutionMode = STEPS_PARALLEL

    MIC_FOLDER = 'micrographs'

    # -------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputTiltSeries', params.PointerParam,
                      pointerClass=SetOfTiltSeries, important=True,
                      label='Tilt-series',
                      help='Tilt-series to unstack. Their alignment (if any) is kept as '
                           'metadata, the images are not interpolated.')
        form.addParam('maxTiltAngle', params.FloatParam, default=-1,
                      label='Maximum tilt angle (deg)',
                      help='Only unstack tilt images with |tilt angle| <= this value. '
                           'A negative value keeps all of them.')
        form.addParam('binning', params.IntParam, default=1,
                      label='Binning factor',
                      validators=[params.GE(1)],
                      help='Downsample the micrographs by this integer factor (Fourier '
                           'cropping). Pickers often work better on binned images. The picks '
                           'are localized at any binning.')
        form.addParam('batchSize', params.IntParam, default=8,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Tilt-series per step',
                      validators=[params.GE(1)],
                      help='Tilt-series are processed in batches of this size; batches run '
                           'in parallel on the available threads.')
        form.addParallelSection(threads=4, mpi=0)

    # -------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        tsIds = sorted(self.inputTiltSeries.get().getTSIds())
        stepIds = [self._insertFunctionStep(self.unstackStep, i, batch,
                                            prerequisites=[], needsGPU=False)
                   for i, batch in enumerate(batches(tsIds, self.batchSize.get()))]
        self._insertFunctionStep(self.createOutputStep, prerequisites=stepIds, needsGPU=False)

    # --------------------------- STEPS functions ----------------------------
    def unstackStep(self, batchIndex, tsIds):
        """ Write the micrographs of a batch of tilt-series, and a JSON manifest
        with the tilt images written for each tilt-series. """
        manifest = {}
        for tsId in tsIds:
            try:
                manifest[tsId] = self._unstackTiltSeries(tsId)
            except Exception as e:
                logger.error(redStr('Tilt-series %s could not be unstacked: %s' % (tsId, e)))
                logger.error(traceback.format_exc())
        with open(self._getManifestFn(batchIndex), 'w') as f:
            json.dump(manifest, f)

    def createOutputStep(self):
        inSet = self.inputTiltSeries.get()
        written = {}
        for fn in sorted(os.listdir(self._getExtraPath())):
            if fn.startswith('manifest_') and fn.endswith('.json'):
                with open(self._getExtraPath(fn)) as f:
                    written.update(json.load(f))

        outSet = SetOfTiltSeriesMicrographs.create(self._getPath())
        outSet.setSamplingRate(self._getOutputSamplingRate())
        outSet.setAcquisition(self._getAcquisition(inSet))
        outSet.setTiltSeries(self.inputTiltSeries)

        for ts in inSet.iterItems(orderBy='_tsId'):
            tsId = ts.getTsId()
            indices = set(written.get(tsId, []))
            for ti in ts.iterItems(orderBy='_index'):
                if ti.getIndex() not in indices:
                    continue
                tiltId = getTiltId(tsId, ti.getIndex())
                mic = TiltSeriesMicrograph(location=self._getMicrographFn(tiltId))
                mic.setMicName(tiltId)
                mic.copyFromTiltImage(ti)
                mic.setTsId(tsId)
                outSet.append(mic)

        if not outSet.getSize():
            raise RuntimeError('No micrograph could be extracted. Check the log for errors.')
        self._defineOutputs(**{ParapickUnstackOutputs.micrographs.name: outSet})
        self._defineSourceRelation(self.inputTiltSeries, outSet)

    # --------------------------- UTILS functions ----------------------------
    def _unstackTiltSeries(self, tsId):
        with self._lock:  # Database access is not thread safe
            ts = self.inputTiltSeries.get().getTiltSeriesFromTsId(tsId)
            tilts = [(ti.getIndex(), ti.getFileName()) for ti in ts.iterItems(orderBy='_index')
                     if self._isSelected(ti)]
        binning = self.binning.get()
        pixelSize = self._getOutputSamplingRate()
        done = []
        for index, fileName in tilts:
            outFn = self._getMicrographFn(getTiltId(tsId, index))
            if not os.path.exists(outFn):  # Resume
                image = readSection(fileName, index)
                writeMrc(outFn, binImage(image, binning), pixelSize)
            done.append(index)
        logger.info('%s: %d micrographs' % (tsId, len(done)))
        return done

    def _isSelected(self, tiltImage):
        maxAngle = self.maxTiltAngle.get()
        return (tiltImage.isEnabled() and
                (maxAngle < 0 or abs(tiltImage.getTiltAngle()) <= maxAngle))

    def _getOutputSamplingRate(self):
        return self.inputTiltSeries.get().getSamplingRate() * self.binning.get()

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

    def _getManifestFn(self, batchIndex):
        return self._getExtraPath('manifest_%04d.json' % batchIndex)

    # --------------------------- INFO functions -----------------------------
    def _validate(self):
        errors = []
        inSet = self.inputTiltSeries.get()
        if inSet is not None and inSet.getDim() is None:
            errors.append('The dimensions of the input tilt-series are unknown.')
        return errors

    def _summary(self):
        summary = []
        out = getattr(self, ParapickUnstackOutputs.micrographs.name, None)
        if out is None:
            summary.append('Output not ready yet.')
        else:
            summary.append('%d micrographs from %d tilt-series, %.3f Å/px.'
                           % (out.getSize(), len(out.getTSIds()), out.getSamplingRate()))
        return summary

    def _methods(self):
        methods = ['The tilt images of %d tilt-series were extracted as individual '
                   'micrographs' % self.inputTiltSeries.get().getSize()]
        if self.binning.get() > 1:
            methods[0] += ', binned by %d with Fourier cropping' % self.binning.get()
        methods[0] += ', to be picked as single-particle micrographs and localized with ParaPick.'
        return methods
