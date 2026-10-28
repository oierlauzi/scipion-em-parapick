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
"""End-to-end test on a synthetic dataset with known particle positions.

Tilt-series with random rigid alignments are imported, unstacked and "picked" by
projecting known 3D points (Scipion's own projection model, see
``tomo.protocols.ProtTomoProjectCoordinates``), with missed and false picks. The picks
are imported as EMAN box files and localized; outputs are compared to the truth.

Needs ParaPick installed and the imod (alignment import) and eman2 (coordinate import)
plugins.
"""

import math
import os
import shutil

import mrcfile
import numpy as np
from pwem.protocols import ProtImportCoordinates
from pyworkflow.tests import BaseTest, setupTestProject
from tomo import constants as tomoConst
from tomo.protocols import ProtImportTomograms, ProtImportTs

from parapick.convert.geometry import projectScipion
from parapick.convert.naming import getTiltId
from parapick.protocols import (ParapickLocalizeOutputs, ParapickProtLocalize,
                                ParapickProtUnstack, ParapickUnstackOutputs)

TS_IDS = ['TS_01', 'TS_02']
WIDTH, HEIGHT = 201, 180      # Odd width on purpose (half-pixel conventions)
TS_SAMPLING = 10.0            # Å/px
ANGLES = np.arange(-60.0, 60.0 + 1e-6, 3.0)
N_PARTICLES = 10
BINNING = 2
TOMO_SAMPLING = 20.0          # Tomograms binned by 2 with respect to the tilt-series
TOMO_DIMS = (100, 90, 40)
RECALL = 0.85
FALSE_PICKS = 2               # Per micrograph
BOX = 10                      # px, even


def _rotation(degrees, shift):
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return np.array([[c, -s, shift[0]], [s, c, shift[1]], [0, 0, 1]])


def _placePoints(rng):
    points = []
    while len(points) < N_PARTICLES:
        p = rng.uniform((-60, -50, -15), (60, 50, 15))
        if all(np.linalg.norm(p - q) > 20 for q in points):
            points.append(p)
    return np.array(points)


class TestParapickWorkflow(BaseTest):

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.rng = np.random.default_rng(42)
        cls.dataDir = cls.proj.getPath('synthetic')
        cls.truth = {}       # tsId -> (K, 3) centred TS pixels
        cls.matrices = {}    # tsId -> list of 3x3 alignment matrices (index order)
        cls._writeInputs()

    # ----------------------------------------------------------------- data
    @classmethod
    def _writeInputs(cls):
        tsDir = os.path.join(cls.dataDir, 'ts')
        tomoDir = os.path.join(cls.dataDir, 'tomos')
        for d in (tsDir, tomoDir):
            os.makedirs(d, exist_ok=True)
        for tsId in TS_IDS:
            stack = cls.rng.normal(size=(len(ANGLES), HEIGHT, WIDTH)).astype(np.float32)
            with mrcfile.new(os.path.join(tsDir, tsId + '.mrc'), overwrite=True) as mrc:
                mrc.set_data(stack)
                mrc.voxel_size = TS_SAMPLING
            # Tilt axis close to the X axis, as IMOD alignments of many microscopes
            matrices = [_rotation(cls.rng.normal(85, 2), cls.rng.normal(0, 4, 2))
                        for _ in ANGLES]
            with open(os.path.join(tsDir, tsId + '.xf'), 'w') as f:
                for m in matrices:
                    f.write('%f %f %f %f %f %f\n' % (m[0, 0], m[0, 1], m[1, 0], m[1, 1],
                                                     m[0, 2], m[1, 2]))
            cls.matrices[tsId] = matrices
            cls.truth[tsId] = _placePoints(cls.rng)
            with mrcfile.new(os.path.join(tomoDir, tsId + '.mrc'), overwrite=True) as mrc:
                mrc.set_data(np.zeros(TOMO_DIMS[::-1], dtype=np.float32))
                mrc.voxel_size = TOMO_SAMPLING

    def _projections(self, tsId, index):
        """ Truth projected on the raw tilt image ``index`` (1-based), TS pixels. """
        return projectScipion(self.truth[tsId], self.matrices[tsId][index - 1],
                              ANGLES[index - 1], (WIDTH, HEIGHT))

    def _writeBoxFiles(self, mics):
        """ Simulated picker output: EMAN .box files on the binned micrographs. """
        boxDir = os.path.join(self.dataDir, 'boxes')
        shutil.rmtree(boxDir, ignore_errors=True)
        os.makedirs(boxDir)
        micW, micH = mics.getDim()[:2]
        for mic in mics:
            picks = self._projections(mic.getTsId(), mic.getTiltIndex()) / BINNING
            picks = picks[self.rng.random(len(picks)) < RECALL]
            false = self.rng.uniform((0, 0), (micW, micH), size=(FALSE_PICKS, 2))
            with open(os.path.join(boxDir, mic.getMicName() + '.box'), 'w') as f:
                for x, y in np.concatenate([picks, false]):
                    f.write('%d\t%d\t%d\t%d\n' % (round(x) - BOX // 2, round(y) - BOX // 2,
                                                  BOX, BOX))
        return boxDir

    # ----------------------------------------------------------------- protocols
    def _importTiltSeries(self):
        prot = self.newProtocol(ProtImportTs,
                                filesPath=os.path.join(self.dataDir, 'ts'),
                                filesPattern='{TS}.mrc',
                                anglesFrom=0,  # Range
                                minAngle=ANGLES[0], maxAngle=ANGLES[-1], stepAngle=3.0,
                                voltage=300, sphericalAberration=2.7, amplitudeContrast=0.1,
                                magnification=50000, samplingRate=TS_SAMPLING,
                                tiltAxisAngle=85, dosePerFrame=2.0)
        self.launchProtocol(prot)
        from imod.protocols import ProtImodImportTransformationMatrix
        protTm = self.newProtocol(ProtImodImportTransformationMatrix,
                                  filesPath=os.path.join(self.dataDir, 'ts'),
                                  filesPattern='{TS}.xf',
                                  inputSetOfTiltSeries=getattr(prot, prot.OUTPUT_NAME),
                                  binningTM=1, binningTS=1)
        self.launchProtocol(protTm)
        return getattr(protTm, 'TiltSeries')

    def _importTomograms(self):
        prot = self.newProtocol(ProtImportTomograms,
                                filesPath=os.path.join(self.dataDir, 'tomos'),
                                filesPattern='*.mrc',
                                samplingRate=TOMO_SAMPLING)
        self.launchProtocol(prot)
        return getattr(prot, 'Tomograms')

    def test_workflow(self):
        tsSet = self._importTiltSeries()
        self.assertTrue(tsSet.hasAlignment())

        # Unstack
        protUnstack = self.newProtocol(ParapickProtUnstack, inputTiltSeries=tsSet,
                                       binning=BINNING, batchSize=1, numberOfThreads=2)
        self.launchProtocol(protUnstack)
        mics = getattr(protUnstack, ParapickUnstackOutputs.micrographs.name)
        self.assertEqual(mics.getSize(), len(TS_IDS) * len(ANGLES))
        self.assertAlmostEqual(mics.getSamplingRate(), TS_SAMPLING * BINNING)
        self.assertEqual(tuple(mics.getDim()[:2]), (WIDTH // BINNING, HEIGHT // BINNING))
        mic = mics.getFirstItem()
        self.assertEqual(mic.getMicName(), getTiltId(mic.getTsId(), mic.getTiltIndex()))
        self.assertEqual(mics.getTiltSeries().getSize(), len(TS_IDS))

        # "Pick"
        protPicks = self.newProtocol(ProtImportCoordinates,
                                     importFrom=ProtImportCoordinates.IMPORT_FROM_EMAN,
                                     filesPath=self._writeBoxFiles(mics), filesPattern='*.box',
                                     inputMicrographs=mics, boxSize=BOX)
        self.launchProtocol(protPicks)
        coords = getattr(protPicks, 'outputCoordinates')

        # Localize (tilt-series deduced from the micrographs)
        protLocalize = self.newProtocol(ParapickProtLocalize,
                                        inputCoordinates=coords,
                                        inputTomograms=self._importTomograms(),
                                        particleDiameter=100, boxSize=100, minScore=0.5,
                                        batchSize=1,
                                        useGpu=bool(shutil.which('nvidia-smi')))
        self.launchProtocol(protLocalize)
        self._checkTiltSeriesCoordinates(protLocalize)
        self._checkLandmarks(protLocalize)
        self._checkCoordinates3D(protLocalize)

    # ----------------------------------------------------------------- checks
    def _matchTruth(self, tsId, positions):
        """ Distance from each truth point to the nearest position, in TS pixels. """
        truth = self.truth[tsId]
        self.assertEqual(len(positions), len(truth), 'particles of %s' % tsId)
        d = np.linalg.norm(truth[:, None, :] - np.asarray(positions)[None], axis=-1)
        return d.min(axis=1), d.argmin(axis=0)

    def _checkTiltSeriesCoordinates(self, prot):
        out = getattr(prot, ParapickLocalizeOutputs.tiltSeriesCoordinates.name)
        for tsId in TS_IDS:
            positions = [c.getPosition(TS_SAMPLING)
                         for c in out.iterItems(where='_tsId="%s"' % tsId)]
            errors, _ = self._matchTruth(tsId, positions)
            # Picks are rounded to binned pixels (±1 TS pixel)
            self.assertLess(errors.max(), 1.0, 'TS coordinate errors (px): %s' % errors)

    def _checkLandmarks(self, prot):
        out = getattr(prot, ParapickLocalizeOutputs.landmarkModels.name)
        self.assertEqual(out.getSize(), len(TS_IDS))
        for model in out:
            tsId = model.getTsId()
            self.assertFalse(model.applyTSTransformation())
            rows = model.retrieveInfoTable()
            chains = {int(r[3]) for r in rows}
            self.assertEqual(len(chains), N_PARTICLES)
            # Each chain follows the projections of one particle on the raw tilt images
            self.assertGreater(len(rows), 0.9 * RECALL * N_PARTICLES * len(ANGLES))
            for chain in chains:
                landmarks = [r for r in rows if int(r[3]) == chain]
                indices = [int(r[2]) for r in landmarks]
                self.assertEqual(len(indices), len(set(indices)), 'one pick per tilt')
                projections = np.array([self._projections(tsId, i) for i in indices])
                xy = np.array([[float(r[0]), float(r[1])] for r in landmarks])
                distances = np.linalg.norm(projections - xy[:, None, :], axis=-1)
                particle = np.argmin(distances.mean(axis=0))
                self.assertLess(distances[:, particle].max(), 1.5)

    def _checkCoordinates3D(self, prot):
        out = getattr(prot, ParapickLocalizeOutputs.coordinates3D.name)
        self.assertEqual(out.getBoxSize(), 5)
        for tsId in TS_IDS:
            positions = [np.array(c.getPosition(tomoConst.SCIPION)) * TOMO_SAMPLING / TS_SAMPLING
                         for c in out.iterCoordinates(volume=None) if c.getTomoId() == tsId]
            errors, _ = self._matchTruth(tsId, positions)
            self.assertLess(errors.max(), 1.0)
