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
"""Tests of the statistics behind the localize viewer (no GUI, no project)."""

import json
import os
import tempfile
import unittest

import numpy as np

from parapick.convert import geometry, star
from parapick.viewers.stats import LocalizeStats, binAngles


def _writeRun(extra):
    """ A fake run: 2 tilt-series of 3 tilts, 2 particles each. """
    inputDir = os.path.join(extra, 'parapick_input')
    batch = os.path.join(extra, 'batch_0000')
    os.makedirs(inputDir)
    os.makedirs(batch)
    angles = [-3.0, 0.02, 3.0]
    geos = [star.TiltSeriesGeometry(tsId, 2.0, (100, 100), (100, 100, 50),
                                    [('%s_%03d' % (tsId, i + 1),
                                      geometry.RelionTilt(0, a, 0, 0, 0))
                                     for i, a in enumerate(angles)], 300, 2.7, 0.1)
            for tsId in ('TS_1', 'TS_2')]
    star.writeTomogramsStar(os.path.join(inputDir, 'tomograms.star'), geos)
    report = {'tilt_series': [
        {'tilt_series': tsId, 'n_tilts': 3, 'n_picks': 6, 'iterations': 10,
         'converged': tsId == 'TS_1', 'background_fraction': 0.1, 'elapsed_seconds': 1,
         'tilts': [{'tilt_id': '%s_%03d' % (tsId, i + 1), 'n_picks': n, 'sigma_angst': s}
                   for i, (n, s) in enumerate([(2, 5.0), (2, 4.0), (0, 99.0)])]}
        for tsId in ('TS_1', 'TS_2')]}
    with open(os.path.join(batch, 'localize_report.json'), 'w') as f:
        json.dump(report, f)
    with open(os.path.join(batch, 'particles.star'), 'w') as f:
        f.write('data_particles\nloop_\n_rlnTomoName\n_rlnCenteredCoordinateXAngst\n'
                '_rlnCenteredCoordinateYAngst\n_rlnCenteredCoordinateZAngst\n'
                '_rlnAutopickFigureOfMerit\n_ppkEffectiveCount\n_ppkParticleId\n'
                'TS_1 0 0 0 0.66 2 1\nTS_1 10 0 0 0.33 1 2\n'
                'TS_2 0 0 60 0.66 2 1\nTS_2 0 0 0 0.66 2 2\n')
    with open(os.path.join(batch, 'assignments.star'), 'w') as f:
        f.write('data_assignments\nloop_\n_rlnTomoName\n_ppkParticleId\n_rlnMicrographName\n'
                '_rlnCoordinateX\n_rlnCoordinateY\n_ppkPickIndex\n_ppkResponsibility\n'
                '_ppkResidualXAngst\n_ppkResidualYAngst\n'
                'TS_1 1 TS_1_001 1 1 0 0.9 3 4\nTS_1 1 TS_1_002 1 1 1 0.9 0 1\n'
                'TS_1 2 TS_1_001 1 1 2 0.9 0 0\n')


class TestLocalizeStats(unittest.TestCase):
    def test_statistics(self):
        with tempfile.TemporaryDirectory() as extra:
            _writeRun(extra)
            stats = LocalizeStats(extra)
            angles, sigmas = stats.perTilt('sigma')
            # Tilts without picks are excluded (their sigma is not fitted)
            np.testing.assert_allclose(sorted(sigmas), [4, 4, 5, 5])
            angles, detection = stats.perTilt('detection')
            self.assertEqual(sorted(detection.tolist()), [0, 0, 0, 0, 0.5, 1.0])
            _, residuals = stats.residualsPerTilt()
            np.testing.assert_allclose(sorted(residuals), [0, 1, 5])
            np.testing.assert_allclose(stats.chainLengths(), [2, 1, 0, 0])
            # TS_2 particle 1 is at z = 60 Å, outside the 100 Å thick volume
            np.testing.assert_array_equal(stats.insideVolume(), [True, True, False, True])
            self.assertEqual([t.converged for t in stats.tiltSeries], [True, False])
            self.assertAlmostEqual(stats.halfThickness(), 50.0)

    def test_angle_bins(self):
        centres, groups = binAngles([-3.01, -2.98, 0.02, 3.0], [1, 2, 3, 4], 1.0)
        np.testing.assert_allclose(centres, [-3, 0, 3])
        self.assertEqual([g.tolist() for g in groups], [[1, 2], [3], [4]])
