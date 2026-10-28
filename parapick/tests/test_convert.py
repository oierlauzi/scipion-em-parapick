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
"""Unit tests of the conversions (no Scipion project or ParaPick needed)."""

import math
import os
import tempfile
import unittest

import numpy as np

from parapick.convert import geometry, star
from parapick.convert.images import binImage, binnedShape, fourierCrop, readSection, writeMrc
from parapick.convert.naming import batches, getTiltId, parseTiltId


def _alignment(rng, scale=1.0):
    psi = math.radians(rng.uniform(-180, 180))
    shift = rng.normal(0, 20, 2)
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[scale * c, -s, shift[0]], [s, scale * c, shift[1]], [0, 0, 1]])


class TestGeometry(unittest.TestCase):
    def test_relion_matches_scipion_projection(self):
        """ RELION 5 geometry reproduces Scipion's projection of centred points. """
        rng = np.random.default_rng(1)
        pixelSize = 2.5
        for imageSize in [(512, 400), (511, 401), (927, 1024)]:
            volume = [geometry.evenSize(imageSize[0]), geometry.evenSize(imageSize[1]),
                      geometry.evenSize(301)]
            for _ in range(20):
                matrix = _alignment(rng)
                tilt = rng.uniform(-60, 60)
                points = rng.uniform(-150, 150, (6, 3))
                expected = geometry.projectScipion(points, matrix, tilt, imageSize)
                relion = geometry.tiltImageToRelion(matrix, tilt, pixelSize, imageSize)
                actual = geometry.projectRelion(points * pixelSize, relion, pixelSize,
                                                imageSize, volume)
                np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_identity_alignment(self):
        relion = geometry.tiltImageToRelion(None, 30.0, 2.0, (100, 101))
        self.assertAlmostEqual(relion.zRot, 0.0)
        self.assertAlmostEqual(relion.yTilt, 30.0)
        self.assertAlmostEqual(relion.xShiftAngst, 0.0)
        self.assertAlmostEqual(relion.yShiftAngst, 1.0)  # (101/2 - 50) px * 2 Å/px

    def test_rigidity(self):
        rng = np.random.default_rng(2)
        self.assertTrue(geometry.isRigid(_alignment(rng)))
        self.assertTrue(geometry.isRigid(None))
        self.assertFalse(geometry.isRigid(_alignment(rng, scale=1.05)))

    def test_aligned_extent(self):
        np.testing.assert_allclose(geometry.alignedExtent((400, 300), 0), (400, 300))
        np.testing.assert_allclose(geometry.alignedExtent((400, 300), -90), (300, 400), atol=1e-9)

    def test_even_size(self):
        self.assertEqual(geometry.evenSize(301), 302)
        self.assertEqual(geometry.evenSize(300), 300)
        self.assertEqual(geometry.evenSize(299.2), 300)


class TestNaming(unittest.TestCase):
    def test_round_trip(self):
        for tsId in ['TS_01', '001', 'a_b_c', 'TS.43']:
            self.assertEqual(parseTiltId(getTiltId(tsId, 7)), (tsId, 7))
        self.assertEqual(parseTiltId('/data/TS_01_012.mrc'), ('TS_01', 12))
        self.assertEqual(parseTiltId('3@TS_01_012.mrcs'), ('TS_01', 12))
        self.assertIsNone(parseTiltId('TS_01'))

    def test_batches(self):
        self.assertEqual(batches('abcde', 2), [['a', 'b'], ['c', 'd'], ['e']])
        self.assertEqual(batches('abc', 0), [['a', 'b', 'c']])
        self.assertEqual(batches([], 3), [])


class TestImages(unittest.TestCase):
    def test_binning_scales_coordinates(self):
        """ A blob at (x, y) moves to (x, y) / binning and the mean is kept. """
        height, width = 120, 151
        yy, xx = np.mgrid[0:height, 0:width]
        x0, y0 = 60.0, 46.0
        image = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * 6.0 ** 2))
        self.assertEqual(binnedShape((height, width), 2), (60, 75))
        out = binImage(image, 2)
        self.assertEqual(out.shape, (60, 75))
        self.assertAlmostEqual(out.mean(), image[:, :150].mean(), places=5)
        y, x = np.unravel_index(np.argmax(out), out.shape)
        self.assertEqual((x, y), (30, 23))
        np.testing.assert_allclose(fourierCrop(image, image.shape), image, atol=1e-6)

    def test_mrc_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            fn = os.path.join(tmp, 'mic.mrc')
            data = np.arange(12, dtype=np.float32).reshape(3, 4)
            writeMrc(fn, data, 3.5)
            np.testing.assert_array_equal(readSection(fn + ':mrc', 1), data)


class TestStar(unittest.TestCase):
    def test_tomograms_and_picks(self):
        with tempfile.TemporaryDirectory() as tmp:
            tilt = geometry.RelionTilt(0.0, -3.0, 85.0, 1.5, -2.5)
            geo = star.TiltSeriesGeometry('001', 1.35, (100, 90), (100, 90, 300),
                                          [('001_001', tilt)], 300.0, 2.7, 0.1)
            fn = os.path.join(tmp, 'tomograms.star')
            star.writeTomogramsStar(fn, [geo])
            rows = star.readStarTable(fn, 'global')
            self.assertEqual(rows[0]['rlnTomoName'], '001')  # Not parsed as a number
            self.assertEqual(rows[0]['rlnTomoSizeZ'], '300')
            tsRows = star.readStarTable(os.path.join(tmp, rows[0]['rlnTomoTiltSeriesStarFile']),
                                        '001')
            self.assertAlmostEqual(float(tsRows[0]['rlnTomoZRot']), 85.0)

            picksFn = os.path.join(tmp, 'picks.star')
            star.writePicksStar(picksFn, [star.Pick('001', '001_001', 10, 20, None),
                                          star.Pick('001', '001_001', 11, 21, 0.5)], 2.7)
            picks = star.readStarTable(picksFn, 'picks')
            self.assertEqual(len(picks), 2)
            self.assertNotIn('rlnAutopickFigureOfMerit', picks[0])
            self.assertEqual(picks[1]['rlnMicrographPixelSize'], '2.700000')

    def test_read_parapick_outputs(self):
        """ Files as written by ParaPick (starfile). """
        with tempfile.TemporaryDirectory() as tmp:
            fn = os.path.join(tmp, 'particles.star')
            with open(fn, 'w') as f:
                f.write('# Created by the starfile Python package\n\ndata_particles\n\nloop_\n'
                        '_rlnTomoName #1\n_rlnCenteredCoordinateXAngst #2\n'
                        '_rlnCenteredCoordinateYAngst #3\n_rlnCenteredCoordinateZAngst #4\n'
                        '_rlnAutopickFigureOfMerit #5\n_ppkEffectiveCount #6\n'
                        '_ppkParticleId #7\n'
                        'TS_01\t1.5\t-2.0\t30.25\t0.9\t36.0\t4\n')
            particles = star.readParticlesStar(fn)
            self.assertEqual(particles[0], star.Particle('TS_01', 4, 1.5, -2.0, 30.25, 0.9, 36.0))
