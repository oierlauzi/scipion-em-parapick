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
"""Conversion of tilt-series CTFs to single-particle CTFs (no Scipion project needed)."""

import unittest

from tomo.objects import CTFTomo

from parapick.objects import TiltCtfLookup, ctfFromTiltCtf


def _tiltCtf(index, acqOrder, defocusU=20000.0, defocusV=21000.0, angle=30.0, enabled=True):
    ctf = CTFTomo(index=index, acqOrder=acqOrder)
    ctf.setDefocusU(defocusU)
    ctf.setDefocusV(defocusV)
    ctf.setDefocusAngle(angle)
    ctf.setResolution(6.5)
    ctf.setFitQuality(0.8)
    ctf.setPsdFile('%d@psd.mrcs' % index)
    ctf.setEnabled(enabled)
    return ctf


class _Series:
    """ Stands in for a CTFTomoSeries. """

    def __init__(self, tsId, ctfs):
        self._tsId, self._ctfs = tsId, ctfs

    def getTsId(self):
        return self._tsId

    def iterItems(self):
        return iter(self._ctfs)


class _Set:
    """ Stands in for a SetOfCTFTomoSeries. """

    def __init__(self, *series):
        self._series = series

    def iterItems(self):
        return iter(self._series)


class _TiltImage:
    def __init__(self, tsId, index, acqOrder):
        self._tsId, self._index, self._acqOrder = tsId, index, acqOrder

    def getTsId(self):
        return self._tsId

    def getIndex(self):
        return self._index

    def getAcquisitionOrder(self):
        return self._acqOrder


class TestTiltCtf(unittest.TestCase):
    def test_values_are_copied_and_standardized(self):
        ctf = ctfFromTiltCtf(_tiltCtf(3, 5))
        self.assertEqual(ctf.getClassName(), 'CTFModel')
        # EMX convention: defocusU >= defocusV, angle rotated by 90 when swapped
        self.assertEqual(ctf.getDefocus(), (21000.0, 20000.0, 120.0))
        self.assertAlmostEqual(ctf.getDefocusRatio(), 21000.0 / 20000.0)
        self.assertEqual(ctf.getResolution(), 6.5)
        self.assertEqual(ctf.getFitQuality(), 0.8)
        self.assertEqual(ctf.getPsdFile(), '3@psd.mrcs')
        self.assertFalse(ctf.hasPhaseShift())

    def test_phase_shift_is_kept(self):
        tiltCtf = _tiltCtf(1, 1)
        tiltCtf.setPhaseShift(45.0)
        self.assertEqual(ctfFromTiltCtf(tiltCtf).getPhaseShift(), 45.0)

    def test_missing_defocus_gives_none(self):
        tiltCtf = CTFTomo(index=1, acqOrder=1)
        self.assertIsNone(ctfFromTiltCtf(tiltCtf))

    def test_lookup_matches_by_acquisition_order(self):
        lookup = TiltCtfLookup(_Set(_Series('TS_01', [
            _tiltCtf(1, 2, defocusU=10000.0, defocusV=10000.0),
            _tiltCtf(2, 1, defocusU=30000.0, defocusV=30000.0),
            _tiltCtf(3, 3, enabled=False),
        ])))
        self.assertEqual(lookup.getTsIds(), {'TS_01'})
        # Index 1 was acquired second: matched by acquisition order, not by index
        self.assertEqual(lookup.get(_TiltImage('TS_01', 1, 2)).getDefocusU(), 10000.0)
        self.assertEqual(lookup.get(_TiltImage('TS_01', 2, 1)).getDefocusU(), 30000.0)
        self.assertIsNone(lookup.get(_TiltImage('TS_01', 3, 3)))  # disabled
        self.assertIsNone(lookup.get(_TiltImage('TS_02', 1, 1)))  # no CTF series
        # Each call returns a new object, free to be given a micrograph
        first = _TiltImage('TS_01', 1, 2)
        self.assertIsNot(lookup.get(first), lookup.get(first))

    def test_lookup_falls_back_to_index(self):
        lookup = TiltCtfLookup(_Set(_Series('TS_01', [
            _tiltCtf(1, None, defocusU=10000.0, defocusV=10000.0),
            _tiltCtf(2, None, defocusU=30000.0, defocusV=30000.0),
        ])))
        self.assertEqual(lookup.get(_TiltImage('TS_01', 2, 7)).getDefocusU(), 30000.0)


if __name__ == '__main__':
    unittest.main()
