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
"""Micrographs that remember the tilt image they come from.

The micrographs unstacked from tilt-series are a plain ``SetOfMicrographs`` of plain
``Micrograph`` items, so that any single-particle picker and viewer accepts them: some
check the class names (the Xmipp picker and viewer only accept items of class
``Micrograph`` in a ``SetOfMicrographs``). What ParaPick needs is stored as extra
attributes, which Scipion saves and loads like any other:

* on each micrograph, the tilt-series id, the stack index, the acquisition order and the
  tilt angle of its source tilt image (:func:`setTiltImageInfo`);
* on the set, a pointer to the source ``SetOfTiltSeries`` (:func:`setTiltSeriesPointer`).

:class:`TiltCtfLookup` turns the per-tilt CTFs of a ``SetOfCTFTomoSeries`` into the
single-particle ``CTFModel`` of each micrograph.

:class:`TiltSeriesMicrograph` and :class:`SetOfTiltSeriesMicrographs` carry the same
attributes under the same names. They are kept only so that projects made by earlier
versions still load. New outputs do not use them.
"""

import pwem.objects as emobj
from pyworkflow.object import Float, Integer, Pointer, String

TS_ID_FIELD = '_tsId'          # Same name as TiltImage, for compatibility
ACQ_ORDER_FIELD = '_acqOrder'  # Same name as TiltImage, for compatibility
TILT_INDEX_FIELD = '_tiltIndex'
TILT_ANGLE_FIELD = '_tiltAngle'
TILT_SERIES_FIELD = '_tiltSeriesPointer'


def _getValue(obj, field):
    attr = getattr(obj, field, None)
    return None if attr is None else attr.get()


def setTiltImageInfo(micrograph, tiltImage):
    """ Record on a micrograph the identity of the tilt image it comes from. """
    setattr(micrograph, TS_ID_FIELD, String(tiltImage.getTsId()))
    setattr(micrograph, TILT_INDEX_FIELD, Integer(tiltImage.getIndex()))
    setattr(micrograph, ACQ_ORDER_FIELD, Integer(tiltImage.getAcquisitionOrder()))
    setattr(micrograph, TILT_ANGLE_FIELD, Float(tiltImage.getTiltAngle()))


def getTsId(micrograph):
    """ Tilt-series id of the source tilt image, or None if not recorded. """
    return _getValue(micrograph, TS_ID_FIELD)


def getTiltIndex(micrograph):
    """ 1-based index of the source tilt image in its stack (``TiltImage.getIndex``), as
    used by landmark models, or None if not recorded. """
    return _getValue(micrograph, TILT_INDEX_FIELD)


def getAcquisitionOrder(micrograph):
    return _getValue(micrograph, ACQ_ORDER_FIELD)


def getTiltAngle(micrograph):
    return _getValue(micrograph, TILT_ANGLE_FIELD)


def setTiltSeriesPointer(micrographs, tiltSeries):
    """ Record on a set of micrographs the SetOfTiltSeries (or a pointer to it) that they
    were unstacked from. """
    pointer = Pointer()
    if tiltSeries.isPointer():
        pointer.copy(tiltSeries, copyId=False)
    else:
        pointer.set(tiltSeries)
    setattr(micrographs, TILT_SERIES_FIELD, pointer)


def getTiltSeriesPointer(micrographs):
    """ Pointer to the SetOfTiltSeries the micrographs were unstacked from, or None. """
    pointer = getattr(micrographs, TILT_SERIES_FIELD, None)
    return pointer if pointer is not None and pointer.hasValue() else None


def ctfFromTiltCtf(tiltCtf):
    """ A single-particle ``CTFModel`` with the values of a tilt image's ``CTFTomo``.

    Defocus U/V/angle (standardized, as SPA programs expect), phase shift, PSD file,
    resolution and fit quality are copied. The defocus is the one estimated for the
    whole tilt image: the defocus gradient across a tilted image is not modelled.
    Returns None if the tilt image has no defocus estimate.
    """
    defocusU, defocusV, defocusAngle = (tiltCtf.getDefocusU(), tiltCtf.getDefocusV(),
                                        tiltCtf.getDefocusAngle())
    if defocusU is None or defocusV is None or defocusAngle is None:
        return None
    ctf = emobj.CTFModel()
    ctf.setStandardDefocus(defocusU, defocusV, defocusAngle)
    if tiltCtf.hasPhaseShift():
        ctf.setPhaseShift(tiltCtf.getPhaseShift())
    if tiltCtf.getPsdFile():
        ctf.setPsdFile(tiltCtf.getPsdFile())
    if tiltCtf.getResolution() is not None:
        ctf.setResolution(tiltCtf.getResolution())
    if tiltCtf.getFitQuality() is not None:
        ctf.setFitQuality(tiltCtf.getFitQuality())
    return ctf


class TiltCtfLookup:
    """ The CTF of each tilt image, from a ``SetOfCTFTomoSeries``.

    A tilt image is matched to its CTF by acquisition order or, for CTF series that do
    not record it, by index, as ``CTFTomoSeries.getCtfTomoFromTi`` does. Disabled CTFs
    and CTFs without a defocus estimate are left out.
    """

    def __init__(self, ctfSeriesSet):
        self._series = {}  # tsId -> (key field, {key value: CTFModel})
        for series in ctfSeriesSet.iterItems():
            tsId = series.getTsId()  # Read first: iterated items are reused objects
            byAcqOrder, byIndex = {}, {}
            for tiltCtf in series.iterItems():
                ctf = ctfFromTiltCtf(tiltCtf) if tiltCtf.isEnabled() else None
                acqOrder, index = tiltCtf.getAcquisitionOrder(), tiltCtf.getIndex()
                if acqOrder is not None:
                    byAcqOrder[acqOrder] = ctf
                if index is not None:
                    byIndex[index] = ctf
            self._series[tsId] = ((ACQ_ORDER_FIELD, byAcqOrder) if byAcqOrder
                                  else (TILT_INDEX_FIELD, byIndex))

    def getTsIds(self):
        return set(self._series)

    def get(self, tiltImage):
        """ A new CTFModel for the tilt image, or None if it has no (enabled) CTF. """
        field, ctfs = self._series.get(tiltImage.getTsId(), (None, {}))
        key = (tiltImage.getAcquisitionOrder() if field == ACQ_ORDER_FIELD
               else tiltImage.getIndex())
        ctf = ctfs.get(key)
        return None if ctf is None else ctf.clone()


class TiltSeriesMicrograph(emobj.Micrograph):
    """ A micrograph extracted from a tilt image of a tilt-series.

    Deprecated: only for projects made by earlier versions (see the module docstring).
    """

    def __init__(self, location=None, **kwargs):
        super().__init__(location, **kwargs)
        self._tsId = String()
        self._tiltIndex = Integer()
        self._acqOrder = Integer()
        self._tiltAngle = Float()

    def getTsId(self):
        return self._tsId.get()

    def setTsId(self, tsId):
        self._tsId.set(tsId)

    def getTiltIndex(self):
        """ 1-based index of the source tilt image in its stack (``TiltImage.getIndex``),
        as used by landmark models. """
        return self._tiltIndex.get()

    def setTiltIndex(self, index):
        self._tiltIndex.set(index)

    def getAcquisitionOrder(self):
        return self._acqOrder.get()

    def setAcquisitionOrder(self, acqOrder):
        self._acqOrder.set(acqOrder)

    def getTiltAngle(self):
        return self._tiltAngle.get()

    def setTiltAngle(self, angle):
        self._tiltAngle.set(angle)

    def copyFromTiltImage(self, tiltImage):
        """ Record the identity of the source tilt image. """
        self.setTsId(tiltImage.getTsId())
        self.setTiltIndex(tiltImage.getIndex())
        self.setAcquisitionOrder(tiltImage.getAcquisitionOrder())
        self.setTiltAngle(tiltImage.getTiltAngle())


class SetOfTiltSeriesMicrographs(emobj.SetOfMicrographs):
    """ Micrographs extracted from the tilt images of a set of tilt-series.

    Deprecated: only for projects made by earlier versions (see the module docstring).
    """
    ITEM_TYPE = TiltSeriesMicrograph

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._tiltSeriesPointer = Pointer()

    def getTiltSeries(self, asPointer=False):
        """ Source SetOfTiltSeries (or a pointer to it). """
        return self._tiltSeriesPointer if asPointer else self._tiltSeriesPointer.get()

    def setTiltSeries(self, tiltSeries):
        """ :param tiltSeries: a SetOfTiltSeries or a pointer to it. """
        if tiltSeries.isPointer():
            self._tiltSeriesPointer.copy(tiltSeries, copyId=False)
        else:
            self._tiltSeriesPointer.set(tiltSeries)

    def copyInfo(self, other):
        super().copyInfo(other)
        if isinstance(other, SetOfTiltSeriesMicrographs):
            self._tiltSeriesPointer.copy(other._tiltSeriesPointer, copyId=False)

    def getTSIds(self):
        return self.getUniqueValues(TS_ID_FIELD)

    def iterTiltSeriesMicrographs(self, tsId, **kwargs):
        return self.iterItems(where='%s="%s"' % (TS_ID_FIELD, tsId), **kwargs)
