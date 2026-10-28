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

A :class:`SetOfTiltSeriesMicrographs` is a regular ``SetOfMicrographs``, so it can
be fed to any single-particle picker. Each item records the tilt-series id, the stack
index, the acquisition order and the tilt angle of its source tilt image, and the
set points to the source ``SetOfTiltSeries``.
"""

import pwem.objects as emobj
from pyworkflow.object import Float, Integer, Pointer, String

TS_ID_FIELD = '_tsId'          # Same name as TiltImage, for compatibility
ACQ_ORDER_FIELD = '_acqOrder'  # Same name as TiltImage, for compatibility
TILT_INDEX_FIELD = '_tiltIndex'
TILT_ANGLE_FIELD = '_tiltAngle'


class TiltSeriesMicrograph(emobj.Micrograph):
    """ A micrograph extracted from a tilt image of a tilt-series. """

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
    """ Micrographs extracted from the tilt images of a set of tilt-series. """
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
