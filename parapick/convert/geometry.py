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
"""Tilt-series geometry: Scipion <-> RELION 5 (the convention read by ParaPick).

**Scipion** (see ``tomo.protocols.ProtTomoProjectCoordinates``). A point ``X``, in
pixels of the tilt-series and relative to the centre of the reconstructed volume,
projects onto the *raw* (non-aligned) tilt image as::

    y = inv(M) @ [cos(t) X + sin(t) Z, Y, 1] + (W/2, H/2)

where ``M`` is the 3x3 alignment matrix of the tilt image and ``t`` its tilt angle.

**RELION 5** (as implemented by ParaPick)::

    y = floor(S/2) + shift_A / p + [Rz(ZRot) Ry(YTilt) Rx(XTilt) (X + o)]_xy

with ``o = V/2 - floor(V/2)`` for a volume of ``V`` pixels. Both models agree when:

* ``XTilt = 0``, ``YTilt = t`` and ``Rz(ZRot)`` is the linear part of ``inv(M)``;
* ``shift_A = (inv(M)[:2, 2] + S/2 - floor(S/2)) * p``;
* the declared volume size is even (``o = 0``), see :func:`evenSize`.

This requires ``inv(M)`` to be a rotation. Other alignments (e.g. IMOD stretching)
are approximated by their closest rotation, see :func:`isRigid`.
"""

import math
from collections import namedtuple

import numpy as np

RelionTilt = namedtuple('RelionTilt', ['xTilt', 'yTilt', 'zRot', 'xShiftAngst', 'yShiftAngst'])


def _inverseAlignment(matrix):
    if matrix is None:
        return np.eye(3)
    return np.linalg.inv(np.asarray(matrix, dtype=float))


def _closestRotation(linear):
    u, _, vt = np.linalg.svd(linear)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def isRigid(matrix, tolerance=1e-3):
    """ Whether a tilt image alignment matrix is a rotation plus a shift. """
    linear = _inverseAlignment(matrix)[:2, :2]
    return (np.allclose(linear @ linear.T, np.eye(2), atol=tolerance)
            and np.linalg.det(linear) > 0)


def tiltImageToRelion(matrix, tiltAngle, pixelSize, imageSize):
    """ RELION 5 per-tilt alignment of a Scipion tilt image.

    :param matrix: 3x3 alignment matrix of the tilt image (``None`` for identity).
    :param tiltAngle: tilt angle in degrees.
    :param pixelSize: tilt-series pixel size (Å/px).
    :param imageSize: ``(W, H)`` of the tilt images in pixels.
    :return: :class:`RelionTilt` (angles in degrees, shifts in Å).
    """
    inverse = _inverseAlignment(matrix)
    rotation = _closestRotation(inverse[:2, :2])
    size = np.asarray(imageSize, dtype=float)
    shift = (inverse[:2, 2] + size / 2.0 - np.floor(size / 2.0)) * pixelSize
    zRot = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
    return RelionTilt(0.0, float(tiltAngle), zRot, float(shift[0]), float(shift[1]))


def alignedExtent(imageSize, zRot):
    """ ``(X, Y)`` extent (px) of a ``(W, H)`` tilt image rotated by ``zRot`` degrees,
    i.e. the footprint of the raw image in the aligned (tomogram) frame. """
    c, s = abs(math.cos(math.radians(zRot))), abs(math.sin(math.radians(zRot)))
    width, height = imageSize
    return c * width + s * height, s * width + c * height


def evenSize(size):
    """ Smallest even integer >= ``size``, so the volume centre is a pixel corner in
    both conventions. """
    return int(2 * math.ceil(float(size) / 2.0 - 1e-9))


def _rotation(axis, degrees):
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def projectScipion(points, matrix, tiltAngle, imageSize):
    """ Project centred points (px, shape (K, 3)) onto a raw tilt image, as Scipion
    does. Returns (K, 2) pixel coordinates. """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    t = math.radians(tiltAngle)
    projected = np.stack([math.cos(t) * points[:, 0] + math.sin(t) * points[:, 2],
                          points[:, 1], np.ones(len(points))], axis=-1)
    raw = projected @ _inverseAlignment(matrix).T
    return raw[:, :2] + np.asarray(imageSize, dtype=float) / 2.0


def projectRelion(points, tilt, pixelSize, imageSize, volumeSize):
    """ Project points (centred Å, shape (K, 3)) with the RELION 5 model. Returns
    (K, 2) pixel coordinates. ``volumeSize`` is in pixels. """
    points = np.atleast_2d(np.asarray(points, dtype=float)) / pixelSize
    volume = np.asarray(volumeSize, dtype=float)
    offset = volume / 2.0 - np.floor(volume / 2.0)
    rotation = _rotation('z', tilt.zRot) @ _rotation('y', tilt.yTilt) @ _rotation('x', tilt.xTilt)
    projected = (points + offset) @ rotation.T
    shift = np.array([tilt.xShiftAngst, tilt.yShiftAngst]) / pixelSize
    return np.floor(np.asarray(imageSize, dtype=float) / 2.0) + shift + projected[:, :2]
