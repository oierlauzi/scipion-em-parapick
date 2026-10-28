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
"""Minimal STAR reading and writing for the files exchanged with ParaPick.

Values are kept as strings when read (no type guessing), so identifiers such as
``001`` survive the round trip.
"""

import os
import shlex
from collections import namedtuple

TiltSeriesGeometry = namedtuple('TiltSeriesGeometry', [
    'tsId',         # name of the tilt-series (rlnTomoName)
    'pixelSize',    # Å/px of the tilt-series
    'imageSize',    # (W, H) of the tilt images, pixels
    'volumeSize',   # (X, Y, Z) of the volume to localize in, pixels (even)
    'tilts',        # list of (tiltId, geometry.RelionTilt)
    'voltage', 'sphericalAberration', 'amplitudeContrast',
])

Pick = namedtuple('Pick', ['tsId', 'tiltId', 'x', 'y', 'score'])
""" A 2D pick in pixels of the picked image (x = column, y = row, from the first pixel). """

TILT_SERIES_DIR = 'tilt_series'


# ------------------------------------------------------------------------- writing
def _format(value):
    if isinstance(value, float):
        return '%.6f' % value
    text = str(value)
    if not text or any(c.isspace() for c in text) or text[0] in '"\'_#' or text.startswith('data_'):
        return '"%s"' % text.replace('"', "'")
    return text


def _writeLoop(f, block, columns, rows):
    f.write('\ndata_%s\n\nloop_\n' % block)
    for i, column in enumerate(columns, start=1):
        f.write('_%s #%d\n' % (column, i))
    for row in rows:
        f.write(' '.join(_format(v) for v in row) + '\n')
    f.write('\n')


def writeTomogramsStar(path, geometries):
    """ Write ``tomograms.star`` and one ``tilt_series/<tsId>.star`` per tilt-series
    (paths in ``tomograms.star`` are relative to it). """
    root = os.path.dirname(os.path.abspath(path))
    os.makedirs(os.path.join(root, TILT_SERIES_DIR), exist_ok=True)
    rows = []
    for geo in geometries:
        tsStar = os.path.join(TILT_SERIES_DIR, '%s.star' % geo.tsId)
        with open(os.path.join(root, tsStar), 'w') as f:
            _writeLoop(f, geo.tsId,
                       ['rlnMicrographName', 'rlnTomoXTilt', 'rlnTomoYTilt', 'rlnTomoZRot',
                        'rlnTomoXShiftAngst', 'rlnTomoYShiftAngst'],
                       [(tiltId, t.xTilt, t.yTilt, t.zRot, t.xShiftAngst, t.yShiftAngst)
                        for tiltId, t in geo.tilts])
        rows.append((geo.tsId, tsStar, float(geo.pixelSize),
                     int(geo.volumeSize[0]), int(geo.volumeSize[1]), int(geo.volumeSize[2]),
                     int(geo.imageSize[0]), int(geo.imageSize[1]),
                     float(geo.voltage), float(geo.sphericalAberration),
                     float(geo.amplitudeContrast)))
    with open(path, 'w') as f:
        _writeLoop(f, 'global',
                   ['rlnTomoName', 'rlnTomoTiltSeriesStarFile', 'rlnTomoTiltSeriesPixelSize',
                    'rlnTomoSizeX', 'rlnTomoSizeY', 'rlnTomoSizeZ',
                    'ppkTiltImageSizeX', 'ppkTiltImageSizeY',
                    'rlnVoltage', 'rlnSphericalAberration', 'rlnAmplitudeContrast'],
                   rows)


def writePicksStar(path, picks, pixelSize):
    """ Write a ParaPick picks STAR file (``data_picks``). ``pixelSize`` is the pixel
    size of the picked images. Scores are written only if every pick has one. """
    picks = list(picks)
    withScores = bool(picks) and all(p.score is not None for p in picks)
    columns = ['rlnTomoName', 'rlnMicrographName', 'rlnCoordinateX', 'rlnCoordinateY',
               'rlnMicrographPixelSize']
    if withScores:
        columns.append('rlnAutopickFigureOfMerit')
    rows = []
    for p in picks:
        row = [p.tsId, p.tiltId, float(p.x), float(p.y), float(pixelSize)]
        if withScores:
            row.append(float(p.score))
        rows.append(row)
    with open(path, 'w') as f:
        _writeLoop(f, 'picks', columns, rows)


# ------------------------------------------------------------------------- reading
def readStarBlocks(path):
    """ Read every block of a STAR file as ``{name: [ {label: str} ]}``. Key-value
    blocks are returned as a single row. """
    blocks = {}
    block = None
    labels = []
    inLoop = False
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('data_'):
                block = blocks.setdefault(line[5:], [])
                labels, inLoop = [], False
            elif line == 'loop_':
                labels, inLoop = [], True
            elif line.startswith('_'):
                parts = line[1:].split(None, 1)
                if inLoop:
                    labels.append(parts[0])
                else:
                    if not block:
                        block.append({})
                    block[0][parts[0]] = _unquote(parts[1].split('#')[0].strip()
                                                  if len(parts) > 1 else '')
            elif inLoop and block is not None:
                values = shlex.split(line, comments=False, posix=True)
                block.append(dict(zip(labels, values)))
    return blocks


def _unquote(text):
    return shlex.split(text)[0] if text else text


def readStarTable(path, block):
    """ Rows of one block (a list of ``{label: str}``); an empty list if missing. """
    return readStarBlocks(path).get(block, [])


Particle = namedtuple('Particle', ['tsId', 'particleId', 'x', 'y', 'z', 'score', 'support'])
""" A localized particle: centred coordinates in Å. """

Assignment = namedtuple('Assignment', [
    'tsId', 'particleId', 'tiltId', 'x', 'y', 'pickIndex', 'responsibility',
    'residualX', 'residualY'])
""" A pick explained by a particle: coordinates in pixels of the picked image,
residuals in Å (image frame). """


def _float(row, label, default=None):
    value = row.get(label)
    return default if value is None else float(value)


def readParticlesStar(path):
    particles = []
    for i, row in enumerate(readStarTable(path, 'particles'), start=1):
        particles.append(Particle(
            tsId=row['rlnTomoName'],
            particleId=int(row.get('ppkParticleId', i)),
            x=float(row['rlnCenteredCoordinateXAngst']),
            y=float(row['rlnCenteredCoordinateYAngst']),
            z=float(row['rlnCenteredCoordinateZAngst']),
            score=_float(row, 'rlnAutopickFigureOfMerit'),
            support=_float(row, 'ppkEffectiveCount')))
    return particles


def readAssignmentsStar(path):
    return [Assignment(tsId=row['rlnTomoName'],
                       particleId=int(row['ppkParticleId']),
                       tiltId=row['rlnMicrographName'],
                       x=float(row['rlnCoordinateX']),
                       y=float(row['rlnCoordinateY']),
                       pickIndex=int(row.get('ppkPickIndex', -1)),
                       responsibility=_float(row, 'ppkResponsibility'),
                       residualX=_float(row, 'ppkResidualXAngst', 0.0),
                       residualY=_float(row, 'ppkResidualYAngst', 0.0))
            for row in readStarTable(path, 'assignments')]
