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
"""Names that link micrographs, files and ParaPick tilt identifiers to tilt images."""

import re

TILT_ID_PATTERN = re.compile(r'^(?P<tsId>.+)_(?P<index>\d{3,})$')


def getTiltId(tsId, tiltIndex):
    """ Canonical identifier of a tilt image: ``<tsId>_<index>``, with the 1-based
    stack index of the tilt image zero-padded to 3 digits. It is used as micrograph
    name, file basename and ParaPick tilt id, so pickers that match micrographs by
    basename and ParaPick's file-stem matching both work. """
    return '%s_%03d' % (tsId, int(tiltIndex))


def parseTiltId(name):
    """ Inverse of :func:`getTiltId`. Accepts paths and extensions.

    :return: ``(tsId, tiltIndex)`` or ``None`` if ``name`` does not follow the pattern.
    """
    base = name.replace('\\', '/').rsplit('/', 1)[-1]
    base = base.split('@')[-1]
    for ext in ('.mrcs', '.mrc', '.st', '.tif', '.tiff', '.em', '.spi', '.hdf'):
        if base.lower().endswith(ext):
            base = base[:-len(ext)]
            break
    match = TILT_ID_PATTERN.match(base)
    if match is None:
        return None
    return match.group('tsId'), int(match.group('index'))


def batches(items, size):
    """ Split ``items`` into consecutive lists of at most ``size`` elements
    (a single batch when ``size`` <= 0). """
    items = list(items)
    if size <= 0:
        return [items] if items else []
    return [items[i:i + size] for i in range(0, len(items), size)]
