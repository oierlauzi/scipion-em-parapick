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
"""Reading tilt images and writing micrographs."""

import os

import mrcfile
import numpy as np

MRC_EXTENSIONS = ('.mrc', '.mrcs', '.st', '.ali', '.rec', '.map')


def _stripType(fileName):
    """ Remove a Scipion/RELION ``:type`` suffix (e.g. ``file.mrc:mrcs``). """
    head, sep, tail = fileName.rpartition(':')
    return head if sep and tail and '/' not in tail else fileName


def readSection(fileName, index):
    """ Read the ``index``-th (1-based) 2D section of an image stack as float32.

    MRC files are always read as a stack of sections, whatever their header says
    (tilt-series are often flagged as volumes).
    """
    fileName = _stripType(fileName)
    if fileName.lower().endswith(MRC_EXTENSIONS):
        with mrcfile.mmap(fileName, mode='r', permissive=True) as mrc:
            data = mrc.data
            section = data if data.ndim == 2 else data[index - 1]
            return np.array(section, dtype=np.float32)
    from pwem.emlib.image import ImageHandler  # Other formats
    return np.asarray(ImageHandler().read((index, fileName)).getData(), dtype=np.float32)


def binnedShape(shape, binning):
    """ ``(H, W)`` after binning by an integer factor (trailing pixels are dropped). """
    return tuple(max(1, n // int(binning)) for n in shape)


def binImage(image, binning):
    """ Bin by an integer factor: drop the trailing pixels that do not fill a bin, then
    crop the Fourier transform. Output pixel ``k`` samples input pixel ``k * binning``
    exactly, on both axes. """
    binning = int(binning)
    if binning == 1:
        return np.asarray(image, dtype=np.float32)
    h, w = binnedShape(image.shape, binning)
    return fourierCrop(np.asarray(image)[:h * binning, :w * binning], (h, w))


def fourierCrop(image, shape):
    """ Downsample ``image`` to ``shape`` by cropping its Fourier transform.

    Output pixel ``k`` samples the input at ``k * n_in / n_out``, so coordinates scale
    by the size ratio. The mean value is preserved.
    """
    image = np.asarray(image, dtype=np.float32)
    height, width = image.shape
    h, w = shape
    if (h, w) == (height, width):
        return image
    if h > height or w > width:
        raise ValueError('Fourier cropping cannot upsample')
    spectrum = np.fft.rfft2(image)
    rows = np.r_[0:(h + 1) // 2, height - h // 2:height]
    cropped = spectrum[rows, :w // 2 + 1]
    out = np.fft.irfft2(cropped, s=(h, w)) * (h * w) / float(height * width)
    return out.astype(np.float32)


def writeMrc(fileName, image, pixelSize):
    """ Write a 2D float32 MRC with the given pixel size (Å/px). """
    os.makedirs(os.path.dirname(os.path.abspath(fileName)), exist_ok=True)
    tmp = fileName + '.tmp'
    with mrcfile.new(tmp, overwrite=True) as mrc:
        mrc.set_data(np.asarray(image, dtype=np.float32))
        mrc.voxel_size = pixelSize
    os.replace(tmp, fileName)  # Never leave half-written micrographs behind
