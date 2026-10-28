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

PARAPICK = 'parapick'

# Supported versions of the ParaPick package (PyPI: parapick)
V0_2_0 = '0.2.0'
VERSIONS = [V0_2_0]
DEFAULT_VERSION = V0_2_0

# Scipion variables
PARAPICK_ENV_ACTIVATION = 'PARAPICK_ENV_ACTIVATION'
"""Command that activates the conda environment holding ParaPick."""
PARAPICK_SOURCE = 'PARAPICK_SOURCE'
"""Optional pip requirement or local path to install ParaPick from (development)."""
PARAPICK_JAX_EXTRA = 'PARAPICK_JAX_EXTRA'
"""Optional ParaPick extra selecting the JAX backend: 'cuda12', 'cuda13' or 'cpu'."""

PYTHON_VERSION = '3.12'


def getEnvName(version):
    return '%s-%s' % (PARAPICK, version)


DEFAULT_ACTIVATION_CMD = 'conda activate %s' % getEnvName(DEFAULT_VERSION)
