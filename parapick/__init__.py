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
"""Scipion plugin for ParaPick.

ParaPick localizes particles in 3D from 2D picks made on the individual images of
cryo-ET tilt-series, without reconstructing a tomogram. See
https://github.com/oierlauzi/parapick
"""

import os
import shlex
import shutil

import pwem
import pyworkflow.utils as pwutils
from pyworkflow import TOMO

from .constants import (PARAPICK, VERSIONS, DEFAULT_VERSION, PARAPICK_ENV_ACTIVATION,
                        PARAPICK_SOURCE, PARAPICK_JAX_EXTRA, DEFAULT_ACTIVATION_CMD,
                        PYTHON_VERSION, getEnvName)

__version__ = '0.1.0'
_references = ['deIsidroGomez2026parapick']


class Plugin(pwem.Plugin):
    _url = 'https://github.com/scipion-em/scipion-em-parapick'
    _supportedVersions = VERSIONS
    _processingField = [TOMO]

    @classmethod
    def _defineVariables(cls):
        cls._defineVar(PARAPICK_ENV_ACTIVATION, DEFAULT_ACTIVATION_CMD)
        cls._defineVar(PARAPICK_SOURCE, '')
        cls._defineVar(PARAPICK_JAX_EXTRA, '')

    @classmethod
    def getEnviron(cls):
        """ Environment for ParaPick: Scipion's PYTHONPATH must not leak into the
        conda environment. """
        environ = pwutils.Environ(os.environ)
        environ.pop('PYTHONPATH', None)
        return environ

    @classmethod
    def getDependencies(cls):
        return [] if cls.getCondaActivationCmd() else ['conda']

    @classmethod
    def defineBinaries(cls, env):
        for version in cls._supportedVersions:
            cls._addParapickPackage(env, version, default=(version == DEFAULT_VERSION))

    @classmethod
    def _getJaxExtra(cls):
        """ ParaPick extra installing the right JAX backend. JAX's CUDA wheels bundle
        the CUDA libraries, so only an NVIDIA driver is needed. """
        extra = (cls.getVar(PARAPICK_JAX_EXTRA) or '').strip()
        if extra:
            return None if extra.lower() == 'cpu' else extra
        return 'cuda12' if shutil.which('nvidia-smi') else None

    @classmethod
    def _getRequirement(cls, version):
        extra = cls._getJaxExtra()
        source = (cls.getVar(PARAPICK_SOURCE) or '').strip()
        suffix = '[%s]' % extra if extra else ''
        if not source:
            return shlex.quote('parapick%s==%s' % (suffix, version))
        if os.path.isdir(source):  # Local clone: editable install for development
            return '-e ' + shlex.quote(os.path.abspath(source) + suffix)
        return shlex.quote(source + suffix)

    @classmethod
    def _addParapickPackage(cls, env, version, default=False):
        envName = getEnvName(version)
        installed = 'parapick_%s_installed' % version
        cmd = ' '.join([
            cls.getCondaActivationCmd(),
            'conda create -y -n %s python=%s &&' % (envName, PYTHON_VERSION),
            'conda activate %s &&' % envName,
            'python -m pip install %s &&' % cls._getRequirement(version),
            'parapick --version &&',
            'touch %s' % installed,
        ])
        envPath = os.environ.get('PATH', '')  # conda is likely there
        env.addPackage(PARAPICK, version=version, tar='void.tgz',
                       commands=[(cmd, installed)],
                       neededProgs=cls.getDependencies(),
                       vars={'PATH': envPath} if envPath else None,
                       default=default)

    @classmethod
    def getActivationCmd(cls):
        return '%s %s' % (cls.getCondaActivationCmd(), cls.getVar(PARAPICK_ENV_ACTIVATION))

    @classmethod
    def getProgram(cls, gpu=True):
        """ Shell prefix that runs the ParaPick CLI. With ``gpu``, the GPUs assigned to
        the step by Scipion are exposed through the %(GPU)s placeholder. """
        program = '%s && ' % cls.getActivationCmd()
        if gpu:
            program += 'CUDA_VISIBLE_DEVICES=%(GPU)s '
        return program + PARAPICK

    @classmethod
    def runParapick(cls, protocol, args, gpu=True, cwd=None):
        protocol.runJob(cls.getProgram(gpu=gpu), args, env=cls.getEnviron(), cwd=cwd,
                        numberOfMpi=1, numberOfThreads=1)
