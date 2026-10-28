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
"""Statistics of a ParaPick localization run, read from its output files.

Everything here is plain numpy, independent of the GUI, so it can be tested and reused
(e.g. from a notebook).
"""

import glob
import json
import math
import os
from collections import defaultdict, namedtuple

import numpy as np

from ..convert import star

TiltStats = namedtuple('TiltStats', [
    'tsId', 'tiltId', 'angle', 'nPicks', 'sigma', 'nAssigned', 'nParticles'])
""" One tilt image: picks, fitted sigma (Å) and picks assigned to particles. """

TsStats = namedtuple('TsStats', [
    'tsId', 'nTilts', 'nPicks', 'nParticles', 'iterations', 'converged',
    'backgroundFraction', 'elapsed'])


class LocalizeStats:
    """ Statistics of the ParaPick runs found in ``outputDir`` (``batch_*`` folders).

    :param outputDir: the ``extra`` folder of the localize protocol.
    :param tiltAngles: optional ``{(tsId, tiltId): angle}``. If not given, the angles are
        read from the ParaPick input STAR files kept in ``outputDir``.
    """

    def __init__(self, outputDir, tiltAngles=None, inputFolder='parapick_input'):
        self.outputDir = outputDir
        batchDirs = sorted(d for d in glob.glob(os.path.join(outputDir, 'batch_*'))
                           if os.path.isdir(d))
        self._reports = []
        self.particles = []
        self.assignments = []
        for d in batchDirs:
            report = os.path.join(d, 'localize_report.json')
            if os.path.exists(report):
                with open(report) as f:
                    self._reports.append(json.load(f))
            if os.path.exists(os.path.join(d, 'particles.star')):
                self.particles += star.readParticlesStar(os.path.join(d, 'particles.star'))
            if os.path.exists(os.path.join(d, 'assignments.star')):
                self.assignments += star.readAssignmentsStar(os.path.join(d, 'assignments.star'))
        self.volumeSizes = {}  # tsId -> (X, Y, Z) Å
        self.tiltAngles = dict(tiltAngles or {})
        self._readInputGeometry(os.path.join(outputDir, inputFolder))
        self.tilts = self._collectTilts()
        self.tiltSeries = self._collectTiltSeries()

    # ------------------------------------------------------------------ reading
    def _readInputGeometry(self, folder):
        fn = os.path.join(folder, 'tomograms.star')
        if not os.path.exists(fn):
            return
        for row in star.readStarTable(fn, 'global'):
            tsId = row['rlnTomoName']
            pixel = float(row['rlnTomoTiltSeriesPixelSize'])
            self.volumeSizes[tsId] = tuple(
                float(row[k]) * pixel for k in ('rlnTomoSizeX', 'rlnTomoSizeY', 'rlnTomoSizeZ'))
            tsStar = os.path.join(folder, row['rlnTomoTiltSeriesStarFile'])
            if os.path.exists(tsStar):
                for tilt in star.readStarTable(tsStar, tsId):
                    self.tiltAngles.setdefault((tsId, tilt['rlnMicrographName']),
                                               float(tilt['rlnTomoYTilt']))

    def _collectTilts(self):
        assigned = defaultdict(int)
        for a in self.assignments:
            assigned[(a.tsId, a.tiltId)] += 1
        particles = defaultdict(int)
        for p in self.particles:
            particles[p.tsId] += 1
        tilts = []
        for report in self._reports:
            for ts in report.get('tilt_series', []):
                tsId = ts['tilt_series']
                for t in ts.get('tilts', []):
                    tiltId = t['tilt_id']
                    sigma = t.get('sigma_angst')
                    tilts.append(TiltStats(
                        tsId=tsId, tiltId=tiltId,
                        angle=self.tiltAngles.get((tsId, tiltId), float('nan')),
                        nPicks=int(t.get('n_picks', 0)),
                        # Tilts without picks keep the initial sigma: not a fit result
                        sigma=float(sigma) if sigma is not None and t.get('n_picks') else float('nan'),
                        nAssigned=assigned[(tsId, tiltId)],
                        nParticles=particles[tsId]))
        return tilts

    def _collectTiltSeries(self):
        particles = defaultdict(int)
        for p in self.particles:
            particles[p.tsId] += 1
        result = []
        for report in self._reports:
            for ts in report.get('tilt_series', []):
                result.append(TsStats(
                    tsId=ts['tilt_series'], nTilts=ts.get('n_tilts', 0),
                    nPicks=ts.get('n_picks', 0), nParticles=particles[ts['tilt_series']],
                    iterations=ts.get('iterations', 0), converged=bool(ts.get('converged')),
                    backgroundFraction=ts.get('background_fraction', float('nan')),
                    elapsed=ts.get('elapsed_seconds', float('nan'))))
        return sorted(result, key=lambda t: _naturalKey(t.tsId))

    # ------------------------------------------------------------------ per tilt
    def tiltAngleArray(self):
        return np.array([t.angle for t in self.tilts], dtype=float)

    def perTilt(self, quantity):
        """ ``(angles, values)`` of a per-tilt quantity: ``'sigma'`` (Å), ``'picks'``,
        ``'assigned'`` (picks assigned to a particle) or ``'detection'`` (fraction of the
        particles of the tilt-series that have an assigned pick on the tilt). """
        if quantity == 'sigma':
            values = [t.sigma for t in self.tilts]
        elif quantity == 'picks':
            values = [t.nPicks for t in self.tilts]
        elif quantity == 'assigned':
            values = [t.nAssigned for t in self.tilts]
        elif quantity == 'detection':
            values = [t.nAssigned / float(t.nParticles) if t.nParticles else float('nan')
                      for t in self.tilts]
        else:
            raise ValueError('Unknown quantity %s' % quantity)
        return _finite(self.tiltAngleArray(), np.array(values, dtype=float))

    def residualsPerTilt(self):
        """ ``(angles, |residual|)`` of every assigned pick, in Å. """
        angles = np.array([self.tiltAngles.get((a.tsId, a.tiltId), float('nan'))
                           for a in self.assignments], dtype=float)
        residuals = np.array([math.hypot(a.residualX, a.residualY) for a in self.assignments],
                             dtype=float)
        return _finite(angles, residuals)

    # ------------------------------------------------------------------ per particle
    def scores(self):
        return np.array([p.score for p in self.particles if p.score is not None], dtype=float)

    def chainLengths(self):
        """ Number of assigned picks (tilts) of every particle. """
        counts = defaultdict(int)
        for a in self.assignments:
            counts[(a.tsId, a.particleId)] += 1
        return np.array([counts[(p.tsId, p.particleId)] for p in self.particles], dtype=float)

    def coordinates(self):
        """ ``(N, 3)`` particle centres (Å, relative to the volume centre). """
        return np.array([[p.x, p.y, p.z] for p in self.particles], dtype=float).reshape(-1, 3)

    def insideVolume(self):
        """ Boolean mask of the particles inside the volume they were localized in
        (the tomogram, when one was given). ``True`` when the volume is unknown. """
        mask = []
        for p in self.particles:
            size = self.volumeSizes.get(p.tsId)
            mask.append(size is None or all(abs(c) <= s / 2.0
                                            for c, s in zip((p.x, p.y, p.z), size)))
        return np.array(mask, dtype=bool)

    def halfThickness(self):
        """ Median half thickness (Å) of the volumes, or None. """
        if not self.volumeSizes:
            return None
        return float(np.median([s[2] for s in self.volumeSizes.values()])) / 2.0


def binAngles(angles, values, binWidth):
    """ Group ``values`` by tilt angle rounded to multiples of ``binWidth`` (degrees).

    :return: ``(centres, [values of each bin])`` sorted by angle.
    """
    angles = np.asarray(angles, dtype=float)
    values = np.asarray(values, dtype=float)
    if binWidth <= 0 or len(angles) == 0:
        keys = angles
    else:
        keys = np.round(angles / binWidth) * binWidth
    centres = np.unique(keys)
    return centres, [values[keys == c] for c in centres]


def _finite(angles, values):
    keep = np.isfinite(angles) & np.isfinite(values)
    return angles[keep], values[keep]


def _naturalKey(text):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', text)]


def tiltAnglesFromTiltSeries(tsSet, tsIds=None):
    """ ``{(tsId, tiltId): angle}`` from a SetOfTiltSeries (for runs that did not keep
    the ParaPick input files). """
    from ..convert.naming import getTiltId
    angles = {}
    for ts in tsSet.iterItems():
        tsId = ts.getTsId()
        if tsIds is not None and tsId not in tsIds:
            continue
        for ti in ts.iterItems():
            angles[(tsId, getTiltId(tsId, ti.getIndex()))] = ti.getTiltAngle()
    return angles


__all__ = ['LocalizeStats', 'TiltStats', 'TsStats', 'binAngles', 'tiltAnglesFromTiltSeries']
