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
import numpy as np
import pyworkflow.protocol.params as params
from pyworkflow.viewer import DESKTOP_TKINTER, WEB_DJANGO, ProtocolViewer

from ..protocols import ParapickProtLocalize
from .stats import LocalizeStats, binAngles, tiltAnglesFromTiltSeries

BOX_PLOT, INTERVAL_PLOT = 0, 1
COLOR = '#3b6ea8'
COLOR_ALT = '#c8553d'


class ParapickLocalizeViewer(ProtocolViewer):
    """ Statistics of a ParaPick localization: per-tilt fit quality, picks and particle
    detection as a function of the tilt angle, and particle scores and positions. Per-tilt
    quantities are pooled over all the tilt-series, grouped by tilt angle. """
    _label = 'analyze results'
    _targets = [ParapickProtLocalize]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def _defineParams(self, form):
        form.addSection(label='Per tilt')
        form.addParam('tiltPlotType', params.EnumParam, default=BOX_PLOT,
                      choices=['Box plot', 'Interval'],
                      display=params.EnumParam.DISPLAY_HLIST,
                      label='Plot type',
                      help='How the values of all tilt-series are summarized at each tilt '
                           'angle. Box plot: quartiles, whiskers at 1.5 IQR and outliers. '
                           'Interval: median, 25-75% (dark) and 5-95% (light) ranges.')
        form.addParam('angleBin', params.FloatParam, default=1.0,
                      label='Tilt angle bin (deg)',
                      help='Tilt angles of different tilt-series are grouped by rounding to '
                           'multiples of this value. 0: no grouping.')
        form.addParam('displaySigma', params.LabelParam,
                      label='Per-tilt sigma',
                      help='Fitted standard deviation of the picks around the projected '
                           'particles, per tilt image. It measures how well each tilt agrees '
                           'with the 3D model (picking precision and alignment errors).')
        form.addParam('displayResiduals', params.LabelParam,
                      label='Residuals of the assigned picks',
                      help='Distance between each assigned pick and the projection of its '
                           'particle.')
        form.addParam('displayDetection', params.LabelParam,
                      label='Picks and detection rate',
                      help='Picks per tilt image, and fraction of the particles of the '
                           'tilt-series that have a pick assigned on the tilt image (an '
                           'estimate of the picker recall).')

        form.addSection(label='Particles')
        form.addParam('displayScores', params.LabelParam,
                      label='Scores and chain lengths',
                      help='Distribution of the particle scores (fraction of tilts supporting '
                           'the particle) and of the number of picks assigned to each '
                           'particle.')
        form.addParam('displayDepth', params.LabelParam,
                      label='Particle positions',
                      help='Distribution of the particle Z (depth) and X-Y positions, '
                           'relative to the centre of the volume. Dashed lines: volume '
                           'boundaries (the tomogram, when given).')
        form.addParam('displayTiltSeries', params.LabelParam,
                      label='Per tilt-series summary',
                      help='Particles, background fraction and iterations per tilt-series. '
                           'Red: the fit did not converge within the maximum iterations.')

    def _getVisualizeDict(self):
        return {
            'displaySigma': self._showSigma,
            'displayResiduals': self._showResiduals,
            'displayDetection': self._showDetection,
            'displayScores': self._showScores,
            'displayDepth': self._showDepth,
            'displayTiltSeries': self._showTiltSeries,
        }

    # -------------------------------------------------------------- data
    def _getStats(self):
        if getattr(self, '_stats', None) is None:
            extra = self.protocol._getExtraPath()
            stats = LocalizeStats(extra)
            if any(np.isnan(t.angle) for t in stats.tilts):
                # Runs that did not keep the ParaPick input files: use the tilt-series
                tsSet = self.protocol._getTiltSeriesSet()
                if tsSet is not None:
                    stats = LocalizeStats(extra, tiltAngles=tiltAnglesFromTiltSeries(
                        tsSet, {t.tsId for t in stats.tiltSeries}))
            self._stats = stats
        return self._stats

    @staticmethod
    def _plotter(rows=1, cols=1, title='', figsize=None):
        from pyworkflow.gui.plotter import Plotter
        return Plotter(x=rows, y=cols, windowTitle=title, mainTitle=title, figsize=figsize)

    @staticmethod
    def _done(plotter):
        plotter.getFigure().tight_layout()
        return [plotter]

    def _nothing(self, what):
        return [self.errorMessage('No %s found: has the protocol finished?' % what,
                                  title='ParaPick')]

    # -------------------------------------------------------------- plots
    def _tiltPlot(self, ax, angles, values, color=COLOR):
        binWidth = self.angleBin.get()
        centres, groups = binAngles(angles, values, binWidth)
        if len(centres) == 0:
            return
        step = np.min(np.diff(centres)) if len(centres) > 1 else 1.0
        if self.tiltPlotType.get() == BOX_PLOT:
            # Individual outliers only add noise when there are many values per bin
            showFliers = max(len(g) for g in groups) <= 500
            ax.boxplot(groups, positions=centres, widths=0.7 * step, manage_ticks=False,
                       patch_artist=True, showfliers=showFliers,
                       boxprops=dict(facecolor=color, alpha=0.5, edgecolor=color),
                       medianprops=dict(color='black'),
                       whiskerprops=dict(color=color), capprops=dict(color=color),
                       flierprops=dict(marker='.', markersize=3, alpha=0.4,
                                       markeredgecolor=color))
        else:
            q = np.array([np.percentile(g, [5, 25, 50, 75, 95]) for g in groups])
            ax.fill_between(centres, q[:, 0], q[:, 4], color=color, alpha=0.2, lw=0,
                            label='5-95%')
            ax.fill_between(centres, q[:, 1], q[:, 3], color=color, alpha=0.45, lw=0,
                            label='25-75%')
            ax.plot(centres, q[:, 2], color=color, marker='.', label='median')
            ax.legend(loc='best', fontsize=8)
        ax.set_xlim(centres[0] - step, centres[-1] + step)
        ax.grid(True, alpha=0.3)
        self._clipOutliers(ax, groups)

    def _clipOutliers(self, ax, groups):
        """ Keep a few extreme values (e.g. a failed tilt) from squashing the plot: the
        y axis ends above the highest whisker (box plot) or 95th percentile (interval),
        and the number of values off the plot is shown. """
        tops = []
        for g in groups:
            q1, q3, p95 = np.percentile(g, [25, 75, 95])
            if self.tiltPlotType.get() == BOX_PLOT:
                tops.append(np.max(g[g <= q3 + 1.5 * (q3 - q1)]))
            else:
                tops.append(p95)
        top = max(tops)
        bottom = min(0.0, min(np.min(g) for g in groups))
        if top <= bottom:
            return
        top = bottom + 1.15 * (top - bottom)
        hidden = int(sum(np.sum(g > top) for g in groups))
        ax.set_ylim(bottom, top)
        if hidden:
            ax.set_title('%s\n(%d values above %.3g not shown)' % (ax.get_title(), hidden, top),
                         fontsize=ax.title.get_fontsize())

    def _showSigma(self, e=None):
        stats = self._getStats()
        angles, sigmas = stats.perTilt('sigma')
        if not len(sigmas):
            return self._nothing('per-tilt statistics')
        plotter = self._plotter(title='Per-tilt sigma (%d tilt-series)' % len(stats.tiltSeries))
        ax = plotter.createSubPlot('Fitted sigma of the picks', 'Tilt angle (deg)', 'Sigma (Å)')
        self._tiltPlot(ax, angles, sigmas)
        return self._done(plotter)

    def _showResiduals(self, e=None):
        stats = self._getStats()
        angles, residuals = stats.residualsPerTilt()
        if not len(residuals):
            return self._nothing('pick assignments')
        plotter = self._plotter(title='Residuals (%d assigned picks)' % len(residuals))
        ax = plotter.createSubPlot('Distance of the assigned picks to their particle '
                                   'projection', 'Tilt angle (deg)', 'Residual (Å)')
        self._tiltPlot(ax, angles, residuals)
        return self._done(plotter)

    def _showDetection(self, e=None):
        stats = self._getStats()
        angles, picks = stats.perTilt('picks')
        if not len(picks):
            return self._nothing('per-tilt statistics')
        plotter = self._plotter(2, 1, title='Picks and detection per tilt', figsize=(8, 8))
        ax = plotter.createSubPlot('Picks per tilt image', 'Tilt angle (deg)', 'Picks')
        self._tiltPlot(ax, angles, picks)
        detAngles, detection = stats.perTilt('detection')
        ax = plotter.createSubPlot('Particles with an assigned pick on the tilt image',
                                   'Tilt angle (deg)', 'Fraction of particles')
        self._tiltPlot(ax, detAngles, detection, color=COLOR_ALT)
        ax.set_ylim(0, 1.05)
        return self._done(plotter)

    def _showScores(self, e=None):
        stats = self._getStats()
        scores = stats.scores()
        if not len(scores):
            return self._nothing('particles')
        plotter = self._plotter(1, 2, title='Particles (%d)' % len(scores), figsize=(11, 4.5))
        ax = plotter.createSubPlot('Score', 'Score (fraction of supporting tilts)', 'Particles')
        ax.hist(scores, bins=np.linspace(0, max(1.0, scores.max()), 41), color=COLOR)
        minScore = self.protocol.minScore.get()
        if minScore is not None and minScore > 0:
            ax.axvline(minScore, color='black', ls='--', lw=1)
        lengths = stats.chainLengths()
        ax = plotter.createSubPlot('Assigned picks per particle', 'Tilt images', 'Particles')
        if lengths.size:
            ax.hist(lengths, bins=np.arange(-0.5, lengths.max() + 1.5, 1), color=COLOR_ALT)
        return self._done(plotter)

    def _showDepth(self, e=None):
        stats = self._getStats()
        xyz = stats.coordinates()
        if not len(xyz):
            return self._nothing('particles')
        inside = stats.insideVolume()
        title = 'Particle positions: %d particles, %d outside the volume' % (
            len(xyz), int((~inside).sum()))
        plotter = self._plotter(1, 2, title=title, figsize=(11, 4.5))
        ax = plotter.createSubPlot('Depth', 'Z (Å)', 'Particles')
        ax.hist(xyz[:, 2], bins=60, color=COLOR)
        half = stats.halfThickness()
        if half:
            for z in (-half, half):
                ax.axvline(z, color='black', ls='--', lw=1)
        ax = plotter.createSubPlot('X-Y (all tilt-series)', 'X (Å)', 'Y (Å)')
        ax.scatter(xyz[inside, 0], xyz[inside, 1], s=2, alpha=0.3, color=COLOR, label='inside')
        if (~inside).any():
            ax.scatter(xyz[~inside, 0], xyz[~inside, 1], s=4, alpha=0.6, color=COLOR_ALT,
                       label='outside')
            ax.legend(loc='best', fontsize=8, markerscale=3)
        if stats.volumeSizes:
            sx, sy, _ = np.median(np.array(list(stats.volumeSizes.values())), axis=0) / 2.0
            ax.plot([-sx, sx, sx, -sx, -sx], [-sy, -sy, sy, sy, -sy], 'k--', lw=1)
        ax.set_aspect('equal', adjustable='datalim')
        return self._done(plotter)

    def _showTiltSeries(self, e=None):
        stats = self._getStats()
        series = stats.tiltSeries
        if not series:
            return self._nothing('per tilt-series statistics')
        names = [t.tsId for t in series]
        colors = [COLOR if t.converged else COLOR_ALT for t in series]
        x = np.arange(len(series))
        width = max(8, 0.3 * len(series))
        plotter = self._plotter(3, 1, title='Per tilt-series summary', figsize=(width, 10))
        for title, ylabel, values in (
                ('Particles', 'Particles', [t.nParticles for t in series]),
                ('Background (false positive) fraction', 'Fraction',
                 [t.backgroundFraction for t in series]),
                ('EM iterations', 'Iterations', [t.iterations for t in series])):
            ax = plotter.createSubPlot(title, '', ylabel)
            ax.bar(x, values, color=colors)
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=90, fontsize=7)
            ax.grid(True, axis='y', alpha=0.3)
        from matplotlib.patches import Patch
        plotter.getFigure().axes[0].legend(
            handles=[Patch(color=COLOR, label='converged'),
                     Patch(color=COLOR_ALT, label='not converged')],
            loc='upper right', fontsize=8)
        return self._done(plotter)
