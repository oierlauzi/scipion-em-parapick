=======================
Scipion ParaPick plugin
=======================

`ParaPick <https://github.com/oierlauzi/parapick>`_ localizes particles in 3D from 2D
picks made on the individual images of cryo-ET tilt-series, without reconstructing a
tomogram. Any single-particle picker can be used for the 2D picking.

Workflow
--------

1. **parapick - unstack tilt-series**: ``SetOfTiltSeries`` → ``SetOfTiltSeriesMicrographs``.
   Every enabled tilt image becomes a micrograph named ``<tsId>_<index>`` (1-based stack
   index), optionally binned. It is a regular ``SetOfMicrographs``, so any SPA picker
   accepts it, and each micrograph remembers its tilt-series, stack index, acquisition
   order and tilt angle.
2. Pick the micrographs with any SPA picker (Topaz, crYOLO, Xmipp, Gautomatch, ...).
3. **parapick - localize**: picks (``SetOfCoordinates``) + ``SetOfTiltSeries`` (optional,
   taken from the micrographs by default) + ``SetOfTomograms`` (optional) →

   - ``SetOfTiltSeriesCoordinates``: particle centres in Å relative to the centre of the
     volume, with the ParaPick score;
   - ``SetOfLandmarkModels``: for each particle, the chain of picks it explains (at most
     one per tilt image), in the raw tilt-image frame, with residuals;
   - ``SetOfCoordinates3D`` on the tomograms, if given.

The **particle diameter** is required: half of it is the initial spread of the model.
It must be smaller than the distance between neighbouring particles; ParaPick's own
default (5% of the volume size) merges neighbours in crowded samples. The STAR files
given to ParaPick are kept in ``extra/parapick_input`` to rerun it by hand.

**Analyze results** (localize viewer) plots, pooling all tilt-series by tilt angle (box
plot or median/percentile interval): per-tilt sigma of the fit, residuals of the assigned
picks, picks per tilt and fraction of particles detected per tilt; and the particle score,
picks per particle, depth and X-Y distributions (marking particles outside the tomogram,
which are kept), plus a per tilt-series summary (particles, background fraction,
iterations and convergence).

Picks on micrographs from other protocols are matched to tilt images through the
``_tsId`` / ``_acqOrder`` attributes or the ``<tsId>_<index>`` name. Tilt-series are
processed in batches (one step per batch).

Installation
------------

.. code-block:: bash

    scipion3 installp -p scipion-em-parapick      # or, from a clone:  -p . --devel
    scipion3 installb parapick-0.2.0

ParaPick is installed with pip in its own conda environment (``parapick-<version>``).
Variables (Scipion config or environment):

- ``PARAPICK_ENV_ACTIVATION``: command that activates the environment
  (default ``conda activate parapick-<version>``).
- ``PARAPICK_JAX_EXTRA``: JAX backend, ``cuda12``, ``cuda13`` or ``cpu``. By default
  ``cuda12`` if ``nvidia-smi`` is found, otherwise CPU. The CUDA wheels bundle CUDA, so
  only the NVIDIA driver is required.
- ``PARAPICK_SOURCE``: install ParaPick from a pip requirement or a local clone
  (editable) instead of PyPI, e.g. for development.

Geometry
--------

ParaPick reads the tilt-series geometry as RELION 5 STAR files, which the plugin writes
from the Scipion alignment of each tilt image (``parapick/convert/geometry.py``). The
conversion reproduces Scipion's projection of tilt-series coordinates exactly for rigid
alignments; alignments with stretching are approximated by their closest rotation (a
warning is logged).

Tests
-----

.. code-block:: bash

    scipion3 tests parapick.tests.test_convert     # conversions, no data needed
    scipion3 tests parapick.tests.test_viewers     # statistics behind the plots
    scipion3 tests parapick.tests.test_protocols   # synthetic end-to-end (needs imod, eman2)
