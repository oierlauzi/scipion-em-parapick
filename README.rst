=======================
Scipion ParaPick plugin
=======================

`ParaPick <https://github.com/oierlauzi/parapick>`_ localizes particles in 3D from 2D
picks made on the individual images of cryo-ET tilt-series, without reconstructing a
tomogram. Any single-particle picker can be used for the 2D picking.

Workflow
--------

1. **parapick - unstack tilt-series**: ``SetOfTiltSeries`` → ``SetOfMicrographs``.
   Every enabled tilt image becomes a micrograph named ``<tsId>_<index>`` (1-based stack
   index), optionally binned. It is a plain ``SetOfMicrographs``, so any SPA picker and
   viewer accepts it (Xmipp's accept no other class), and each micrograph remembers its
   tilt-series, stack index, acquisition order and tilt angle in extra attributes.
   Given the CTF estimated on the tilt-series (``SetOfCTFTomoSeries``, optional), it also
   outputs the CTF of each micrograph as a ``SetOfCTF``, for pickers and other
   single-particle programs that use it. Each micrograph keeps the defocus estimated for
   its whole tilt image; the defocus gradient across a tilted image is not modelled.
2. Pick the micrographs with any SPA picker (Topaz, crYOLO, Xmipp, Gautomatch, ...).
3. **parapick - localize**: picks (``SetOfCoordinates``) + ``SetOfTiltSeries`` (optional,
   taken from the micrographs by default) + ``SetOfTomograms`` (optional) →

   - ``SetOfTiltSeriesCoordinates``: particle centres in Å relative to the centre of the
     volume, with the ParaPick score;
   - ``SetOfLandmarkModels``: for each particle, the chain of picks it explains (at most
     one per tilt image), in the raw tilt-image frame, with residuals;
   - ``SetOfCoordinates3D`` on the tomograms, if given.

Instead of an SPA picker (steps 1 and 2), **parapick - template matching (Leopard-EM)**
picks every tilt image by 2D template matching: ``SetOfTiltSeries`` + ``SetOfCTFTomoSeries``
+ template ``Volume`` (at the pixel size of the tilt-series) → ``SetOfMicrographs`` (the tilt
images, as from unstack) and ``SetOfCoordinates`` (the peaks, scored by their z-score),
which go straight into localize. The parameters of each tilt (low-pass, angular and defocus
steps, defocus range) are derived by ParaPick from its dose, its CTF and the tilt of the
sample; the thickness comes from the tomograms if given. "Tilts per tilt-series" searches
only those with the highest predicted SNR. It needs the optional binary
``parapick-leopard-<version>`` (see Installation). The template matching is done by
`Leopard-EM <https://github.com/Lucaslab-Berkeley/Leopard-EM>`_ (Lucas lab, UC Berkeley);
please cite it when you use this protocol: Giammar MD, Dickerson JL, Hall LN, Lucas BA
(2026). Leopard-EM: an extensible 2D template-matching package to accelerate in situ
structural biology. *Acta Cryst. D* 82, 30–42. https://doi.org/10.1107/S2059798325009982

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
    scipion3 installb parapick-leopard-0.2.0   # optional: Leopard-EM, for template matching

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
