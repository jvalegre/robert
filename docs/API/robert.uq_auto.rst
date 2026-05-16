uq_auto
=======

Automatic uncertainty selection for **regression**. :mod:`robert.uq_auto` scores
multiple uncertainty candidates on training out-of-fold absolute residuals, fits an
optional post-hoc scaler, and writes calibrated ``{y}_pred_uq_auto`` values plus
metadata under ``PREDICT/uq_auto_metadata.json``.

**Candidates** (configured via ``uq_auto_candidates``):

* ``cv_sd`` — spread across repeated CV refits.
* ``conformal`` — constant split-conformal half-width (requires ``conformal_enable``).
* ``meta_total`` — combined meta-model uncertainty (requires ``uq_enable_meta`` during fit).

**Scalers** (``uq_auto_scaler``): ``none``, ``global_multiplicative`` (default), or
``isotonic``.

**Selection** uses a weighted composite of coverage, sharpness, and Gaussian NLL on
an inner hold-out split (see ``uq_auto_metric_weights``). When scores tie, preference
order is ``cv_sd`` → ``conformal`` → ``meta_total``.

User-facing configuration, output columns, and ``predict(..., return_uncertainty="auto")``
are documented in :doc:`robert.api`.

.. automodule:: robert.uq_auto
   :members:
