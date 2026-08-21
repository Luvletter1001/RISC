"""Small config post-processing helpers for OpenRSD training switches."""


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ('true', '1', 'yes', 'y', 'on'):
            return True
        if lowered in ('false', '0', 'no', 'n', 'off'):
            return False
    return bool(value)


def sync_openrsd_feature_switches(cfg):
    """Propagate top-level OpenRSD switches into nested model config.

    MMEngine applies ``--cfg-options`` after executing the config file, so
    assignments such as ``use_ccl_loss=use_ccl`` are not re-evaluated when the
    CLI changes ``use_ccl``. This sync keeps the short top-level knobs usable.
    """
    if 'model' not in cfg:
        return cfg

    model = cfg.model
    if 'use_declip' in cfg:
        model.use_declip_support = _as_bool(cfg.use_declip)

    if 'use_ccl' in cfg and 'bbox_head' in model:
        model.bbox_head.use_ccl_loss = _as_bool(cfg.use_ccl)

    return cfg
