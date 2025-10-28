# Temporary shim to allow future move of configuration loading
try:
    from config import load_default_params  # type: ignore
except Exception:  # pragma: no cover
    def load_default_params():  # type: ignore
        raise ImportError("Original config.load_default_params not found")
