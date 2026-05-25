# Lazy top-level imports.  Eager imports here trigger compute_polarizabilities.py
# which instantiates Potassium39() (-> ARC SQLite) at class-definition time.
# Worker subprocesses importing kamo.constants must not touch ARC.
# Python 3.7+ module __getattr__ defers these until first access.

_lazy = {
    'Potassium39':             '.atom_properties.k39',
    'GaussianBeam':            '.gaussian_beam',
    'ComputeLightShift':       '.light_shift',
    'ComputePolarizabilities': '.light_shift',
    'bec':                     '.BEC_properties',
}

def __getattr__(name):
    if name in _lazy:
        import importlib
        mod = importlib.import_module(_lazy[name], __name__)
        val = getattr(mod, name)
        globals()[name] = val  # cache so subsequent accesses skip __getattr__
        return val
    raise AttributeError(f"module 'kamo' has no attribute {name!r}")