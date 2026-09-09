"""Array backend: numpy on the CPU, torch on the GPU, one set of call sites.

The split-step loop is an FFT pair plus a handful of elementwise operations over a
few hundred thousand complex points, repeated once per slice.  That is the shape a
GPU is built for, and :class:`ArrayBackend` is the thin layer that lets
:mod:`kamo.imaging.bpm` run the same loop on either device.

The abstraction is deliberately minimal.  Everything that happens ONCE -- sizing
the grid, building the angular spectrum, assembling the result -- stays in numpy
and is moved across at the boundary.  Only the hot loop is backend-agnostic, and it
needs just ``fft2``, ``ifft2``, ``exp``, ``sqrt``, elementwise arithmetic (which
torch already spells the way numpy does) and two reductions.

Precision
---------
Consumer NVIDIA parts run FP64 at 1/64 of their FP32 rate, so ``"single"`` is the
useful GPU mode and is the default *on the GPU only* -- the CPU path stays double,
so nothing that runs today changes.  For a unitary propagator the accumulated error
over ``n`` slices grows like ``sqrt(n) * eps``, which is ~2e-6 over 180 slices in
single precision, several orders below the 1e-3-level quantities the forward model
reports.  Reductions (the density-weighted mean intensity, the peak saturation) are
accumulated in double whatever the storage, since those sum hundreds of millions of
terms.  :func:`kamo.imaging.bpm.check_backend` runs a case both ways and prints the
difference rather than asking anyone to take this on trust.
"""

from __future__ import annotations

import numpy as np


class ArrayBackend:
    """Minimal array backend.  ``ArrayBackend()`` stays on the CPU.

    Parameters
    ----------
    device : {'cpu', 'gpu', 'cuda', 'auto'}
        ``'auto'`` picks the GPU when torch reports a usable CUDA device and falls
        back silently to numpy otherwise.  ``'gpu'``/``'cuda'`` raise if there is
        none, so a script that means to use the GPU fails loudly instead of
        quietly running 30x slower.
    precision : {'single', 'double', None}
        ``None`` means double on the CPU and single on the GPU -- see the module
        docstring.
    """

    def __init__(self, device: str = "cpu", precision=None):
        self.torch = None
        self.device = None
        if device in ("auto", "gpu", "cuda"):
            try:
                import torch
                if torch.cuda.is_available():
                    self.torch = torch
                    self.device = torch.device("cuda")
                    self.name = f"torch:{torch.cuda.get_device_name(0)}"
                elif device != "auto":
                    raise RuntimeError(
                        "device=%r but torch.cuda.is_available() is False" % device)
            except ImportError:
                if device != "auto":
                    raise RuntimeError("device=%r but torch is not installed" % device)
        elif device not in ("cpu", "numpy"):
            raise ValueError("device must be 'cpu', 'gpu', 'cuda' or 'auto'")
        if self.torch is None:
            self.name = "numpy"
        if precision is None:
            precision = "single" if self.torch is not None else "double"
        if precision not in ("single", "double"):
            raise ValueError("precision must be 'single', 'double' or None")
        self.precision = precision
        self._set_dtypes()

    def _set_dtypes(self):
        single = self.precision == "single"
        if self.torch is not None:
            t = self.torch
            self.cdtype = t.complex64 if single else t.complex128
            self.rdtype = t.float32 if single else t.float64
        else:
            self.cdtype = np.complex64 if single else np.complex128
            self.rdtype = np.float32 if single else np.float64

    @property
    def on_gpu(self) -> bool:
        return self.torch is not None

    # --------------------------------------------------------------- transfers

    def complex(self, a):
        """Move a host array onto the device as the working complex type."""
        if self.torch is None:
            return np.asarray(a, dtype=self.cdtype)
        return self.torch.as_tensor(np.ascontiguousarray(a),
                                    dtype=self.cdtype, device=self.device)

    def real(self, a):
        """Move a host array onto the device as the working real type."""
        if self.torch is None:
            return np.asarray(a, dtype=self.rdtype)
        return self.torch.as_tensor(np.ascontiguousarray(a),
                                    dtype=self.rdtype, device=self.device)

    def numpy(self, a) -> np.ndarray:
        """Bring a device array back to the host."""
        if self.torch is None or not isinstance(a, self.torch.Tensor):
            return np.asarray(a)
        return a.detach().cpu().numpy()

    # -------------------------------------------------------------- operations

    def fft2(self, a):
        if self.torch is None:
            import scipy.fft as sfft
            return sfft.fft2(a, workers=-1)
        return self.torch.fft.fft2(a)

    def ifft2(self, a):
        if self.torch is None:
            import scipy.fft as sfft
            return sfft.ifft2(a, workers=-1)
        return self.torch.fft.ifft2(a)

    def exp(self, a):
        return np.exp(a) if self.torch is None else self.torch.exp(a)

    def sqrt(self, a):
        return np.sqrt(a) if self.torch is None else self.torch.sqrt(a)

    def abs2(self, a):
        """``|a|^2`` as a REAL array, without forming the complex intermediate."""
        if self.torch is None:
            return np.abs(a) ** 2
        return self.torch.real(a * self.torch.conj(a))

    def fsum(self, a) -> float:
        """Sum to a Python float, accumulating in double whatever the storage."""
        if self.torch is None:
            return float(np.sum(np.asarray(a, dtype=np.float64)))
        return float(a.double().sum().item())

    def fmax(self, a) -> float:
        if self.torch is None:
            return float(np.max(a))
        return float(a.max().item())

    # Device-RESIDENT reductions.  The split-step loop accumulates a
    # density-weighted intensity and tracks a peak saturation once per slice; doing
    # that with ``.item()`` would force a host sync every slice and stall the
    # pipeline.  These keep the running totals on the device, and
    # :meth:`to_float` is called once, at the end.

    def dsum(self, a):
        """Sum, accumulated in double, left on the device as a 0-d array."""
        if self.torch is None:
            return np.sum(a, dtype=np.float64)
        return a.double().sum()

    def dmax(self, a):
        """Maximum, left on the device as a 0-d array."""
        return np.max(a) if self.torch is None else a.max()

    def maximum(self, a, b):
        """Elementwise (here: scalar) maximum of two device values."""
        return np.maximum(a, b) if self.torch is None else self.torch.maximum(a, b)

    def to_float(self, a) -> float:
        if self.torch is None or not isinstance(a, self.torch.Tensor):
            return float(a)
        return float(a.item())

    def zeros_like_scalar(self):
        """A device-resident double zero, for starting an accumulator."""
        if self.torch is None:
            return np.float64(0.0)
        return self.torch.zeros((), dtype=self.torch.float64, device=self.device)

    def sync(self):
        """Block until queued device work is done -- needed before timing."""
        if self.torch is not None:
            self.torch.cuda.synchronize()

    def free_memory(self):
        if self.torch is not None:
            self.torch.cuda.empty_cache()

    def __repr__(self):
        return f"ArrayBackend({self.name}, {self.precision})"


def as_backend(spec) -> ArrayBackend:
    """Coerce ``None`` / a string / an :class:`ArrayBackend` into a backend.

    ``None`` and ``'cpu'`` give the numpy backend, so every existing call site
    keeps the behaviour it has today.
    """
    if spec is None:
        return ArrayBackend("cpu")
    if isinstance(spec, ArrayBackend):
        return spec
    return ArrayBackend(str(spec))


def array_exp(a):
    """``exp(a)`` for a numpy array or a torch tensor, without knowing which.

    Susceptibility sources build their density profile from whatever transverse
    coordinate arrays the propagator hands them, which are device arrays on the GPU
    path.  This lets them stay array-generic without carrying a backend around.
    """
    mod = type(a).__module__
    if mod.split(".")[0] == "torch":
        import torch
        return torch.exp(a)
    return np.exp(a)
