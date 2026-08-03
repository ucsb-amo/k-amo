"""Scattering-length backends (engines).

* :class:`~kamo.scattering.backends.empirical.EmpiricalBackend` — closed-form
  tabulated-resonance model ``a(B) = a_bg * prod_i (1 - Delta_i/(B - B0_i))``.
  Trustworthy given a verified resonance table.
* :class:`~kamo.scattering.backends.mqdt.MQDTBackend` — van der Waals MQDT with
  singlet/triplet frame transformation.  Frame-transform machinery is complete
  and self-validating; the multichannel ``a(B)`` output is **gated as
  UNVALIDATED** pending literature parameters.
"""

from .empirical import EmpiricalBackend

__all__ = ["EmpiricalBackend", "CoupledChannelsBackend"]


def __getattr__(name):
    if name == "CoupledChannelsBackend":
        from .coupled_channels import CoupledChannelsBackend
        return CoupledChannelsBackend
    raise AttributeError(name)
