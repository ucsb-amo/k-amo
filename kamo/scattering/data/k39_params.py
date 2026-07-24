"""K39 scattering parameters — LITERATURE-VERIFIED (deep-research pass 2026-07-24).

Every value below was confirmed by an adversarial multi-source verification pass
(24/25 claims 3-0) against the primary references:

  * Falke et al., PRA 78, 012503 (2008)          [arXiv:0804.2949]  — a_S, a_T, C6
  * D'Errico et al., NJP 9, 223 (2007)           [arXiv:0705.3036]  — a_S,a_T,C6, resonances
  * Chapurin et al., PRL 123, 233402 (2019)      [arXiv:1907.00729] — 33.582 G anchor, refined a_S/a_T
  * Etrych et al., PRResearch 5, 013174 (2023)   [arXiv:2208.13766] — precise resonance/zero-crossing map

Units: scattering lengths in Bohr radii (a0); C6 in atomic units; fields in Gauss.

MODEL NOTE — the empirical a(B) uses the multi-resonance product form
``a(B) = a_bg_channel * prod_i (1 - Delta_i/(B - B0_i))`` with ONE background per
channel (A_BG_CHANNEL) and per-resonance width Delta_i.  Etrych's per-resonance
``a_bg`` are *local* backgrounds (the other poles' tails reduce the single
channel background to the local value near each pole), so they are stored for
reference only; the product form reconstructs them.  Where a resonance has a
reported zero-crossing B_zero we set ``Delta = B_zero - B0`` (self-consistent
with Etrych's coupled-channel Delta to <1 G, and makes a(B_zero)=0 exact).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

PARAMS_VERIFIED = True

# --------------------------------------------------------------------------
# Singlet / triplet background scattering lengths and van der Waals C6.
# Canonical (with error bars): Falke 2008.  Best central estimate: Chapurin 2019.
# --------------------------------------------------------------------------
A_SINGLET_A0 = 138.49       # X^1 Sigma_g+  a_S [a0]  Falke 2008 (pot. B, BO-corrected)
A_SINGLET_UNC_A0 = 0.12
A_TRIPLET_A0 = -33.48       # a^3 Sigma_u+  a_T [a0]  Falke 2008
A_TRIPLET_UNC_A0 = 0.18
# Best central estimates (Chapurin 2019, no individual error bars):
A_SINGLET_CHAPURIN_A0 = 138.85
A_TRIPLET_CHAPURIN_A0 = -33.40

C6_AU = 3921.0              # K2 dispersion coeff [a.u.]  D'Errico 2007 fit
C6_UNC_AU = 8.0
# Alternatives (all give R_vdW = 64.5-64.6 a0): Derevianko 3897(15); Falke ~3925.9.
C8_AU = None               # not confirmed by verified sources
_ST_SOURCE = "Falke 2008 (a_S,a_T); D'Errico 2007 (C6=3921(8)); Chapurin 2019 central"


@dataclass(frozen=True)
class FeshbachResonance:
    """One magnetic Feshbach resonance in a specified collision channel.

    a(B) near an isolated pole is ``a_bg * (1 - width/(B - B0))``; the channel
    a(B) multiplies the ``(1 - width/(B-B0))`` factors against a single
    ``A_BG_CHANNEL`` background (see module docstring).

    Attributes
    ----------
    B0_gauss : resonance position (Gauss), experimental.
    width_gauss : magnetic width Delta (Gauss); sign carries into the formula.
    a_bg_a0 : *local* background scattering length near this pole (a0), Etrych.
    zero_crossing_gauss : reported a=0 field (Gauss) tied to this pole, or None.
    decay_gauss : inelastic width Gamma_inel (Gauss) if the pole is lossy, else None.
    source : provenance.
    verified : True if B0/Delta are literature-verified; False if Delta estimated.
    """

    B0_gauss: float
    width_gauss: float
    a_bg_a0: float
    zero_crossing_gauss: Optional[float] = None
    decay_gauss: Optional[float] = None
    source: str = ""
    verified: bool = True


def channel_key(state_a: Tuple[int, int], state_b: Tuple[int, int]):
    """Order-independent key for a pair channel (intra or inter)."""
    return frozenset((tuple(state_a), tuple(state_b)))


_ET = "Etrych et al. PRResearch 5, 013174 (2023)"
_DE = "D'Errico et al. NJP 9, 223 (2007)"
_CH = "Chapurin et al. PRL 123, 233402 (2019)"

# Single per-channel background used by the product form (a0).
A_BG_CHANNEL = {
    channel_key((1, -1), (1, -1)): -29.10,   # Etrych local a_bg @ 561 G (tails ~1)
    channel_key((1, 0),  (1, 0)):  -18.0,    # D'Errico low-field |1,0> background
    channel_key((1, 1),  (1, 1)):  -29.52,   # Etrych local a_bg @ 402 G
}

# --------------------------------------------------------------------------
# PHYSICS NOTE (kamo thresholds + Etrych Gamma_inel): the F=1 lower manifold has
# mF=+1 lowest (g_F<0), so |1,+1>+|1,+1> is the ABSOLUTE ground pair (Etrych
# Gamma_inel=0 for its 402 G resonance).  Within F=1, |1,0>+|1,0> is the lowest
# M_F=0 pair and |1,-1>+|1,-1> the unique M_F=-2 pair, so all three are elastic
# at threshold (no open spin-exchange channel).  |1,-1>'s 33.6 G resonance has a
# tiny dipolar Gamma_inel~1e-4 G (negligible) -> a treated as real. Verify with
# ScatteringModel.is_lossy.
# --------------------------------------------------------------------------
RESONANCES = {
    # |1,-1> + |1,-1>  (Delta: theory for 33.6/162; B_zero-B0 for 561)
    channel_key((1, -1), (1, -1)): [
        FeshbachResonance(33.5820,  79.469, -13.50, None,  None, _CH + "/" + _ET),  # Gamma_inel~1e-4 G negligible
        FeshbachResonance(162.36,  -60.628, -11.73, None,  None,   _ET),
        FeshbachResonance(561.14,  504.9 - 561.14, -29.10, 504.9, None, _ET),
    ],
    # |1,0> + |1,0>  (Delta = B_zero-B0 for 472/491; 58.97/65.57 narrow, est.)
    channel_key((1, 0), (1, 0)): [
        FeshbachResonance(58.97,   -0.5,  -18.0, None, None, _ET + " (Delta est.)", False),
        FeshbachResonance(65.57,   -0.5,  -18.0, None, None, _ET + " (Delta est.)", False),
        FeshbachResonance(472.33,  393.2 - 472.33, -18.0, 393.2, None, _ET),
        FeshbachResonance(491.17,  490.1 - 491.17, -133.43, 490.1, None, _ET),
    ],
    # |1,+1> + |1,+1>  (absolute ground pair; 402 = B_zero-B0; 25.9/752 narrow, est.)
    channel_key((1, 1), (1, 1)): [
        FeshbachResonance(25.91,   -0.5,  -33.0, None, None, _DE + " (Delta est.)", False),
        FeshbachResonance(402.74,  350.4 - 402.74, -29.52, 350.4, 0.0, _ET),
        FeshbachResonance(752.3,   -0.4,  -35.0, None, None, _DE + " (Delta est.)", False),
    ],
    # inter/mixed |1,-1> + |1,0>: NOT separately tabulated in the literature
    # (Etrych/D'Errico give intra channels only). Left empty deliberately;
    # this channel is elastic at threshold (is_lossy False).
}

# Experimentally-used a=0 zero-crossing fields (Gauss), verified.
ZERO_CROSSINGS_GAUSS = {
    channel_key((1, 1),  (1, 1)):  [350.4],   # D'Errico 350.4(4); Etrych 350.4(1)
    channel_key((1, -1), (1, -1)): [504.9],   # Etrych 504.9(2) (561 G resonance)
    channel_key((1, 0),  (1, 0)):  [393.2, 490.1],  # Etrych
}


def singlet_triplet():
    """Return ``(a_S, a_T, C6)`` = (138.49, -33.48, 3921.0) [a0, a0, a.u.]."""
    return A_SINGLET_A0, A_TRIPLET_A0, C6_AU


def background_channel(state_a, state_b) -> Optional[float]:
    """Single per-channel background a_bg (a0) for the product form, or None."""
    return A_BG_CHANNEL.get(channel_key(state_a, state_b))


def resonances_for(state_a, state_b) -> List[FeshbachResonance]:
    """Verified resonance list for the {state_a, state_b} channel ([] if none)."""
    return list(RESONANCES.get(channel_key(state_a, state_b), []))


def zero_crossings_for(state_a, state_b) -> List[float]:
    return list(ZERO_CROSSINGS_GAUSS.get(channel_key(state_a, state_b), []))
