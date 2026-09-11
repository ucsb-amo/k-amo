"""Measured 39K ground-state Feshbach resonances, zero crossings and dimer energies.

Literature database (deep-research pass 2026-09-10) used as the *input* to the
coupled-channels calibration (:mod:`kamo.scattering.calibration`) and as the
source of the empirical backend's resonance table.  Every number was read from
the cited table/page of the primary source (arXiv LaTeX/PDF, cross-checked
against the published version where they differ).

Conventions
-----------
* States are low-field ``|F, mF>`` labels, ``(F, mF)`` tuples.  Papers using
  letters map as ``a=|1,1>, b=|1,0>, c=|1,-1>``.
* Fields in Gauss.  ``a(B) = a_bg (1 - Delta/(B - B0))``, so ``B_zero = B0 + Delta``.
* ``pole_strength = a_bg*Delta`` (a0 G): the model-independent residue,
  ``a ~ -a_bg*Delta/(B - B0)`` at the pole.  Etrych's experimental
  "effective width" column is this product.
* ``theory`` columns are the MOLSCAT characterisation of Etrych et al. 2023
  (Tiemann 2020 "model 3" potentials + second-order spin-orbit), which is also
  the sensible default for ``Delta``/``a_bg``: experiments measure ``B0`` (and at
  best ``a_bg*Delta``), not the split.
* ``gamma_inel`` is Etrych's signed ``Gamma_inel = -2 a_bg Delta / a_res`` (G).

Sources
-------
* Etrych et al., PRResearch 5, 013174 (2023), arXiv:2208.13766 — Tables I, II, IV
  (published version; adds interstate zero crossings missing from v1).
* Chapurin et al., PRL 123, 233402 (2019), arXiv:1907.00729 (+ Suppl.).
* Roy et al., PRL 111, 053202 (2013), arXiv:1303.3843 (Suppl. table).
* D'Errico et al., NJP 9, 223 (2007), arXiv:0705.3036 — Table 1 (column is -Delta).
* Tanzi et al., PRA 98, 062712 (2018), arXiv:1810.12453 — |1,0>+|1,-1> at 113.76 G.
* Etrych et al., PRX 15, 021070 (2025), arXiv:2402.14816 — 526.16(3) G refinement.
* Spin-mixture CC scattering lengths: Lavoine 2021, Hammond 2022, Eid 2025
  (Tiemann 2020 potentials), Sanz 2022 and Frolian 2022 (Roy 2013 potentials).
* Zaheer et al., arXiv:2608.05512 (2026) — trap-depth-extrapolated positions, p-wave 52.4 G.
* Higher-partial-wave |1,-1> features: Xie 2020 (PRL 125, 243401), Zhang 2025
  (NJP 27, 063002), Fouche 2019 (PRA 99, 022701).
* Laskowski & Mehta, arXiv:2307.16654 (2023) — independent CC on the Falke 2008
  curves (a_S = 138.808, a_T = -33.391 a0), a check on :mod:`..tiemann`.
* Tiemann et al., PRResearch 2, 013366 (2020), arXiv:1912.07395 — Table I
  (d-wave features, |2,-2> data), Table IV (a_S, a_T).
* Lysebo & Veseth, PRA 81, 032702 (2010) — Table II (theory only).
* Zaccanti et al., Nat. Phys. 5, 586 (2009); Fletcher et al., Science 355, 377
  (2017); Fattori et al., PRL 101, 190405 (2008); Eigen et al., PRX 6, 041058 (2016).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

State = Tuple[int, int]

_ET = "Etrych 2023 PRR 5,013174"
_CH = "Chapurin 2019 PRL 123,233402"
_ROY = "Roy 2013 PRL 111,053202"
_DE = "D'Errico 2007 NJP 9,223"
_TZ = "Tanzi 2018 PRA 98,062712"
_TI = "Tiemann 2020 PRR 2,013366"
_LV = "Lysebo & Veseth 2010 PRA 81,032702"


@dataclass(frozen=True)
class MeasuredResonance:
    """One measured magnetic Feshbach resonance (position ``B0 +- B0_unc``, Gauss)."""

    state_a: State
    state_b: State
    B0: float
    B0_unc: float
    source: str
    arxiv: str = ""
    method: str = ""
    partial_wave: str = "s"
    # coupled-channels characterisation reported alongside (Etrych Table IV)
    B0_theory: Optional[float] = None
    width_theory: Optional[float] = None          # Delta (G)
    a_bg_theory: Optional[float] = None           # local background (a0)
    gamma_inel_theory: Optional[float] = None     # signed Gamma_inel (G)
    zero_crossing_theory: Optional[float] = None  # B_zero (G)
    # measured pole strength a_bg*Delta (a0 G), if reported
    pole_strength: Optional[float] = None
    pole_strength_unc: Optional[float] = None
    lossy: bool = False                           # open s-wave inelastic channel
    use_in_fit: bool = True
    notes: str = ""

    @property
    def channel(self) -> frozenset:
        return frozenset((tuple(self.state_a), tuple(self.state_b)))

    @property
    def label(self) -> str:
        (Fa, ma), (Fb, mb) = sorted((tuple(self.state_a), tuple(self.state_b)), reverse=True)
        return f"|{Fa},{ma:+d}>+|{Fb},{mb:+d}>"


@dataclass(frozen=True)
class MeasuredZeroCrossing:
    state_a: State
    state_b: State
    B_zero: float
    B_zero_unc: float
    source: str
    notes: str = ""


# --------------------------------------------------------------------------
# Best-value set: one entry per resonance.  These are the calibration inputs.
# --------------------------------------------------------------------------
_R = MeasuredResonance
RESONANCES: List[MeasuredResonance] = [
    # |1,+1> + |1,+1>   (absolute ground pair)
    _R((1, 1), (1, 1), 25.91, 0.06, _ET, "2208.13766", "loss spectroscopy",
       B0_theory=25.879, width_theory=-0.465, a_bg_theory=-33.10, gamma_inel_theory=0.0),
    _R((1, 1), (1, 1), 402.74, 0.01, _ET, "2208.13766", "Ramsey bound-state spectroscopy",
       B0_theory=402.554, width_theory=-51.291, a_bg_theory=-29.52, gamma_inel_theory=0.0,
       zero_crossing_theory=350.714, pole_strength=1530.0, pole_strength_unc=20.0,
       notes="constrained fit (Delta = B_zero - B0); Fletcher 2017 gives 402.70(3)"),
    _R((1, 1), (1, 1), 752.3, 0.1, _DE, "0705.3036", "loss spectroscopy",
       B0_theory=752.255, width_theory=-0.397, a_bg_theory=-35.30, gamma_inel_theory=0.0,
       notes="Etrych quotes D'Errico's position"),
    # |1,0> + |1,0>
    _R((1, 0), (1, 0), 58.97, 0.12, _ET, "2208.13766", "loss spectroscopy",
       B0_theory=58.949, width_theory=-6.648, a_bg_theory=-29.22, gamma_inel_theory=-5e-6,
       notes="overlaps 65.57; Roy 2013 gives 58.92(3)(10); a d-wave resonance sits at 60.1 G"),
    _R((1, 0), (1, 0), 65.57, 0.23, _ET, "2208.13766", "loss spectroscopy",
       B0_theory=65.573, width_theory=-3.361, a_bg_theory=-41.87, gamma_inel_theory=-1.83e-4,
       notes="Roy 2013 gives 65.67(5)"),
    _R((1, 0), (1, 0), 472.33, 0.01, _ET, "2208.13766", "bound-state spectroscopy",
       B0_theory=472.118, width_theory=-117.78, a_bg_theory=-16.96, gamma_inel_theory=-8e-6,
       zero_crossing_theory=393.635, pole_strength=2040.0, pole_strength_unc=20.0),
    _R((1, 0), (1, 0), 491.17, 0.07, _ET, "2208.13766", "loss spectroscopy",
       B0_theory=490.930, width_theory=-1.045, a_bg_theory=-133.43, gamma_inel_theory=-2e-6,
       zero_crossing_theory=489.931, pole_strength=140.0, pole_strength_unc=30.0,
       notes="narrow, on the wing of 472.33; pole strength is the effective a'_bg*Delta (App. C)"),
    # |1,-1> + |1,-1>
    _R((1, -1), (1, -1), 33.5820, 0.0014, _CH, "1907.00729", "dimer binding energies (tuned CC)",
       B0_theory=33.568, width_theory=79.469, a_bg_theory=-13.50, gamma_inel_theory=1.0e-4,
       pole_strength=-1073.5,
       notes="uncertainty = fit (+) 0.5 mG calibration; no zero crossing"),
    _R((1, -1), (1, -1), 162.36, 0.02, _ET, "2208.13766", "bound-state spectroscopy",
       B0_theory=162.347, width_theory=-60.628, a_bg_theory=-11.73, gamma_inel_theory=-1.8e-4,
       pole_strength=760.0, pole_strength_unc=20.0, notes="no zero crossing"),
    _R((1, -1), (1, -1), 561.14, 0.02, _ET, "2208.13766", "bound-state spectroscopy",
       B0_theory=560.935, width_theory=-55.358, a_bg_theory=-29.10, gamma_inel_theory=-9e-6,
       zero_crossing_theory=504.717, pole_strength=1660.0, pole_strength_unc=20.0,
       notes="Etrych Table III prints 561.14(1); Roy 2013 gives 560.72(20)"),
    # |1,+1> + |1,0>
    _R((1, 1), (1, 0), 25.81, 0.06, _ET, "2208.13766", "loss spectroscopy (mixture)",
       B0_theory=25.785, width_theory=-1.345, a_bg_theory=-35.35, gamma_inel_theory=-5e-6),
    _R((1, 1), (1, 0), 39.81, 0.06, _ET, "2208.13766", "loss spectroscopy (mixture)",
       B0_theory=39.835, width_theory=-2.061, a_bg_theory=-40.99, gamma_inel_theory=-5.6e-5),
    _R((1, 1), (1, 0), 445.42, 0.03, _ET, "2208.13766", "loss + rf association",
       B0_theory=445.293, width_theory=-37.569, a_bg_theory=-30.35, gamma_inel_theory=-9e-6,
       zero_crossing_theory=407.479, pole_strength=1110.0, pole_strength_unc=40.0),
    # |1,+1> + |1,-1>   (lossy: decays to |1,0>+|1,0>)
    _R((1, 1), (1, -1), 77.6, 0.4, _ET, "2208.13766", "loss spectroscopy (mixture)",
       B0_theory=77.725, width_theory=-95.738, a_bg_theory=-27.35, gamma_inel_theory=-8.738e-3,
       lossy=True, use_in_fit=False, notes="strongly inelastic"),
    _R((1, 1), (1, -1), 501.6, 0.3, _ET, "2208.13766", "loss spectroscopy (mixture)",
       B0_theory=501.480, width_theory=-18.304, a_bg_theory=-49.05, gamma_inel_theory=-3.364e-3,
       zero_crossing_theory=486.845, lossy=True, use_in_fit=False),
    # |1,0> + |1,-1>
    _R((1, 0), (1, -1), 113.76, 0.01, _TZ, "1810.12453", "loss spectroscopy (mixture)",
       B0_theory=113.768, width_theory=-19.215, a_bg_theory=-39.33, gamma_inel_theory=-2.95e-4,
       zero_crossing_theory=97.793, pole_strength=715.0, pole_strength_unc=7.0,
       notes="Aarhus Bose-polaron resonance; values via Etrych Table IV"),
    _R((1, 0), (1, -1), 526.16, 0.03, "Etrych 2025 PRX 15,021070", "2402.14816",
       "thermal-gas dimer binding energies",
       B0_theory=525.995, width_theory=-28.249, a_bg_theory=-31.04, gamma_inel_theory=-9e-6,
       zero_crossing_theory=497.648, pole_strength=970.0, pole_strength_unc=50.0,
       notes="refines Etrych 2023's 526.21(5); theory/pole strength from Etrych 2023"),
]

# --------------------------------------------------------------------------
# Other measurements of the same resonances (kept for comparison, not fitted).
# --------------------------------------------------------------------------
ALTERNATIVE_MEASUREMENTS: List[MeasuredResonance] = [
    _R((1, 1), (1, 1), 25.85, 0.10, _DE, "0705.3036", use_in_fit=False),
    _R((1, 1), (1, 1), 403.4, 0.7, _DE, "0705.3036", "loss (asymmetric fit centre)", use_in_fit=False),
    _R((1, 1), (1, 1), 401.5, 0.5, _DE, "0705.3036", "molecule association", use_in_fit=False),
    _R((1, 1), (1, 1), 402.6, 0.2, _ROY, "1303.3843", use_in_fit=False),
    _R((1, 1), (1, 1), 402.50, 0.03, "Zaccanti 2009 Nat.Phys. 5,586", "0904.4453", use_in_fit=False),
    _R((1, 1), (1, 1), 402.70, 0.03, "Fletcher 2017 Science 355,377", "1608.04377",
       "thermal-gas expansion", use_in_fit=False),
    _R((1, 0), (1, 0), 59.3, 0.6, _DE, "0705.3036", use_in_fit=False),
    _R((1, 0), (1, 0), 58.92, 0.10, _ROY, "1303.3843", notes="0.03 stat, 0.10 syst", use_in_fit=False),
    _R((1, 0), (1, 0), 66.0, 0.9, _DE, "0705.3036", use_in_fit=False),
    _R((1, 0), (1, 0), 65.67, 0.05, _ROY, "1303.3843", use_in_fit=False),
    _R((1, 0), (1, 0), 471.0, 0.4, _ROY, "1303.3843", use_in_fit=False),
    _R((1, -1), (1, -1), 32.6, 1.5, _DE, "0705.3036", use_in_fit=False),
    _R((1, -1), (1, -1), 33.64, 0.15, _ROY, "1303.3843", use_in_fit=False),
    _R((1, -1), (1, -1), 162.8, 0.9, _DE, "0705.3036", use_in_fit=False),
    _R((1, -1), (1, -1), 162.35, 0.18, _ROY, "1303.3843", use_in_fit=False),
    _R((1, -1), (1, -1), 562.2, 1.5, _DE, "0705.3036", use_in_fit=False),
    _R((1, -1), (1, -1), 560.72, 0.20, _ROY, "1303.3843", use_in_fit=False),
    _R((1, 0), (1, -1), 526.21, 0.05, _ET, "2208.13766", "loss + rf association", use_in_fit=False),
    _R((1, 0), (1, -1), 113.83, 0.0, "Jorgensen 2016 PRL 117,055302 (arXiv v1)", "1604.07883",
       "dimer spectroscopy", use_in_fit=False, notes="no uncertainty given; Delta=-15.93, a_bg=-45.24"),
    _R((1, 1), (1, 0), 25.9, 0.4, "Zaheer 2026", "2608.05512", "loss, zero-trap-depth extrap.",
       use_in_fit=False),
    _R((1, 1), (1, 0), 40.7, 0.5, "Zaheer 2026", "2608.05512", "loss, zero-trap-depth extrap.",
       use_in_fit=False, notes="large ac-Stark shift of the dimer at 1064 nm; low weight"),
]

# --------------------------------------------------------------------------
# Non-s-wave features (need l>0 + spin-spin coupling; outside the s-wave model).
# --------------------------------------------------------------------------
HIGHER_PARTIAL_WAVE: List[MeasuredResonance] = [
    _R((1, 0), (1, 0), 60.1, 0.1, _ROY, "1303.3843", partial_wave="d", use_in_fit=False,
       notes="theory 60.5 G, -Delta=-0.6"),
    _R((1, -1), (1, -1), 395.1, 1.0, _TI + " (obs. Fouche 2019 PRA 99,022701)", "1912.07395",
       "inelastic-rate peak", partial_wave="d", use_in_fit=False, B0_theory=395.260),
    _R((2, -2), (2, -2), 125.94, 0.14, _TI, "1912.07395", "inelastic-rate peak",
       partial_wave="d", use_in_fit=False, B0_theory=126.060),
    _R((2, -2), (2, -2), 188.72, 0.05, _TI, "1912.07395", partial_wave="d", use_in_fit=False,
       B0_theory=188.720),
    _R((2, -2), (2, -2), 227.71, 0.60, _TI, "1912.07395", partial_wave="d", use_in_fit=False,
       B0_theory=227.840),
    _R((1, 0), (1, -1), 52.4, 0.1, "Zaheer 2026", "2608.05512", "loss", partial_wave="p",
       use_in_fit=False, B0_theory=52.6, notes="p-wave spin-exchange; theory Simoni (priv. comm.)"),
    _R((1, -1), (1, -1), 63.609, 0.004, "Xie 2020 PRL 125,243401", "2008.00396", "L2 loss peak",
       partial_wave="d", use_in_fit=False, notes="Zhang 2025 sees it at 63.67(9) and calls it g-wave"),
    _R((1, -1), (1, -1), 53.62, 0.07, "Zhang 2025 NJP 27,063002", "2505.08183", "loss",
       partial_wave="?", use_in_fit=False),
    _R((1, -1), (1, -1), 107.60, 0.02, "Zhang 2025 NJP 27,063002", "2505.08183", "loss",
       partial_wave="g", use_in_fit=False, B0_theory=106.82, notes="MQDT assignment"),
    _R((1, -1), (1, -1), 113.00, 0.13, "Zhang 2025 NJP 27,063002", "2505.08183", "loss",
       partial_wave="?", use_in_fit=False, notes="strongly temperature dependent"),
    _R((1, -1), (1, -1), 138.00, 0.19, "Zhang 2025 NJP 27,063002", "2505.08183", "loss",
       partial_wave="d", use_in_fit=False, B0_theory=141.84, notes="MQDT assignment"),
]

# --------------------------------------------------------------------------
# Other groups' coupled-channels scattering lengths at specific fields (a0).
# (state_a, state_b, B [G], a [a0], potentials/model, source)
# --------------------------------------------------------------------------
THEORY_POINTS = [
    ((1, -1), (1, 0), 56.830, -53.2, "Tiemann 2020", "Lavoine 2021 PRL 127,203402"),
    ((1, -1), (1, -1), 56.830, 33.4, "Tiemann 2020", "Lavoine 2021 PRL 127,203402"),
    ((1, 0), (1, 0), 56.830, 83.4, "Tiemann 2020", "Lavoine 2021 PRL 127,203402"),
    ((1, -1), (1, 0), 54.690, -54.2, "Tiemann 2020", "Hammond 2022 PRL 128,083401"),
    ((1, -1), (1, -1), 54.690, 37.9, "Tiemann 2020", "Hammond 2022 PRL 128,083401"),
    ((1, 0), (1, 0), 54.690, 36.9, "Tiemann 2020", "Hammond 2022 PRL 128,083401"),
    ((1, -1), (1, 0), 56.85, -53.2, "Tiemann 2020", "Eid 2025 PRA 112,063322"),
    ((1, -1), (1, -1), 56.85, 33.3, "Tiemann 2020", "Eid 2025 PRA 112,063322"),
    ((1, 0), (1, 0), 56.85, 84.3, "Tiemann 2020", "Eid 2025 PRA 112,063322"),
    ((1, -1), (1, 0), 57.280, -52.9, "Roy 2013 (Simoni)", "Sanz 2022 PRL 128,013201"),
    ((1, -1), (1, -1), 57.280, 32.5, "Roy 2013 (Simoni)", "Sanz 2022 PRL 128,013201"),
    ((1, 0), (1, 0), 57.280, 109.0, "Roy 2013 (Simoni)", "Sanz 2022 PRL 128,013201"),
    ((1, -1), (1, 0), 56.000, -53.5, "Roy 2013 (Simoni)", "Sanz 2022 PRL 128,013201"),
    ((1, -1), (1, -1), 56.000, 35.1, "Roy 2013 (Simoni)", "Sanz 2022 PRL 128,013201"),
    ((1, 0), (1, 0), 56.000, 57.9, "Roy 2013 (Simoni)", "Sanz 2022 PRL 128,013201"),
    ((1, 0), (1, 1), 374.29, -13.8, "Roy 2013", "Frolian 2022 Nature 608,293"),
    ((1, 0), (1, 0), 374.29, -4.9, "Roy 2013", "Frolian 2022 Nature 608,293"),
    ((1, 1), (1, 1), 374.29, 24.6, "Roy 2013", "Frolian 2022 Nature 608,293"),
    ((1, 0), (1, 1), 385.62, -10.8, "Roy 2013", "Frolian 2022 Nature 608,293"),
    ((1, 0), (1, 0), 385.62, -2.3, "Roy 2013", "Frolian 2022 Nature 608,293"),
    ((1, 1), (1, 1), 385.62, 61.0, "Roy 2013", "Frolian 2022 Nature 608,293"),
    ((1, 0), (1, 1), 397.01, -6.3, "Roy 2013", "Frolian 2022 Nature 608,293"),
    ((1, 0), (1, 0), 397.01, 1.3, "Roy 2013", "Frolian 2022 Nature 608,293"),
    ((1, 1), (1, 1), 397.01, 252.7, "Roy 2013", "Frolian 2022 Nature 608,293"),
]

# --------------------------------------------------------------------------
# Theory-only positions (not observed) that complete the empirical table.
# --------------------------------------------------------------------------
THEORY_ONLY: List[MeasuredResonance] = [
    _R((1, 0), (1, 0), 825.0, 0.0, _LV, "", "CC theory", use_in_fit=False,
       width_theory=-0.0361, a_bg_theory=-32.41),
    _R((1, 0), (1, 0), 832.4, 0.0, _LV, "", "CC theory", use_in_fit=False,
       width_theory=-0.5249, a_bg_theory=-36.31),
    _R((1, 0), (1, -1), 719.0, 0.0, _LV, "", "CC theory", use_in_fit=False,
       width_theory=-0.9820, a_bg_theory=-36.94),
]

ZERO_CROSSINGS: List[MeasuredZeroCrossing] = [
    MeasuredZeroCrossing((1, 1), (1, 1), 350.4, 0.1, "Fattori 2008 PRL 101,190405",
                         "Eigen 2016 refines to 350.45(3)"),
    MeasuredZeroCrossing((1, 0), (1, 0), 393.2, 0.2, _ET),
    MeasuredZeroCrossing((1, 0), (1, 0), 490.1, 0.2, _ET, "thermalization rate"),
    MeasuredZeroCrossing((1, -1), (1, -1), 504.9, 0.2, _ET),
    MeasuredZeroCrossing((1, 0), (1, 0), 43.0, 2.0, _ROY, "zero of the 58.9 G resonance; CC 44.3(2)"),
]

# Chapurin 2019 Suppl.: |1,-1>+|1,-1> Feshbach-dimer binding energies near 33.58 G,
# free space (confinement shift removed).  (B [G], B_unc, E_b/h [kHz], E_unc)
DIMER_BINDING_ENERGIES_1m1 = [
    (33.7420, 0.0003, 2.103, 0.056), (33.7978, 0.0004, 3.901, 0.078),
    (33.8494, 0.0005, 6.008, 0.085), (33.9575, 0.0003, 12.187, 0.057),
    (34.0078, 0.0004, 15.621, 0.067), (34.0622, 0.0007, 20.052, 0.122),
    (34.1644, 0.0004, 29.832, 0.083), (34.2663, 0.0006, 41.760, 0.103),
    (34.4812, 0.0004, 74.295, 0.093), (34.5940, 0.0004, 95.307, 0.137),
    (35.0060, 0.0005, 200.205, 0.406), (35.3198, 0.0005, 308.628, 0.436),
    (35.7593, 0.0006, 507.916, 0.582), (36.1582, 0.0007, 742.175, 1.071),
    (36.7303, 0.0005, 1167.237, 1.031),   # excluded from Chapurin's final fit
]

# Literature singlet/triplet scattering lengths (a0) for comparison with the fit
SINGLET_TRIPLET_LITERATURE = {
    "D'Errico 2007": (138.90, 0.15, -33.3, 0.3),
    "Falke 2008 (model A, no BO corr.)": (138.80, None, -33.41, None),
    "Falke 2008 (model B, BO corr.)": (138.49, 0.12, -33.48, 0.18),
    "Chapurin 2019": (138.85, None, -33.40, None),
    "Tiemann 2020 (model 3)": (138.759, 0.020, -33.413, 0.025),
    "Laskowski & Mehta 2023 (Falke 2008 curves, CC)": (138.808, None, -33.391, None),
}

# Warning from Zaheer 2026 (arXiv:2608.05512): in a 1064 nm optical trap the
# 33.6 G and 39.8 G resonances shift by ~+85-95 mG per uK of trap depth
# (anomalously large dimer polarizability) - relevant when comparing
# optical-trap measurements with free-space theory.


def _key(a, b):
    return frozenset((tuple(a), tuple(b)))


def resonances_for(state_a, state_b, include_alternatives: bool = False) -> List[MeasuredResonance]:
    """Best-value resonances of the ``{a, b}`` channel (optionally + alternatives)."""
    pool = RESONANCES + (ALTERNATIVE_MEASUREMENTS if include_alternatives else [])
    k = _key(state_a, state_b)
    return sorted((r for r in pool if r.channel == k), key=lambda r: r.B0)


def fit_set(include_lossy: bool = False) -> List[MeasuredResonance]:
    """Resonances used to calibrate the coupled-channels model."""
    return [r for r in RESONANCES if r.use_in_fit or (include_lossy and r.lossy)]


def channels() -> List[Tuple[State, State]]:
    """Distinct channels with measured resonances, as ``(state_a, state_b)``."""
    seen, out = set(), []
    for r in RESONANCES:
        if r.channel not in seen:
            seen.add(r.channel)
            out.append((tuple(r.state_a), tuple(r.state_b)))
    return out
