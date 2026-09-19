"""39K: :class:`Potassium39`, the default atom of kamo.

The species-agnostic machinery is :class:`~kamo.atom_properties.alkali.PortalAlkali`;
this subclass adds what only potassium has: the curated hyperfine module
(:mod:`kamo.atom_properties.hyperfine`, measured 2022 survey values with
theory and extrapolation fill-in) and the 39K scattering-length tables
(:mod:`kamo.scattering`).
"""

from __future__ import annotations

import arc

from kamo.atom_properties.alkali import (PortalAlkali, _angular_factor,  # noqa: F401
                                         _own_manifolds, _crossings,
                                         _PORTAL_SIGN_TOLERANCE)


class Potassium39(PortalAlkali, arc.Potassium39):
    """ARC's Potassium39 with UDel-portal E1 data and kamo's structure methods.

    See :mod:`kamo.atom_properties.alkali` for the portal, energy and Zeeman
    conventions shared by every alkali. Potassium-specific: hyperfine
    constants come from :mod:`kamo.atom_properties.hyperfine` (Allegrini,
    Arimondo & Orozco 2022 measurements, portal theory where better, n*^3
    extrapolation beyond) instead of the portal-then-ARC ladder;
    :meth:`get_scattering_length` reads the 39K coupled-channels tables.

    Energies are ARC's tabulated NIST levels (``preferQuantumDefects=False``).
    ARC's default computes every K level from Rydberg quantum defects instead,
    which puts D2 2.55 GHz and D1 4.71 GHz too high and 3D 67 cm^-1 too low.
    Pass ``preferQuantumDefects=True`` to get those energies back (before
    2026-09 kamo used them).
    """

    species = "K39"

    def __init__(self, use_portal=True, portal_species="K1", preferQuantumDefects=False):
        super().__init__(use_portal=use_portal, portal_species=portal_species,
                         preferQuantumDefects=preferQuantumDefects)

    def hyperfine_constants(self, n, l, j):
        """Hyperfine ``A`` and ``B`` from :func:`kamo.atom_properties.hyperfine.hyperfine_constants`
        (39K), whatever ``use_portal`` is: kamo's structure calculations
        (:mod:`kamo.hamiltonian`) always use the curated constants, as they did
        before 2026-09. Only :meth:`getHFSCoefficients` honours
        ``use_portal=False`` by returning ARC's table (Arimondo 1977: its 4P_1/2
        A is 28.85 MHz, against 27.793(71) MHz recommended now)."""
        from kamo.atom_properties.hyperfine import hyperfine_constants
        return hyperfine_constants(int(n), int(l), float(j), iso=39)

    def get_scattering_length(self, f, mf, b,
                            f2=None, mf2=None,
                            interp=True,
                            method='table',
                            return_complex=False):
        """s-wave scattering length (a0) of the pair |f,mf> + |f2,mf2> at field b (G).

        Thin wrapper around :func:`kamo.scattering.lookup.scattering_length`.

        Args:
            f (int), mf (int): hyperfine state of the first atom.
            b (float or array): magnetic field in Gauss.
            f2 (int), mf2 (int), optional: state of the second atom.  Omit both
                for two atoms in |f,mf>.
            interp (bool, optional): method='kokkelmans' only; interpolate the
                0.5 G table instead of taking the nearest point.
            method (str, optional):
                'table': (default) the calibrated coupled-channels model, precomputed for all
                    36 pairs of ground states on 0-1000 G and shipped with kamo.
                    Instant, and the only method covering every pair.  Accurate to
                    ~1e-3 relative above 0.01 G and exact at b = 0; in
                    0 < b < 0.01 G unresolved channel-opening jumps make it
                    indicative only -- use 'cc' there.
                'kokkelmans': S. Kokkelmans' tables on the
                    Tweezers G: drive, same-state pairs only, 1-1000 G.  Needs the
                    share mounted, and differs from the calibrated model by ~2% at
                    some fields.  NOTE: this default disagrees with 'table' below;
                    pass method explicitly if it matters.
                'cc': the same model computed directly (~1.5 s setup, then
                    ~40-90 ms per field, memoised).
                'empirical': measured-resonance model.  Instant, but only for the
                    F=1 channels with measured resonances.
            return_complex (bool, optional): return a_re - i a_im (lossy
                channels, e.g. F=2) instead of the real part.

        Returns:
            float for scalar b, else ndarray of b's shape.

        Raises:
            ValueError: invalid states, only one of f2/mf2 given, no data for
                the pair with this method, or b outside the method's range.
        """
        from kamo.scattering.lookup import scattering_length
        if (f2 is None) != (mf2 is None):
            raise ValueError("give both f2 and mf2, or neither (same-state pair)")
        second = None if f2 is None else (f2, mf2)
        return scattering_length((f, mf), second, b, method=method, interp=interp,
                                 return_complex=return_complex)
    
