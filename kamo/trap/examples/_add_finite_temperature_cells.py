"""Insert the finite-temperature demo section into trap_demo.ipynb (nbformat; writes
the notebook, does not execute it).  Idempotent: cells tagged ``finite-T`` are
replaced.  Run from anywhere: ``python _add_finite_temperature_cells.py``."""

import pathlib

import nbformat

HERE = pathlib.Path(__file__).resolve().parent
NB = HERE / "trap_demo.ipynb"
TAG = "finite-T"

MD_INTRO = r"""## 5b. Finite temperature

`solve(trap, N, "gp", T_K=...)` puts a Hartree-Fock thermal cloud in equilibrium with the GP condensate,
on the real potential and truncated at the escape saddle. Two facts set the picture for this trap:

* the ideal harmonic $T_c$ for $N = 500$ is 152 nK, and the trap is only 208 nK deep: $\eta = U/k_BT$ is 1.4 at $T_c$,
  so *every* thermal cloud worth modelling is evaporating at its edge and "$T_c$" is not a number this trap has;
* only ~3 single-particle states sit below the condensate's own $\mu - V_{\min}$ (which is 95% zero-point energy),
  so the textbook semiclassical thermal cloud (`thermal="lda"`) is low by 2-6x in $N_{\rm th}$. The default
  `thermal="hybrid"` uses calibrated discrete levels below $E_0 + 2\hbar\omega_{\max}$ and a semiclassical tail above.

The returned `TrapCloud` carries both components on one grid; `info` reports $\eta$, the truncated fraction,
the seam diagnostics and whether the result is `trustworthy`."""

CODE_TC = """from kamo.trap import critical_temperature
from kamo.BEC_properties.thermal import IdealHarmonicBoseGas

print(critical_temperature(trap, N_ATOMS, a_scattering=gp.a_scattering).summary())
ref = IdealHarmonicBoseGas.from_trap(trap, N_ATOMS, 30e-9, a_scattering=gp.a_scattering)
print()
print(ref.summary())          # the closed-form ideal gas: the reference a bimodal fit assumes"""

CODE_SOLVE = """T_NK = 30.0                                            # nK; eta = U/kT ~ 6.8 here
warm = solve(trap, N_ATOMS, "gp", T_K=T_NK * 1e-9, condensate_options=dict(points_per_scale=3.0))
print(warm.summary())
print()
print(warm.info.summary())"""

MD_PLOT = r"""The condensate is barely changed by the thermal cloud (the thermal mean field on it is ~1.5% of $k_BT$), but the
thermal wings extend ~30 µm along the beam, to the escape saddles. On a linear scale they are invisible; the
bimodal profile is a log-scale object."""

CODE_PLOT = r"""fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), layout="constrained")
tplt.plot_bimodal_profile(warm, "z", log=True, ax=axes[0])
tplt.plot_column_density(warm, "x", component="condensate", ax=axes[1], n=(161, 161))
tplt.plot_column_density(warm, "x", component="thermal", ax=axes[2], n=(161, 161),
                         half_width=(2.5e-6, 2.5e-6))
plt.show()

fig, ax = plt.subplots(figsize=(7, 3.2), layout="constrained")
s = np.linspace(-32e-6, 32e-6, 641)
c = warm.centroid
for comp, ls in (("condensate", "-"), ("thermal", "--")):
    ax.semilogy(s * 1e6, warm.density(c[0] + s, c[1], c[2], component=comp) * 1e-6 + 1e-3, ls,
                label=comp)                                  # + 1e-3: a floor so log(0) does not blank the axis
ax.axvline(trap.escape_saddle[0] * 1e6, color="tab:red", ls=":", lw=0.9, label="escape saddle")
ax.axvline(-trap.escape_saddle[0] * 1e6, color="tab:red", ls=":", lw=0.9)
ax.set_ylim(1e6, None)                                   # 1e6 cm^-3: the bottom of the wings
ax.set_xlabel(r"$x$ ($\mu$m, along the beam)"); ax.set_ylabel(r"$n$ (cm$^{-3}$)")
ax.set_title(f"axial cut at {T_NK:.0f} nK: the thermal cloud fills the basin out to the saddles")
ax.legend(fontsize="small")
plt.show()"""

MD_SWEEP = r"""### The condensate fraction against temperature

The solver object caches its spectrum and grid, so a sweep costs one cold solve plus a warm one per point.
The ideal harmonic gas (semiclassical and the exact finite-$N$ sum) is drawn for reference: the real trap holds
*more* condensate than the harmonic ideal gas at the same $T$, because truncation at the saddle removes the
high-lying thermal states. Results are cached in `./output` so re-running the notebook is cheap."""

CODE_SWEEP = r"""import os, warnings
from kamo.trap import FiniteTemperatureSolver, ModelValidityWarning

os.makedirs("output", exist_ok=True)                    # next to the notebook (the kernel's cwd)
T_grid_nK = np.array([20.0, 30.0, 45.0, 60.0, 80.0, 100.0])
cache = "output/trap_demo_condensate_fraction.npz"
key = f"N={N_ATOMS},f={F_RADIAL},T={T_grid_nK.tolist()}"
hit = False
if os.path.exists(cache):
    with np.load(cache) as d:
        if str(d["key"]) == key:
            frac, eta, hit = d["frac"], d["eta"], True
            print("cache hit:", cache)
if not hit:
    ft = FiniteTemperatureSolver(trap, a_scattering=gp.a_scattering,
                                 condensate_options=dict(points_per_scale=3.0))
    frac, eta = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ModelValidityWarning)      # eta < 5 warns, on purpose
        for T in T_grid_nK:
            c = ft.solve(N_ATOMS, T * 1e-9)
            frac.append(c.condensate_fraction); eta.append(c.info.eta)
            print(f"{T:6.1f} nK: N0/N = {c.condensate_fraction:.3f}, eta = {c.info.eta:.2f}, "
                  f"{c.info.passes} passes, {c.info.wall_time_s:.0f} s")
    frac, eta = np.array(frac), np.array(eta)
    np.savez_compressed(cache, key=key, frac=frac, eta=eta, T_nK=T_grid_nK)

T_dense = np.linspace(5, 160, 200)
ideal = np.array([IdealHarmonicBoseGas.from_trap(trap, N_ATOMS, T * 1e-9).condensate_fraction for T in T_dense])
exact = np.array([IdealHarmonicBoseGas.from_trap(trap, N_ATOMS, T * 1e-9).exact_condensate_fraction()
                  for T in T_dense])
fig, ax = plt.subplots(figsize=(6, 3.6), layout="constrained")
ax.plot(T_dense, ideal, color="0.6", lw=1.2, label="ideal harmonic, semiclassical ($1 - (T/T_c)^3$)")
ax.plot(T_dense, exact, color="0.3", lw=1.2, ls="--", label="ideal harmonic, exact finite-$N$ sum")
ax.plot(T_grid_nK, frac, "o-", color="tab:blue", label="real trap, hybrid HF (truncated at the saddle)")
for T, f, e in zip(T_grid_nK, frac, eta):
    ax.annotate(f"$\\eta$={e:.1f}", (T, f), textcoords="offset points", xytext=(4, 6), fontsize=7)
ax.axvline(trap.depth_K * 1e9, color="tab:red", ls=":", lw=0.9, label="trap depth")
ax.set_xlabel("T (nK)"); ax.set_ylabel(r"$N_0/N$"); ax.set_ylim(0, 1.02)
ax.set_title(f"condensate fraction, N = {N_ATOMS}: no sharp transition, and the depth is the scale")
ax.legend(fontsize="x-small")
plt.show()"""

APPROX_OLD = "* Zero temperature."
APPROX_NEW = (r"* Finite temperature is Hartree-Fock: one-way mean-field coupling by default "
              r"(`mean_field_feedback=True` adds $2gn_{\rm th}$ to the condensate; the anomalous average "
              r"is never included, quantum depletion is $2\times10^{-4}$ here), a truncated quasi-equilibrium "
              r"rather than a true equilibrium below $\eta \approx 5$ (the solver warns), and a separable "
              r"product basis calibrated on the soft axis -- a crossed trap far from separable is flagged by "
              r"`seam_count_ratio`.")


def build():
    nb = nbformat.read(NB, as_version=4)
    nb.cells = [c for c in nb.cells if TAG not in c.get("metadata", {}).get("tags", [])]
    # the approximations cell: replace the "Zero temperature" bullet
    for c in nb.cells:
        if c.cell_type == "markdown" and "## Approximations" in c.source and APPROX_OLD in c.source:
            lines = c.source.split("\n")
            out, skip = [], False
            for line in lines:
                if line.startswith(APPROX_OLD):
                    out.append(APPROX_NEW)
                    skip = True
                    continue
                if skip and line.startswith("  "):
                    continue
                skip = False
                out.append(line)
            c.source = "\n".join(out)
    # insert after the last cell of section 5 (the density line cuts) = before "## 6."
    idx = next(i for i, c in enumerate(nb.cells)
               if c.cell_type == "markdown" and c.source.lstrip().startswith("## 6."))
    new = [nbformat.v4.new_markdown_cell(MD_INTRO), nbformat.v4.new_code_cell(CODE_TC),
           nbformat.v4.new_code_cell(CODE_SOLVE), nbformat.v4.new_markdown_cell(MD_PLOT),
           nbformat.v4.new_code_cell(CODE_PLOT), nbformat.v4.new_markdown_cell(MD_SWEEP),
           nbformat.v4.new_code_cell(CODE_SWEEP)]
    for c in new:
        c.metadata["tags"] = [TAG]
    nb.cells[idx:idx] = new
    nbformat.write(nb, NB)
    print(f"wrote {NB} with {len(new)} finite-T cells at index {idx}")


if __name__ == "__main__":
    build()
