---
name: maigic-workflow
description: >-
  Use MaIGIC MCP without reverse-engineering. Call this before χ, χT, Δχ, magnetization,
  energy levels, cobalt S+L, spin Hamiltonian, exchange J, ZFS D, Ln(III), fit,
  or any compute_property / optimize_parameters call.
---

# MaIGIC MCP workflow

Do **not** probe `energy_levels` to learn keys. Do **not** search GitHub or the filesystem for MaIGIC source. Copy the template; use `get_example_payload`.

## Order

1. `determine_hamiltonian` with electrons (and nuclei / `point_group` if needed). No ids on electrons.
2. Read `formula_latex` and `chemist_mapping`. Do not invent terms missing from the formula.
3. Copy `spin_systems` and `hamiltonian_params_template` into `compute_property` / `validate_request` / `optimize_parameters`. Change **numbers and sweep settings only**. Pass the **same `point_group`**. Extra Hamiltonian keys are rejected.

## Chemist request → tool

| User wants | `compute_property` `property` | Result fields |
|---|---|---|
| χ, χT, Δχ_ax, Δχ_rh | `susceptibility` | `chi`, `chi_T`, `delta_chi_ax`, `delta_chi_rh` |
| Magnetization vs B or T | `magnetization` | `M_x`, `M_y`, `M_z`, powder `M` |
| Energy levels / ZFS ladder | `energy_levels` | `E_cm_inv` |

**Units:** see NON-NEGOTIABLE SUSCEPTIBILITY UNIT RULE below. Fit χT column is `chi_t`, not `chi_T`.

`originIon` = Ln(III) name from `list_lanthanide_ions`, or null. Never `Co(II)`.

For high-spin Co(II) S=3/2 L=1: `get_example_payload(example="co_sl_axial")`, then susceptibility.

Other examples: `s_half`, `s_one_zfs`, `two_spins_exchange` (J>0 AF), `gd_j`, `fit_zfs` (pass the whole object to `optimize_parameters`; χT column is `chi_t` not `chi_T`; `maxiter` is inside payload).

## GUI H(L) names → MCP fields

There is **no** API key `sigma_SL`. Nest Tmin / numPoints / σ inside `hamiltonian_params`. Every `compute_property` recomputes (`recomputed: true`); read `params_echo`. Optional `force_refit` is accepted and ignored. Sweep points are also in `grid_csv`.

| GUI | MCP field | Notes |
|---|---|---|
| **σ¹** (Orbital g-factor) | `sigma_L["1"]` | μ_B σ B·L̂. **Not** σ^{SL}. |
| **λ¹¹** (cm⁻¹) | `lambda_SL["1"]` | Digits only. Never `"1-1"`. |
| **σ^{SL}_1** | `lambda_sigma_SL["1"]` | H_SOC = λ × σ^{SL} × Ŝ·L̂. Default 0 zeros SOC. |
| **Σ_k^L** | `sigma_CF["1_2"]` | CF scale. Not σ^{SL}. |
| **B_k^q** | `B_kq["1_2_0"]` | Axial Δ. |

## Other conventions

- Exchange H = J Ŝ_i·Ŝ_j, keys `"1-2"`. **J > 0 antiferromagnetic**.
- Ln(III): `list_lanthanide_ions`; `type="J"` + `originIon` fills Landé `g_J` in the template.
- Nuclei: `nuclei=[{nucleus:"1H"}]`. gamma is rad s⁻¹ T⁻¹, not MHz/T.
- `numPoints` ≥ 2. Prefer 10–20.

If `validate_request` or `compute_property` returns a warning that λ is set but `lambda_sigma_SL` is 0, set chemist σ there — do not probe spectra to discover this.

## NON-NEGOTIABLE SUSCEPTIBILITY UNIT RULE

`chi` and `delta_chi_ax` are not numerically comparable.

| Field | Unit | Basis | System |
|---|---|---|---|
| `chi` | `cm^3 mol^-1` | per mole | cgs |
| `chi_T` | `cm^3 K mol^-1` | per mole | cgs |
| `delta_chi_ax` | `m^3 ion^-1` | per ion | SI |
| `delta_chi_rh` | `m^3 ion^-1` | per ion | SI |

Hard rules:

- Never compare the raw numerical value of `chi` with `delta_chi_ax`.
- Never call `delta_chi_ax = 1e-31` zero because the exponent is small.
- Never use a generic threshold such as `abs(x) < 1e-6` for `delta_chi`.
- Read `quantity_metadata` returned by `compute_property`.
- Use `is_exact_zero` only for exact numerical zero.
- To compare with `chi`, first convert: `delta_chi_cgs = delta_chi_si * N_A * 1e6 / (4*pi)`.
- In the final answer, always write the unit next to every value.
- Do not repeat the computation merely because `delta_chi` has magnitude `1e-31`.
