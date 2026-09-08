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

**Units (do not mix):** `chi` = cgs **cm³ mol⁻¹** (per mole). `chi_T` = chi×T. `delta_chi_ax` / `delta_chi_rh` = SI **m³ per ion** = `4π × Δχ_cgs[cm³ mol⁻¹] / (N_A × 10⁶)` already applied. |Δχ| ~ 10⁻³¹ is physical, not noise — do not compare it to χ. Inverse: `Δχ_cgs = delta_chi_ax × N_A × 10⁶ / (4π)`. M: μB. See `conventions` on the susceptibility response.

`originIon` = Ln(III) name from `list_lanthanide_ions`, or null. Never `Co(II)`.

For high-spin Co(II) S=3/2 L=1: `get_example_payload(example="co_sl_axial")`, then susceptibility.

Other examples: `s_half`, `s_one_zfs`, `two_spins_exchange` (J>0 AF), `gd_j`, `fit_zfs` (pass the whole object to `optimize_parameters`; χT column is `chi_t` not `chi_T`; `maxiter` is inside payload).

## S–L chemist names → MCP fields

| Chemist | MCP field | Notes |
|---|---|---|
| λ (cm⁻¹) | `lambda_SL["1"]` | Digit keys only. **Never `"1-1"`.** |
| σ of Ŝ·L̂ | `lambda_sigma_SL["1"]` | Template default is **0**. H_SOC = λ × σ × Ŝ·L̂. If λ ≠ 0 and this stays 0, SOC energies are **identically zero**. |
| orbital reduction on B·L̂ | `sigma_L["1"]` | Default 1. **Not** chemist σ of S–L. |
| CF scale σ_k | `sigma_CF["1_2"]` | Default 1. **Not** chemist σ of S–L. |
| axial Δ (cm⁻¹) | `B_kq["1_2_0"]` | There is **no** field named Δ. Pass `point_group` (e.g. D4h). |

## Other conventions

- Exchange H = J Ŝ_i·Ŝ_j, keys `"1-2"`. **J > 0 antiferromagnetic**.
- Ln(III): `list_lanthanide_ions`; `type="J"` + `originIon` fills Landé `g_J` in the template.
- Nuclei: `nuclei=[{nucleus:"1H"}]`. gamma is rad s⁻¹ T⁻¹, not MHz/T.
- `numPoints` ≥ 2. Prefer 10–20.

If `validate_request` or `compute_property` returns a warning that λ is set but `lambda_sigma_SL` is 0, set chemist σ there — do not probe spectra to discover this.
