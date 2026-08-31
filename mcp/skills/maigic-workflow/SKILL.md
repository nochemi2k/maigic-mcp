---
name: maigic-workflow
description: >-
  Use MaIGIC MCP without reverse-engineering. Call this before χ, χT, Δχ, magnetization,
  energy levels, cobalt S+L, spin Hamiltonian, or any compute_property / optimize_parameters call.
---

# MaIGIC MCP workflow

Do **not** probe `energy_levels` to learn keys. Do **not** search GitHub or the filesystem for MaIGIC source. Copy the template; use `get_example_payload`.

## Order

1. `determine_hamiltonian` with electrons (and nuclei / `point_group` if needed).
2. Read `formula_latex` and `chemist_mapping`. Do not invent terms missing from the formula.
3. Copy `spin_systems` and `hamiltonian_params_template` into `compute_property` / `validate_request` / `optimize_parameters`. Change **numbers and sweep settings only**. Extra Hamiltonian keys are rejected.

For high-spin Co(II) S=3/2 L=1: `get_example_payload(example="co_sl_axial")`, then `compute_property(property="susceptibility")`.

## S–L chemist names → MCP fields

| Chemist | MCP field | Notes |
|---|---|---|
| λ (cm⁻¹) | `lambda_SL["1"]` | Digit keys only. **Never `"1-1"`.** |
| σ of Ŝ·L̂ | `lambda_sigma_SL["1"]` | Template default is **0**. H_SOC = λ × σ × Ŝ·L̂. If λ ≠ 0 and this stays 0, SOC energies are **identically zero**. |
| orbital reduction on B·L̂ | `sigma_L["1"]` | Default 1. **Not** chemist σ of S–L. |
| CF scale σ_k | `sigma_CF["1_2"]` | Default 1. **Not** chemist σ of S–L. |
| axial Δ (cm⁻¹) | `B_kq["1_2_0"]` | There is **no** field named Δ. Pass `point_group` (e.g. D4h). |

## Schema traps

- `numPoints` ≥ 2 (1 is rejected). Prefer 10–20 unless the user wants a dense curve.
- `lambda_SL` / `lambda_sigma_SL` keys match `^\d+$` (`"1"`), not `"1-1"`.
- Hilbert dimension is ∏(2s+1) over centers (and L if L>0). Keep it modest.

If `validate_request` or `compute_property` returns a warning that λ is set but `lambda_sigma_SL` is 0, set chemist σ there — do not probe spectra to discover this.
