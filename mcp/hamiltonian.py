"""Port of App.vue hamiltonianSelectionFunc / determineHamiltonian / formula expansion."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "maigic" / "maigic" / "src" / "data"

with (DATA_DIR / "nuclei_info.json").open(encoding="utf-8") as _f:
    NUCLEI_INFO: dict[str, dict[str, float]] = json.load(_f)
with (DATA_DIR / "symmetry_info.json").open(encoding="utf-8") as _f:
    SYMMETRY_INFO: dict[str, Any] = json.load(_f)
with (DATA_DIR / "hamiltonian_info.json").open(encoding="utf-8") as _f:
    HAMILTONIAN_INFO: dict[str, Any] = json.load(_f)

SWEEP_KEYS = {
    "calculationMode",
    "fixedB",
    "fixedT",
    "Tmin",
    "Tmax",
    "Bmin",
    "Bmax",
    "numPoints",
    "tip_correction",
    "diamagnetic_correction",
    "computeE",
    "computeM",
    "computeChi",
    "computeChiAx",
    "computeChiRh",
    "computeChiT",
}

TERM_PARAM_BLOCKS = {
    "H(S)_e-B": ["g", "gType", "gAniso"],
    "H(S)_S-S": ["D", "E"],
    "H(S)_e-e": ["J_ex"],
    "H(L)_e-B": ["sigma_L"],
    "H(L)_S-L": ["lambda_SL", "lambda_sigma_SL"],
    "H(L)_CF": ["sigma_CF", "B_kq"],
    "H(J)_e-B": ["g_J", "gJType", "gJAniso"],
    "H(J)_e-e": ["J_ex_J"],
    "H(J)_CF": ["sigma_CF", "B_kq"],
    "H(I)_n-B": [],
    "H(I)_e-n": ["A_hf"],
}

MCP_INSTRUCTIONS = (
    "REQUIRED WORKFLOW — do this before every calculation, fit, or modelling step: "
    "1) Call determine_hamiltonian with the complex (electrons, nuclei, point_group). "
    "2) Read formula_latex and chemist_mapping. That is the Hamiltonian. Do not invent ZFS, "
    "exchange, CF, hyperfine, or orbital terms that are missing from it. "
    "3) Copy spin_systems and hamiltonian_params_template into compute_property / "
    "optimize_parameters / validate_request. Change only numbers and sweep settings. "
    "Passing keys that are not in the template is rejected. "
    "Call list_nuclei before naming nuclei. Call list_lanthanide_ions for Ln(III) S, L, J, g_J. "
    "If originIon is a Ln(III) name, B_kq is multiplied by Stevens θ_k; if null, θ_k = 1. "
    "parameter_fixed_state: True = hold fixed, False = fit; at least one key must be False. "
    "Keep numPoints at 10–20 (minimum 2 — the schema rejects 1) unless the user asks for a dense curve. "
    "Keep optimize maxiter ≤ 20 and few data points. "
    "Hilbert-space dimension is the product of (2s+1) over all centers (and L if L>0). "
    "S–L mapping (do not guess, do not probe energy_levels to learn keys): "
    "chemist λ (cm⁻¹) → lambda_SL['1'] (digit keys only, never '1-1'); "
    "chemist σ of the S–L term → lambda_sigma_SL['1'] (template default 0; H_SOC = λ×σ×Ŝ·L̂, so if λ≠0 and this is 0, SOC is identically zero); "
    "sigma_L['1'] is orbital Zeeman μ_B σ B·L̂ (default 1), not chemist σ of S–L; "
    "sigma_CF['1_2'] is the CF scale on B_2^q (default 1), not chemist σ of S–L; "
    "there is no parameter named Δ — axial crystal-field Δ is B_kq['1_2_0']; pass point_group (D4h, C2v, …) to restrict Stevens operators. "
    "High-spin Co(II) with L=1: S=3/2, L=1; call get_example_payload(example='co_sl_axial') and edit numbers."
)

CHEMIST_MAPPING_SL: dict[str, str] = {
    "lambda_SL": (
        "Chemist λ of Ŝ·L̂ (cm⁻¹). Keys are center ids as digits only, e.g. '1'. Never '1-1'."
    ),
    "lambda_sigma_SL": (
        "Chemist σ of the S–L term (dimensionless). H_SOC = λ × σ × Ŝ·L̂. "
        "Template default is 0: if you set λ but leave this at 0, SOC energies stay identically 0."
    ),
    "sigma_L": (
        "Orbital reduction on μ_B σ B·L̂ (orbital Zeeman), default 1. "
        "Not the chemist σ of S–L; that belongs in lambda_sigma_SL."
    ),
    "sigma_CF": (
        "Per-rank CF scale on B_k^q, keys '{id}_{k}' e.g. '1_2'. Default 1. Not the chemist σ of S–L."
    ),
    "B_kq": (
        "Stevens B_k^q (cm⁻¹), keys '{id}_{k}_{q}' e.g. '1_2_0'. "
        "There is no field named Δ. Axial crystal-field Δ is usually B_kq['1_2_0']. "
        "Pass point_group (D4h, C2v, …) so unused q are off."
    ),
    "do_not": (
        "Do not call energy_levels to reverse-engineer key formats or which field is SOC. "
        "Copy hamiltonian_params_template keys. For Co(II) S=3/2 L=1 use get_example_payload('co_sl_axial')."
    ),
}


def chemist_mapping_for_spec(term_selection: dict[str, bool]) -> dict[str, str] | None:
    if not (
        term_selection.get("H(L)_S-L")
        or term_selection.get("H(L)_e-B")
        or term_selection.get("H(L)_CF")
    ):
        return None
    return dict(CHEMIST_MAPPING_SL)


def _float_map(d: dict[str, Any] | None) -> dict[str, float]:
    return {str(k): float(v) for k, v in (d or {}).items()}


def physics_warnings_from_hp(
    hp: dict[str, Any],
    spec: dict[str, Any] | None = None,
) -> list[str]:
    """Warnings the backend will not raise as errors (e.g. SOC identically zero)."""
    warnings: list[str] = []
    if spec:
        warnings.extend(spec.get("warnings") or [])
    lam = _float_map(hp.get("lambda_SL"))
    lsig = _float_map(hp.get("lambda_sigma_SL"))
    ids = set(lam) | set(lsig)
    for cid in sorted(ids, key=lambda x: int(x) if str(x).isdigit() else str(x)):
        lv = lam.get(cid, 0.0)
        sv = lsig.get(cid, 0.0)
        if abs(lv) > 0 and abs(sv) == 0:
            warnings.append(
                f"Center {cid}: lambda_SL={lv} but lambda_sigma_SL=0. "
                "H_SOC = λ × σ × Ŝ·L̂, so spin-orbit is identically zero. "
                "Chemist σ of S–L belongs in lambda_sigma_SL (not sigma_L / sigma_CF)."
            )
        elif abs(lv) == 0 and abs(sv) > 0:
            warnings.append(
                f"Center {cid}: lambda_sigma_SL={sv} but lambda_SL=0. "
                "H_SOC is still zero. Set lambda_SL to chemist λ (cm⁻¹)."
            )
    return warnings


def build_co_sl_axial_example() -> dict[str, Any]:
    """High-spin Co(II): S=3/2, L=1, axial CF. Edit numbers, then compute_property."""
    spec = determine_hamiltonian_spec(
        electrons=[{"type": "S", "S": 1.5, "L": 1.0, "originIon": None}],
        point_group="D4h",
    )
    hp = deepcopy(spec["hamiltonian_params_template"])
    hp["lambda_SL"]["1"] = 152.4
    hp["lambda_sigma_SL"]["1"] = 1.35
    hp["sigma_L"]["1"] = 1.35
    hp.setdefault("B_kq", {})["1_2_0"] = -632.0
    hp["calculationMode"] = "fixedB"
    hp["fixedB"] = 7.0
    hp["Tmin"] = 200.0
    hp["Tmax"] = 300.0
    hp["numPoints"] = 12
    hp["computeChi"] = True
    hp["computeChiAx"] = True
    hp["computeChiT"] = True
    return {
        "description": (
            "High-spin Co(II) S=3/2 L=1, axial CF. χT and Δχ_ax vs T at 7 T. "
            "λ → lambda_SL; chemist σ of S–L → lambda_sigma_SL (not sigma_L/sigma_CF); "
            "Δ → B_kq['1_2_0']. Copy into compute_property(property='susceptibility')."
        ),
        "point_group": "D4h",
        "spin_systems": spec["spin_systems"],
        "hamiltonian_params": hp,
        "chemist_mapping": spec.get("chemist_mapping"),
        "notes": [
            "λ = lambda_SL['1'] = 152.4 cm⁻¹ (digit key '1', never '1-1')",
            "σ of S–L = lambda_sigma_SL['1'] = 1.35; if this stays 0, SOC is identically zero",
            "sigma_L['1'] = 1.35 is orbital Zeeman reduction (same number as chemist σ, different term)",
            "Δ axial CF → B_kq['1_2_0'] = -632 cm⁻¹; there is no parameter named Δ",
            "numPoints must be ≥ 2",
        ],
    }


def register_maigic_skill(mcp: Any) -> None:
    skill_dir = Path(__file__).resolve().parent / "skills" / "maigic-workflow"
    if not (skill_dir / "SKILL.md").is_file():
        return
    try:
        from fastmcp.server.providers.skills import SkillProvider

        mcp.add_provider(SkillProvider(skill_dir))
    except Exception:
        return


def parse_quantum_number(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if "/" in text:
        num, den = text.split("/", 1)
        return float(num) / float(den)
    return float(text)


def lookup_nucleus(name: str) -> tuple[str, dict[str, float]]:
    raw = name.strip()
    if raw in NUCLEI_INFO:
        return raw, NUCLEI_INFO[raw]
    superscripts = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
    ascii_name = raw.translate(superscripts)
    for key in NUCLEI_INFO:
        if key.translate(superscripts) == ascii_name:
            return key, NUCLEI_INFO[key]
    known = ", ".join(sorted(NUCLEI_INFO, key=lambda n: n.translate(superscripts)))
    raise ValueError(f"Unknown nucleus '{name}'. Known: {known}")


def hamiltonian_selection_func(
    s_list: list[float],
    l_list: list[float],
    j_list: list[float],
    i_list: list[float],
) -> dict[str, list[str]]:
    """Same rules as App.vue hamiltonianSelectionFunc."""
    hamiltonian: dict[str, list[str]] = {
        "H(S)": [],
        "H(L)": [],
        "H(J)": [],
        "H(I)": [],
    }
    if len(s_list) == 1 and s_list[0] == 0.5:
        hamiltonian["H(S)"] = ["e-B"]
    elif len(s_list) == 1 and s_list[0] > 0.5:
        hamiltonian["H(S)"] = ["e-B", "S-S"]
    elif len(s_list) > 1 and max(s_list) == 0.5:
        hamiltonian["H(S)"] = ["e-B", "e-e"]
    elif len(s_list) > 1 and max(s_list) > 0.5:
        hamiltonian["H(S)"] = ["e-B", "S-S", "e-e"]

    if s_list and l_list and max(l_list) > 0:
        hamiltonian["H(L)"] = ["e-B", "S-L", "CF"]

    if len(j_list) == 1:
        hamiltonian["H(J)"] = ["e-B", "CF"]
    elif len(j_list) > 1:
        hamiltonian["H(J)"] = ["e-B", "CF", "e-e"]

    if i_list:
        hamiltonian["H(I)"] = ["n-B", "e-n"]

    return hamiltonian


def all_cf_labels() -> list[str]:
    return [f"k={k}, q={q}" for k in (2, 4, 6) for q in range(-k, k + 1)]


def allowed_cf_labels(point_group: str | None) -> list[str] | None:
    if not point_group:
        return None
    group = point_group.strip()
    family = SYMMETRY_INFO["group_to_symmetry"].get(group)
    if not family:
        known = sorted(SYMMETRY_INFO["group_to_symmetry"])
        raise ValueError(f"Unknown point group '{group}'. Known: {known}")
    coefs = SYMMETRY_INFO["symmetry_to_coefs"][family]
    return [f"k={k}, q={q}" for k, q in coefs]


def parse_cf_label(label: str) -> tuple[int, int]:
    match = re.fullmatch(r"k=(-?\d+), q=(-?\d+)", label.strip())
    if not match:
        raise ValueError(f"Bad CF label '{label}'")
    return int(match.group(1)), int(match.group(2))


def _idx(center_id: int, use_indices: bool) -> str:
    return f"_{center_id}" if use_indices else ""


def expand_formula_latex(
    spin_systems: list[dict[str, Any]],
    term_selection: dict[str, bool],
    cf_selection: dict[str, dict[str, bool]],
) -> str:
    """Same expansion as App.vue expandedHamiltonianLaTeX."""
    terms: list[str] = []
    use_indices = len(spin_systems) > 1
    electrons = [s for s in spin_systems if s["type"] in {"S", "J"}]
    nuclei = [s for s in spin_systems if s["type"] == "I"]

    for sys in electrons:
        block = sys["type"]
        if not term_selection.get(f"H({block})_e-B"):
            continue
        info = HAMILTONIAN_INFO.get(block, {}).get("e-B")
        if not info:
            continue
        expr = info["formula"]
        expr = expr.replace(r"\hat S^i", rf"\hat{{S}}{_idx(sys['id'], use_indices)}")
        expr = expr.replace(r"\hat J^i", rf"\hat{{J}}{_idx(sys['id'], use_indices)}")
        expr = expr.replace("g^i", f"g{_idx(sys['id'], use_indices)}")
        expr = expr.replace(r"\vec B", r"\vec{B}")
        terms.append(expr)

    for sys in spin_systems:
        if sys["type"] != "S" or sys.get("L", 0) <= 0:
            continue
        if term_selection.get("H(L)_e-B"):
            info = HAMILTONIAN_INFO.get("L", {}).get("e-B")
            if info:
                expr = info["formula"]
                expr = expr.replace(r"\hat L^i", rf"\hat{{L}}{_idx(sys['id'], use_indices)}")
                expr = expr.replace(r"\vec B", r"\vec{B}")
                expr = expr.replace(r"\sigma^i", rf"\sigma{_idx(sys['id'], use_indices)}")
                terms.append(expr)
        if term_selection.get("H(L)_S-L"):
            info = HAMILTONIAN_INFO.get("L", {}).get("S-L")
            if info:
                expr = info["formula"]
                expr = expr.replace(r"\hat S^i", rf"\hat{{S}}{_idx(sys['id'], use_indices)}")
                expr = expr.replace(r"\hat L^j", rf"\hat{{L}}{_idx(sys['id'], use_indices)}")
                expr = expr.replace(r"\lambda^{ij}", rf"\lambda{_idx(sys['id'], use_indices)}")
                expr = expr.replace(r"\sigma^{ij}", rf"\sigma{_idx(sys['id'], use_indices)}")
                terms.append(expr)

    if term_selection.get("H(S)_S-S"):
        for sys in spin_systems:
            if sys["type"] != "S" or sys["S"] <= 0.5:
                continue
            info = HAMILTONIAN_INFO.get("S", {}).get("S-S")
            if not info:
                continue
            expr = info["formula"]
            expr = expr.replace(r"\hat S^i", rf"\hat{{S}}{_idx(sys['id'], use_indices)}")
            expr = expr.replace("D^{ii}", f"D{_idx(sys['id'], use_indices)}")
            terms.append(expr)

    if len(electrons) >= 2 and (
        term_selection.get("H(S)_e-e") or term_selection.get("H(J)_e-e")
    ):
        for a, e1 in enumerate(electrons):
            for e2 in electrons[a + 1 :]:
                if e1["type"] != e2["type"]:
                    continue
                info = HAMILTONIAN_INFO.get(e1["type"], {}).get("e-e")
                if not info:
                    continue
                expr = info["formula"]
                if e1["type"] == "S":
                    expr = expr.replace(r"\hat S^i", rf"\hat{{S}}{_idx(e1['id'], use_indices)}")
                    expr = expr.replace(r"\hat S^j", rf"\hat{{S}}{_idx(e2['id'], use_indices)}")
                else:
                    expr = expr.replace(r"\hat J^i", rf"\hat{{J}}{_idx(e1['id'], use_indices)}")
                    expr = expr.replace(r"\hat J^j", rf"\hat{{J}}{_idx(e2['id'], use_indices)}")
                expr = expr.replace("J^{ij}", f"J_{{{e1['id']}{e2['id']}}}")
                terms.append(expr)

    if term_selection.get("H(I)_e-n") and nuclei:
        for elec in electrons:
            for nuc in nuclei:
                if elec["type"] == "S":
                    term_key, label = "Se-n", "S"
                elif elec["type"] == "J":
                    term_key, label = "Je-n", "J"
                else:
                    continue
                info = HAMILTONIAN_INFO.get("I", {}).get(term_key)
                if not info:
                    continue
                expr = info["formula"]
                for op in ("S", "L", "J"):
                    expr = expr.replace(
                        rf"\hat {op}^i",
                        rf"\hat{{{label}}}{_idx(elec['id'], use_indices)}",
                    )
                expr = expr.replace(r"\hat I^{ij}", rf"\hat{{I}}{_idx(nuc['id'], use_indices)}")
                expr = expr.replace("A^{ij}", f"A_{{{elec['id']}{nuc['id']}}}")
                terms.append(expr)
                if elec["type"] == "S" and elec.get("L", 0) > 0:
                    info_l = HAMILTONIAN_INFO.get("I", {}).get("Le-n")
                    if info_l:
                        expr_l = info_l["formula"]
                        expr_l = expr_l.replace(
                            r"\hat L^i", rf"\hat{{L}}{_idx(elec['id'], use_indices)}"
                        )
                        expr_l = expr_l.replace(
                            r"\hat I^{ij}", rf"\hat{{I}}{_idx(nuc['id'], use_indices)}"
                        )
                        expr_l = expr_l.replace("A^{ij}", f"A_{{L,{elec['id']}{nuc['id']}}}")
                        terms.append(expr_l)

    if term_selection.get("H(I)_n-B"):
        for nuc in nuclei:
            info = HAMILTONIAN_INFO.get("I", {}).get("n-B")
            if not info:
                continue
            expr = info["formula"]
            expr = expr.replace(r"\hat I^i", rf"\hat{{I}}{_idx(nuc['id'], use_indices)}")
            expr = expr.replace("g^i", f"g{_idx(nuc['id'], use_indices)}")
            expr = expr.replace(r"\vec B", r"\vec{B}")
            terms.append(expr)

    def handle_cf(kind: str, term_key: str) -> None:
        for sys in spin_systems:
            applicable = (kind == "L" and sys["type"] == "S" and sys.get("L", 0) > 0) or (
                kind == "J" and sys["type"] == "J"
            )
            if not applicable:
                continue
            if not term_selection.get(f"H({kind})_{term_key}"):
                continue
            cf_sel = cf_selection.get(f"H({kind})_{term_key}", {})
            if not any(cf_sel.values()):
                continue
            info = HAMILTONIAN_INFO.get(kind, {}).get(term_key)
            if not info:
                continue
            expr = info["formula"].replace("^i", f"^{{{sys['id']}}}")
            op = "L" if kind == "L" else "J"
            suffix = f"_{{{sys['id']}}}" if use_indices else ""
            expr = expr.replace(r"( \hat L )", rf"(\hat{{{op}}}{suffix})")
            expr = expr.replace(r"( \hat J )", rf"(\hat{{J}}{suffix})")
            selected = [lab for lab, on in cf_sel.items() if on]
            if selected:
                kq = ", ".join(selected)
                expr = expr + rf" \quad ({kq})"
            terms.append(expr)

    handle_cf("L", "CF")
    handle_cf("J", "CF")
    return " + ".join(terms) if terms else ""


def build_spin_systems(
    electrons: list[dict[str, Any]],
    nuclei: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    systems: list[dict[str, Any]] = []
    next_id = 1
    for electron in electrons:
        kind = electron.get("type")
        if kind == "S":
            if electron.get("S") is None:
                raise ValueError("S-type electron requires S")
            systems.append(
                {
                    "id": next_id,
                    "type": "S",
                    "S": parse_quantum_number(electron.get("S")),
                    "L": parse_quantum_number(electron.get("L", 0)),
                    "originIon": electron.get("originIon") or None,
                }
            )
            next_id += 1
        elif kind == "J":
            if electron.get("J") is None:
                raise ValueError("J-type electron requires J")
            systems.append(
                {
                    "id": next_id,
                    "type": "J",
                    "J": parse_quantum_number(electron.get("J")),
                    "originIon": electron.get("originIon") or None,
                }
            )
            next_id += 1
        else:
            raise ValueError(f"Electron type must be 'S' or 'J', got {kind!r}")
    for nucleus in nuclei:
        label, info = lookup_nucleus(nucleus["nucleus"])
        systems.append(
            {
                "id": next_id,
                "type": "I",
                "I": info["I"],
                "gamma": info["gamma"],
                "shieldingType": nucleus.get("shieldingType", "None"),
                "shieldingScalar": float(nucleus.get("shieldingScalar", 0.0)),
                "shieldingDiagonal": list(nucleus.get("shieldingDiagonal") or [0.0, 0.0, 0.0]),
                "shieldingEulerAngles": list(
                    nucleus.get("shieldingEulerAngles") or [0.0, 0.0, 0.0]
                ),
                "nucleus": label,
            }
        )
        next_id += 1
    if not systems:
        raise ValueError("Need at least one electron or nucleus.")
    return systems


def _active_term_keys(
    computed_terms: dict[str, list[str]],
    disabled_blocks: list[str],
    disabled_terms: list[str],
) -> dict[str, bool]:
    selection: dict[str, bool] = {}
    disabled_term_set = set(disabled_terms)
    disabled_block_set = set(disabled_blocks)
    for block, terms in computed_terms.items():
        for term in terms:
            key = f"{block}_{term}"
            on = block not in disabled_block_set and key not in disabled_term_set
            selection[key] = on
    return selection


def _cf_selection(
    computed_terms: dict[str, list[str]],
    term_selection: dict[str, bool],
    point_group: str | None,
) -> dict[str, dict[str, bool]]:
    allowed = allowed_cf_labels(point_group)
    selection: dict[str, dict[str, bool]] = {}
    for block, terms in computed_terms.items():
        if "CF" not in terms:
            continue
        key = f"{block}_CF"
        if not term_selection.get(key):
            continue
        labels = all_cf_labels()
        if allowed is None:
            selection[key] = {lab: True for lab in labels}
        else:
            selection[key] = {lab: lab in allowed for lab in labels}
    return selection


def _template_and_keys(
    spin_systems: list[dict[str, Any]],
    term_selection: dict[str, bool],
    cf_selection: dict[str, dict[str, bool]],
) -> tuple[dict[str, Any], list[dict[str, str]], list[str]]:
    template: dict[str, Any] = {
        "calculationMode": "fixedB",
        "fixedB": 1.0,
        "fixedT": 298.15,
        "Tmin": 10.0,
        "Tmax": 300.0,
        "Bmin": 0.1,
        "Bmax": 10.0,
        "numPoints": 12,
        "tip_correction": 0.0,
        "diamagnetic_correction": 0.0,
        "gType": {},
        "gAniso": {},
        "g": {},
        "D": {},
        "E": {},
        "J_ex": {},
        "sigma_L": {},
        "lambda_SL": {},
        "lambda_sigma_SL": {},
        "gJType": {},
        "gJAniso": {},
        "g_J": {},
        "J_ex_J": {},
        "A_hf": {},
        "sigma_CF": {},
        "B_kq": {},
        "computeE": True,
        "computeM": True,
        "computeChi": True,
        "computeChiAx": True,
        "computeChiRh": True,
        "computeChiT": True,
    }
    fit_keys: list[dict[str, str]] = [
        {"key": "tip_correction", "meaning": "TIP added to χ diagonal", "unit": "cm^3 mol^-1"},
        {
            "key": "diamagnetic_correction",
            "meaning": "diamagnetic correction added to χ diagonal",
            "unit": "cm^3 mol^-1",
        },
    ]
    allowed_leaf: set[str] = set()

    s_ids = [s["id"] for s in spin_systems if s["type"] == "S"]
    j_ids = [s["id"] for s in spin_systems if s["type"] == "J"]
    i_ids = [s["id"] for s in spin_systems if s["type"] == "I"]
    l_ids = [s["id"] for s in spin_systems if s["type"] == "S" and s.get("L", 0) > 0]

    if term_selection.get("H(S)_e-B"):
        for sid in s_ids:
            key = str(sid)
            template["gType"][key] = "isotropic"
            template["g"][key] = 2.0023
            template["gAniso"][key] = [2.0023, 2.0023, 2.0023]
            fit_keys.append({"key": f"g_{sid}", "meaning": f"isotropic g of S-center {sid}", "unit": ""})
            fit_keys.append({"key": f"gAniso_{sid}_0", "meaning": f"g_x of S-center {sid}", "unit": ""})
            fit_keys.append({"key": f"gAniso_{sid}_1", "meaning": f"g_y of S-center {sid}", "unit": ""})
            fit_keys.append({"key": f"gAniso_{sid}_2", "meaning": f"g_z of S-center {sid}", "unit": ""})
            allowed_leaf.add(f"g.{key}")
            allowed_leaf.add(f"gType.{key}")
            allowed_leaf.add(f"gAniso.{key}")

    if term_selection.get("H(S)_S-S"):
        for sys in spin_systems:
            if sys["type"] != "S" or sys["S"] <= 0.5:
                continue
            key = str(sys["id"])
            template["D"][key] = 0.0
            template["E"][key] = 0.0
            fit_keys.append(
                {"key": f"D_{sys['id']}", "meaning": f"axial ZFS D of S-center {sys['id']}", "unit": "cm^-1"}
            )
            fit_keys.append(
                {"key": f"E_{sys['id']}", "meaning": f"rhombic ZFS E of S-center {sys['id']}", "unit": "cm^-1"}
            )
            allowed_leaf.add(f"D.{key}")
            allowed_leaf.add(f"E.{key}")

    if term_selection.get("H(S)_e-e"):
        for i, a in enumerate(s_ids):
            for b in s_ids[i + 1 :]:
                pair = f"{a}-{b}"
                template["J_ex"][pair] = 0.0
                fit_keys.append(
                    {
                        "key": f"J_ex_{pair}",
                        "meaning": f"exchange J between S-centers {a} and {b}",
                        "unit": "cm^-1",
                    }
                )
                allowed_leaf.add(f"J_ex.{pair}")

    if term_selection.get("H(L)_e-B"):
        for sid in l_ids:
            template["sigma_L"][str(sid)] = 1.0
            fit_keys.append(
                {
                    "key": f"sigma_L_{sid}",
                    "meaning": (
                        f"orbital Zeeman reduction σ_L of center {sid} (μ_B σ B·L̂). "
                        "Not chemist σ of Ŝ·L̂ (that is lambda_sigma_SL)."
                    ),
                    "unit": "",
                }
            )
            allowed_leaf.add(f"sigma_L.{sid}")

    if term_selection.get("H(L)_S-L"):
        for sid in l_ids:
            template["lambda_SL"][str(sid)] = 0.0
            template["lambda_sigma_SL"][str(sid)] = 0.0
            fit_keys.append(
                {
                    "key": f"lambda_SL_{sid}",
                    "meaning": (
                        f"chemist λ of Ŝ·L̂ for center {sid} (cm⁻¹). "
                        f"Key is digit '{sid}', never '{sid}-{sid}'."
                    ),
                    "unit": "cm^-1",
                }
            )
            fit_keys.append(
                {
                    "key": f"lambda_sigma_SL_{sid}",
                    "meaning": (
                        f"chemist σ of Ŝ·L̂ for center {sid} (dimensionless). "
                        "H_SOC = λ × σ × Ŝ·L̂. Template default 0: leaving this at 0 zeros SOC."
                    ),
                    "unit": "",
                }
            )
            allowed_leaf.add(f"lambda_SL.{sid}")
            allowed_leaf.add(f"lambda_sigma_SL.{sid}")

    if term_selection.get("H(J)_e-B"):
        for sid in j_ids:
            key = str(sid)
            template["gJType"][key] = "isotropic"
            template["g_J"][key] = 2.0023
            template["gJAniso"][key] = [2.0023, 2.0023, 2.0023]
            fit_keys.append({"key": f"g_J_{sid}", "meaning": f"isotropic g_J of J-center {sid}", "unit": ""})
            fit_keys.append({"key": f"gJAniso_{sid}_0", "meaning": f"g_x of J-center {sid}", "unit": ""})
            fit_keys.append({"key": f"gJAniso_{sid}_1", "meaning": f"g_y of J-center {sid}", "unit": ""})
            fit_keys.append({"key": f"gJAniso_{sid}_2", "meaning": f"g_z of J-center {sid}", "unit": ""})
            allowed_leaf.add(f"g_J.{key}")
            allowed_leaf.add(f"gJType.{key}")
            allowed_leaf.add(f"gJAniso.{key}")

    if term_selection.get("H(J)_e-e"):
        for i, a in enumerate(j_ids):
            for b in j_ids[i + 1 :]:
                pair = f"{a}-{b}"
                template["J_ex_J"][pair] = 0.0
                fit_keys.append(
                    {
                        "key": f"J_ex_J_{pair}",
                        "meaning": f"exchange J between J-centers {a} and {b}",
                        "unit": "cm^-1",
                    }
                )
                allowed_leaf.add(f"J_ex_J.{pair}")

    if term_selection.get("H(I)_e-n"):
        for eid in s_ids:
            for nid in i_ids:
                pair = f"e{eid}-n{nid}"
                template["A_hf"][pair] = 0.0
                fit_keys.append(
                    {
                        "key": f"A_hf_SI_{pair}",
                        "meaning": f"hyperfine A (S–I) {pair}",
                        "unit": "MHz",
                    }
                )
                allowed_leaf.add(f"A_hf.{pair}")
        for eid in l_ids:
            for nid in i_ids:
                pair = f"L-e{eid}-n{nid}"
                template["A_hf"][pair] = 0.0
                fit_keys.append(
                    {
                        "key": f"A_hf_LI_e{eid}-n{nid}",
                        "meaning": f"hyperfine A (L–I) e{eid}–n{nid}",
                        "unit": "MHz",
                    }
                )
                allowed_leaf.add(f"A_hf.{pair}")
        for eid in j_ids:
            for nid in i_ids:
                pair = f"J-e{eid}-n{nid}"
                template["A_hf"][pair] = 0.0
                fit_keys.append(
                    {
                        "key": f"A_hf_JI_e{eid}-n{nid}",
                        "meaning": f"hyperfine A (J–I) e{eid}–n{nid}",
                        "unit": "MHz",
                    }
                )
                allowed_leaf.add(f"A_hf.{pair}")

    def add_cf(kind: str, center_ids: list[int], fit_prefix_sigma: str, fit_prefix_b: str) -> None:
        key = f"H({kind})_CF"
        if not term_selection.get(key):
            return
        selected = [lab for lab, on in cf_selection.get(key, {}).items() if on]
        ranks = sorted({parse_cf_label(lab)[0] for lab in selected})
        for sid in center_ids:
            for k in ranks:
                sigma_key = f"{sid}_{k}"
                template["sigma_CF"][sigma_key] = 1.0
                fit_keys.append(
                    {
                        "key": f"{fit_prefix_sigma}{sid}_{k}",
                        "meaning": f"CF scale σ_{k} ({kind}) of center {sid}",
                        "unit": "",
                    }
                )
                allowed_leaf.add(f"sigma_CF.{sigma_key}")
            for lab in selected:
                k, q = parse_cf_label(lab)
                b_key = f"{sid}_{k}_{q}"
                template["B_kq"][b_key] = 0.0
                fit_keys.append(
                    {
                        "key": f"{fit_prefix_b}{sid}_{k}_{q}",
                        "meaning": f"Stevens B_{k}^{q} ({kind}) of center {sid}",
                        "unit": "cm^-1",
                    }
                )
                allowed_leaf.add(f"B_kq.{b_key}")

    add_cf("L", l_ids, "sigma_CF_", "B_kq_")
    add_cf("J", j_ids, "sigma_CF_J_", "B_kq_J_")

    # Drop empty dicts that are not part of the formula so the LLM does not fill them.
    interaction_dicts = [
        "g",
        "gType",
        "gAniso",
        "D",
        "E",
        "J_ex",
        "sigma_L",
        "lambda_SL",
        "lambda_sigma_SL",
        "g_J",
        "gJType",
        "gJAniso",
        "J_ex_J",
        "A_hf",
        "sigma_CF",
        "B_kq",
    ]
    for name in interaction_dicts:
        if not template[name]:
            del template[name]

    return template, fit_keys, sorted(allowed_leaf)


def determine_from_spin_systems(
    spin_systems: list[dict[str, Any]],
    point_group: str | None = None,
    disabled_terms: list[str] | None = None,
    disabled_blocks: list[str] | None = None,
) -> dict[str, Any]:
    s_list = [s["S"] for s in spin_systems if s["type"] == "S"]
    l_list = [s.get("L", 0.0) for s in spin_systems if s["type"] == "S"]
    j_list = [s["J"] for s in spin_systems if s["type"] == "J"]
    i_list = [s["I"] for s in spin_systems if s["type"] == "I"]

    computed = hamiltonian_selection_func(s_list, l_list, j_list, i_list)
    term_selection = _active_term_keys(
        computed, disabled_blocks or [], disabled_terms or []
    )
    cf_sel = _cf_selection(computed, term_selection, point_group)
    latex = expand_formula_latex(spin_systems, term_selection, cf_sel)
    template, fit_keys, allowed_leaf = _template_and_keys(
        spin_systems, term_selection, cf_sel
    )

    active_terms = [key for key, on in term_selection.items() if on]
    blocks = sorted({key.split("_")[0] for key in active_terms})
    cf_on = {
        key: [lab for lab, on in labels.items() if on] for key, labels in cf_sel.items()
    }
    warnings: list[str] = []
    if (term_selection.get("H(L)_CF") or term_selection.get("H(J)_CF")) and not point_group:
        warnings.append(
            "No point_group: every Stevens operator k=2,4,6 is ON (same as the GUI with empty symmetry). "
            "Pass a Schoenflies symbol (D4h, C2v, ...) to restrict B_kq."
        )

    family = None
    if point_group:
        family = SYMMETRY_INFO["group_to_symmetry"].get(point_group.strip())

    out: dict[str, Any] = {
        "formula_latex": f"H = {latex}" if latex else "",
        "blocks": blocks,
        "terms": {block: [t for t in computed[block] if term_selection.get(f"{block}_{t}")] for block in computed},
        "term_selection": term_selection,
        "cf_components": cf_on,
        "point_group": point_group,
        "symmetry_family": family,
        "spin_systems": spin_systems,
        "hamiltonian_params_template": template,
        "allowed_parameter_keys": fit_keys,
        "allowed_hamiltonian_leaves": allowed_leaf,
        "warnings": warnings,
        "must_use": (
            "Copy spin_systems and hamiltonian_params_template into compute_property / "
            "optimize_parameters. Change numerical values and sweep settings only. "
            "Do not add D, E, J_ex, B_kq, A_hf, or other keys that are absent from the template — "
            "they are not in this Hamiltonian. "
            "lambda_SL / lambda_sigma_SL keys are center ids as digits ('1'), never '1-1'. "
            "numPoints ≥ 2. Do not probe energy_levels to learn the API."
        ),
    }
    mapping = chemist_mapping_for_spec(term_selection)
    if mapping:
        out["chemist_mapping"] = mapping
    return out


def determine_hamiltonian_spec(
    electrons: list[dict[str, Any]],
    nuclei: list[dict[str, Any]] | None = None,
    point_group: str | None = None,
    disabled_terms: list[str] | None = None,
    disabled_blocks: list[str] | None = None,
) -> dict[str, Any]:
    spin_systems = build_spin_systems(electrons, nuclei or [])
    return determine_from_spin_systems(
        spin_systems,
        point_group=point_group,
        disabled_terms=disabled_terms,
        disabled_blocks=disabled_blocks,
    )


def _leaf_keys_in_params(hp: dict[str, Any]) -> set[str]:
    leaves: set[str] = set()
    for name, value in hp.items():
        if name in SWEEP_KEYS:
            continue
        if isinstance(value, dict):
            for sub in value:
                if value[sub] in (None, {}, []):
                    continue
                # keep zeros: they still mean "this term is present"
                leaves.add(f"{name}.{sub}")
        elif value not in (None, {}, []):
            leaves.add(name)
    return leaves


def extra_hamiltonian_keys(
    hp: dict[str, Any],
    spec: dict[str, Any],
) -> list[str]:
    allowed = set(spec["allowed_hamiltonian_leaves"])
    extras = []
    for leaf in sorted(_leaf_keys_in_params(hp)):
        if leaf.split(".", 1)[0] in SWEEP_KEYS:
            continue
        if leaf not in allowed:
            extras.append(leaf)
    return extras


def apply_template_defaults(hp: dict[str, Any], template: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(hp)
    for key, value in template.items():
        if key in SWEEP_KEYS:
            merged.setdefault(key, value)
            continue
        if isinstance(value, dict):
            merged.setdefault(key, {})
            if not isinstance(merged[key], dict):
                merged[key] = {}
            for sub, subval in value.items():
                merged[key].setdefault(sub, deepcopy(subval))
        else:
            merged.setdefault(key, deepcopy(value))
    return merged
