"""MaIGIC MCP server over Streamable HTTP.

Imports the FastAPI compute layer (compute.py + models.py) directly.
Requires MCP_TOKEN. Listens on MCP_HOST:MCP_PORT, path /mcp.
"""

from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastmcp import FastMCP
from fastmcp.server.auth.providers.jwt import StaticTokenVerifier
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.requests import Request
from starlette.responses import JSONResponse

ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT / "maigic" / "maigic" / "backend"
DATA_DIR = ROOT / "maigic" / "maigic" / "src" / "data"

sys.path.insert(0, str(BACKEND_DIR))

from compute import (  # noqa: E402
    cf_params,
    get_energy_levels_b_const,
    get_energy_levels_t_const,
    get_magnetic_susceptibility_b_const,
    get_magnetic_susceptibility_t_const,
    get_magnetization_vector_b_const,
    get_magnetization_vector_t_const,
    inv_cm_to_j,
)
from models import (  # noqa: E402
    BaseRequest,
    HamiltonianParameters,
    ParameterFixedState,
    SpinSystem,
)
from optimize import optimize  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hamiltonian import (  # noqa: E402
    LANTHANIDE_IONS,
    MCP_INSTRUCTIONS,
    SUSCEPTIBILITY_CONVENTIONS,
    apply_template_defaults,
    build_co_sl_axial_example,
    determine_from_spin_systems,
    determine_hamiltonian_spec,
    extra_hamiltonian_keys,
    lande_g,
    physics_warnings_from_hp,
    register_maigic_skill,
)

PropertyName = Literal["susceptibility", "magnetization", "energy_levels"]
FitObservable = Literal[
    "M", "M_x", "M_y", "M_z", "chi", "chi_t", "delta_chi_ax", "delta_chi_rh"
]
MAX_HILBERT_DIM = 512
MAX_OPT_HILBERT_DIM = 64
MAX_OPT_POINTS = 40
MAX_OPT_ITER = 50

AVOGADRO = 6.02214e23
FOUR_PI = 4.0 * np.pi
DELTA_CHI_CGS_TO_SI_PER_ION = FOUR_PI / (AVOGADRO * 1e6)
DELTA_CHI_SI_PER_ION_TO_CGS = 1.0 / DELTA_CHI_CGS_TO_SI_PER_ION

def _http_auth() -> StaticTokenVerifier:
    token = os.environ.get("MCP_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "MCP_TOKEN is empty. Copy .env.example to .env and set a long random token."
        )
    return StaticTokenVerifier(
        tokens={token: {"client_id": "maigic-client", "scopes": ["mcp:access"]}},
        required_scopes=["mcp:access"],
    )


mcp = FastMCP(
    "MaIGIC",
    instructions=MCP_INSTRUCTIONS,
    auth=_http_auth(),
)
register_maigic_skill(mcp)


@mcp.custom_route("/health", methods=["GET"])
async def health(_request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


def _load_json(name: str) -> Any:
    with (DATA_DIR / name).open(encoding="utf-8") as f:
        return json.load(f)


NUCLEI_INFO: dict[str, dict[str, float]] = _load_json("nuclei_info.json")
SYMMETRY_INFO: dict[str, Any] = _load_json("symmetry_info.json")


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (float, np.floating)) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def _hilbert_dim(spin_systems: list[dict[str, Any]]) -> int:
    dim = 1
    for spin in spin_systems:
        kind = spin["type"]
        if kind == "S":
            dim *= int(round(2 * spin["S"] + 1))
            if spin.get("L", 0) > 0:
                dim *= int(round(2 * spin["L"] + 1))
        elif kind == "J":
            dim *= int(round(2 * spin["J"] + 1))
        elif kind == "I":
            dim *= int(round(2 * spin["I"] + 1))
        else:
            raise ValueError(f"Unknown spin type: {kind}")
    return dim


def _formula_kwargs(payload: Any) -> dict[str, Any]:
    return {
        "point_group": getattr(payload, "point_group", None) or None,
        "disabled_terms": getattr(payload, "disabled_terms", None) or [],
        "disabled_blocks": getattr(payload, "disabled_blocks", None) or [],
    }


def _formula_spec(
    spin_systems: list[dict[str, Any]],
    point_group: str | None = None,
    disabled_terms: list[str] | None = None,
    disabled_blocks: list[str] | None = None,
) -> dict[str, Any]:
    return determine_from_spin_systems(
        spin_systems,
        point_group=point_group or None,
        disabled_terms=disabled_terms or [],
        disabled_blocks=disabled_blocks or [],
    )


def _energies_to_cm(values: Any) -> Any:
    """Backend diagonalisation returns Joules; report cm⁻¹ as well."""
    if not values:
        return values
    if isinstance(values[0], (list, tuple)):
        return [[e / inv_cm_to_j for e in row] for row in values]
    return [e / inv_cm_to_j for e in values]


def _to_base_request(
    spin_systems: list[SpinSystem],
    hamiltonian_params: HamiltonianParameters,
    parameter_fixed_state: dict[str, bool] | None = None,
) -> BaseRequest:
    return BaseRequest(
        spin_systems=spin_systems,
        hamiltonian_params=hamiltonian_params,
        parameter_fixed_state=ParameterFixedState(
            parameter_fixed_state or {}
        ),
    )


def _dump_spin_systems(value: Any) -> list[dict[str, Any]]:
    """AllSpinSystems is a RootModel; FastMCP may also pass a plain list."""
    if value is None:
        raise ValueError(
            "spin_systems is missing. Copy spin_systems from determine_hamiltonian."
        )
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
    elif isinstance(value, list):
        dumped = [
            item.model_dump() if hasattr(item, "model_dump") else dict(item)
            for item in value
        ]
    else:
        raise ValueError(
            f"spin_systems must be a list, got {type(value).__name__}."
        )
    if isinstance(dumped, dict) and "root" in dumped:
        dumped = dumped["root"]
    if not isinstance(dumped, list):
        raise ValueError(
            f"spin_systems dump is {type(dumped).__name__}, expected a list."
        )
    return dumped


def _dump_hamiltonian_params(value: Any) -> dict[str, Any]:
    if value is None:
        raise ValueError(
            "hamiltonian_params is required. Copy hamiltonian_params_template "
            "from determine_hamiltonian (or get_example_payload)."
        )
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(exclude_unset=True)
    elif isinstance(value, dict):
        dumped = value
    else:
        raise ValueError(
            f"hamiltonian_params must be an object, got {type(value).__name__}."
        )
    if not isinstance(dumped, dict):
        raise ValueError(
            f"hamiltonian_params dump is {type(dumped).__name__}, expected an object."
        )
    return dumped


def _prepare(
    request: BaseRequest,
    force: bool,
    *,
    max_dim: int = MAX_HILBERT_DIM,
    point_group: str | None = None,
    disabled_terms: list[str] | None = None,
    disabled_blocks: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], int, dict[str, Any]]:
    spin_systems = _dump_spin_systems(getattr(request, "spin_systems", None))
    spec = _formula_spec(spin_systems, point_group, disabled_terms, disabled_blocks)
    hp_user = _dump_hamiltonian_params(getattr(request, "hamiltonian_params", None))
    extras = extra_hamiltonian_keys(hp_user, spec)
    if extras:
        raise ValueError(
            "hamiltonian_params contains terms that are not in the determined Hamiltonian: "
            f"{extras}. Formula: {spec['formula_latex']}. "
            "Call determine_hamiltonian and copy hamiltonian_params_template; "
            "do not add D, E, J_ex, B_kq, A_hf, or other keys missing from that template."
        )
    hp = apply_template_defaults(
        deepcopy(hp_user),
        spec["hamiltonian_params_template"],
    )
    for key in (
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
    ):
        hp.setdefault(key, {})
    dim = _hilbert_dim(spin_systems)
    if dim > max_dim and not force:
        raise ValueError(
            f"Hilbert-space dimension is {dim} (limit {max_dim}). "
            "Reduce S/L/J/I, drop unused nuclei, or pass force=true."
        )
    return deepcopy(spin_systems), deepcopy(hp), dim, spec


def _run_prepare(
    request: BaseRequest,
    force: bool,
    **kwargs: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any], int, dict[str, Any]]:
    """Never leak a raw `cannot unpack NoneType` to the MCP client."""
    try:
        prepared = _prepare(request, force, **kwargs)
    except TypeError as exc:
        raise ValueError(
            f"Internal prepare error ({exc}). "
            "Pass the canonical payload "
            "{spin_systems, hamiltonian_params, point_group?} from "
            "determine_hamiltonian or get_example_payload."
        ) from exc
    if prepared is None:
        raise ValueError(
            "Hamiltonian prepare returned nothing. "
            "Copy spin_systems + hamiltonian_params from determine_hamiltonian."
        )
    try:
        spin_systems, hp, dim, spec = prepared
    except TypeError as exc:
        raise ValueError(
            f"Internal prepare error ({exc}). "
            "Pass the canonical payload "
            "{spin_systems, hamiltonian_params, point_group?} from "
            "determine_hamiltonian or get_example_payload."
        ) from exc
    return spin_systems, hp, dim, spec


# Sweep / Hamiltonian keys that LLMs often put next to spin_systems (invalid).
_HP_MUST_NEST = frozenset(
    {
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
        "computeE",
        "computeM",
        "computeChi",
        "computeChiAx",
        "computeChiRh",
        "computeChiT",
    }
)

# Names that are not API fields. GUI labels → real keys.
_HP_ALIASES = {
    "sigma_SL": (
        "There is no field 'sigma_SL'. GUI σ^{SL}_1 (S–L multiplier) is "
        "hamiltonian_params.lambda_sigma_SL['1']. GUI σ¹ (orbital g-factor) is sigma_L['1']."
    ),
    "sigmaSL": (
        "There is no field 'sigmaSL'. GUI σ^{SL}_1 → lambda_sigma_SL['1']. "
        "GUI σ¹ → sigma_L['1']."
    ),
    "soc_sigma": "Use lambda_sigma_SL (GUI σ^{SL}), not soc_sigma.",
    "g_orbital": "Use sigma_L (GUI σ¹, orbital Zeeman μ_B σ B·L̂), not g_orbital.",
    "sigmaL": "Use sigma_L (GUI σ¹). Do not confuse with lambda_sigma_SL (GUI σ^{SL}).",
    "Delta": "There is no field Δ. Axial CF is B_kq['1_2_0'].",
    "delta": "There is no field Δ. Axial CF is B_kq['1_2_0'].",
    "lambda": "Use lambda_SL (GUI λ¹¹). Keys are digits '1', never '1-1'.",
}


def _reject_hp_aliases(hp: dict[str, Any]) -> None:
    for alias, hint in _HP_ALIASES.items():
        if alias in hp:
            raise ValueError(f"Unknown hamiltonian_params field {alias!r}. {hint}")


def _params_echo(
    hp: dict[str, Any],
    spin_systems: list[dict[str, Any]],
    point_group: str | None,
) -> dict[str, Any]:
    """Canonical values actually used for this call (after template merge). Always recomputed."""
    return {
        "recomputed": True,
        "point_group": point_group,
        "spin_systems": spin_systems,
        "calculationMode": hp.get("calculationMode"),
        "fixedB": hp.get("fixedB"),
        "fixedT": hp.get("fixedT"),
        "Tmin": hp.get("Tmin"),
        "Tmax": hp.get("Tmax"),
        "Bmin": hp.get("Bmin"),
        "Bmax": hp.get("Bmax"),
        "numPoints": hp.get("numPoints"),
        "g": hp.get("g") or {},
        "sigma_L": hp.get("sigma_L") or {},
        "lambda_SL": hp.get("lambda_SL") or {},
        "lambda_sigma_SL": hp.get("lambda_sigma_SL") or {},
        "sigma_CF": hp.get("sigma_CF") or {},
        "B_kq": hp.get("B_kq") or {},
        "D": hp.get("D") or {},
        "E": hp.get("E") or {},
        "J_ex": hp.get("J_ex") or {},
        "g_J": hp.get("g_J") or {},
        "gui_sigma_map": {
            "sigma_L": "GUI σ¹ (Orbital g-factor), μ_B σ B·L̂",
            "lambda_SL": "GUI λ¹¹ (cm⁻¹), SOC constant",
            "lambda_sigma_SL": "GUI σ^{SL}, H_SOC = λ × σ^{SL} × S·L. Not sigma_L.",
            "sigma_CF": "GUI Σ_k^L, CF rank scale. Not σ^{SL}.",
        },
    }


def _grid_csv(result: dict[str, Any]) -> str | None:
    """CSV of 1-D sweep columns so an agent can save the grid without inventing a file."""
    prefer = (
        "T",
        "B",
        "chi",
        "chi_T",
        "delta_chi_ax",
        "delta_chi_rh",
        "M",
        "M_x",
        "M_y",
        "M_z",
    )
    cols: list[str] = []
    for name in prefer:
        values = result.get(name)
        if isinstance(values, list) and values and not isinstance(values[0], (list, tuple)):
            cols.append(name)
    if not cols:
        return None
    n = len(result[cols[0]])
    lines = [",".join(cols)]
    for i in range(n):
        lines.append(",".join(f"{result[c][i]:.12g}" for c in cols))
    return "\n".join(lines)


def _as_float_list(values: Any) -> list[Any]:
    if values is None:
        return []
    if isinstance(values, list):
        return values
    return [values]


def _series_abs_max(values: Any) -> float:
    if not isinstance(values, list):
        try:
            return abs(float(values))
        except (TypeError, ValueError):
            return 0.0
    finite_values = []
    for x in values:
        if x is None:
            continue
        try:
            fx = float(x)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fx):
            finite_values.append(abs(fx))
    return max(finite_values, default=0.0)


def _susceptibility_quantity_metadata(
    result: dict[str, Any],
) -> dict[str, Any]:
    """Units and comparison rules so the LLM does not infer physics from the exponent."""
    delta_ax = _as_float_list(result.get("delta_chi_ax", []))
    delta_rh = _as_float_list(result.get("delta_chi_rh", []))

    def delta_metadata(
        field: str,
        values: list[Any],
        anisotropy_name: str,
    ) -> dict[str, Any]:
        cgs_values = [
            None
            if value is None
            else float(value) * DELTA_CHI_SI_PER_ION_TO_CGS
            for value in values
        ]
        present = [value for value in values if value is not None]
        exact_zero = bool(present) and all(float(value) == 0.0 for value in present)
        return {
            "field": field,
            "quantity": anisotropy_name,
            "value_unit": "m^3 ion^-1",
            "value_system": "SI",
            "value_basis": "per_ion",
            "equivalent_cgs_values": cgs_values,
            "equivalent_cgs_unit": "cm^3 mol^-1",
            "conversion": (
                "delta_chi_cgs[cm^3 mol^-1] = "
                "delta_chi_si[m^3 ion^-1] * N_A * 1e6 / (4*pi)"
            ),
            "N_A": AVOGADRO,
            "max_abs_value": _series_abs_max(values),
            "is_exact_zero": exact_zero,
            "interpretation": (
                "A small numerical value is expected because this field is "
                "SI per ion. Do not interpret 1e-31 as zero and do not "
                "compare the raw value with chi."
            ),
            "comparison_group": "susceptibility_anisotropy_si_per_ion",
            "must_not_compare_raw_with": ["chi", "chi_T"],
        }

    return {
        "chi": {
            "field": "chi",
            "quantity": "powder-average molar susceptibility",
            "unit": "cm^3 mol^-1",
            "system": "cgs",
            "basis": "per_mole",
            "comparison_group": "susceptibility_cgs_per_mole",
        },
        "chi_T": {
            "field": "chi_T",
            "quantity": "chi multiplied by T",
            "unit": "cm^3 K mol^-1",
            "system": "cgs",
            "basis": "per_mole",
            "comparison_group": "chi_T_cgs_per_mole",
        },
        "delta_chi_ax": delta_metadata(
            "delta_chi_ax",
            delta_ax,
            "axial susceptibility anisotropy",
        ),
        "delta_chi_rh": delta_metadata(
            "delta_chi_rh",
            delta_rh,
            "rhombic susceptibility anisotropy",
        ),
        "rules": [
            "chi is cgs and per mole",
            "chi_T is cgs and per mole",
            "delta_chi_ax is SI and per ion",
            "delta_chi_rh is SI and per ion",
            "raw delta_chi values must not be compared with raw chi values",
            "the exponent of delta_chi is not a zero test",
            "use is_exact_zero rather than abs(value) < a generic threshold",
            "do not recompute merely because delta_chi has magnitude ~1e-31",
        ],
    }


def _fit_parameter_catalog(
    spin_systems: list[dict[str, Any]],
    point_group: str | None = None,
    disabled_terms: list[str] | None = None,
    disabled_blocks: list[str] | None = None,
) -> list[dict[str, str]]:
    spec = _formula_spec(spin_systems, point_group, disabled_terms, disabled_blocks)
    return spec["allowed_parameter_keys"]


class ElectronSpec(BaseModel):
    """One magnetic centre, as on the Electrons tab of the Vue app."""

    type: Literal["S", "J"] = Field(..., description="'S' (uncoupled S, L) or 'J' (total angular momentum).")
    S: float | str | None = Field(default=None, description="Spin quantum number if type='S' (e.g. 0.5 or '7/2').")
    L: float | str | None = Field(default=0, description="Orbital quantum number if type='S'. 0 unless an SL model.")
    J: float | str | None = Field(default=None, description="Total angular momentum if type='J' (e.g. 7.5 or '15/2').")
    originIon: str | None = Field(
        default=None,
        description=(
            "Only a name from list_lanthanide_ions, e.g. 'Dy(III)', for Stevens θ_k. "
            "Null for 3d ions. 'Co(II)', 'Fe(III)', 'Ni(II)' are rejected — use null."
        ),
    )


class NucleusSpec(BaseModel):
    """One nucleus, as on the Nuclei tab. Name is looked up in MaIGIC's nuclei table."""

    nucleus: str = Field(..., description="Nucleus label, e.g. '1H' or '¹H'.")
    shieldingType: Literal["None", "Scalar", "Tensor"] = "None"
    shieldingScalar: float = 0.0
    shieldingDiagonal: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    shieldingEulerAngles: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])


class DetermineHamiltonianRequest(BaseModel):
    electrons: list[ElectronSpec] = Field(
        default_factory=list,
        description="Magnetic centres from the Electrons tab.",
    )
    nuclei: list[NucleusSpec] = Field(
        default_factory=list,
        description="Nuclei from the Nuclei tab. Omit if you are not modelling hyperfine/nuclear Zeeman.",
    )
    point_group: str | None = Field(
        default=None,
        description="Schoenflies symbol from the Symmetry tab, e.g. 'D4h', 'C2v', 'Oh'. Empty = no restriction.",
    )
    disabled_terms: list[str] = Field(
        default_factory=list,
        description="Optional terms to uncheck, e.g. 'H(I)_n-B', 'H(S)_S-S'. Default: all applicable terms ON.",
    )
    disabled_blocks: list[str] = Field(
        default_factory=list,
        description="Optional blocks to uncheck: 'H(S)', 'H(L)', 'H(J)', 'H(I)'.",
    )


class ComputeRequest(BaseModel):
    """Payload for compute_property / validate_request. Use output of determine_hamiltonian.

    Canonical shape only: {spin_systems, hamiltonian_params, point_group?}.
    Tmin/Tmax/numPoints/g/sigma_L/lambda_SL/lambda_sigma_SL/B_kq belong inside hamiltonian_params.
    """

    model_config = ConfigDict(extra="forbid")

    spin_systems: list[SpinSystem] = Field(
        ...,
        description=(
            "Spin centers. S-type: {id, type:'S', S, L, originIon?}. "
            "J-type: {id, type:'J', J, originIon?}. "
            "I-type: {id, type:'I', I, gamma, shieldingType, nucleus, ...}. "
            "ids must be unique integers >= 1. Fractions like '7/2' are accepted."
        ),
    )
    hamiltonian_params: HamiltonianParameters = Field(
        ...,
        description=(
            "Required. Copy hamiltonian_params_template from determine_hamiltonian, then set values. "
            "Do not add terms that are not in that template. "
            "calculationMode 'fixedB' sweeps T; 'fixedT' sweeps B. "
            "GUI σ¹ → sigma_L; GUI λ¹¹ → lambda_SL; GUI σ^{SL} → lambda_sigma_SL (not sigma_SL)."
        ),
    )
    point_group: str | None = Field(
        default=None,
        description="Schoenflies point group used in determine_hamiltonian (e.g. 'D4h'). Restricts CF B_kq.",
    )
    disabled_terms: list[str] = Field(
        default_factory=list,
        description="Same term keys disabled in determine_hamiltonian, e.g. 'H(S)_S-S'.",
    )
    disabled_blocks: list[str] = Field(
        default_factory=list,
        description="Same blocks disabled in determine_hamiltonian, e.g. 'H(I)'.",
    )
    force: bool = Field(
        default=False,
        description="Allow Hilbert-space dimension above the default safety limit. Does not enable caching.",
    )
    force_refit: bool = Field(
        default=False,
        description=(
            "Accepted for agents that want an explicit invalidate flag. Ignored: "
            "every compute_property call always recomputes (recomputed is always true)."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _canonical_nested_payload(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        leaked = sorted(k for k in data if k in _HP_MUST_NEST)
        if leaked:
            raise ValueError(
                "Canonical payload is {spin_systems, hamiltonian_params, point_group?}. "
                f"These fields must sit inside hamiltonian_params, not next to spin_systems: {leaked}."
            )
        hp = data.get("hamiltonian_params")
        if hp is None:
            raise ValueError(
                "hamiltonian_params is required. Copy hamiltonian_params_template from "
                "determine_hamiltonian (or get_example_payload) and put Tmin/Tmax/numPoints/"
                "sigma_L/lambda_SL/lambda_sigma_SL/B_kq inside it."
            )
        if isinstance(hp, dict):
            _reject_hp_aliases(hp)
        return data


class OptimizeRequest(BaseModel):
    """Fit Hamiltonian parameters to experimental curves (backend /api/optimize)."""

    spin_systems: list[SpinSystem] = Field(
        ...,
        description="Same spin_systems as compute_property.",
    )
    hamiltonian_params: HamiltonianParameters = Field(
        ...,
        description="Starting Hamiltonian from determine_hamiltonian template, with numerical values filled.",
    )
    point_group: str | None = Field(
        default=None,
        description="Same point_group as determine_hamiltonian.",
    )
    disabled_terms: list[str] = Field(default_factory=list)
    disabled_blocks: list[str] = Field(default_factory=list)
    parameter_fixed_state: dict[str, bool] = Field(
        ...,
        description=(
            "True = hold fixed, False = include in the fit. "
            "Keys from list_fit_parameter_keys (e.g. 'D_1', 'g_1', 'J_ex_1-2'). "
            "At least one value must be False."
        ),
    )
    reference_data: dict[str, list[float]] = Field(
        ...,
        description=(
            "Experimental columns of equal length. Must include B (tesla) "
            "and T (kelvin). Observable units are: "
            "M/M_x/M_y/M_z in mu_B; "
            "chi in cm^3 mol^-1 cgs per mole; "
            "chi_t in cm^3 K mol^-1 cgs per mole; "
            "delta_chi_ax and delta_chi_rh in m^3 ion^-1 SI per ion. "
            "Do not provide delta_chi in cm^3 mol^-1."
        ),
    )
    maxiter: int = Field(
        default=10,
        ge=1,
        le=MAX_OPT_ITER,
        description="Nelder–Mead iteration cap. Each step re-diagonalises; keep small.",
    )
    method: Literal["Nelder-Mead"] = Field(
        default="Nelder-Mead",
        description="Optimizer. Only Nelder-Mead is supported by the backend.",
    )
    force: bool = Field(
        default=False,
        description="Allow Hilbert-space dimension above the optimize safety limit.",
    )

    @model_validator(mode="before")
    @classmethod
    def _canonical_nested_payload(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        leaked = sorted(k for k in data if k in _HP_MUST_NEST)
        if leaked:
            raise ValueError(
                "Put sweep/Hamiltonian keys inside hamiltonian_params, not next to spin_systems: "
                f"{leaked}."
            )
        hp = data.get("hamiltonian_params")
        if isinstance(hp, dict):
            _reject_hp_aliases(hp)
        return data


@mcp.tool
def determine_hamiltonian(payload: DetermineHamiltonianRequest) -> dict[str, Any]:
    """Build the spin Hamiltonian the same way the MaIGIC GUI does (Determine Hamiltonian).

    Call this BEFORE compute_property, validate_request, list_fit_parameter_keys, or
    optimize_parameters. The GUI rules are:

    H(S): one S=1/2 → e-B only; one S>1/2 → e-B + ZFS (S-S); several S=1/2 → e-B + exchange;
    several with max S>1/2 → e-B + ZFS + exchange.
    H(L): if any S-center has L>0 → orbital Zeeman, spin-orbit, crystal field.
    H(J): one J → e-B + CF; several J → plus exchange.
    H(I): if any nuclei → nuclear Zeeman + hyperfine.

    Returns formula_latex, chemist_mapping (when L>0), spin_systems (with assigned ids),
    and hamiltonian_params_template. Copy spin_systems + template into later tools.
    Do not add Hamiltonian keys that are not in the template.

    S–L chemist mapping (do not probe energy_levels to learn this):
    λ (cm⁻¹) → lambda_SL['1'] (digit keys only, never '1-1');
    σ of Ŝ·L̂ → lambda_sigma_SL['1'] (default 0; H_SOC = λ×σ×Ŝ·L̂);
    sigma_L is orbital Zeeman, not chemist σ; axial Δ is B_kq['1_2_0'].
    For Co(II) S=3/2 L=1 call get_example_payload(example='co_sl_axial').
    """
    spec = determine_hamiltonian_spec(
        electrons=[e.model_dump() for e in payload.electrons],
        nuclei=[n.model_dump() for n in payload.nuclei],
        point_group=payload.point_group,
        disabled_terms=payload.disabled_terms,
        disabled_blocks=payload.disabled_blocks,
    )
    spec["hilbert_dimension"] = _hilbert_dim(spec["spin_systems"])
    return _jsonable(spec)


@mcp.tool
def list_nuclei() -> dict[str, Any]:
    """List nuclei known to MaIGIC (spin I and gyromagnetic ratio).

    gamma is in rad s⁻¹ T⁻¹, matching the Vue app's nuclei_info.json.
    Use these values when constructing I-type spin systems.
    """
    return {
        "gamma_unit": "rad s^-1 T^-1",
        "I_unit": "hbar",
        "nuclei": {
            name: {"I": data["I"], "gamma": data["gamma"]}
            for name, data in NUCLEI_INFO.items()
        },
    }


@mcp.tool
def list_lanthanide_ions() -> dict[str, Any]:
    """Ground-term S, L, J and Landé g_J for Ln(III) ions supported by MaIGIC crystal-field θ_k.

    For a J-multiplet model use type='J' with this J and g_J.
    For an uncoupled SL model use type='S' with this S and L, and set originIon to the ion name
    (required for Stevens θ_k when using B_kq). If originIon is null, θ_k defaults to 1.
    Eu(III) has J=0 (non-magnetic ground multiplet).
    """
    ions = {}
    for name, data in LANTHANIDE_IONS.items():
        ions[name] = {
            **data,
            "g_J": lande_g(data["S"], data["L"], data["J"]),
            "stevens_theta_L": cf_params[name]["L"],
            "stevens_theta_J": cf_params[name]["J"],
        }
    return {"ions": ions}


@mcp.tool
def get_crystal_field_terms(point_group: str) -> dict[str, Any]:
    """Allowed Stevens operators (k, q) for a crystallographic point group.

    B_kq keys are '{centerId}_{k}_{q}', e.g. '1_2_0' for B_2^0 on center 1.
    sigma_CF keys are '{centerId}_{k}', e.g. '1_2'.
    With originIon set to a Ln(III) name, B_kq is scaled by Stevens θ_k; otherwise θ_k = 1.
    """
    group = point_group.strip()
    mapping = SYMMETRY_INFO["group_to_symmetry"]
    if group not in mapping:
        return {
            "error": f"Unknown point group '{group}'",
            "known_groups": sorted(mapping.keys()),
        }
    family = mapping[group]
    coefs = SYMMETRY_INFO["symmetry_to_coefs"][family]
    return {
        "point_group": group,
        "symmetry_family": family,
        "allowed_kq": [{"k": k, "q": q} for k, q in coefs],
        "B_kq_key_example": "1_{k}_{q}",
        "sigma_CF_key_example": "1_{k}",
        "theta_k": "Stevens θ_k from originIon if set, else 1",
    }


@mcp.tool
def get_example_payload(
    example: Literal[
        "s_half",
        "s_one_zfs",
        "gd_j",
        "two_spins_exchange",
        "fit_zfs",
        "co_sl_axial",
    ] = "s_half",
) -> dict[str, Any]:
    """Return a complete valid payload. Copy spin_systems + hamiltonian_params into the next tool.

    example → next call:
      s_half             → compute_property(property='susceptibility')  Curie S=1/2 χ vs T
      s_one_zfs          → compute_property(property='energy_levels')   S=1, D vs B
      two_spins_exchange → compute_property(property='susceptibility')  dimer, H=J Ŝ1·Ŝ2, J>0 AF
      gd_j               → compute_property(property='susceptibility')  Gd(III) J=7/2 χ vs T
                           (for M vs B: calculationMode='fixedT', property='magnetization')
      co_sl_axial        → compute_property(property='susceptibility')  Co(II) S=3/2 L=1 χT and Δχ_ax
      fit_zfs            → optimize_parameters(payload=<this object>)   fit D to χT (column 'chi_t')

    Edit numbers only. Do not reverse-engineer keys. Do not probe energy_levels.
    """
    examples: dict[str, dict[str, Any]] = {
        "s_half": {
            "description": "Isotropic S=1/2, g=2, χ vs T at B=0.1 T",
            "spin_systems": [
                {"id": 1, "type": "S", "S": 0.5, "L": 0.0, "originIon": None}
            ],
            "hamiltonian_params": {
                "calculationMode": "fixedB",
                "fixedB": 0.1,
                "Tmin": 2.0,
                "Tmax": 300.0,
                "numPoints": 12,
                "gType": {"1": "isotropic"},
                "g": {"1": 2.0023},
            },
        },
        "s_one_zfs": {
            "description": "S=1 with axial ZFS D=3 cm⁻¹, energy levels vs B at T=2 K",
            "spin_systems": [
                {"id": 1, "type": "S", "S": 1.0, "L": 0.0, "originIon": None}
            ],
            "hamiltonian_params": {
                "calculationMode": "fixedT",
                "fixedT": 2.0,
                "Bmin": 0.0,
                "Bmax": 10.0,
                "numPoints": 21,
                "gType": {"1": "isotropic"},
                "g": {"1": 2.0},
                "D": {"1": 3.0},
                "E": {"1": 0.0},
            },
        },
        "gd_j": {
            "description": "Gd(III) as J=7/2 isotropic, χ vs T",
            "spin_systems": [
                {"id": 1, "type": "J", "J": 3.5, "originIon": "Gd(III)"}
            ],
            "hamiltonian_params": {
                "calculationMode": "fixedB",
                "fixedB": 0.1,
                "Tmin": 2.0,
                "Tmax": 300.0,
                "numPoints": 12,
                "gJType": {"1": "isotropic"},
                "g_J": {"1": 2.0},
            },
        },
        "two_spins_exchange": {
            "description": (
                "Two S=1/2, Heisenberg H=J Ŝ1·Ŝ2 with J_ex['1-2']=+10 cm⁻¹ (antiferromagnetic; "
                "χT→0 at low T). Next: compute_property(property='susceptibility'). J<0 is ferromagnetic."
            ),
            "spin_systems": [
                {"id": 1, "type": "S", "S": 0.5, "L": 0.0, "originIon": None},
                {"id": 2, "type": "S", "S": 0.5, "L": 0.0, "originIon": None},
            ],
            "hamiltonian_params": {
                "calculationMode": "fixedB",
                "fixedB": 0.1,
                "Tmin": 2.0,
                "Tmax": 300.0,
                "numPoints": 16,
                "gType": {"1": "isotropic", "2": "isotropic"},
                "g": {"1": 2.0023, "2": 2.0023},
                "J_ex": {"1-2": 10.0},
            },
        },
        "fit_zfs": {
            "description": (
                "Fit axial D of S=1 to χT vs T. Pass this object to optimize_parameters. "
                "Reference χT is synthetic (Curie-like); replace with experiment."
            ),
            "spin_systems": [
                {"id": 1, "type": "S", "S": 1.0, "L": 0.0, "originIon": None}
            ],
            "hamiltonian_params": {
                "calculationMode": "fixedB",
                "fixedB": 0.1,
                "Tmin": 2.0,
                "Tmax": 300.0,
                "numPoints": 8,
                "gType": {"1": "isotropic"},
                "g": {"1": 2.0},
                "D": {"1": 1.0},
                "E": {"1": 0.0},
            },
            "parameter_fixed_state": {"g_1": True, "D_1": False, "E_1": True},
            "reference_data": {
                "T": [2.0, 10.0, 25.0, 50.0, 100.0, 200.0, 300.0],
                "B": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                "chi_t": [0.8, 0.95, 0.99, 1.0, 1.0, 1.0, 1.0],
            },
            "maxiter": 8,
        },
        "co_sl_axial": build_co_sl_axial_example(),
    }
    return examples[example]


@mcp.tool
def list_fit_parameter_keys(
    spin_systems: list[SpinSystem],
    point_group: str | None = None,
    disabled_terms: list[str] | None = None,
    disabled_blocks: list[str] | None = None,
) -> dict[str, Any]:
    """List parameter_fixed_state keys that belong to the determined Hamiltonian.

    Call determine_hamiltonian first. Only those terms appear here (no D for S=1/2, no extra B_kq).
    True = hold fixed during optimize_parameters; False = fit.
    """
    dumped = [s.model_dump() if hasattr(s, "model_dump") else s for s in spin_systems]
    spec = _formula_spec(dumped, point_group, disabled_terms, disabled_blocks)
    catalog = spec["allowed_parameter_keys"]
    return {
        "convention": "True = fixed, False = optimize",
        "formula_latex": spec["formula_latex"],
        "n_keys": len(catalog),
        "keys": catalog,
    }


@mcp.tool
def validate_request(payload: ComputeRequest) -> dict[str, Any]:
    """Validate spin systems + Hamiltonian without running a (possibly slow) diagonalisation.

    Rejects Hamiltonian keys that are not in the formula from determine_hamiltonian.
    """
    request = _to_base_request(payload.spin_systems, payload.hamiltonian_params)
    spin_systems, hp, dim, spec = _run_prepare(request, force=True, **_formula_kwargs(payload))
    kinds: dict[str, int] = {}
    for spin in spin_systems:
        kinds[spin["type"]] = kinds.get(spin["type"], 0) + 1
    return {
        "ok": True,
        "formula_latex": spec["formula_latex"],
        "terms": spec["terms"],
        "hilbert_dimension": dim,
        "n_centers": len(spin_systems),
        "center_counts": kinds,
        "calculationMode": hp["calculationMode"],
        "numPoints": hp["numPoints"],
        "above_safety_limit": dim > MAX_HILBERT_DIM,
        "safety_limit": MAX_HILBERT_DIM,
        "fit_parameter_keys": [e["key"] for e in spec["allowed_parameter_keys"]],
        "params_echo": _params_echo(hp, spin_systems, payload.point_group),
        "warnings": physics_warnings_from_hp(hp, spec),
    }


@mcp.tool
def compute_property(
    property: PropertyName,
    payload: ComputeRequest,
) -> dict[str, Any]:
    """Run a MaIGIC backend calculation.

    Call determine_hamiltonian first. Pass its spin_systems + hamiltonian_params_template
    (numbers filled) and the same point_group. Extra Hamiltonian terms are rejected.

    property (pick from the chemist request):
      - susceptibility: χ vs T (fixedB) or vs B (fixedT).
        result.chi: cgs molar susceptibility, unit cm^3 mol^-1, per mole.
        result.chi_T: chi multiplied by temperature, unit cm^3 K mol^-1, per mole.
        result.delta_chi_ax and result.delta_chi_rh: susceptibility anisotropies,
        unit m^3 ion^-1, SI, per ion.
        Conversion already applied:
          delta_chi_si[m^3 ion^-1] = 4*pi * delta_chi_cgs[cm^3 mol^-1] / (N_A * 1e6).
        Values around 1e-31 m^3 ion^-1 are physically meaningful in this unit convention.
        Never compare raw delta_chi values with raw chi values.
        Never classify delta_chi as zero from its decimal exponent.
        Use quantity_metadata.is_exact_zero. Do not recompute because of the exponent.
      - magnetization: M_x, M_y, M_z and powder-mean result.M (μB). Use calculationMode='fixedT' for M vs B.
      - energy_levels: eigenvalues. fixedB → one spectrum; fixedT → vs B. result.E_cm_inv.
        Do not use this to learn API keys.

    Hamiltonian D, E, J_ex, λ, B_kq in cm⁻¹; A_hf in MHz. T in K, B in T.
    GUI H(L) names → API (do not invent sigma_SL):
      σ¹ (Orbital g-factor) → sigma_L['1']
      λ¹¹ (cm⁻¹) → lambda_SL['1']
      σ^{SL}_1 → lambda_sigma_SL['1']  (H_SOC = λ × σ^{SL} × S·L; default 0)
      Σ_k^L → sigma_CF['1_k']; B_k^q → B_kq['1_k_q']
    Every call is recomputed from this payload (no result cache). Check params_echo.numPoints
    against what you sent. Canonical payload: {spin_systems, hamiltonian_params, point_group?}.
    Co(II) S=3/2 L=1: get_example_payload('co_sl_axial').
    """
    request = _to_base_request(payload.spin_systems, payload.hamiltonian_params)
    spin_systems, hp, dim, spec = _run_prepare(
        request, force=payload.force, **_formula_kwargs(payload)
    )
    mode = hp["calculationMode"]

    if property == "susceptibility":
        raw = (
            get_magnetic_susceptibility_b_const(spin_systems, hp)
            if mode == "fixedB"
            else get_magnetic_susceptibility_t_const(spin_systems, hp)
        )
        result = {
            "chi": raw["\\chi"],
            "delta_chi_ax": raw["\\Delta \\chi_{ax}"],
            "delta_chi_rh": raw["\\Delta \\chi_{rh}"],
        }
        axis_key = "T" if mode == "fixedB" else "B"
        result[axis_key] = raw[axis_key]
        if mode == "fixedB":
            result["chi_T"] = [c * t for c, t in zip(raw["\\chi"], raw["T"])]
        else:
            result["chi_T"] = [c * hp["fixedT"] for c in raw["\\chi"]]
        units = {
            "chi": "cm^3 mol^-1 (cgs, per mole; powder average of the susceptibility tensor)",
            "chi_T": "cm^3 K mol^-1 (cgs, per mole; equals chi * T)",
            "delta_chi_ax": (
                "m^3 ion^-1 (SI, per ion). "
                "Already 4*pi * (chi_zz-0.5*(chi_xx+chi_yy)) / (N_A*1e6). "
                "Not cm^3/mol. See quantity_metadata; use is_exact_zero, not the exponent."
            ),
            "delta_chi_rh": (
                "m^3 ion^-1 (SI, per ion). "
                "Already 4*pi * (chi_xx-chi_yy) / (N_A*1e6). "
                "Not cm^3/mol. Do not compare raw values with chi."
            ),
        }
    elif property == "magnetization":
        raw = (
            get_magnetization_vector_b_const(spin_systems, hp)
            if mode == "fixedB"
            else get_magnetization_vector_t_const(spin_systems, hp)
        )
        result = {
            "M_x": raw["M_x"],
            "M_y": raw["M_y"],
            "M_z": raw["M_z"],
        }
        axis_key = "T" if mode == "fixedB" else "B"
        result[axis_key] = raw[axis_key]
        result["M"] = [
            (x + y + z) / 3.0 for x, y, z in zip(raw["M_x"], raw["M_y"], raw["M_z"])
        ]
        units = {"M_x": "mu_B", "M_y": "mu_B", "M_z": "mu_B", "M": "mu_B (isotropic mean)"}
    elif property == "energy_levels":
        if mode == "fixedB":
            raw_e = get_energy_levels_b_const(spin_systems, hp)
            result = {
                "B": hp["fixedB"],
                "E_J": raw_e,
                "E_cm_inv": _energies_to_cm(raw_e),
            }
        else:
            raw = get_energy_levels_t_const(spin_systems, hp)
            result = {
                "B": raw["B"],
                "E_J": raw["E"],
                "E_cm_inv": _energies_to_cm(raw["E"]),
            }
        units = {"E_J": "J", "E_cm_inv": "cm^-1"}
    else:
        raise ValueError(f"Unknown property: {property}")

    payload_out: dict[str, Any] = {
        "property": property,
        "formula_latex": spec["formula_latex"],
        "terms": spec["terms"],
        "calculationMode": mode,
        "hilbert_dimension": dim,
        "numPoints": hp["numPoints"],
        "recomputed": True,
        "params_echo": _params_echo(hp, spin_systems, payload.point_group),
        "units": units,
        "result": result,
        "warnings": physics_warnings_from_hp(hp, spec),
    }
    csv_text = _grid_csv(result)
    if csv_text is not None:
        payload_out["grid_csv"] = csv_text
    if property == "susceptibility":
        payload_out["conventions"] = SUSCEPTIBILITY_CONVENTIONS
        payload_out["quantity_metadata"] = _susceptibility_quantity_metadata(result)
        payload_out["agent_instruction"] = (
            "Report delta_chi_ax and delta_chi_rh as SI per-ion quantities. "
            "A value near 1e-31 m^3 ion^-1 is not automatically zero. "
            "Do not compare raw delta_chi values with chi or chi_T. "
            "Use quantity_metadata and is_exact_zero. "
            "Do not recompute because of the exponent."
        )
    return _jsonable(payload_out)


@mcp.tool
def optimize_parameters(payload: OptimizeRequest) -> dict[str, Any]:
    """Fit unfixed Hamiltonian parameters to experimental B, T, and observable columns.

    Call determine_hamiltonian first. Pass the whole request as payload (not maxiter at the top level).
    Ready-made: get_example_payload(example='fit_zfs').
    reference_data: arrays of equal length. Must include B (tesla) AND T (kelvin), plus at least one of
    M, M_x, M_y, M_z, chi, chi_t, delta_chi_ax, delta_chi_rh.
    reference_data units:
      M, M_x, M_y, M_z: mu_B
      chi: cm^3 mol^-1, cgs, per mole
      chi_t: cm^3 K mol^-1, cgs, per mole
      delta_chi_ax / delta_chi_rh: m^3 ion^-1, SI, per ion (not cm^3 mol^-1)
    χT column is 'chi_t' (lowercase t) — not 'chi_T' from compute_property.
    parameter_fixed_state keys from list_fit_parameter_keys; True=fixed, False=fit; ≥1 False.
    maxiter inside payload, default 10, keep ≤20. Nelder–Mead only.
    """
    allowed_obs = {
        "M",
        "M_x",
        "M_y",
        "M_z",
        "chi",
        "chi_t",
        "delta_chi_ax",
        "delta_chi_rh",
    }
    data = payload.reference_data
    if "B" not in data or "T" not in data:
        raise ValueError("reference_data must include both 'B' (T) and 'T' (K) arrays.")
    n = len(data["B"])
    if n != len(data["T"]):
        raise ValueError("reference_data 'B' and 'T' must have the same length.")
    if n < 2:
        raise ValueError("Need at least two data points.")
    if n > MAX_OPT_POINTS:
        raise ValueError(
            f"reference_data has {n} points (limit {MAX_OPT_POINTS}). Downsample the curve."
        )
    for key, values in data.items():
        if len(values) != n:
            raise ValueError(f"Column '{key}' length {len(values)} != {n}.")
    observables = [k for k in data if k not in {"B", "T"}]
    unknown = [k for k in observables if k not in allowed_obs]
    if unknown:
        raise ValueError(
            f"Unknown observable column(s) {unknown}. Allowed: {sorted(allowed_obs)}."
        )
    if not observables:
        raise ValueError(
            "reference_data needs at least one observable besides B and T "
            f"({sorted(allowed_obs)})."
        )

    active = sorted(k for k, fixed in payload.parameter_fixed_state.items() if not fixed)
    if not active:
        raise ValueError(
            "parameter_fixed_state must set at least one key to False (parameter to fit)."
        )

    request = _to_base_request(
        payload.spin_systems,
        payload.hamiltonian_params,
        payload.parameter_fixed_state,
    )
    spin_systems, hp, dim, spec = _run_prepare(
        request,
        force=payload.force,
        max_dim=MAX_OPT_HILBERT_DIM,
        **_formula_kwargs(payload),
    )

    stdout_buf = io.StringIO()
    with redirect_stdout(stdout_buf):
        optimized = optimize(
            reference_data=data,
            spin_systems=spin_systems,
            hamiltonian_params=hp,
            parameter_fixed_state=payload.parameter_fixed_state,
            return_only_active_params=False,
            maxiter=payload.maxiter,
            method=payload.method,
        )

    from optimize import get_value_by_key

    fitted = {key: get_value_by_key(key, optimized) for key in active}
    return _jsonable(
        {
            "ok": True,
            "formula_latex": spec["formula_latex"],
            "terms": spec["terms"],
            "hilbert_dimension": dim,
            "n_points": n,
            "observables": observables,
            "maxiter": payload.maxiter,
            "method": payload.method,
            "fitted_values": fitted,
            "hamiltonian_params": HamiltonianParameters.model_validate(optimized).model_dump(),
            "warnings": physics_warnings_from_hp(hp, spec),
        }
    )


if __name__ == "__main__":
    host = os.environ.get("MCP_HOST", "0.0.0.0")
    port = int(os.environ.get("MCP_PORT", "8000"))
    mcp.run(
        transport="http",
        host=host,
        port=port,
        path="/mcp",
        stateless_http=True,
        host_origin_protection=False,
        uvicorn_config={"timeout_keep_alive": 300},
    )
