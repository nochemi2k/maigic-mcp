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
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastmcp import FastMCP
from fastmcp.server.auth.providers.jwt import StaticTokenVerifier
from pydantic import BaseModel, Field
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
    apply_template_defaults,
    determine_from_spin_systems,
    determine_hamiltonian_spec,
    extra_hamiltonian_keys,
)

PropertyName = Literal["susceptibility", "magnetization", "energy_levels"]
FitObservable = Literal[
    "M", "M_x", "M_y", "M_z", "chi", "chi_t", "delta_chi_ax", "delta_chi_rh"
]
MAX_HILBERT_DIM = 512
MAX_OPT_HILBERT_DIM = 64
MAX_OPT_POINTS = 40
MAX_OPT_ITER = 50

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
    instructions=(
        "REQUIRED WORKFLOW — do this before every calculation, fit, or modelling step: "
        "1) Call determine_hamiltonian with the complex (electrons, nuclei, point_group). "
        "2) Read formula_latex. That is the Hamiltonian. Do not invent ZFS, exchange, CF, "
        "hyperfine, or orbital terms that are missing from it. "
        "3) Copy spin_systems and hamiltonian_params_template into compute_property / "
        "optimize_parameters / validate_request. Change only numbers and sweep settings. "
        "Passing keys that are not in the template is rejected. "
        "Call list_nuclei before naming nuclei. Call list_lanthanide_ions for Ln(III) S, L, J, g_J. "
        "If originIon is a Ln(III) name, B_kq is multiplied by Stevens θ_k; if null, θ_k = 1. "
        "parameter_fixed_state: True = hold fixed, False = fit; at least one key must be False. "
        "Keep numPoints at 10–20 unless the user asks for a dense curve. "
        "Keep optimize maxiter ≤ 20 and few data points. "
        "Hilbert-space dimension is the product of (2s+1) over all centers (and L if L>0)."
    ),
    auth=_http_auth(),
)


@mcp.custom_route("/health", methods=["GET"])
async def health(_request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


def _load_json(name: str) -> Any:
    with (DATA_DIR / name).open(encoding="utf-8") as f:
        return json.load(f)


NUCLEI_INFO: dict[str, dict[str, float]] = _load_json("nuclei_info.json")
SYMMETRY_INFO: dict[str, Any] = _load_json("symmetry_info.json")

# Ground-term data for free Ln(III) ions (Russell–Saunders).
LANTHANIDE_IONS: dict[str, dict[str, Any]] = {
    "Ce(III)": {"config": "4f1", "term": "2F5/2", "S": 0.5, "L": 3, "J": 2.5},
    "Pr(III)": {"config": "4f2", "term": "3H4", "S": 1.0, "L": 5, "J": 4.0},
    "Nd(III)": {"config": "4f3", "term": "4I9/2", "S": 1.5, "L": 6, "J": 4.5},
    "Pm(III)": {"config": "4f4", "term": "5I4", "S": 2.0, "L": 6, "J": 4.0},
    "Sm(III)": {"config": "4f5", "term": "6H5/2", "S": 2.5, "L": 5, "J": 2.5},
    "Eu(III)": {"config": "4f6", "term": "7F0", "S": 3.0, "L": 3, "J": 0.0},
    "Gd(III)": {"config": "4f7", "term": "8S7/2", "S": 3.5, "L": 0, "J": 3.5},
    "Tb(III)": {"config": "4f8", "term": "7F6", "S": 3.0, "L": 3, "J": 6.0},
    "Dy(III)": {"config": "4f9", "term": "6H15/2", "S": 2.5, "L": 5, "J": 7.5},
    "Ho(III)": {"config": "4f10", "term": "5I8", "S": 2.0, "L": 6, "J": 8.0},
    "Er(III)": {"config": "4f11", "term": "4I15/2", "S": 1.5, "L": 6, "J": 7.5},
    "Tm(III)": {"config": "4f12", "term": "3H6", "S": 1.0, "L": 5, "J": 6.0},
    "Yb(III)": {"config": "4f13", "term": "2F7/2", "S": 0.5, "L": 3, "J": 7.5},
}


def _lande_g(S: float, L: float, J: float) -> float | None:
    if J <= 0:
        return None
    return 1.0 + (J * (J + 1) + S * (S + 1) - L * (L + 1)) / (2.0 * J * (J + 1))


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


def _prepare(
    request: BaseRequest,
    force: bool,
    *,
    max_dim: int = MAX_HILBERT_DIM,
    point_group: str | None = None,
    disabled_terms: list[str] | None = None,
    disabled_blocks: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], int, dict[str, Any]]:
    spin_systems = request.spin_systems.model_dump()
    spec = _formula_spec(spin_systems, point_group, disabled_terms, disabled_blocks)
    hp_user = request.hamiltonian_params.model_dump(exclude_unset=True)
    extras = extra_hamiltonian_keys(hp_user, spec)
    if extras:
        raise ValueError(
            "hamiltonian_params contains terms that are not in the determined Hamiltonian: "
            f"{extras}. Formula: {spec['formula_latex']}. "
            "Call determine_hamiltonian and copy hamiltonian_params_template; "
            "do not add D, E, J_ex, B_kq, A_hf, or other keys missing from that template."
        )
    hp = apply_template_defaults(
        request.hamiltonian_params.model_dump(),
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
    return spin_systems, hp, dim, spec


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
        description="Ln(III) label such as 'Dy(III)' for Stevens θ_k. Null → θ_k = 1.",
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
    """Payload for compute_property / validate_request. Use output of determine_hamiltonian."""

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
        default_factory=HamiltonianParameters,
        description=(
            "Copy hamiltonian_params_template from determine_hamiltonian, then set values. "
            "Do not add terms that are not in that template. "
            "calculationMode 'fixedB' sweeps T; 'fixedT' sweeps B."
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
        description="Allow Hilbert-space dimension above the default safety limit.",
    )


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
            "Experimental columns of equal length. Must include B (tesla) and T (kelvin), "
            "plus at least one of M, M_x, M_y, M_z, chi, chi_t, delta_chi_ax, delta_chi_rh. "
            "chi in cm^3 mol^-1, chi_t in cm^3 K mol^-1, M in μB."
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

    Returns formula_latex, spin_systems (with assigned ids), and hamiltonian_params_template.
    Copy those two objects into later tools. Do not add Hamiltonian keys that are not in the template.
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
            "g_J": _lande_g(data["S"], data["L"], data["J"]),
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
        "s_half", "s_one_zfs", "gd_j", "two_spins_exchange", "fit_zfs"
    ] = "s_half",
) -> dict[str, Any]:
    """Return a complete, valid payload you can edit and pass to compute_property or optimize_parameters."""
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
            "description": "Two S=1/2 coupled by J_ex = -10 cm⁻¹ (antiferromagnetic)",
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
                "J_ex": {"1-2": -10.0},
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
    spin_systems, hp, dim, spec = _prepare(request, force=True, **_formula_kwargs(payload))
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
        "warnings": spec["warnings"],
    }


@mcp.tool
def compute_property(
    property: PropertyName,
    payload: ComputeRequest,
) -> dict[str, Any]:
    """Run a MaIGIC backend calculation.

    Call determine_hamiltonian first and pass its spin_systems + hamiltonian_params_template
    (with numbers filled). Extra Hamiltonian terms not in that formula are rejected.

    property:
      - susceptibility: χ, Δχ_ax, Δχ_rh vs T (fixedB) or vs B (fixedT)
      - magnetization: M_x, M_y, M_z in μB
      - energy_levels: eigenvalues. fixedB → one spectrum at fixedB;
        fixedT → spectrum vs B. Also returned converted to cm⁻¹.

    Units: T in K, B in T, χ in cm³ mol⁻¹, Δχ in 1e-6 cm³ mol⁻¹ (SI, includes 4π from cgs),
    M in μB. Hamiltonian parameters D, E, J_ex, λ, B_kq are in cm⁻¹; A_hf in MHz.
    """
    request = _to_base_request(payload.spin_systems, payload.hamiltonian_params)
    spin_systems, hp, dim, spec = _prepare(
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
            "chi": "cm^3 mol^-1",
            "delta_chi_ax": "1e-6 cm^3 mol^-1 (SI; 4pi cgs->SI)",
            "delta_chi_rh": "1e-6 cm^3 mol^-1 (SI; 4pi cgs->SI)",
            "chi_T": "cm^3 K mol^-1",
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

    return _jsonable(
        {
            "property": property,
            "formula_latex": spec["formula_latex"],
            "terms": spec["terms"],
            "calculationMode": mode,
            "hilbert_dimension": dim,
            "numPoints": hp["numPoints"],
            "units": units,
            "result": result,
        }
    )


@mcp.tool
def optimize_parameters(payload: OptimizeRequest) -> dict[str, Any]:
    """Fit unfixed Hamiltonian parameters to experimental B, T, and observable columns.

    Call determine_hamiltonian first. Only fit keys that appear in that formula
    (see allowed_parameter_keys / list_fit_parameter_keys). Extra Hamiltonian terms are rejected.
    Uses the same Nelder–Mead loop as POST /api/optimize.
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
    spin_systems, hp, dim, spec = _prepare(
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
