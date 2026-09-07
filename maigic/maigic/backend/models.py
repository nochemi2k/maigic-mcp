import pydantic
import typing
from decimal import Decimal
import re

# ========================================
# Magnetic Susceptibility Results
# ========================================

class MagneticSusceptibilityTConstResult(pydantic.BaseModel):
    """Result of get_magnetic_susceptibility_t_const: χ vs B at fixed T"""
    model_config = pydantic.ConfigDict(populate_by_name=True)

    B: typing.List[float] = pydantic.Field(..., description="Magnetic field values (T)")
    chi: typing.List[float] = pydantic.Field(..., alias='\\chi', description="Molar magnetic susceptibility χ (cm³·mol⁻¹)")
    delta_chi_ax: typing.List[float] = pydantic.Field(..., alias='\\Delta \\chi_{ax}', description="Axial susceptibility anisotropy Δχ_ax (SI 10⁻⁶ m³ mol⁻¹ = 4π × Δχ_cgs)")
    delta_chi_rh: typing.List[float] = pydantic.Field(..., alias='\\Delta \\chi_{rh}', description="Rhombic susceptibility anisotropy Δχ_rh (SI 10⁻⁶ m³ mol⁻¹ = 4π × Δχ_cgs)")

class MagneticSusceptibilityBConstResult(pydantic.BaseModel):
    """Result of get_magnetic_susceptibility_b_const: χ vs T at fixed B"""
    model_config = pydantic.ConfigDict(populate_by_name=True)

    T: typing.List[float] = pydantic.Field(..., description="Temperature values (K)")
    chi: typing.List[float] = pydantic.Field(..., alias='\\chi', description="Molar magnetic susceptibility χ (cm³·mol⁻¹)")
    delta_chi_ax: typing.List[float] = pydantic.Field(..., alias='\\Delta \\chi_{ax}', description="Axial susceptibility anisotropy Δχ_ax (SI 10⁻⁶ m³ mol⁻¹ = 4π × Δχ_cgs)")
    delta_chi_rh: typing.List[float] = pydantic.Field(..., alias='\\Delta \\chi_{rh}', description="Rhombic susceptibility anisotropy Δχ_rh (SI 10⁻⁶ m³ mol⁻¹ = 4π × Δχ_cgs)")

# ========================================
# Magnetization Vector Results
# ========================================

class MagnetizationVectorTConstResult(pydantic.BaseModel):
    """Result of get_magnetization_vector_t_const: M vs B at fixed T"""
    B: typing.List[float] = pydantic.Field(..., description="Magnetic field values (T)")
    M_x: typing.List[float] = pydantic.Field(..., description="X-component of magnetization (μB)")
    M_y: typing.List[float] = pydantic.Field(..., description="Y-component of magnetization (μB)")
    M_z: typing.List[float] = pydantic.Field(..., description="Z-component of magnetization (μB)")

class MagnetizationVectorBConstResult(pydantic.BaseModel):
    """Result of get_magnetization_vector_b_const: M vs T at fixed B"""
    T: typing.List[float] = pydantic.Field(..., description="Temperature values (K)")
    M_x: typing.List[float] = pydantic.Field(..., description="X-component of magnetization (μB)")
    M_y: typing.List[float] = pydantic.Field(..., description="Y-component of magnetization (μB)")
    M_z: typing.List[float] = pydantic.Field(..., description="Z-component of magnetization (μB)")

# ========================================
# Energy Levels Results
# ========================================

class EnergyLevelsTConstResult(pydantic.BaseModel):
    """Result of get_energy_levels_t_const: Energy levels vs B at fixed T"""
    B: typing.List[float] = pydantic.Field(..., description="Magnetic field values (T)")
    E: typing.List[typing.List[float]] = pydantic.Field(
        ...,
        description="Energy levels at each field point; E[i][j] = j-th energy level (cm⁻¹) at B[i]"
    )

class EnergyLevelsBConstResult(pydantic.BaseModel):
    """Result of get_energy_levels_b_const: Energy levels at fixed B and T"""
    E: typing.List[float] = pydantic.Field(..., description="Energy levels (cm⁻¹) at fixed magnetic field")

# ========================================
# Spin System Variants (Discriminated Union)
# ========================================

class SSpinSystem(pydantic.BaseModel):
    """Electron spin system with explicit S and L quantum numbers (S-type)"""
    model_config = pydantic.ConfigDict(populate_by_name=True)

    id: int = pydantic.Field(..., ge=1, description="Unique identifier for the spin system")
    type: typing.Literal['S'] = pydantic.Field(..., description="Spin system type indicator")
    S: float = pydantic.Field(..., gt=0, description="Electron spin quantum number S (e.g., 0.5, 1, 3/2)")
    L: float = pydantic.Field(..., ge=0, description="Orbital angular momentum quantum number L")
    originIon: typing.Optional[str] = pydantic.Field(
        None,
        description="Original ion label (e.g., 'Gd³⁺', 'Fe³⁺') if applicable"
    )

    @pydantic.field_validator('S', 'L', mode='before')
    @classmethod
    def parse_fraction_strings(cls, v, info: pydantic.ValidationInfo):
        """Accept fractional strings like '3/2' and convert to float"""
        if isinstance(v, str):
            if '/' in v:
                num, den = v.split('/')
                return float(Decimal(num) / Decimal(den))
            return float(v)
        return v


class JSpinSystem(pydantic.BaseModel):
    """Electron spin system with total angular momentum J (J-type, LS-coupled)"""
    model_config = pydantic.ConfigDict(populate_by_name=True)

    id: int = pydantic.Field(..., ge=1, description="Unique identifier for the spin system")
    type: typing.Literal['J'] = pydantic.Field(..., description="Spin system type indicator")
    J: float = pydantic.Field(..., gt=0, description="Total angular momentum quantum number J (e.g., 0.5, 1, 7/2)")
    originIon: typing.Optional[str] = pydantic.Field(
        None,
        description="Original ion label (e.g., 'Eu³⁺', 'Tb³⁺') if applicable"
    )

    @pydantic.field_validator('J', mode='before')
    @classmethod
    def parse_fraction_strings(cls, v, info: pydantic.ValidationInfo):
        """Accept fractional strings like '7/2' and convert to float"""
        if isinstance(v, str):
            if '/' in v:
                num, den = v.split('/')
                return float(Decimal(num) / Decimal(den))
            return float(v)
        return v


class ISpinSystem(pydantic.BaseModel):
    """Nuclear spin system (I-type) with shielding tensors"""
    model_config = pydantic.ConfigDict(populate_by_name=True)

    id: int = pydantic.Field(..., ge=1, description="Unique identifier for the spin system")
    type: typing.Literal['I'] = pydantic.Field(..., description="Spin system type indicator")
    I: float = pydantic.Field(..., gt=0, description="Nuclear spin quantum number I (e.g., 1/2, 3/2, 5/2)")
    gamma: float = pydantic.Field(
        ...,
        description=(
            "Gyromagnetic ratio in rad s⁻¹ T⁻¹, as returned by list_nuclei / "
            "determine_hamiltonian (e.g. ¹H ≈ 2.6752e8). Do not use MHz/T (42.576)."
        ),
    )
    shieldingType: typing.Literal['None', 'Scalar', 'Tensor'] = pydantic.Field(
        ..., 
        description="Type of shielding tensor representation"
    )
    shieldingScalar: float = pydantic.Field(
        0.0,
        ge=0,
        description="Isotropic shielding constant σ_iso (ppm)"
    )
    shieldingDiagonal: typing.List[float] = pydantic.Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        min_length=3,
        max_length=3,
        description="Principal values of shielding tensor [σ_xx, σ_yy, σ_zz] (ppm)"
    )
    shieldingEulerAngles: typing.List[float] = pydantic.Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        min_length=3,
        max_length=3,
        description="Euler angles (α, β, γ) defining tensor orientation (radians)"
    )
    nucleus: str = pydantic.Field(..., description="Nucleus identifier (e.g., '¹H')")

    @pydantic.field_validator('I', mode='before')
    @classmethod
    def parse_fraction_strings(cls, v, info: pydantic.ValidationInfo):
        """Accept fractional strings like '3/2' and convert to float"""
        if isinstance(v, str):
            if '/' in v:
                num, den = v.split('/')
                return float(Decimal(num) / Decimal(den))
            return float(v)
        return v

    @pydantic.field_validator('shieldingDiagonal', 'shieldingEulerAngles')
    @classmethod
    def check_length(cls, v, info: pydantic.ValidationInfo):
        if len(v) != 3:
            raise ValueError("Must contain exactly 3 elements")
        return v


# ========================================
# Discriminated Union with explicit discriminator
# ========================================

# Critical V2 change: discriminator must be attached to the Union type via typing.Annotated
SpinSystem = typing.Annotated[
    typing.Union[SSpinSystem, JSpinSystem, ISpinSystem],
    pydantic.Field(discriminator='type')
]
"""Discriminated union of all supported spin system types (discriminator='type')"""


# ========================================
# Root Model for Spin System Collection
# ========================================

class AllSpinSystems(pydantic.RootModel[typing.List[SpinSystem]]):
    """Container for a collection of heterogeneous spin systems"""
    root: typing.List[SpinSystem] = pydantic.Field(
        ...,
        description="Ordered list of spin systems (electronic S/J-type and nuclear I-type)",
    )

    model_config = pydantic.ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": [
                {"id": 1, "type": "S", "S": 1.5, "L": 0.0, "originIon": "Gd3+"},
                {"id": 2, "type": "I", "I": 0.5, "gamma": 267520000.0, "shieldingType": "None",
                 "shieldingScalar": 0.0, "shieldingDiagonal": [0,0,0],
                 "shieldingEulerAngles": [0,0,0], "nucleus": "¹H"}
            ]
        }
    )

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item):
        return self.root[item]

    def __len__(self):
        return len(self.root)

# ========================================
# Helper Types & Validators
# ========================================

AnisotropyType = typing.Literal["isotropic", "anisotropic"]
CalculationMode = typing.Literal["fixedB", "fixedT"]

# Key format validators (unchanged)
def validate_exchange_key(key: str) -> bool:
    """Validates exchange coupling keys like '1-2', '3-5'"""
    return bool(re.fullmatch(r"\d+-\d+", key))

def validate_hfine_key(key: str) -> bool:
    """Validates hyperfine coupling keys like 'e1-n3', 'L-e1-n3', 'J-e2-n3'"""
    return bool(re.fullmatch(r"(L-|J-)?e\d+-n\d+", key))

def validate_cf_sigma_key(key: str) -> bool:
    """Validates CF sigma keys like '1_2', '2_4', '3_6'"""
    return bool(re.fullmatch(r"\d+_[246]", key))

def validate_cf_bkq_key(key: str) -> bool:
    """Validates CF B_kq keys like '1_2_-2', '2_4_3', '3_6_6'"""
    return bool(re.fullmatch(r"\d+_[246]_[-+]?\d+", key))

def validate_gtensor(value: typing.List[float]) -> typing.List[float]:
    """Validates g-tensor has exactly 3 components"""
    if len(value) != 3:
        raise ValueError("g-tensor must contain exactly 3 components [gx, gy, gz]")
    return value


# ========================================
# Main Hamiltonian Parameters Model
# ========================================

class HamiltonianParameters(pydantic.BaseModel):
    """Complete Hamiltonian parameter set for spin system calculations"""
    
    model_config = pydantic.ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "calculationMode": "fixedB",
                "fixedB": 1.0,
                "fixedT": 298.15,
                "Tmin": 10,
                "Tmax": 300,
                "Bmin": 0.1,
                "Bmax": 10.0,
                "numPoints": 50,
                "tip_correction": 0,
                "diamagnetic_correction": 0.0,
                "gType": {"1": "isotropic"},
                "gAniso": {"1": [2.0023, 2.0023, 2.0023]},
                "g": {"1": 2.0023},
                "D": {"1": 0.0},
                "E": {"1": 0.0},
                "J_ex": {},
                "sigma_L": {"1": 1.0},
                "lambda_SL": {"1": 0.0},
                "lambda_sigma_SL": {"1": 0.0},
                "g_J": {},
                "J_ex_J": {},
                "A_hf": {"e1-n3": 0.0, "L-e1-n3": 0.0, "J-e2-n3": 0.0},
                "sigma_CF": {"1_2": 1.0, "1_4": 1.0, "1_6": 1.0},
                "B_kq": {
                    "1_2_-2": 0.0, "1_2_-1": 0.0, "1_2_0": 0.0, "1_2_1": 0.0, "1_2_2": 0.0,
                    "1_4_-4": 0.0, "1_4_-3": 0.0, "1_4_-2": 0.0, "1_4_-1": 0.0, "1_4_0": 0.0,
                    "1_4_1": 0.0, "1_4_2": 0.0, "1_4_3": 0.0, "1_4_4": 0.0,
                    "1_6_-6": 0.0, "1_6_-5": 0.0, "1_6_-4": 0.0, "1_6_-3": 0.0, "1_6_-2": 0.0,
                    "1_6_-1": 0.0, "1_6_0": 0.0, "1_6_1": 0.0, "1_6_2": 0.0, "1_6_3": 0.0,
                    "1_6_4": 0.0, "1_6_5": 0.0, "1_6_6": 0.0
                },
                "computeE": True,
                "computeM": True,
                "computeChi": True,
                "computeChiAx": True,
                "computeChiRh": True,
                "computeChiT": True
            }
        }
    )

    # --- Calculation Mode & Fixed Values ---
    calculationMode: CalculationMode = pydantic.Field(
        default="fixedB",
        description="Calculation mode: 'fixedB' (sweep T) or 'fixedT' (sweep B)"
    )
    fixedB: float = pydantic.Field(
        default=1.0,
        ge=0.0,
        description="Fixed magnetic field strength (T) used when calculationMode='fixedB'"
    )
    fixedT: float = pydantic.Field(
        default=298.15,
        gt=0.0,
        description="Fixed temperature (K) used when calculationMode='fixedT'"
    )

    # --- Sweep Ranges & Resolution ---
    Tmin: float = pydantic.Field(
        default=10.0,
        gt=0.0,
        description="Minimum temperature for sweeps (K)"
    )
    Tmax: float = pydantic.Field(
        default=300.0,
        gt=0.0,
        description="Maximum temperature for sweeps (K)"
    )
    Bmin: float = pydantic.Field(
        default=0.1,
        ge=0.0,
        description="Minimum magnetic field for sweeps (T)"
    )
    Bmax: float = pydantic.Field(
        default=10.0,
        gt=0.0,
        description="Maximum magnetic field for sweeps (T)"
    )
    numPoints: int = pydantic.Field(
        default=50,
        ge=2,
        le=10000,
        description="Number of points in temperature/field sweeps (minimum 2; 1 is rejected)"
    )

    # --- Corrections ---
    tip_correction: float = pydantic.Field(
        default=0.0,
        description="Tip correction term added to susceptibility tensor diagonal elements"
    )
    diamagnetic_correction: float = pydantic.Field(
        default=0.0,
        description="Diamagnetic correction term added to susceptibility tensor diagonal elements"
    )

    # --- g-Factor Parameters (S-type centers) ---
    gType: typing.Dict[str, AnisotropyType] = pydantic.Field(
        default_factory=dict,
        description="g-factor anisotropy type per S-center ID: {'1': 'isotropic', '2': 'anisotropic', ...}"
    )
    gAniso: typing.Dict[str, typing.List[float]] = pydantic.Field(
        default_factory=dict,
        description="Anisotropic g-tensor components per S-center ID: {'1': [gx, gy, gz], ...}"
    )
    g: typing.Dict[str, float] = pydantic.Field(
        default_factory=dict,
        description="Isotropic g-factor per S-center ID: {'1': 2.0023, ...}"
    )

    # --- Zero-pydantic.Field Splitting (S-type centers) ---
    D: typing.Dict[str, float] = pydantic.Field(
        default_factory=dict,
        description="Axial ZFS parameter D (Dzz) per S-center ID: {'1': 0.1, ...} (cm⁻¹)"
    )
    E: typing.Dict[str, float] = pydantic.Field(
        default_factory=dict,
        description="Rhombic ZFS parameter E (Dxx-Dyy) per S-center ID: {'1': 0.03, ...} (cm⁻¹)"
    )

    # --- Exchange Coupling (S-type centers) ---
    J_ex: typing.Dict[typing.Annotated[str, pydantic.Field(pattern=r"^\d+-\d+$")], float] = pydantic.Field(
        default_factory=dict,
        description=(
            "Heisenberg exchange H = J Ŝ_i·Ŝ_j (cm⁻¹), keys '{i}-{j}' e.g. {'1-2': 10}. "
            "J > 0 is antiferromagnetic (singlet below for two S=1/2). J < 0 is ferromagnetic."
        )
    )

    # --- Orbital Angular Momentum Parameters (L contributions) ---
    sigma_L: typing.Dict[str, float] = pydantic.Field(
        default_factory=dict,
        description=(
            "Orbital reduction on μ_B σ B·L̂ (orbital Zeeman) per L-center ID: {'1': 0.9, ...}. "
            "Default 1. Not the chemist σ of Ŝ·L̂ — that is lambda_sigma_SL."
        )
    )
    lambda_SL: typing.Dict[typing.Annotated[str, pydantic.Field(pattern=r"^\d+$")], float] = pydantic.Field(
        default_factory=dict,
        description=(
            "Chemist λ of Ŝ·L̂ (cm⁻¹) per center ID. Keys are digits only, e.g. {'1': 152.4}. "
            "Never use '1-1'. H_SOC = lambda_SL × lambda_sigma_SL × Ŝ·L̂."
        )
    )
    lambda_sigma_SL: typing.Dict[typing.Annotated[str, pydantic.Field(pattern=r"^\d+$")], float] = pydantic.Field(
        default_factory=dict,
        description=(
            "Chemist σ of Ŝ·L̂ (dimensionless) per center ID, e.g. {'1': 1.35}. "
            "Keys are digits only, never '1-1'. Template default is 0: if λ is set and this stays 0, "
            "SOC is identically zero. Not sigma_L (orbital Zeeman) and not sigma_CF (CF scale)."
        )
    )

    # --- g-Factor Parameters (J-type centers) ---
    gJType: typing.Dict[str, AnisotropyType] = pydantic.Field(
        default_factory=dict,
        description="g-factor anisotropy type per J-center ID: {'1': 'isotropic', ...}"
    )
    gJAniso: typing.Dict[str, typing.List[float]] = pydantic.Field(
        default_factory=dict,
        description="Anisotropic g-tensor for J-centers: {'1': [gx, gy, gz], ...}"
    )
    g_J: typing.Dict[str, float] = pydantic.Field(
        default_factory=dict,
        description=(
            "Landé g_J per J-center ID: {'1': 1.333, ...}. "
            "For Ln(III) copy the value from list_lanthanide_ions (Dy(III)=4/3, Gd(III)=2). "
            "The electron g 2.0023 is wrong for a lanthanide multiplet."
        )
    )

    # --- Exchange Coupling (J-type centers) ---
    J_ex_J: typing.Dict[typing.Annotated[str, pydantic.Field(pattern=r"^\d+-\d+$")], float] = pydantic.Field(
        default_factory=dict,
        description=(
            "Heisenberg exchange H = J Ĵ_i·Ĵ_j (cm⁻¹), keys '{i}-{j}' e.g. {'1-2': 0.3}. "
            "J > 0 is antiferromagnetic."
        )
    )

    # --- Hyperfine Coupling (I-type centers) ---
    A_hf: typing.Dict[typing.Annotated[str, pydantic.Field(pattern=r"^(L-|J-)?e\d+-n\d+$")], float] = pydantic.Field(
        default_factory=dict,
        description="Hyperfine coupling constants: {'e1-n3': 150, 'L-e1-n3': 20, 'J-e2-n3': 5, ...} (MHz)"
    )

    # --- Crystal pydantic.Field Parameters ---
    sigma_CF: typing.Dict[typing.Annotated[str, pydantic.Field(pattern=r"^\d+_[246]$")], float] = pydantic.Field(
        default_factory=dict,
        description=(
            "CF scale σ_k on B_k^q per electron & rank k: {'1_2': 1.0, '1_4': 0.8, '1_6': 0.5, ...}. "
            "Default 1. Not the chemist σ of Ŝ·L̂ (that is lambda_sigma_SL)."
        )
    )
    B_kq: typing.Dict[typing.Annotated[str, pydantic.Field(pattern=r"^\d+_[246]_[-+]?\d+$")], float] = pydantic.Field(
        default_factory=dict,
        description=(
            "Stevens CF parameters B_k^q per electron, rank k, and projection q: "
            "{'1_2_-2': 0.1, '1_2_0': 0.5, '1_4_4': -0.03, ...} (cm⁻¹). "
            "There is no field named Δ; axial crystal-field Δ is usually B_kq['1_2_0']."
        )
    )

    # --- Computation Flags ---
    computeE: bool = pydantic.Field(default=True, description="Compute energy levels")
    computeM: bool = pydantic.Field(default=True, description="Compute magnetization")
    computeChi: bool = pydantic.Field(default=True, description="Compute isotropic susceptibility χ")
    computeChiAx: bool = pydantic.Field(default=True, description="Compute axial anisotropy Δχ_ax")
    computeChiRh: bool = pydantic.Field(default=True, description="Compute rhombic anisotropy Δχ_rh")
    computeChiT: bool = pydantic.Field(default=True, description="Compute χ·T product")

    # ========================================
    # Validators (V2 style)
    # ========================================

    @pydantic.field_validator("gAniso", "gJAniso")
    @classmethod
    def validate_gtensor_components(cls, v, info: pydantic.ValidationInfo):
        for key, tensor in v.items():
            if len(tensor) != 3:
                raise ValueError(
                    f"g-tensor for center {key} must have exactly 3 components, got {len(tensor)}"
                )
            if not all(isinstance(x, (int, float)) for x in tensor):
                raise ValueError(f"g-tensor components must be numeric: {tensor}")
        return v

    @pydantic.model_validator(mode='after')
    def validate_calculation_ranges(self):
        if self.Tmin >= self.Tmax:
            raise ValueError("Tmin must be less than Tmax")
        if self.Bmin >= self.Bmax:
            raise ValueError("Bmin must be less than Bmax")
        return self

    @pydantic.model_validator(mode='after')
    def validate_mode_consistency(self):
        mode = self.calculationMode
        if mode == "fixedB":
            if self.Tmin is None or self.Tmax is None:
                raise ValueError("Tmin and Tmax required for calculationMode='fixedB'")
        elif mode == "fixedT":
            if self.Bmin is None or self.Bmax is None:
                raise ValueError("Bmin and Bmax required for calculationMode='fixedT'")
        return self

class ParameterFixedState(pydantic.RootModel[typing.Dict[str, bool]]):
    """
    Tracks which Hamiltonian parameters are fixed (True) vs. optimized (False).
    
    Convention:
    - True (checked)  = Parameter is FIXED (not included in optimization)
    - False (unchecked) = Parameter is OPTIMIZABLE (included in fitting)
    
    Keys follow dynamic patterns generated by the frontend:
    - g_<id>, gAniso_<id>_<0|1|2>, D_<id>, E_<id>          (S-type g/ZFS)
    - sigma_L_<id>, lambda_SL_<id>, lambda_sigma_SL_<id>   (L-type orbital)
    - g_J_<id>, gJAniso_<id>_<0|1|2>                        (J-type g)
    - J_ex_<pair>, J_ex_J_<pair>                            (exchange, pair="1-2")
    - A_hf_SI_<pair>, A_hf_LI_<pair>, A_hf_JI_<pair>       (hyperfine)
    - sigma_CF_<id>_<k>, B_kq_<id>_<k>_<q>                  (CF for L)
    - sigma_CF_J_<id>_<k>, B_kq_J_<id>_<k>_<q>              (CF for J)
    - tip_correction, diamagnetic_correction                (corrections)
    
    Example JSON input:
    {
        "g_1": true,
        "gAniso_1_0": true,
        "gAniso_1_1": true,
        "gAniso_1_2": true,
        "D_1": true,
        "E_1": true,
        "J_ex_1-2": false,
        "A_hf_SI_e1-n2": true,
        "tip_correction": true,
        "diamagnetic_correction": true
    }
    """
    
    root: typing.Dict[str, bool] = pydantic.Field(
        default_factory=dict,
        description="Flat dictionary: parameter_key -> fixed_state (True=fixed, False=optimize)"
    )
    
    model_config = pydantic.ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "g_1": True,
                "gAniso_1_0": True,
                "gAniso_1_1": True,
                "gAniso_1_2": True,
                "D_1": True,
                "E_1": True,
                "sigma_L_1": True,
                "lambda_SL_1": True,
                "lambda_sigma_SL_1": True,
                "g_J_1": True,
                "gJAniso_1_0": True,
                "J_ex_1-2": False,
                "A_hf_SI_e1-n2": True,
                "A_hf_LI_e1-n2": True,
                "A_hf_JI_e1-n2": True,
                "sigma_CF_1_2": True,
                "B_kq_1_2_0": False,
                "sigma_CF_J_1_4": True,
                "B_kq_J_1_4_2": False,
                "tip_correction": True,
                "diamagnetic_correction": True
            }
        }
    )
    
    # Dict-like access methods for convenience
    def __getitem__(self, key: str) -> bool:
        return self.root[key]
    
    def __setitem__(self, key: str, value: bool) -> None:
        self.root[key] = value
    
    def get(self, key: str, default: bool = True) -> bool:
        """Get fixed state for parameter key; defaults to True (fixed) if missing."""
        return self.root.get(key, default)
    
    def is_fixed(self, key: str) -> bool:
        """Convenience: returns True if parameter is fixed (should not be optimized)."""
        return self.root.get(key, True)
    
    def is_optimizable(self, key: str) -> bool:
        """Convenience: returns True if parameter should be optimized."""
        return not self.root.get(key, True)
    
    # Make iteration work naturally
    def __iter__(self):
        return iter(self.root)
    
    def keys(self):
        return self.root.keys()
    
    def items(self):
        return self.root.items()
    
    def values(self):
        return self.root.values()

class BaseRequest(pydantic.BaseModel):
    spin_systems : AllSpinSystems
    hamiltonian_params : HamiltonianParameters
    parameter_fixed_state: ParameterFixedState


class OptimizationParamsRequest(pydantic.BaseModel):
    base_request : BaseRequest
    maxiter : typing.Optional[int] = pydantic.Field(default=100)
    method : typing.Optional[typing.Literal['Nelder-Mead']] = pydantic.Field(default='Nelder-Mead')
