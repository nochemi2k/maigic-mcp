import numpy as np
import typing

from functools import reduce

mu_B = 9.274e-24
mu_N = 5.0508e-27
k = 1.3806e-23
n_a = 6.02214e23
h = 6.62607e-34
c = 299792458 

inv_cm_to_j = 100 * h * c
mhz_to_j = 1e6 * h

cf_params = {
    'Ce(III)' : {
        'L' : [-2/45, 2/495, -4/3861],
        'J' : [-2/35, 2/315, 0]
    },
    'Pr(III)' : {
        'L' : [-2/135, -4/10395, 2/81081],
        'J' : [-52/2475, -4/5445, 272/4459455]
    },
    'Nd(III)' : {
        'L' : [-2/495, -2/16335, -10/891891],
        'J' : [-7/1089, -136/467181, -1615/42513471]
    },
    'Pm(III)' : {
        'L' : [2/495, 2/16335, 10/891891],
        'J' : [14/1815, 952/2335905, 2584/42513471]
    },
    'Sm(III)' : {
        'L' : [2/135, 4/10395, -2/81081],
        'J' : [13/315, 26/10395, 0]
    },
    'Eu(III)' : {
        'L' : [2/45, -2/495, 4/3861],
        'J' : [0, 0, 0]
    },
    'Gd(III)' : {
        'L' : [0, 0, 0],
        'J' : [0, 0, 0]
    },
    'Tb(III)' : {
        'L' : [-2/45, 2/495, -4/3861],
        'J' : [-1/99, 2/16335, -1/891891]
    },
    'Dy(III)' : {
        'L' : [-2/135, -4/10395, 2/81081],
        'J' : [-2/315, -8/135135, 4/3864861]
    },
    'Ho(III)' : {
        'L' : [-2/495, -2/16335, -10/891891],
        'J' : [-1/450, -1/30030, -5/3864861]
    },
    'Er(III)' : {
        'L' : [2/495, 2/16335, 10/891891],
        'J' : [4/1575, 2/45045, 8/3864861]
    },
    'Tm(III)' : {
        'L' : [2/135, 4/10395, -2/81081],
        'J' : [1/99, 8/49005, -5/891891]
    },
    'Yb(III)' : {
        'L' : [2/45, -2/495, 4/3861],
        'J' : [2/63, -2/1155, 4/27027]
    },
}

def get_stevens_operator(
    operator: np.ndarray,
    k: int,
    q: int
) -> np.ndarray:
    plus_operator = operator[:, :, 0] + 1j * operator[:, :, 1]
    minus_operator = operator[:, :, 0] - 1j * operator[:, :, 1]
    z_operator = operator[:, :, 2]
    operator_2 = (operator[:, :, 0] @ operator[:, :, 0] +
                  operator[:, :, 1] @ operator[:, :, 1] +
                  operator[:, :, 2] @ operator[:, :, 2])

    if k == 2 and q == -2:
        return (-1j/2) * (plus_operator @ plus_operator - minus_operator @ minus_operator)
    if k == 2 and q == -1:
        return (-1j/4) * (z_operator @ (plus_operator - minus_operator) + (plus_operator - minus_operator) @ z_operator)
    if k == 2 and q == 0:
        return 3 * z_operator @ z_operator - operator_2
    if k == 2 and q == 1:
        return (1/4) * (z_operator @ (plus_operator + minus_operator) + (plus_operator + minus_operator) @ z_operator)
    if k == 2 and q == 2:
        return (1/2) * (plus_operator @ plus_operator + minus_operator @ minus_operator)
    
    if k == 4 and q == -4:
        return (-1j/2) * (np.linalg.matrix_power(plus_operator, 4) - np.linalg.matrix_power(minus_operator, 4))
    if k == 4 and q == -3:
        term = np.linalg.matrix_power(plus_operator, 3) - np.linalg.matrix_power(minus_operator, 3)
        return (-1j/4) * (z_operator @ term + term @ z_operator)
    if k == 4 and q == -2:
        pp_mm = plus_operator @ plus_operator - minus_operator @ minus_operator
        coeff = 7 * z_operator @ z_operator - operator_2 - 5
        return (-1j/4) * (coeff @ pp_mm + pp_mm @ coeff)
    if k == 4 and q == -1:
        diff = plus_operator - minus_operator
        poly_z = (7 * np.linalg.matrix_power(z_operator, 3) -
                  3 * operator_2 @ z_operator - z_operator)
        return (-1j/4) * (poly_z @ diff + diff @ poly_z)
    if k == 4 and q == 0:
        return (35 * np.linalg.matrix_power(z_operator, 4) -
                30 * operator_2 @ np.linalg.matrix_power(z_operator, 2) +
                25 * z_operator @ z_operator +
                3 * operator_2 @ operator_2 - 6 * operator_2)
    if k == 4 and q == 1:
        sum_pm = plus_operator + minus_operator
        poly_z = (7 * np.linalg.matrix_power(z_operator, 3) -
                  3 * operator_2 @ z_operator - z_operator)
        return (1/4) * (poly_z @ sum_pm + sum_pm @ poly_z)
    if k == 4 and q == 2:
        pp_pp = plus_operator @ plus_operator + minus_operator @ minus_operator
        coeff = 7 * z_operator @ z_operator - operator_2 - 5
        return (1/4) * (coeff @ pp_pp + pp_pp @ coeff)
    if k == 4 and q == 3:
        term = np.linalg.matrix_power(plus_operator, 3) + np.linalg.matrix_power(minus_operator, 3)
        return (1/4) * (z_operator @ term + term @ z_operator)
    if k == 4 and q == 4:
        return (1/2) * (np.linalg.matrix_power(plus_operator, 4) + np.linalg.matrix_power(minus_operator, 4))

    if k == 6 and q == -6:
        return (-1j/2) * (np.linalg.matrix_power(plus_operator, 6) - np.linalg.matrix_power(minus_operator, 6))
    if k == 6 and q == -5:
        term = np.linalg.matrix_power(plus_operator, 5) - np.linalg.matrix_power(minus_operator, 5)
        return (-1j/4) * (z_operator @ term + term @ z_operator)
    if k == 6 and q == -4:
        term = np.linalg.matrix_power(plus_operator, 4) - np.linalg.matrix_power(minus_operator, 4)
        coeff = 11 * z_operator @ z_operator - operator_2 - 38
        return (-1j/4) * (term @ coeff + coeff @ term)
    if k == 6 and q == -3:
        term = np.linalg.matrix_power(plus_operator, 3) - np.linalg.matrix_power(minus_operator, 3)
        poly_z = (11 * np.linalg.matrix_power(z_operator, 3) -
                  3 * operator_2 @ z_operator - 59 * z_operator)
        return (-1j/4) * (term @ poly_z + poly_z @ term)
    if k == 6 and q == -2:
        diff_sq = plus_operator @ plus_operator - minus_operator @ minus_operator
        poly_z = (33 * np.linalg.matrix_power(z_operator, 4) -
                  18 * operator_2 @ np.linalg.matrix_power(z_operator, 2) -
                  123 * z_operator @ z_operator +
                  operator_2 @ operator_2 + 10 * operator_2 + 102)
        return (-1j/4) * (diff_sq @ poly_z + poly_z @ diff_sq)
    if k == 6 and q == -1:
        diff = plus_operator - minus_operator
        poly_z = (33 * np.linalg.matrix_power(z_operator, 5) -
                  30 * operator_2 @ np.linalg.matrix_power(z_operator, 3) +
                  15 * np.linalg.matrix_power(z_operator, 3) +
                  5 * operator_2 @ operator_2 @ z_operator -
                  10 * operator_2 @ z_operator + 12 * z_operator)
        return (-1j/4) * (diff @ poly_z + poly_z @ diff)
    if k == 6 and q == 0:
        return (231 * np.linalg.matrix_power(z_operator, 6) -
                315 * operator_2 @ np.linalg.matrix_power(z_operator, 4) +
                735 * np.linalg.matrix_power(z_operator, 4) +
                105 * operator_2 @ operator_2 @ z_operator @ z_operator -
                525 * operator_2 @ z_operator @ z_operator +
                294 * z_operator @ z_operator -
                5 * np.linalg.matrix_power(operator_2, 3) +
                40 * operator_2 @ operator_2 - 60 * operator_2)
    if k == 6 and q == 1:
        sum_pm = plus_operator + minus_operator
        poly_z = (33 * np.linalg.matrix_power(z_operator, 5) -
                  30 * operator_2 @ np.linalg.matrix_power(z_operator, 3) +
                  15 * np.linalg.matrix_power(z_operator, 3) +
                  5 * operator_2 @ operator_2 @ z_operator -
                  10 * operator_2 @ z_operator + 12 * z_operator)
        return (1/4) * (sum_pm @ poly_z + poly_z @ sum_pm)
    if k == 6 and q == 2:
        sum_sq = plus_operator @ plus_operator + minus_operator @ minus_operator
        poly_z = (33 * np.linalg.matrix_power(z_operator, 4) -
                  18 * operator_2 @ np.linalg.matrix_power(z_operator, 2) -
                  123 * z_operator @ z_operator +
                  operator_2 @ operator_2 + 10 * operator_2 + 102)
        return (1/4) * (sum_sq @ poly_z + poly_z @ sum_sq)
    if k == 6 and q == 3:
        term = np.linalg.matrix_power(plus_operator, 3) + np.linalg.matrix_power(minus_operator, 3)
        poly_z = (11 * np.linalg.matrix_power(z_operator, 3) -
                  3 * operator_2 @ z_operator - 59 * z_operator)
        return (1/4) * (term @ poly_z + poly_z @ term)
    if k == 6 and q == 4:
        term = np.linalg.matrix_power(plus_operator, 4) + np.linalg.matrix_power(minus_operator, 4)
        coeff = 11 * z_operator @ z_operator - operator_2 - 38
        return (1/4) * (term @ coeff + coeff @ term)
    if k == 6 and q == 5:
        term = np.linalg.matrix_power(plus_operator, 5) + np.linalg.matrix_power(minus_operator, 5)
        return (1/4) * (z_operator @ term + term @ z_operator)
    if k == 6 and q == 6:
        return (1/2) * (np.linalg.matrix_power(plus_operator, 6) + np.linalg.matrix_power(minus_operator, 6))

    raise ValueError(f"Stevens operator not defined for k={k}, q={q}")

def pauli_matrices_for_spin(s):
    """
    Compute the spin matrices Sx, Sy, Sz for a given spin quantum number s,
    using the standard basis ordering: |s, m=s>, |s, m=s-1>, ..., |s, m=-s>.
    
    Parameters:
    s (float): Spin quantum number (e.g., 0.5, 1, 1.5, ...)
    
    Returns:
    tuple: (Sx, Sy, Sz) as NumPy arrays (with ħ = 1)
    """
    dim = int(2 * s + 1)
    
    # Standard basis: m = s, s-1, ..., -s
    m_values = np.arange(s, -s - 1, -1)  # e.g., for s=0.5: [0.5, -0.5]
    Sz = np.diag(m_values)
    
    # Build S+ (raising operator): <m| S+ |m-1> = sqrt(s(s+1) - m(m-1))
    # But note: in our basis, index 0 = m=s, index 1 = m=s-1, ..., index dim-1 = m=-s
    S_plus = np.zeros((dim, dim), dtype=complex)
    for i in range(dim - 1):
        # Current row i corresponds to m = m_values[i]
        # S+ connects |m-1> (col) to |m> (row), so col = i+1, row = i
        m = m_values[i]  # m after raising
        # The element S_plus[i, i+1] = <m| S+ |m-1> = sqrt(s(s+1) - (m-1)*m)
        # But note: m_prev = m - 1 = m_values[i+1]
        # So: coeff = sqrt(s(s+1) - m_prev*(m_prev + 1)) = sqrt(s(s+1) - m_values[i+1]*(m_values[i+1] + 1))
        m_prev = m_values[i + 1]  # this is m - 1
        S_plus[i, i + 1] = np.sqrt(s * (s + 1) - m_prev * (m_prev + 1))
    
    S_minus = S_plus.conj().T  # Hermitian conjugate
    
    Sx = (S_plus + S_minus) / 2.0
    Sy = (S_plus - S_minus) / (2j)
    
    return np.stack([Sx, Sy, Sz], axis=-1)

def get_operators(
    spin_system : typing.List[typing.Dict[str, typing.Any]]
) -> typing.List[np.ndarray]:
    N = len(spin_system)
    operators = []
    for idx, spin in enumerate(spin_system):
        first_eye_matricies = [
            np.eye(int(round(spin_system[j]['value']*2 + 1)), dtype=complex)
            for j in range(idx)
        ]
        second_eye_matricies = [
            np.eye(int(round(spin_system[j]['value']*2 + 1)), dtype=complex)
            for j in range(idx+1, N)
        ]
        pauli_matricies = pauli_matrices_for_spin(spin['value'])
        operators.append(
            np.stack(
                [
                    reduce(
                        np.kron,
                        first_eye_matricies + [pauli_matricies[:, :, coord]] + second_eye_matricies
                    ) for coord in range(3)
                ], axis=-1
            )
        )
    return operators

def convert_to_inner_spin_systems(
    js_spin_systems : typing.List[typing.Dict[str, typing.Any]]
) -> typing.Tuple[typing.List[typing.Dict[str, typing.Any]], typing.Dict[str, typing.Any]]:

    # Порядок как пришел. Если есть и S и L, то сначала S потом L, подряд

    result = []
    inner_idxs = {}
    for cur in js_spin_systems:
        if cur['type'] == 'S':
            inner_idxs[cur['id']] = {'S' : len(result)}
            result.append({'type' : 'S', 'value' : cur['S']})
            if cur['L'] > 0:
                inner_idxs[cur['id']]['L'] = len(result)
                result.append({'type' : 'L', 'value' : cur['L']})
        elif cur['type'] == 'J':
            inner_idxs[cur['id']] = {'J' : len(result)}
            result.append({'type' : 'J', 'value' : cur['J']})
        elif cur['type'] == 'I':
            inner_idxs[cur['id']] = {'I' : len(result)}
            result.append({'type' : 'I', 'value' : cur['I']})
        else:
            raise ValueError(f'Unknown spin type in {cur} - {cur["type"]}')

    return result, inner_idxs

def R_z(
    theta : float
) -> np.ndarray:
    theta *= np.pi / 180
    return np.asarray(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )

def R_y(
    theta : float
) -> np.ndarray:
    theta *= np.pi / 180
    return np.asarray(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )

def h_s(
    spin_systems : typing.List[typing.Dict[str, typing.Any]],
    hamiltonian_params : typing.Dict[str, typing.Any],
    B_value : float,
    B_axis : typing.Literal['x', 'y', 'z'] = 'z',
    B_bias : float = 0.
):
    def e_b(
        B_x : float,
        B_y : float,
        B_z : float,
        g_i : np.ndarray,
        S : np.ndarray
    ) -> np.ndarray:
        return mu_B * B_x * g_i[0] * S[:, :, 0] + mu_B * B_y * g_i[1] * S[:, :, 1] + mu_B * B_z * g_i[2] * S[:, :, 2]
    
    def s_s(
        D : float,
        E : float,
        S : np.ndarray
    ) -> np.ndarray:
        D_xx = E - (D / 3)
        D_yy = -E - (D / 3)
        D_zz = 2 * D / 3
        return inv_cm_to_j * D_xx * S[:, :, 0] @ S[:, :, 0] +\
                inv_cm_to_j * D_yy * S[:, :, 1] @ S[:, :, 1] +\
                inv_cm_to_j * D_zz * S[:, :, 2] @ S[:, :, 2] 

    def e_e(
        J_ij : float,
        S_i : np.ndarray,
        S_j : np.ndarray,
    ) -> np.ndarray:
        return inv_cm_to_j * J_ij * S_i[:, :, 0] @ S_j[:, :, 0] +\
                inv_cm_to_j * J_ij * S_i[:, :, 1] @ S_j[:, :, 1] +\
                inv_cm_to_j * J_ij * S_i[:, :, 2] @ S_j[:, :, 2]

    inner_spin_systems, inner_idxs = convert_to_inner_spin_systems(spin_systems)
    operators = get_operators(inner_spin_systems)
    
    hamiltonian = np.zeros((operators[0].shape[0], operators[0].shape[0]), dtype=complex)

    B_x, B_y, B_z = 0., 0., 0.

    if B_axis == 'x':
        B_x = B_value + B_bias
    elif B_axis == 'y':
        B_y = B_value + B_bias
    elif B_axis == 'z':
        B_z = B_value + B_bias
    else:
        raise ValueError(f'Unknown axis {B_axis}')
    
    for idx in hamiltonian_params['g'].keys():
        
        if hamiltonian_params['gType'][idx] == 'anisotropic':
            g_i = np.asarray(hamiltonian_params['gAniso'][idx])
        else:
            g_i = np.asarray([hamiltonian_params['g'][idx]] * 3)

        hamiltonian += e_b(
            B_x=B_x,
            B_y=B_y,
            B_z=B_z,
            g_i=g_i,
            S=operators[inner_idxs[int(idx)]['S']]
        )

    for idx in hamiltonian_params['D'].keys():
        hamiltonian += s_s(
            D=hamiltonian_params['D'][idx],
            E=hamiltonian_params['E'][idx],
            S=operators[inner_idxs[int(idx)]['S']]
        )

    for idx_pair in hamiltonian_params['J_ex'].keys():
        i, j = idx_pair.split('-')
        hamiltonian += e_e(
            J_ij=hamiltonian_params['J_ex'][idx_pair],
            S_i=operators[inner_idxs[int(i)]['S']],
            S_j=operators[inner_idxs[int(j)]['S']],
        )
        
    return hamiltonian

def h_l(
    spin_systems : typing.List[typing.Dict[str, typing.Any]],
    hamiltonian_params : typing.Dict[str, typing.Any],
    B_value : float,
    B_axis : typing.Literal['x', 'y', 'z'] = 'z',
    B_bias : float = 0.
):
    def e_b(
        B_x : float,
        B_y : float,
        B_z : float,
        sigma_i : float,
        L : np.ndarray
    ) -> np.ndarray:
        return mu_B * B_x * sigma_i * L[:, :, 0] + mu_B * B_y * sigma_i * L[:, :, 1] + mu_B * B_z * sigma_i * L[:, :, 2]

    def s_l(
        lambda_ij : float,
        sigma_ij : float,
        S : np.ndarray,
        L : np.ndarray
    ) -> np.ndarray:
        return inv_cm_to_j * lambda_ij * sigma_ij * S[:, :, 0] @ L[:, :, 0] +\
                inv_cm_to_j * lambda_ij * sigma_ij * S[:, :, 1] @ L[:, :, 1] +\
                inv_cm_to_j * lambda_ij * sigma_ij * S[:, :, 2] @ L[:, :, 2]

    def cf_term(
        L : np.ndarray,
        k : int,
        q : int, 
        B_k_q : float,
        sigma_k : float,
        theta_k : float
    ) -> np.ndarray:
        return inv_cm_to_j * B_k_q * sigma_k * theta_k * get_stevens_operator(L, k=k, q=q)

    inner_spin_systems, inner_idxs = convert_to_inner_spin_systems(spin_systems)
    operators = get_operators(inner_spin_systems)
    
    hamiltonian = np.zeros((operators[0].shape[0], operators[0].shape[0]), dtype=complex)

    B_x, B_y, B_z = 0., 0., 0.

    if B_axis == 'x':
        B_x = B_value + B_bias
    elif B_axis == 'y':
        B_y = B_value + B_bias
    elif B_axis == 'z':
        B_z = B_value + B_bias
    else:
        raise ValueError(f'Unknown axis {B_axis}')
    
    for idx in hamiltonian_params['sigma_L'].keys():
        hamiltonian += e_b(
            B_x=B_x,
            B_y=B_y,
            B_z=B_z,
            sigma_i=hamiltonian_params['sigma_L'][idx],
            L=operators[inner_idxs[int(idx)]['L']]
        )

    for idx in hamiltonian_params['lambda_SL'].keys():
        hamiltonian += s_l(
            lambda_ij=hamiltonian_params['lambda_SL'][idx],
            sigma_ij=hamiltonian_params['lambda_sigma_SL'][idx],
            L=operators[inner_idxs[int(idx)]['L']],
            S=operators[inner_idxs[int(idx)]['S']],
        )

    idx_to_origin_ion = {
        cur['id'] : cur['originIon'] for cur in spin_systems
        if cur['type'] == 'S'
    }

    for term, B_value in hamiltonian_params['B_kq'].items():
        idx, k, q = map(int, term.split('_'))

        if ('L' not in inner_idxs[idx]) or (inner_idxs[idx]['L'] == 0):
            continue
        
        theta_k = 1.
        if idx_to_origin_ion[idx]:
            theta_k = cf_params[idx_to_origin_ion[idx]]['L'][(k - 2) // 2]
        hamiltonian += cf_term(
            L=operators[inner_idxs[idx]['L']],
            k=k,
            q=q,
            B_k_q=B_value,
            sigma_k=hamiltonian_params['sigma_CF'][f"{idx}_{k}"],
            theta_k=theta_k
        )
        
    return hamiltonian

def h_j(
    spin_systems : typing.List[typing.Dict[str, typing.Any]],
    hamiltonian_params : typing.Dict[str, typing.Any],
    B_value : float,
    B_axis : typing.Literal['x', 'y', 'z'] = 'z',
    B_bias : float = 0.
):
    def e_b(
        B_x : float,
        B_y : float,
        B_z : float,
        g_i : np.ndarray,
        J : np.ndarray
    ) -> np.ndarray:
        return mu_B * B_x * g_i[0] * J[:, :, 0] + mu_B * B_y * g_i[1] * J[:, :, 1] + mu_B * B_z * g_i[2] * J[:, :, 2]

    def e_e(
        J_ij : float,
        J_i : np.ndarray,
        J_j : np.ndarray,
    ) -> np.ndarray:
        return inv_cm_to_j * J_ij * J_i[:, :, 0] @ J_j[:, :, 0] +\
                 inv_cm_to_j * J_ij * J_i[:, :, 1] @ J_j[:, :, 1] +\
                 inv_cm_to_j * J_ij * J_i[:, :, 2] @ J_j[:, :, 2]

    def cf_term(
        J : np.ndarray,
        k : int,
        q : int, 
        B_k_q : float,
        sigma_k : float,
        theta_k : float
    ) -> np.ndarray:
        return inv_cm_to_j * B_k_q * sigma_k * theta_k * get_stevens_operator(J, k=k, q=q)

    inner_spin_systems, inner_idxs = convert_to_inner_spin_systems(spin_systems)
    operators = get_operators(inner_spin_systems)
    
    hamiltonian = np.zeros((operators[0].shape[0], operators[0].shape[0]), dtype=complex)

    B_x, B_y, B_z = 0., 0., 0.

    if B_axis == 'x':
        B_x = B_value + B_bias
    elif B_axis == 'y':
        B_y = B_value + B_bias
    elif B_axis == 'z':
        B_z = B_value + B_bias
    else:
        raise ValueError(f'Unknown axis {B_axis}')
    
    for idx in hamiltonian_params['g_J'].keys():
        if hamiltonian_params['gJType'][idx] == 'anisotropic':
            g_i = np.asarray(hamiltonian_params['gJAniso'][idx])
        else:
            g_i = np.asarray([hamiltonian_params['g_J'][idx]] * 3)
        hamiltonian += e_b(
            B_x=B_x,
            B_y=B_y,
            B_z=B_z,
            g_i=g_i,
            J=operators[inner_idxs[int(idx)]['J']]
        )

    for idx_pair in hamiltonian_params['J_ex_J'].keys():
        i, j = idx_pair.split('-')
        hamiltonian += e_e(
            J_ij=hamiltonian_params['J_ex_J'][idx_pair],
            J_i=operators[inner_idxs[int(i)]['J']],
            J_j=operators[inner_idxs[int(j)]['J']],
        )

    idx_to_origin_ion = {
        cur['id'] : cur['originIon'] for cur in spin_systems
        if cur['type'] == 'J'
    }

    for term, B_value in hamiltonian_params['B_kq'].items():
        idx, k, q = map(int, term.split('_'))

        if ('J' not in inner_idxs[idx]) or (inner_idxs[idx]['J'] == 0):
            continue
        
        theta_k = 1.
        if idx_to_origin_ion[idx]:
            theta_k = cf_params[idx_to_origin_ion[idx]]['J'][(k - 2) // 2]
        hamiltonian += cf_term(
            J=operators[inner_idxs[idx]['J']],
            k=k,
            q=q,
            B_k_q=B_value,
            sigma_k=hamiltonian_params['sigma_CF'][f"{idx}_{k}"],
            theta_k=theta_k
        )

    return hamiltonian

def h_i(
    spin_systems : typing.List[typing.Dict[str, typing.Any]],
    hamiltonian_params : typing.Dict[str, typing.Any],
    B_value : float,
    B_axis : typing.Literal['x', 'y', 'z'] = 'z',
    B_bias : float = 0.
):
    def n_b(
        B : np.ndarray,
        gamma_i : float,
        sigma_i : np.ndarray,
        I : np.ndarray
    ) -> np.ndarray:
        sigma_B = sigma_i @ B
        g_i = gamma_i * h / (2 * mu_N * np.pi)
        return mu_N * g_i * sigma_B[0] * I[:, :, 0] + mu_N * g_i * sigma_B[1] * I[:, :, 1] + mu_N * g_i * sigma_B[2] * I[:, :, 2]

    def e_n(
        A_ij : float,
        I : np.ndarray,
        electron_operator : np.ndarray,
    ) -> np.ndarray:
        return mhz_to_j * A_ij * I[:, :, 0] @ electron_operator[:, :, 0] +\
                 mhz_to_j * A_ij * I[:, :, 1] @ electron_operator[:, :, 1] +\
                 mhz_to_j * A_ij * I[:, :, 2] @ electron_operator[:, :, 2]

        
    inner_spin_systems, inner_idxs = convert_to_inner_spin_systems(spin_systems)
    operators = get_operators(inner_spin_systems)
    
    hamiltonian = np.zeros((operators[0].shape[0], operators[0].shape[0]), dtype=complex)

    B_x, B_y, B_z = 0., 0., 0.

    if B_axis == 'x':
        B_x = B_value + B_bias
    elif B_axis == 'y':
        B_y = B_value + B_bias
    elif B_axis == 'z':
        B_z = B_value + B_bias
    else:
        raise ValueError(f'Unknown axis {B_axis}')

    nuclei_info = {
        cur['id'] : {
            'gamma' : cur['gamma'],
            'shieldingType' : cur['shieldingType'],
            'shieldingScalar' : cur['shieldingScalar'],
            'shieldingDiagonal' : cur['shieldingDiagonal'],
            'shieldingEulerAngles' : cur['shieldingEulerAngles'],
        } for cur in spin_systems if cur['type'] == 'I'
    }
    for idx, values in nuclei_info.items():
        shielding_tensor = np.eye(3)
        if values['shieldingType'] == 'Scalar':
            shielding_tensor *= values['shieldingScalar']
        if values['shieldingType'] == 'Tensor':
            shielding_tensor = np.diag(values['shieldingDiagonal'])
            rotation_matrix = R_z(values['shieldingEulerAngles'][0]) *\
                                R_y(values['shieldingEulerAngles'][1]) *\
                                R_z(values['shieldingEulerAngles'][2])
            shielding_tensor = rotation_matrix @ shielding_tensor @ rotation_matrix.T

        hamiltonian += n_b(
            B=np.asarray([[B_x], [B_y], [B_z]]),
            sigma_i=shielding_tensor,
            gamma_i=values['gamma'],
            I=operators[inner_idxs[idx]['I']],
        )

    for idx_pair in hamiltonian_params['A_hf'].keys():
        if idx_pair.count('-') == 2:
            operator_type = idx_pair[0]
            _, first_comp, second_comp = idx_pair.split('-')
            if first_comp[0] == 'n':
                first_comp, second_comp = second_comp, first_comp
            electron_idx = int(first_comp[1:])
            nuclei_idx = int(second_comp[1:])
        else:
            operator_type = 'S'
            first_comp, second_comp = idx_pair.split('-')
            if first_comp[0] == 'n':
                first_comp, second_comp = second_comp, first_comp
            electron_idx = int(first_comp[1:])
            nuclei_idx = int(second_comp[1:])
        hamiltonian += e_n(
            A_ij=hamiltonian_params['A_hf'][idx_pair],
            I=operators[inner_idxs[nuclei_idx]['I']],
            electron_operator=operators[inner_idxs[electron_idx][operator_type]]
        )
        
    return hamiltonian

def total_hamiltonian(
    spin_systems : typing.List[typing.Dict[str, typing.Any]],
    hamiltonian_params : typing.Dict[str, typing.Any],
    B_value : float,
    B_axis : typing.Literal['x', 'y', 'z'] = 'z',
    B_bias : float = 0.
) -> np.ndarray:
    s_term = h_s(
        spin_systems=spin_systems,
        hamiltonian_params=hamiltonian_params,
        B_value=B_value,
        B_axis=B_axis,
        B_bias=B_bias
    )
    l_term = h_l(
        spin_systems=spin_systems,
        hamiltonian_params=hamiltonian_params,
        B_value=B_value,
        B_axis=B_axis,
        B_bias=B_bias
    )
    j_term = h_j(
        spin_systems=spin_systems,
        hamiltonian_params=hamiltonian_params,
        B_value=B_value,
        B_axis=B_axis,
        B_bias=B_bias
    )
    i_term = h_i(
        spin_systems=spin_systems,
        hamiltonian_params=hamiltonian_params,
        B_value=B_value,
        B_axis=B_axis,
        B_bias=B_bias
    )

    return s_term + l_term + j_term + i_term

def get_energy_levels(
    spin_systems : typing.List[typing.Dict[str, typing.Any]],
    hamiltonian_params : typing.Dict[str, typing.Any],
    B_value : float,
    B_axis : typing.Literal['x', 'y', 'z'] = 'z',
    B_bias : float = 0.
) -> typing.List[float]:
    return np.linalg.eigh(
        total_hamiltonian(
            spin_systems=spin_systems,
            hamiltonian_params=hamiltonian_params,
            B_value=B_value,
            B_axis=B_axis,
            B_bias=B_bias
        )    
    ).eigenvalues.tolist()

def get_statsum(
    spin_systems : typing.List[typing.Dict[str, typing.Any]],
    hamiltonian_params : typing.Dict[str, typing.Any],
    B_value : float,
    T_value : float,
    B_axis : typing.Literal['x', 'y', 'z'] = 'z',
    B_bias : float = 0.
) -> float:
    energy_levels = get_energy_levels(
        spin_systems=spin_systems,
        hamiltonian_params=hamiltonian_params,
        B_value=B_value,
        B_axis=B_axis,
        B_bias=B_bias
    )
    
    return np.sum(np.exp(-np.asarray(energy_levels) / (T_value * k)))

def get_magnetization_vector(
    spin_systems : typing.List[typing.Dict[str, typing.Any]],
    hamiltonian_params : typing.Dict[str, typing.Any],
    B_value : float,
    T_value : float,
) -> typing.List[float]:
    bias = max(B_value*0.01, 0.001)
    
    Z_x_1 = get_statsum(
        spin_systems=spin_systems,
        hamiltonian_params=hamiltonian_params,
        B_value=B_value,
        T_value=T_value,
        B_axis='x',
        B_bias=-bias,
    )
    Z_x_2 = get_statsum(
        spin_systems=spin_systems,
        hamiltonian_params=hamiltonian_params,
        B_value=B_value,
        T_value=T_value,
        B_axis='x',
        B_bias=bias,
    )
    Z_y_1 = get_statsum(
        spin_systems=spin_systems,
        hamiltonian_params=hamiltonian_params,
        B_value=B_value,
        T_value=T_value,
        B_axis='y',
        B_bias=-bias,
    )
    Z_y_2 = get_statsum(
        spin_systems=spin_systems,
        hamiltonian_params=hamiltonian_params,
        B_value=B_value,
        T_value=T_value,
        B_axis='y',
        B_bias=bias,
    )
    Z_z_1 = get_statsum(
        spin_systems=spin_systems,
        hamiltonian_params=hamiltonian_params,
        B_value=B_value,
        T_value=T_value,
        B_axis='z',
        B_bias=-bias,
    )
    Z_z_2 = get_statsum(
        spin_systems=spin_systems,
        hamiltonian_params=hamiltonian_params,
        B_value=B_value,
        T_value=T_value,
        B_axis='z',
        B_bias=bias,
    )

    x_deriv = (np.log(Z_x_2) - np.log(Z_x_1)) / (2 * bias)
    y_deriv = (np.log(Z_y_2) - np.log(Z_y_1)) / (2 * bias)
    z_deriv = (np.log(Z_z_2) - np.log(Z_z_1)) / (2 * bias)

    scaling_factor = k * T_value / mu_B

    return [scaling_factor * x_deriv.item(), scaling_factor * y_deriv.item(), scaling_factor * z_deriv.item()]

def get_magnetization_tensor(
    spin_systems : typing.List[typing.Dict[str, typing.Any]],
    hamiltonian_params : typing.Dict[str, typing.Any],
    B_value : float,
    T_value : float,
) -> typing.List[float]:
    bias = max(B_value*0.01, 0.001)
    
    Z_x_1 = get_statsum(
        spin_systems=spin_systems,
        hamiltonian_params=hamiltonian_params,
        B_value=B_value,
        T_value=T_value,
        B_axis='x',
        B_bias=-bias,
    )
    Z_x_2 = get_statsum(
        spin_systems=spin_systems,
        hamiltonian_params=hamiltonian_params,
        B_value=B_value,
        T_value=T_value,
        B_axis='x',
        B_bias=0,
    )
    Z_x_3 = get_statsum(
        spin_systems=spin_systems,
        hamiltonian_params=hamiltonian_params,
        B_value=B_value,
        T_value=T_value,
        B_axis='x',
        B_bias=bias,
    )
    Z_y_1 = get_statsum(
        spin_systems=spin_systems,
        hamiltonian_params=hamiltonian_params,
        B_value=B_value,
        T_value=T_value,
        B_axis='y',
        B_bias=-bias,
    )
    Z_y_2 = get_statsum(
        spin_systems=spin_systems,
        hamiltonian_params=hamiltonian_params,
        B_value=B_value,
        T_value=T_value,
        B_axis='y',
        B_bias=0,
    )
    Z_y_3 = get_statsum(
        spin_systems=spin_systems,
        hamiltonian_params=hamiltonian_params,
        B_value=B_value,
        T_value=T_value,
        B_axis='y',
        B_bias=bias,
    )
    Z_z_1 = get_statsum(
        spin_systems=spin_systems,
        hamiltonian_params=hamiltonian_params,
        B_value=B_value,
        T_value=T_value,
        B_axis='z',
        B_bias=-bias,
    )
    Z_z_2 = get_statsum(
        spin_systems=spin_systems,
        hamiltonian_params=hamiltonian_params,
        B_value=B_value,
        T_value=T_value,
        B_axis='z',
        B_bias=0,
    )
    Z_z_3 = get_statsum(
        spin_systems=spin_systems,
        hamiltonian_params=hamiltonian_params,
        B_value=B_value,
        T_value=T_value,
        B_axis='z',
        B_bias=bias,
    )

    xx_deriv = (np.log(Z_x_3) - 2*np.log(Z_x_2) + np.log(Z_x_1)) / (bias**2)
    yy_deriv = (np.log(Z_y_3) - 2*np.log(Z_y_2) + np.log(Z_y_1)) / (bias**2)
    zz_deriv = (np.log(Z_z_3) - 2*np.log(Z_z_2) + np.log(Z_z_1)) / (bias**2)

    scaling_factor = n_a * k * T_value / 10

    return [scaling_factor * xx_deriv.item(), scaling_factor * yy_deriv.item(), scaling_factor * zz_deriv.item()]

def get_magnetic_susceptibility(
    spin_systems : typing.List[typing.Dict[str, typing.Any]],
    hamiltonian_params : typing.Dict[str, typing.Any],
    B_value : float,
    T_value : float,
) -> typing.Dict[str, float]:
    diag_values = get_magnetization_tensor(
        spin_systems=spin_systems,
        hamiltonian_params=hamiltonian_params,
        B_value=B_value,
        T_value=T_value
    )
    
    for dim in range(3):
        diag_values[dim] += hamiltonian_params['tip_correction']
        diag_values[dim] += hamiltonian_params['diamagnetic_correction']

    return {
        '\\chi' : (diag_values[0] + diag_values[1] + diag_values[2]) / 3,
        '\\Delta \\chi_{ax}' : 4 * np.pi * (diag_values[2] - 0.5 * (diag_values[0] + diag_values[1])) / (1e6 * n_a),
        '\\Delta \\chi_{rh}' : 4 * np.pi * (diag_values[0] - diag_values[1]) / (1e6 * n_a)
    }

def get_magnetization_vector_b_const(
    spin_systems : typing.List[typing.Dict[str, typing.Any]],
    hamiltonian_params : typing.Dict[str, typing.Any],
) -> typing.Dict[str, typing.List[float]]:
    T_range = np.linspace(
        hamiltonian_params['Tmin'],
        hamiltonian_params['Tmax'],
        hamiltonian_params['numPoints'],
    )
    M_vec_values = [
        get_magnetization_vector(
            spin_systems=spin_systems,
            hamiltonian_params=hamiltonian_params,
            B_value=hamiltonian_params['fixedB'],
            T_value=cur_temp
        ) for cur_temp in T_range
    ]
    return {
        'T' : T_range.tolist(),
        'M_x' : [cur[0].item() for cur in M_vec_values],
        'M_y' : [cur[1] for cur in M_vec_values],
        'M_z' : [cur[2] for cur in M_vec_values],
    }

def get_magnetization_vector_t_const(
    spin_systems : typing.List[typing.Dict[str, typing.Any]],
    hamiltonian_params : typing.Dict[str, typing.Any],
) -> typing.Dict[str, typing.List[float]]:
    B_range = np.linspace(
        hamiltonian_params['Bmin'],
        hamiltonian_params['Bmax'],
        hamiltonian_params['numPoints'],
    )
    M_vec_values = [
        get_magnetization_vector(
            spin_systems=spin_systems,
            hamiltonian_params=hamiltonian_params,
            B_value=cur_B,
            T_value=hamiltonian_params['fixedT']
        ) for cur_B in B_range
    ]
    return {
        'B' : B_range.tolist(),
        'M_x' : [cur[0] for cur in M_vec_values],
        'M_y' : [cur[1] for cur in M_vec_values],
        'M_z' : [cur[2] for cur in M_vec_values],
    }


def get_energy_levels_b_const(
    spin_systems : typing.List[typing.Dict[str, typing.Any]],
    hamiltonian_params : typing.Dict[str, typing.Any],
) -> typing.Dict[str, typing.List[float]]:
    return get_energy_levels(
        spin_systems=spin_systems,
        hamiltonian_params=hamiltonian_params,
        B_value=hamiltonian_params['fixedB'],
        B_axis='z'
    )

def get_energy_levels_t_const(
    spin_systems : typing.List[typing.Dict[str, typing.Any]],
    hamiltonian_params : typing.Dict[str, typing.Any],
) -> typing.Dict[str, typing.List[float]]:
    B_range = np.linspace(
        hamiltonian_params['Bmin'],
        hamiltonian_params['Bmax'],
        hamiltonian_params['numPoints'],
    )
    energy_levels = [
        get_energy_levels(
            spin_systems=spin_systems,
            hamiltonian_params=hamiltonian_params,
            B_value=cur_B,
            B_axis='z'
        ) for cur_B in B_range
    ]
    
    return {
        'B' : B_range.tolist(),
        'E' : energy_levels
    }

def get_magnetic_susceptibility_b_const(
    spin_systems : typing.List[typing.Dict[str, typing.Any]],
    hamiltonian_params : typing.Dict[str, typing.Any],
) -> typing.Dict[str, typing.List[float]]:
    T_range = np.linspace(
        hamiltonian_params['Tmin'],
        hamiltonian_params['Tmax'],
        hamiltonian_params['numPoints'],
    )
    mag_suscept_values = [
        get_magnetic_susceptibility(
            spin_systems=spin_systems,
            hamiltonian_params=hamiltonian_params,
            B_value=hamiltonian_params['fixedB'],
            T_value=cur_temp
        ) for cur_temp in T_range
    ]
    return {
        'T' : T_range.tolist(),
        '\\chi' : [cur['\\chi'] for cur in mag_suscept_values],
        '\\Delta \\chi_{ax}' : [cur['\\Delta \\chi_{ax}'] for cur in mag_suscept_values],
        '\\Delta \\chi_{rh}' : [cur['\\Delta \\chi_{rh}'] for cur in mag_suscept_values],
    }

def get_magnetic_susceptibility_t_const(
    spin_systems : typing.List[typing.Dict[str, typing.Any]],
    hamiltonian_params : typing.Dict[str, typing.Any],
) -> typing.Dict[str, typing.List[float]]:
    B_range = np.linspace(
        hamiltonian_params['Bmin'],
        hamiltonian_params['Bmax'],
        hamiltonian_params['numPoints'],
    )
    mag_suscept_values = [
        get_magnetic_susceptibility(
            spin_systems=spin_systems,
            hamiltonian_params=hamiltonian_params,
            B_value=cur_B,
            T_value=hamiltonian_params['fixedT']
        ) for cur_B in B_range
    ]
    return {
        'B' : B_range.tolist(),
        '\\chi' : [cur['\\chi'] for cur in mag_suscept_values],
        '\\Delta \\chi_{ax}' : [cur['\\Delta \\chi_{ax}'] for cur in mag_suscept_values],
        '\\Delta \\chi_{rh}' : [cur['\\Delta \\chi_{rh}'] for cur in mag_suscept_values],
    }
