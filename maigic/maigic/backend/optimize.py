import re
import typing
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from compute import (
    get_magnetization_vector,
    get_magnetic_susceptibility
)

NAME2CALLABLE = {
    'M_x' : get_magnetization_vector,
    'M_y' : get_magnetization_vector,
    'M_z' : get_magnetization_vector,
    'M' : get_magnetization_vector,
    'chi' : get_magnetic_susceptibility,
    'chi_t' : get_magnetic_susceptibility,
    'delta_chi_ax' : get_magnetic_susceptibility,
    'delta_chi_ax' : get_magnetic_susceptibility,
    'delta_chi_rh' : get_magnetic_susceptibility,
    'delta_chi_rh' : get_magnetic_susceptibility,
}

def get_value_by_key(
    key : str,
    hamiltonian_params : typing.Dict[str, typing.Any]
) -> typing.Any:
    if re.fullmatch(r'g_[\d]+', key):
        return hamiltonian_params['g'][key.split('_')[1]]
        
    if re.fullmatch(r'gAniso_[\d]+_0', key):
        return hamiltonian_params['gAniso'][key.split('_')[1]][0]
    if re.fullmatch(r'gAniso_[\d]+_1', key):
        return hamiltonian_params['gAniso'][key.split('_')[1]][1]
    if re.fullmatch(r'gAniso_[\d]+_2', key):
        return hamiltonian_params['gAniso'][key.split('_')[1]][2]

    if re.fullmatch(r'D_[\d]+', key):
        return hamiltonian_params['D'][key.split('_')[1]]
    if re.fullmatch(r'E_[\d]+', key):
        return hamiltonian_params['E'][key.split('_')[1]]

    if re.fullmatch(r'sigma_L_[\d]+', key):
        return hamiltonian_params['sigma_L'][key.split('_')[2]]
    if re.fullmatch(r'lambda_SL_[\d]+', key):
        return hamiltonian_params['lambda_SL'][key.split('_')[2]]
    if re.fullmatch(r'lambda_sigma_SL_[\d]+', key):
        return hamiltonian_params['lambda_sigma_SL'][key.split('_')[3]]

    if re.fullmatch(r'g_J_[\d]+', key):
        return hamiltonian_params['g_J'][key.split('_')[2]]
        
    if re.fullmatch(r'gJAniso_[\d]+_0', key):
        return hamiltonian_params['gJAniso'][key.split('_')[1]][0]
    if re.fullmatch(r'gJAniso_[\d]+_1', key):
        return hamiltonian_params['gJAniso'][key.split('_')[1]][1]
    if re.fullmatch(r'gJAniso_[\d]+_2', key):
        return hamiltonian_params['gJAniso'][key.split('_')[1]][2]

    if re.fullmatch(r'J_ex_J_[a-z]*[0-9]*\-[a-z]*[0-9]*', key):
        return hamiltonian_params['J_ex_J'][key.split('_')[-1]]
    if re.fullmatch(r'J_ex_[a-z]*[0-9]*\-[a-z]*[0-9]*', key):
        return hamiltonian_params['J_ex'][key.split('_')[-1]]

    if re.fullmatch(r'A_hf_SI_[a-z]*[0-9]*\-[a-z]*[0-9]*', key):
        return hamiltonian_params['A_hf'][key.split('_')[-1]]
    if re.fullmatch(r'A_hf_LI_[a-z]*[0-9]*\-[a-z]*[0-9]*', key):
        return hamiltonian_params['A_hf']["L-" + key.split('_')[-1]]
    if re.fullmatch(r'A_hf_JI_[a-z]*[0-9]*\-[a-z]*[0-9]*', key):
        return hamiltonian_params['A_hf']["J-" + key.split('_')[-1]]

    if re.fullmatch(r'sigma_CF_[0-9]*_[0-9]*', key):
        i, k = key.split('_')[-2:]
        return hamiltonian_params['sigma_CF'][i + "_" + k]
    if re.fullmatch(r'sigma_CF_J_[0-9]*_[0-9]*', key):
        i, k = key.split('_')[-2:]
        return hamiltonian_params['sigma_CF'][i + "_" + k]

    if re.fullmatch(r'B_kq_[0-9]*_[0-9]*_[0-9]*', key):
        i, k, q = key.split('_')[-3:]
        return hamiltonian_params['B_kq'][i + "_" + k + "_" + q]
    if re.fullmatch(r'B_kq_J_[0-9]*_[0-9]*_[0-9]*', key):
        i, k, q = key.split('_')[-3:]
        return hamiltonian_params['B_kq'][i + "_" + k + "_" + q]

    if key == 'tip_correction':
        return hamiltonian_params[key]
    if key == 'diamagnetic_correction':
        return hamiltonian_params[key]

    raise ValueError(f'Unknown key: {key}')

def set_value_by_key(
    key : str,
    value : float,
    hamiltonian_params : typing.Dict[str, typing.Any]
) -> typing.Dict[str, typing.Any]:

    res = deepcopy(hamiltonian_params)
    
    if re.fullmatch(r'g_[\d]+', key):
         res['g'][key.split('_')[1]] = value
        
    if re.fullmatch(r'gAniso_[\d]+_0', key):
        res['gAniso'][key.split('_')[1]][0] = value
    if re.fullmatch(r'gAniso_[\d]+_1', key):
        res['gAniso'][key.split('_')[1]][1] = value
    if re.fullmatch(r'gAniso_[\d]+_2', key):
        res['gAniso'][key.split('_')[1]][2] = value

    if re.fullmatch(r'D_[\d]+', key):
        res['D'][key.split('_')[1]] = value
    if re.fullmatch(r'E_[\d]+', key):
        res['E'][key.split('_')[1]] = value

    if re.fullmatch(r'sigma_L_[\d]+', key):
        res['sigma_L'][key.split('_')[2]] = value
    if re.fullmatch(r'lambda_SL_[\d]+', key):
        res['lambda_SL'][key.split('_')[2]] = value
    if re.fullmatch(r'lambda_sigma_SL_[\d]+', key):
        res['lambda_sigma_SL'][key.split('_')[3]] = value

    if re.fullmatch(r'g_J_[\d]+', key):
        res['g_J'][key.split('_')[2]] = value
        
    if re.fullmatch(r'gJAniso_[\d]+_0', key):
        res['gJAniso'][key.split('_')[1]][0] = value
    if re.fullmatch(r'gJAniso_[\d]+_1', key):
        res['gJAniso'][key.split('_')[1]][1] = value
    if re.fullmatch(r'gJAniso_[\d]+_2', key):
        res['gJAniso'][key.split('_')[1]][2] = value

    if re.fullmatch(r'J_ex_J_[a-z]*[0-9]*\-[a-z]*[0-9]*', key):
        res['J_ex_J'][key.split('_')[-1]] = value
    if re.fullmatch(r'J_ex_[a-z]*[0-9]*\-[a-z]*[0-9]*', key):
        res['J_ex'][key.split('_')[-1]] = value

    if re.fullmatch(r'A_hf_SI_[a-z]*[0-9]*\-[a-z]*[0-9]*', key):
        res['A_hf'][key.split('_')[-1]] = value
    if re.fullmatch(r'A_hf_LI_[a-z]*[0-9]*\-[a-z]*[0-9]*', key):
        res['A_hf']["L-" + key.split('_')[-1]] = value
    if re.fullmatch(r'A_hf_JI_[a-z]*[0-9]*\-[a-z]*[0-9]*', key):
        res['A_hf']["J-" + key.split('_')[-1]] = value

    if re.fullmatch(r'sigma_CF_[0-9]*_[0-9]*', key):
        i, k = key.split('_')[-2:]
        res['sigma_CF'][i + "_" + k] = value
    if re.fullmatch(r'sigma_CF_J_[0-9]*_[0-9]*', key):
        i, k = key.split('_')[-2:]
        res['sigma_CF'][i + "_" + k] = value

    if re.fullmatch(r'B_kq_[0-9]*_[0-9]*_[0-9]*', key):
        i, k, q = key.split('_')[-3:]
        res['B_kq'][i + "_" + k + "_" + q] = value
    if re.fullmatch(r'B_kq_J_[0-9]*_[0-9]*_[0-9]*', key):
        i, k, q = key.split('_')[-3:]
        res['B_kq'][i + "_" + k + "_" + q] = value

    if key == 'tip_correction':
        res[key] = value
    if key == 'diamagnetic_correction':
        res[key] = value

    return res

def _objective(
    reference_data : typing.Dict[str, float],
    spin_systems : typing.List[typing.Dict[str, typing.Any]],
    current_hamiltonian_params : typing.Dict[str, typing.Any],
    eps : float = 1e-3
) -> float:

    def process_prediction_result(
        result : typing.Any,
        key : str,
        T : float
    ) -> float:
        if 'M' == key:
            return np.mean(result)
        if 'M_x' == key:
            return result[0]
        if 'M_y' == key:
            return result[1]
        if 'M_z' == key:
            return result[2]
        if 'chi' == key:
            return result['\\chi']
        if 'chi_t' == key:
            return result['\\chi'] * T
        if 'delta_chi_ax' == key:
            return result['\\Delta \\chi_{ax}']
        if 'delta_chi_rh' == key:
            return result['\\Delta \\chi_{rh}']

    loss = 0.

    targets = set(reference_data.keys()).difference({'B', 'T'})

    reference_data_scale_params = {
        key : {
            'mean' : np.mean(reference_data[key]),
            'std' : 1. if len(reference_data[key]) < 2 else np.std(reference_data[key])
        } for key in targets
    }

    for key in targets:
        for b_value, t_value, target_value in zip(
            reference_data['B'],
            reference_data['T'],
            reference_data[key]
        ):
            predicted_value = process_prediction_result(
                result=NAME2CALLABLE[key](
                    spin_systems=spin_systems,
                    hamiltonian_params=current_hamiltonian_params,
                    B_value=b_value,
                    T_value=t_value
                ),
                key=key,
                T=t_value
            )

            loss += np.abs(target_value - predicted_value) / ((reference_data_scale_params[key]['std'] * np.sqrt(len(targets))) + eps)

            #loss += np.abs(target_value - predicted_value) / (np.abs(target_value) + eps)
    
    return loss
        

def optimize(
    reference_data : typing.Dict[str, float],
    spin_systems : typing.List[typing.Dict[str, typing.Any]],
    hamiltonian_params : typing.Dict[str, typing.Any],
    parameter_fixed_state : typing.Dict[str, bool],
    maxiter : int = 10,
    method : str = 'Nelder-Mead',
    return_only_active_params : bool = False
) -> typing.Dict[str, float]:
    active_params = sorted([k for k, v in parameter_fixed_state.items() if not v])

    print(active_params)
    print(hamiltonian_params)
    print(parameter_fixed_state)
    
    x0 = np.asarray(
        [get_value_by_key(key, hamiltonian_params) for key in active_params],
        dtype=np.float32
    )
    
    def proxy_objective(x : np.ndarray) -> float:
        current_hamiltonian_params = hamiltonian_params
        for param_name, param_value in zip(active_params, x.tolist()):
            current_hamiltonian_params = set_value_by_key(
                key=param_name,
                value=param_value,
                hamiltonian_params=current_hamiltonian_params
            )
        objective_value = _objective(
            reference_data=reference_data,
            spin_systems=spin_systems,
            current_hamiltonian_params=current_hamiltonian_params,
        )
        del current_hamiltonian_params
        return objective_value

    x_optimized = minimize(
        fun=proxy_objective,
        x0=x0,
        method=method,
        options={
            'maxiter' : maxiter
        }
    ).x

    if return_only_active_params:
        return dict(zip(active_params, x_optimized.tolist()))

    optimized_hamiltonian_params = hamiltonian_params
    for param_name, param_value in zip(active_params, x_optimized.tolist()):
        optimized_hamiltonian_params = set_value_by_key(
            key=param_name,
            value=param_value,
            hamiltonian_params=optimized_hamiltonian_params
        )
    return optimized_hamiltonian_params
