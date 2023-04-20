# PoroUtils_complete will be the final version I use when training

import h5py
import numpy as np


def get_data(filename, scenarios, step_size=1, stress_measure=None,
             other_columns=None, plastic_strain_threshold=None,
             two_stage=False):

    data = []
    data_names = []

    if stress_measure is not None:
        # HERE'S WHERE I MAKE AN EDIT: I allow for custom stress measurements or porosity evolution
        if stress_measure == "lode": # hydrostatic, von Mises, Lode 
            princ_stress = _get_data_in_bingo_format(filename,
                                                     ["S_h", "S_vm", "S_l"],
                                                     scenarios, step_size) # NEED to title my hdf5 this way
            # princ_stress[:,0:2] /= 1e6 # edit this out as needed
            stresses = princ_stress
            stress_names = ["Sp", "Sq", "Sl"] # same as when you put in pql stress measure
            
        elif stress_measure == "no_lode": # hydrostatic, von Mises
            princ_stress = _get_data_in_bingo_format(filename,
                                                     ["S_h", "S_vm"],
                                                     scenarios, step_size) # NEED to title my hdf5 this way
            princ_stress /= 1e6 # edit this out as needed
            stresses = princ_stress
            stress_names = ["Sp", "Sq"] # same as when you put in pql stress measure
            
        else:
            princ_stress = _get_data_in_bingo_format(filename,
                                                     ["S11", "S22", "S33"],
                                                     scenarios, step_size)
            # princ_stress /= 1e6 # edit this out as needed
            stresses, stress_names = _stress_transform(princ_stress,
                                                   stress_measure)
        data.append(stresses)
        data_names.extend(stress_names)
        
    if other_columns is not None:
        data.append(_get_data_in_bingo_format(filename, other_columns,
                                              scenarios, step_size))
        data_names.extend(other_columns)

    data = np.hstack(data)

    if plastic_strain_threshold is not None:
        plastic_strain = _get_data_in_bingo_format(filename,
                                                   ["PEEQ"],
                                                   scenarios, step_size,
                                                   two_stage)
        data = _filter_out_elastic(data, plastic_strain,
                                   plastic_strain_threshold)

    return data, data_names


def get_num_available_scenarios(h5_file_name):
    with h5py.File(h5_file_name, "r") as f:
        return f["S_h"].shape[0]


def _get_data_in_bingo_format(h5_file_name, value_names, sims, step_size=1,
                              normalize_by_start=False):
    unformatted_data = []
    with h5py.File(h5_file_name, "r") as f:
        for value in value_names:
            value_data = np.copy(f[value])
            if normalize_by_start:
                value_data -= value_data[:, 0].reshape((-1, 1))
            if value != "Time":
                value_data = value_data[sims]
            else:
                value_data = np.vstack([value_data for _ in range(len(sims))])
            value_data = value_data[:, ::step_size]
            unformatted_data.append(value_data)
    return _reformat_as_bingo(unformatted_data)


def _reformat_as_bingo(input_data_list):
    columns = [_flatten_with_nans(i) for i in input_data_list]
    formatted_data = np.vstack(columns).T
    return formatted_data[:-1]


def _flatten_with_nans(input_data):
    new_col = np.full((input_data.shape[0], 1), np.nan)
    output_data = np.hstack((input_data, new_col)).flatten()
    return output_data


def _filter_out_elastic(data, plastic_strain, pe_threshold):
    filter_ = np.isnan(data[:, 0])
    over_threshold = np.sum(np.abs(plastic_strain), axis=1) >= pe_threshold
    filter_ = np.logical_or(filter_, over_threshold)
    return data[filter_]


def _stress_transform(principle_stresses, transform):
    if transform == "principle_stresses":
        return principle_stresses, ["S1", "S2", "S3"]

    i_1 = np.sum(principle_stresses, axis=1)
    i_2 = principle_stresses[:, 0] * principle_stresses[:, 1] + \
        principle_stresses[:, 1] * principle_stresses[:, 2] + \
        principle_stresses[:, 2] * principle_stresses[:, 0]
    i_3 = principle_stresses[:, 0] * principle_stresses[:, 1] * \
        principle_stresses[:, 2]
    j_2 = (i_1 ** 2) / 3 - i_2
    j_3 = (i_1 ** 3) * 2 / 27 - i_1 * i_2 / 3 + i_3

    if transform == "invariants":
        return np.vstack((i_1, j_2, j_3)).T, ["I1", "J2", "J3"]

    if transform in ["lode_coordinates", "haigh_westergaard"]:
        z = i_1 / np.sqrt(3)
        r = np.sqrt(2 * j_2)
        lode_angle = j_3 / 2 * np.power(3 / j_2, 3/2)
        return np.vstack((z, r, lode_angle)).T, ["Sxsi", "Srho", "Stheta"]

    if transform == "pql":
        p = i_1 / 3
        q = np.sqrt(3 * j_2)
        l = 3 * np.sqrt(3) / 2 * j_3 / np.power(j_2, 3/2)
        return np.vstack((p, q, l)).T, ["Sp", "Sq", "Sl"]
