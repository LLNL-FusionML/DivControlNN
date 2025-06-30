# ------------------------------------------------------------------------------
# Utility Functions for Reading Training Data
# ------------------------------------------------------------------------------

import os
import glob
import h5py
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
def read_data_inputs(inpath, fname):
    """
    Reads input data from an HDF5 file.

    Args:
        inpath (str): Path to the directory containing the file.
        fname (str): Name of the HDF5 file.

    Returns:
        tuple: Arrays for ip, ncore, pinj, fz, and diff.
    """
    filename = os.path.join(inpath, fname)

    with h5py.File(filename, 'r') as hf:
        ip = np.array(hf.get('ip'))
        ncore = np.array(hf.get('ncore'))
        pinj = np.array(hf.get('pinj'))
        fz = np.array(hf.get('fz'))
        diff = np.array(hf.get('diff'))

    return (ip.astype(np.float32), ncore.astype(np.float32),
            pinj.astype(np.float32), fz.astype(np.float32),
            diff.astype(np.float32))

# ------------------------------------------------------------------------------
def read_data_outputs(inpath, fname):
    """
    Reads a lightweight version of example dataset 2 from an HDF5 file.

    Args:
        inpath (str): Path to the directory containing the file.
        fname (str): Name of the HDF5 file.

    Returns:
        tuple: Arrays for qtl, qtr, jl, jr, tel, ter, teu, neu, and rads.
    """
    filename = os.path.join(inpath, fname)

    with h5py.File(filename, 'r') as hf:
        qtl = np.array(hf.get('qtl'))
        qtr = np.array(hf.get('qtr'))
        jl = np.array(hf.get('jl'))
        jr = np.array(hf.get('jr'))
        tel = np.array(hf.get('tel'))
        ter = np.array(hf.get('ter'))
        teu = np.array(hf.get('teu'))
        neu = np.array(hf.get('neu'))
        rads = np.array(hf.get('rads'))
        pinj = np.array(hf.get('pinj'))

        # Normalize radiation data
        rads[:, 1] = rads[:, 1] / rads[:, 0]  # Divertor radiation fraction
        rads[:, 0] = rads[:, 0] / (pinj * 1e6)  # Total radiation fraction

    return (qtl.astype(np.float32), qtr.astype(np.float32),
            jl.astype(np.float32), jr.astype(np.float32),
            tel.astype(np.float32), ter.astype(np.float32),
            teu.astype(np.float32), neu.astype(np.float32),
            rads.astype(np.float32))

# ------------------------------------------------------------------------------
def lsr_standardize(data):
    """
    Standardizes the data by subtracting the mean and dividing by the standard deviation.
    Args:
        data: Input data to be standardized.
    Returns:
        Standardized data, mean, and standard deviation.
    """
    mean = np.mean(data, axis=-2, keepdims=True)
    std = np.std(data, axis=-2, keepdims=True)
    return (data - mean) / std, mean, std

# ------------------------------------------------------------------------------
def lsr_destandardize(data, mean, std):
    """
    De-standardizes the data by reversing the standardization process.
    Args:
        data: Standardized data.
        mean: Mean used during standardization.
        std: Standard deviation used during standardization.
    Returns:
        De-standardized data.
    """
    return data * std + mean

# ------------------------------------------------------------------------------
def standardize(**kwargs):
    """
    Standardizes the input data by subtracting the mean and dividing by the standard deviation.

    Args:
        kwargs (dict): Named arrays to standardize.

    Returns:
        list: Standardized arrays and their respective (mean, std) tuples.
    """
    stds = {}
    for k, v in kwargs.items():
        m, s = v.mean(), v.std()
        stds[k] = (m, s)
        print(f'Standardizing ({k}): {v.shape} : mean = {m}, std = {s}')
        v -= m
        v /= s
        print(f'Result: mean = {v.mean()}, std = {v.std()}')

    return [(v, stds[k]) for k, v in kwargs.items()]

# ------------------------------------------------------------------------------
def destandardize(**kwargs):
    """
    Reverts the standardization process.

    Args:
        kwargs (dict): Named arrays and their respective (mean, std) tuples.

    Returns:
        list: De-standardized arrays.
    """
    for k, (v, stds) in kwargs.items():
        m, s = stds
        v *= s
        v += m

    return [v for k, (v, stds) in kwargs.items()]

# ------------------------------------------------------------------------------
def maxmin_norm(v, mm=None):
    """
    Performs min-max normalization on the input data.

    Args:
        v (array): Input array to normalize.
        mm (tuple, optional): Min and max values for normalization.

    Returns:
        list: Normalized array and its (min, max) tuple.
    """
    if mm is None:
        vmin, vmax = np.floor(v.min()), np.ceil(v.max())
    else:
        vmin, vmax = mm

    v -= vmin
    v /= (vmax - vmin)

    return [v, (vmin, vmax)]

# ------------------------------------------------------------------------------
def maxmin_denorm(**kwargs):
    """
    Reverts the min-max normalization process.

    Args:
        kwargs (dict): Named arrays and their respective (min, max) tuples.

    Returns:
        list: De-normalized arrays.
    """
    for k, (v, mm) in kwargs.items():
        vmin, vmax = mm
        v *= (vmax - vmin)
        v += vmin

    return [v for k, (v, mm) in kwargs.items()]

