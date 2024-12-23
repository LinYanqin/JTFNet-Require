from cProfile import label

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft2, fft2
import os
import nmrglue as ng
import scipy.io as sio

def process_data(data_path, mask_path, direct_dim, indirect_dim,noise_factor):
    # %%%%%%%%%%%% load data %%%%%%%%%%%%%
    dic, data = ng.pipe.read(data_path)
    y_axis, x_axis = data.shape
    fid = data
    label_f = np.fft.fft(fid)
    label_f = label_f / np.max(np.real(label_f))

    # %%%%%%%%%%%% Normalized data %%%%%%%%%%%

    langda=noise_factor
    noise = np.sqrt(langda) * np.random.randn(direct_dim, indirect_dim) + 1j * np.sqrt(langda) * np.random.randn(direct_dim, indirect_dim)
    lable_t = np.fft.ifft2(label_f)
    label_t_noise  = lable_t + noise
    label_f1 = np.fft.fft2(label_t_noise)

    temp = label_f1
    temp = temp / np.max(np.real(temp))
    area = temp[1:10, 1:10] # you can select other area that without signal
    SNR = 1 / np.std(area)

    # %%%%%%%%%%%% generate mask for NUS %%%%%%%%%%%%
    t_label1 = np.fft.ifft(label_f1,axis=1)
    mask_index = np.loadtxt(mask_path, dtype=int)
    mask = np.zeros((1, x_axis))
    l = len(mask_index)
    for k in range(l):
        mask[0, mask_index[k]] = 1
    undersample = t_label1 * mask ## undersampled

    # %%%%%%%%%%% generate_inputs_for_JTF-Net %%%%%%%%%%%%
    input_data = np.zeros((y_axis, 1, x_axis, 2))
    factor = np.zeros((1, y_axis))
    mask_data = np.zeros((y_axis, 1, x_axis))

    for i in range(y_axis):
        FID2 = undersample[i, :]
        f, inputtemp = saveUnderSampledSpectrumToTXT(FID2)
        input_data[i, 0, :, 0] = inputtemp[:, 0]
        input_data[i, 0, :, 1] = inputtemp[:, 1]
        factor[0, i] = f
        mask_data[i, 0, :] = mask
    folder_path = "./temp_data/input"

    parent_folder_path = os.path.dirname(folder_path)

    if not os.path.exists(parent_folder_path):
        os.makedirs(parent_folder_path)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder {folder_path} not exist, begin to build")
    else:
        print(f"Folder {folder_path} exist")
    name_input = folder_path + '/input_data.mat'
    name_mask = folder_path + '/mask.mat'
    name_factor = folder_path + '/factor.mat'

    sio.savemat(name_input, {'input_data': input_data})
    sio.savemat(name_mask, {'mask_data': mask_data})
    sio.savemat(name_factor, {'factor': factor})

    return SNR


def saveUnderSampledSpectrumToTXT(FID):
    F = np.fft.fft(FID)
    temp = np.real(F).flatten()
    factor = np.max(temp)
    F = F / factor
    Size = F.shape
    N1 = Size[0]
    M = np.zeros((N1, 2))
    M[:, 0] = np.real(F)
    M[:, 1] = np.imag(F)
    M = M.astype(np.single)

    return factor, M

def saveSpectrumToTXT(FID, f):
    F = np.fft.fft(FID)
    F = F / f
    Size = F.shape
    N1 = Size[0]
    M = np.zeros((N1, 2))
    M[:, 0] = np.real(F)
    M[:, 1] = np.imag(F)
    M = M.astype(np.single)

    return M