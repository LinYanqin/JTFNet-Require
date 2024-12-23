import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import nmrglue as ng
import os


def create_mask_from_txt(file_path,y_axis, x_axis):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    index = []
    for line in lines:
        values = list(map(int, line.split()))
        values = values[::-1]
        index.append(values)
    num = len(index)
    mask = np.zeros((x_axis, y_axis))
    for i in range(num):
        mask[index[i][0], index[i][1]] = 1
    return mask


def saveUnderSampledSpectrumToTXT(FID):
    F = np.fft.fftshift(np.fft.fft2(FID))
    temp = np.abs(np.real(F.reshape(-1)))
    factor = np.max(temp)
    F = F / factor
    Size = F.shape
    N1, N2 = Size[0], Size[1]
    M = np.zeros((N1, N2, 2))
    M[:, :, 0] = np.real(F)
    M[:, :, 1] = np.imag(F)
    M = M.astype(np.float32)
    return factor, M

def preprocess_BMRB(data_path, mask_path,y_axis,x_axis,z_axis,noise_factor):
    folder_temp = "./input_temp"
    if not os.path.exists(folder_temp):
        os.makedirs(folder_temp)
        print(f"Folder {folder_temp} is not exit，begin to bulit this folder")
    else:
        print(f"Folder {folder_temp} exit")
    dic, fid = ng.pipe.read(data_path)
    mask = create_mask_from_txt(mask_path,y_axis, x_axis)
    R1R2 = fid[::2, ::2, :]
    R1I2 = fid[::2, 1::2, :]
    I1R2 = fid[1::2, ::2, :]
    I1I2 = fid[1::2, 1::2, :]

    langda = noise_factor
    noise = np.sqrt(langda) * np.random.randn(x_axis, y_axis, z_axis) + 1j * np.sqrt(langda) * np.random.randn(x_axis, y_axis, z_axis)
    FID_real1 = R1R2 + 1j * R1I2 + noise
    FID_real2 = I1R2 + 1j * I1I2 + noise
    #
    # FID_real1 = R1R2 + 1j * R1I2
    # FID_real2 = I1R2 + 1j * I1I2


    temp = np.fft.fft2(FID_real1, s=(x_axis, y_axis), axes=(0, 1))
    temp /= np.max(np.real(temp.reshape(-1)))
    area = temp[1:10, 1:10, :10]
    SNR = 1 / np.std(area.reshape(-1))

    input_real1 = np.zeros((z_axis, x_axis, y_axis, 2))
    input_real2 = np.zeros((z_axis, x_axis, y_axis, 2))
    factor1 = np.zeros((1, z_axis))
    factor2 = np.zeros((1, z_axis))
    for i in range(z_axis):
        FID1 = FID_real1[:, :, i]
        FID1_under = FID1*mask
        FID2 = FID_real2[:, :, i]
        FID2_under = FID2*mask
        f, under = saveUnderSampledSpectrumToTXT(FID1_under)
        factor1[0, i] = f
        input_real1[i, :, :, :] = under
        f, under = saveUnderSampledSpectrumToTXT(FID2_under)
        factor2[0, i] = f
        input_real2[i, :, :, :] = under
    input_name1 = folder_temp + '/inputreal1.mat'
    input_name2 = folder_temp + '/inputreal2.mat'
    factor_name1 = folder_temp + '/factor1.mat'
    factor_name2 = folder_temp + '/factor2.mat'
    mask_name = folder_temp + '/mask3D.mat'
    label_name = folder_temp + '/label.mat'
    sio.savemat(input_name1, {'input_real1': input_real1})
    sio.savemat(input_name2, {'input_real2': input_real2})
    sio.savemat(factor_name1, {'factor1': factor1})
    sio.savemat(factor_name2, {'factor2': factor2})
    sio.savemat(mask_name, {'mask': mask})
    sio.savemat(label_name, {'label': fid})
    return SNR
