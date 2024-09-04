import h5py
import numpy    as np
import scipy.io as sio
import tensorflow as tf
import os
import random
import matplotlib.pyplot as plt
y_axis = 64
x_axis = 64

def real2complex(x):
    channel = x.shape[-1] // 2
    if x.shape.ndims == 3:
        return tf.complex(x[:,:,:channel], x[:,:,channel:])
    elif x.shape.ndims == 4:
        return tf.complex(x[:,:,:,:channel], x[:,:,:,channel:])

def complex2real(x):
    x_real = np.real(x)
    x_imag = np.imag(x)
    return np.concatenate([x_real,x_imag], axis=-1)



def gene_full_data_full(x):
    real_data = x[:, 0:x_axis]
    imag_data = x[:, x_axis:x_axis*2]
    full_data = real_data + 1j * imag_data
    return full_data

def gene_full_data(x):
    real_data = x[:, :,:,0]
    imag_data = x[:, :,:,1]
    full_data = real_data + 1j * imag_data
    return full_data

def load_Azi(data_path_1, num_, batch_size, load_gap, next_batch_start,m,direct_num,Y_AXIS, X_AXIS):
    input_t = []
    x_dc = []
    mask = []
    mask_data = sio.loadmat(data_path_1 + '/mask3D.mat')
    mask_data = mask_data['mask']
    for i in range(batch_size):
        mask.append(mask_data)
    factor_data = sio.loadmat(data_path_1 + '/factor%d.mat' % int(m + 1))
    factor = factor_data['factor%d'%int(m+1)]
    for i in range(next_batch_start, next_batch_start + load_gap):

        under_data = sio.loadmat(data_path_1 + '/inputreal%d.mat' % int(m + 1))

        under_data = under_data['input_real%d' % int(m + 1)]
        under_f_data = gene_full_data(under_data)
        input_t.append(np.fft.ifft2(under_f_data))
        x_dc.append(np.fft.ifft2(under_f_data))
        print('loading the %dth/%d' % (int(i), int(num_) - 1))

    x_dc = np.reshape(x_dc, [load_gap * direct_num, Y_AXIS, X_AXIS, 1])

    input_data = np.reshape(input_t, [load_gap * direct_num, Y_AXIS, X_AXIS, 1])
    mask1 = np.reshape(mask, [batch_size, Y_AXIS, X_AXIS, 1])
    input_data = complex2real(input_data)

    next_batch_start = next_batch_start + load_gap
    return input_data, x_dc, mask1, factor, next_batch_start

