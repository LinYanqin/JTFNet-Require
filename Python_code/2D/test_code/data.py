import h5py
import numpy    as np
import scipy.io as sio
from model1 import real2complex, complex2real
y_axis = 1
x_aixs = 176


def gene_full_data(x):
    real_data = x[:,:,:, 0]
    imag_data = x[:,:,:, 1]
    full_data = real_data + 1j * imag_data
    return full_data
def load_real(nb_train, path, x_aixs):
    full_data = sio.loadmat(path + '/label_data.mat')
    input_data = sio.loadmat(path + '/input_data.mat')
    mask_data = sio.loadmat(path + '/mask.mat')
    input_data = input_data['input_data']
    full_data = full_data['label_data']
    mask_data = mask_data['mask_data']
    full_f_data = gene_full_data(full_data)
    input_f_data = gene_full_data(input_data)
    input_t = np.fft.ifft(input_f_data)
    label_t = np.fft.ifft(full_f_data)
    x_dc = np.fft.ifft(input_f_data)
    label_f_all = np.reshape(full_f_data, [nb_train, y_axis, x_aixs, 1])
    label_t_data = np.reshape(label_t, [nb_train, y_axis, x_aixs, 1])
    input_data = np.reshape(input_t, [nb_train, y_axis, x_aixs, 1])
    x_dc = np.reshape(x_dc, [nb_train, y_axis, x_aixs, 1])
    mask = np.reshape(mask_data, [nb_train, y_axis, x_aixs, 1])
    input_data = complex2real(input_data)
    label_t_data = complex2real(label_t_data)
    label_f_data = complex2real(label_f_all)
    return label_f_data, label_t_data, input_data, x_dc, mask

def load_batch(nb_train, path):
    full_data = sio.loadmat(path + '/label_data.mat')
    input_data = sio.loadmat(path + '/input_data.mat')
    mask_data = sio.loadmat(path + '/mask.mat')
    input_data = input_data['input_data']
    full_data = full_data['label_data']
    mask_data = mask_data['mask_data']
    full_f_data = gene_full_data(full_data)
    input_f_data = gene_full_data(input_data)
    input_t = np.fft.ifft(input_f_data)
    label_t = np.fft.ifft(full_f_data)
    x_dc = np.fft.ifft(input_f_data)
    label_f_all = np.reshape(full_f_data, [nb_train, y_axis, x_aixs, 1])
    label_t_data = np.reshape(label_t, [nb_train, y_axis, x_aixs, 1])
    input_data = np.reshape(input_t, [nb_train, y_axis, x_aixs, 1])
    x_dc = np.reshape(x_dc, [nb_train, y_axis, x_aixs, 1])
    mask = np.reshape(mask_data, [nb_train, y_axis, x_aixs, 1])
    input_data = complex2real(input_data)
    label_t_data = complex2real(label_t_data)
    label_f_data = complex2real(label_f_all)
    return label_f_data, label_t_data, input_data, x_dc, mask

def load_test(nb_train, path, y_axis, x_aixs):
    input_data = sio.loadmat(path + '/input_data.mat')
    mask_data = sio.loadmat(path + '/mask.mat')
    input_data = input_data['input_data']
    mask_data = mask_data['mask_data']
    input_f_data = gene_full_data(input_data)
    input_t = np.fft.ifft(input_f_data)
    x_dc = np.fft.ifft(input_f_data)
    input_data = np.reshape(input_t, [nb_train, y_axis, x_aixs, 1])
    x_dc = np.reshape(x_dc, [nb_train, y_axis, x_aixs, 1])
    mask = np.reshape(mask_data, [nb_train, y_axis, x_aixs, 1])
    input_data = complex2real(input_data)
    return input_data, x_dc, mask

def load_test1(nb_train, path):
    full_data = sio.loadmat(path + '/label_data.mat')
    input_data = sio.loadmat(path + '/input_data.mat')
    mask_data = sio.loadmat(path + '/mask.mat')
    input_data = input_data['input_data']
    full_data = full_data['label_data']
    mask_data = mask_data['mask_data']
    full_f_data = gene_full_data(full_data)
    input_f_data = gene_full_data(input_data)
    input_t = np.fft.ifft(input_f_data)
    label_t = np.fft.ifft(full_f_data)
    x_dc = np.fft.ifft(input_f_data)
    label_f_all = np.reshape(full_f_data, [nb_train, y_axis, x_aixs, 1])
    label_t_data = np.reshape(label_t, [nb_train, y_axis, x_aixs, 1])
    input_data = np.reshape(input_t, [nb_train, y_axis, x_aixs, 1])
    x_dc = np.reshape(x_dc, [nb_train, y_axis, x_aixs, 1])
    mask = np.reshape(mask_data, [nb_train, y_axis, x_aixs, 1])
    input_data = complex2real(input_data)
    label_t_data = complex2real(label_t_data)
    label_f_data = complex2real(label_f_all)
    return label_f_data, label_t_data, input_data, x_dc, mask
def load_testSIM(nb_train, path):
    full_data = sio.loadmat(path + '/label_data.mat')
    input_data = sio.loadmat(path + '/input_data.mat')
    mask_data = sio.loadmat(path + '/mask.mat')
    input_data = input_data['input_sim']
    full_data = full_data['label_sim']
    mask_data = mask_data['mask_sim']
    full_f_data = gene_full_data(full_data)
    input_f_data = gene_full_data(input_data)
    input_t = np.fft.ifft(input_f_data)
    label_t = np.fft.ifft(full_f_data)
    x_dc = np.fft.ifft(input_f_data)
    label_f_all = np.reshape(full_f_data, [nb_train, y_axis, x_aixs, 1])
    label_t_data = np.reshape(label_t, [nb_train, y_axis, x_aixs, 1])
    input_data = np.reshape(input_t, [nb_train, y_axis, x_aixs, 1])
    x_dc = np.reshape(x_dc, [nb_train, y_axis, x_aixs, 1])
    mask = np.reshape(mask_data, [nb_train, y_axis, x_aixs, 1])
    input_data = complex2real(input_data)
    label_t_data = complex2real(label_t_data)
    label_f_data = complex2real(label_f_all)
    return label_f_data, label_t_data, input_data, x_dc, mask