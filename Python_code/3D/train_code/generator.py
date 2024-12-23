import numpy as np
import os
import tensorflow as tf
import time
import scipy.io as sio



def gene_full_data(x):
    real_data = x[:,:,0]
    imag_data = x[:,:,1]
    full_data = real_data + 1j * imag_data
    return full_data
def complex2real(x):
    x_real = np.real(x)
    x_imag = np.imag(x)
    return np.concatenate([x_real,x_imag], axis=-1)

def data_generator(f_xtrain, f_ytrain, f_masktrain, y_axis, x_axis, batch_size):
    idx = np.arange(len(f_xtrain))
    np.random.shuffle(idx)
    batches = [idx[range(batch_size*i, min(len(f_xtrain), batch_size*(i+1)))] for i in range(len(f_xtrain)//batch_size)]
    while True:
        for batch in batches:
            Xtrain = []
            Ytrain = []
            Masktrain = []
            X = np.zeros((len(batch), y_axis, x_axis, 2))
            Y = np.zeros((len(batch), y_axis, x_axis, 2))
            t_label = np.zeros((len(batch), y_axis, x_axis, 2))
            mask_data = np.zeros((len(batch), y_axis, x_axis, 1),dtype=np.complex64)
            dc_data = np.zeros((len(batch), y_axis, x_axis, 1),dtype=np.complex64)
            for i in batch:
                Xtrain.append(f_xtrain[i])
                Ytrain.append(f_ytrain[i])
                Masktrain.append(f_masktrain[i])
            for i in range(len(batch)):
                file_path1 = Xtrain[i]
                file_path2 = Ytrain[i]
                file_path3 = Masktrain[i]
                fr1 = sio.loadmat(file_path1)
                x_train_temp = fr1['under']
                fr2 = sio.loadmat(file_path2)
                y_train_temp = fr2['full']
                fr3 = sio.loadmat(file_path3)
                mask_train_temp = fr3['mask']
                full_f_data = gene_full_data(y_train_temp)
                under_f_data = gene_full_data(x_train_temp)
                dc_temp = np.fft.ifft2(under_f_data)
                t_temp = np.fft.ifft2(full_f_data)
                dc_temp = np.expand_dims(dc_temp,axis=-1)
                t_temp = np.expand_dims(t_temp,axis=-1)
                input_t = complex2real(dc_temp)
                label_t = complex2real(t_temp)
                X[i,:,:,:] = input_t[:,:,:]
                t_label[i,:,:,:] = label_t[:,:,:]
                Y[i,:,:,:] = y_train_temp
                mask_data[i,:,:,0] = mask_train_temp
                dc_data[i,:,:,:] = dc_temp[:,:,:]
            yield (X, Y, t_label, mask_data, dc_data)

def TEST_generator(f_xtrain, f_ytrain, f_masktrain, y_axis, x_axis, batch_size):
    idx = np.arange(len(f_xtrain))
    np.random.shuffle(idx)
    batches = [idx[range(batch_size * i, min(len(f_xtrain), batch_size * (i + 1)))] for i in
               range(len(f_xtrain) // batch_size)]
    while True:
        for batch in batches:
            Xtrain = []
            Ytrain = []
            Masktrain = []
            X = np.zeros((len(batch), y_axis, x_axis, 2))
            Y = np.zeros((len(batch), y_axis, x_axis, 2))
            t_label = np.zeros((len(batch), y_axis, x_axis, 2))
            mask_data = np.zeros((len(batch), y_axis, x_axis, 1), dtype=np.complex64)
            dc_data = np.zeros((len(batch), y_axis, x_axis, 1), dtype=np.complex64)
            for i in batch:
                Xtrain.append(f_xtrain[i])
                Ytrain.append(f_ytrain[i])
                Masktrain.append(f_masktrain[i])
            for i in range(len(batch)):
                file_path1 = Xtrain[i]
                file_path2 = Ytrain[i]
                file_path3 = Masktrain[i]
                fr1 = sio.loadmat(file_path1)
                x_train_temp = fr1['under']
                fr2 = sio.loadmat(file_path2)
                y_train_temp = fr2['full']
                fr3 = sio.loadmat(file_path3)
                mask_train_temp = fr3['mask']
                full_f_data = gene_full_data(y_train_temp)
                under_f_data = gene_full_data(x_train_temp)
                dc_temp = np.fft.ifft2(under_f_data)
                t_temp = np.fft.ifft2(full_f_data)
                dc_temp = np.expand_dims(dc_temp, axis=-1)
                t_temp = np.expand_dims(t_temp, axis=-1)
                input_t = complex2real(dc_temp)
                label_t = complex2real(t_temp)
                X[i, :, :, :] = input_t[:, :, :]
                t_label[i, :, :, :] = label_t[:, :, :]
                Y[i, :, :, :] = y_train_temp
                mask_data[i, :, :, 0] = mask_train_temp
                dc_data[i, :, :, :] = dc_temp[:, :, :]
            yield (X, Y, t_label, mask_data,dc_data)