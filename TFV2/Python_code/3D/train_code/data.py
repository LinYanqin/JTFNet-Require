import h5py
import numpy    as np
import scipy.io as sio
import tensorflow as tf
import os
import random
import matplotlib.pyplot as plt
from model3D_1 import real2complex, complex2real
y_axis = 64
x_axis = 64

def load_data(data_path, data_path_1, batch_size):
    # print 'Loading dataset from '+ data_path
    # with h5py.File(os.path.join(data_path,'2Dpoisson5.mat'),'r') as f:
    #     mask = f['mask'][:]
    #     mask = np.fft.ifftshift(mask)
    mask = sio.loadmat(data_path+'/mask25.mat')
    mask = mask['Exp_50']
    # with h5py.File(os.path.join(data_path,'label_12ch_v1.h5'),'r') as f:
    #     data = f['label_12ch'][0:nb_samples]
    # data = np.transpose(data,(0, 2, 3, 1))
    # nb_train = nb_samples // 11 * 10
    # channel = data.shape[-1] // 2
    # data_real = data[:,:,:,:channel]
    # data_imag = data[:,:,:,channel:]
    # data =  data_real + 1j*data_imag
    # train_data = data[:nb_train]
    # validate_data = data[nb_train:]
    train_data = read_all_data(data_path_1+'/train')
    label_data = read_all_data(data_path_1 + '/train')

    validate_data = read_all_data(data_path_1 + '/val')
    validate_data_label = read_all_data(data_path_1 + '/val')
    test_data = 1
    test_data_label = 1
    print('Loading Done.')

    return train_data, validate_data, validate_data_label, label_data, test_data, test_data_label, mask
def read_all_data(path):
    files = os.listdir(path)
    arr = []
    for Dataname in files:
        data = sio.loadmat(path + '/' + Dataname)
        data = data['F']
        arr.append(data)
    return arr

def read_and_decode(filename, batch_size):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    feature = {'train/label':tf.FixedLenFeature([],tf.string)}
    features = tf.parse_single_example(serialized_example, features=feature)
    img = tf.decode_raw(features['train/label'], tf.float64)
    img = tf.reshape(img, shape=[256, 256, 24])
    # img_batch = tf.train.shuffle_batch([img], batch_size=batch_size, num_threads=64, capacity=30, min_after_dequeue=10)
    return img

def setup_inputs(x, mask, batch_size,num):
    mask = np.tile(mask, (num, 1, 1))#复制mask，用于后续相乘
    mask_tf = tf.cast(tf.constant(mask), tf.float32)#变成张量
    mask_tf = tf.reshape(mask_tf,[num,256,256,1])
    mask_tf_c = tf.cast(mask_tf, tf.complex64)#变成复值
    x = np.asarray(x)
    x_complex = tf.cast(x, tf.complex64)
    x_cat = complex2real(x_complex)

    x_cat = tf.reshape(x_cat, [num, 256, 256, 2])
    x_cat = tf.cast(x_cat, tf.float32)
    x_complex = tf.reshape(x_complex,[num,256,256,1])
    #x_complex = tf.transpose(x_complex, [2, 0, 1])
    kx_mask = x_complex * mask_tf_c#采样
    x_u = tf.ifft2d(kx_mask)#逆变换
    x_u_cat = complex2real(x_u)#链接实部虚部
    features= x_u_cat
    labels = x_cat
    masks = mask_tf_c

    return features, labels, kx_mask, masks

def setup_inputs_test(x, mask, norm=None):
    batch = x.shape[0]
    channel = x.shape[-1]
    mask = np.tile(mask, (batch, channel, 1, 1))
    mask = np.transpose(mask, (0, 2, 3, 1))
    kx = np.fft.fft2(x, axes=(1,2), norm=norm)
    kx_mask = kx * mask
    x_u = np.fft.ifft2(kx_mask, axes=(1,2), norm=norm)

    x_u_cat = np.concatenate((np.real(x_u), np.imag(x_u)), axis=-1)
    x_cat = np.concatenate((np.real(x), np.imag(x)), axis=-1)
    mask_c = mask.astype(np.complex64)
    return x_u_cat, x_cat, kx_mask, mask_c

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

def load_batch(data_path_1, num_, batch_size, load_gap, next_batch_start):
    label_t = []
    label_f = []
    input_t = []
    x_dc = []
    mask = []
    # mask_data = sio.loadmat('/data3/ly/dataset1/res3Dby2D/mask3D_8.mat')
    # mask_data = mask_data['mask']
    # for i in range(batch_size):
    #     mask.append(mask_data)
    for i in range(next_batch_start, next_batch_start + load_gap):
        full_data = sio.loadmat(data_path_1 + '/label%d.mat'%(i+1))
        full_data = full_data['label']
        under_data = sio.loadmat(data_path_1 + '/input%d.mat'%(i+1))
        under_data = under_data['input']
        mask_data = sio.loadmat(data_path_1 + '/mask3D_%d.mat'%(i+1))
        mask_data = mask_data['mask_part']

        full_f_data = gene_full_data(full_data)
        under_f_data = gene_full_data(under_data)

        mask.append(mask_data)
        input_t.append(np.fft.ifft2(under_f_data))
        label_t.append(np.fft.ifft2(full_f_data))
        label_f.append(full_f_data)
        x_dc.append(np.fft.ifft2(under_f_data))
        print('loading the %dth/%d' % (int(i), int(num_)-1))
    label_f_data = np.reshape(label_f, [load_gap*4000, y_axis, x_axis, 1])
    x_dc = np.reshape(x_dc, [load_gap*4000, y_axis, x_axis, 1])
    label_t_data = np.reshape(label_t, [load_gap*4000, y_axis, x_axis, 1])
    input_data = np.reshape(input_t, [load_gap*4000, y_axis, x_axis, 1])
    mask1 = np.reshape(mask,[load_gap*4000, y_axis, x_axis, 1])
    input_data = complex2real(input_data)
    label_t_data = complex2real(label_t_data)
    label_f_data = complex2real(label_f_data)
    next_batch_start = next_batch_start + load_gap
    return label_f_data, label_t_data, input_data, x_dc, mask1, next_batch_start

def load_test(data_path_1, num_, batch_size, load_gap, next_batch_start):
    label_t = []
    label_f = []
    input_t = []
    x_dc = []
    mask = []
    mask_data = sio.loadmat('/data3/ly/dataset1/res3Dby2D/mask3D.mat')
    mask_data = mask_data['mask8_6464']
    for i in range(batch_size):
        mask.append(mask_data)
    for i in range(next_batch_start, next_batch_start + load_gap):
        full_data = sio.loadmat(data_path_1 + '/testlabel1.mat')
        full_data = full_data['label']
        under_data = sio.loadmat(data_path_1 + '/testinput1.mat')
        under_data = under_data['input']

        full_f_data = gene_full_data(full_data)
        under_f_data = gene_full_data(under_data)
        # x = range(120)
        # y = range(120)
        # plt.contour(x, y, np.real(full_f_data[1, :, :]), 10)
        # plt.show()
        # plt.contour(x, y, np.real(under_f_data[1, :, :]), 10)
        # plt.show()

        input_t.append(np.fft.ifft2(under_f_data))
        label_t.append(np.fft.ifft2(full_f_data))
        label_f.append(full_f_data)
        x_dc.append(np.fft.ifft2(under_f_data))
        print('loading the %dth/%d' % (int(i), int(num_) - 1))
    label_f_data = np.reshape(label_f, [load_gap * 200, y_axis, x_axis, 1])
    x_dc = np.reshape(x_dc, [load_gap * 200, y_axis, x_axis, 1])
    label_t_data = np.reshape(label_t, [load_gap * 200, y_axis, x_axis, 1])
    input_data = np.reshape(input_t, [load_gap * 200, y_axis, x_axis, 1])
    mask1 = np.reshape(mask, [batch_size, y_axis, x_axis, 1])
    input_data = complex2real(input_data)
    label_t_data = complex2real(label_t_data)
    label_f_data = complex2real(label_f_data)
    next_batch_start = next_batch_start + load_gap
    return label_f_data, label_t_data, input_data, x_dc, mask1, next_batch_start
def load_real(data_path_1, num_, batch_size, load_gap, next_batch_start,m,direct_num):
    label_t = []
    label_f = []
    input_t = []
    x_dc = []
    mask = []
    mask_data = sio.loadmat('/data3/ly/dataset1/res3Dby2D/mask3D_real.mat')
    mask_data = mask_data['mask']
    for i in range(batch_size):
        mask.append(mask_data)
    for i in range(next_batch_start, next_batch_start + load_gap):
        full_data = sio.loadmat(data_path_1 + '/labelreal%d.mat'%int(m+1))
        full_data = full_data['label_real%d'%int(m+1)]
        under_data = sio.loadmat(data_path_1 + '/inputreal%d.mat'%int(m+1))
        under_data = under_data['input_real%d'%int(m+1)]

        full_f_data = gene_full_data(full_data)
        under_f_data = gene_full_data(under_data)
        # x = range(120)
        # y = range(120)
        # plt.contour(x, y, np.real(full_f_data[1, :, :]), 10)
        # plt.show()
        # plt.contour(x, y, np.real(under_f_data[1, :, :]), 10)
        # plt.show()

        input_t.append(np.fft.ifft2(under_f_data))
        label_t.append(np.fft.ifft2(full_f_data))
        label_f.append(full_f_data)
        x_dc.append(np.fft.ifft2(under_f_data))
        print('loading the %dth/%d' % (int(i), int(num_) - 1))
    label_f_data = np.reshape(label_f, [load_gap * direct_num, y_axis, x_axis, 1])
    x_dc = np.reshape(x_dc, [load_gap * direct_num, y_axis, x_axis, 1])
    label_t_data = np.reshape(label_t, [load_gap * direct_num, y_axis, x_axis, 1])
    input_data = np.reshape(input_t, [load_gap * direct_num, y_axis, x_axis, 1])
    mask1 = np.reshape(mask, [batch_size, y_axis, x_axis, 1])
    input_data = complex2real(input_data)
    label_t_data = complex2real(label_t_data)
    label_f_data = complex2real(label_f_data)
    next_batch_start = next_batch_start + load_gap
    return label_f_data, label_t_data, input_data, x_dc, mask1, next_batch_start

def load_SIM(data_path_1, num_, batch_size, load_gap, next_batch_start,m,direct_num):
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

        under_data = sio.loadmat(data_path_1 + '/inputsim%d.mat' % int(m + 1))

        under_data = under_data['input_real%d' % int(m + 1)]
        under_f_data = gene_full_data(under_data)
        input_t.append(np.fft.ifft2(under_f_data))
        x_dc.append(np.fft.ifft2(under_f_data))
        print('loading the %dth/%d' % (int(i), int(num_) - 1))

    x_dc = np.reshape(x_dc, [load_gap * direct_num, y_axis, x_axis, 1])

    input_data = np.reshape(input_t, [load_gap * direct_num, y_axis, x_axis, 1])
    mask1 = np.reshape(mask, [batch_size, y_axis, x_axis, 1])
    input_data = complex2real(input_data)

    next_batch_start = next_batch_start + load_gap
    return input_data, x_dc, mask1, factor, next_batch_start

def load_Azi(data_path_1, num_, batch_size, load_gap, next_batch_start,m,direct_num,Y_AXIS, X_AXIS):
    input_t = []
    x_dc = []
    mask = []
    mask_data = sio.loadmat(data_path_1 + '/mask3D.mat',verify_compressed_data_integrity=False)
    mask_data = mask_data['mask']
    for i in range(batch_size):
        mask.append(mask_data)
    factor_data = sio.loadmat(data_path_1 + '/factor%d.mat' % int(m + 1),verify_compressed_data_integrity=False)
    factor = factor_data['factor%d'%int(m+1)]
    for i in range(next_batch_start, next_batch_start + load_gap):

        under_data = sio.loadmat(data_path_1 + '/inputreal%d.mat' % int(m + 1),verify_compressed_data_integrity=False)

        under_data = under_data['input_real%d' % int(m + 1)]
        under_f_data = gene_full_data(under_data)
        input_t.append(np.fft.ifft2(under_f_data))
        x_dc.append(np.fft.ifft2(under_f_data))
        print('loading the %dth/%d' % (int(i), int(num_) - 1))

    x_dc = np.reshape(x_dc, [load_gap * direct_num, X_AXIS, Y_AXIS, 1])

    input_data = np.reshape(input_t, [load_gap * direct_num, X_AXIS, Y_AXIS, 1])
    mask1 = np.reshape(mask, [batch_size, X_AXIS, Y_AXIS, 1])
    input_data = complex2real(input_data)

    next_batch_start = next_batch_start + load_gap
    return input_data, x_dc, mask1, factor, next_batch_start

