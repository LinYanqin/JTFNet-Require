import tensorflow as tf
import numpy as np
from model3D_combine import getModel
from losses import mae
from data import load_Azi
from scipy.io import savemat
import os

num_FDN = 1
num_SDN = 4
y_axis = 64
x_axis = 64
direct = 732
z_axis = direct
load_gap = 1
nb_everygap =direct
nb_test_samples = direct

model_path = './model3D_flex'
BATCH_SIZE = direct
os.environ["CUDA_VISIBLE_DEVICES"]="1"
regularization_rate = 1e-7
def test():
    x = tf.placeholder(tf.float32, shape=(BATCH_SIZE, y_axis, x_axis, 2), name='x_input')
    y_ = tf.placeholder(tf.float32, shape=(BATCH_SIZE, y_axis, x_axis, 2), name='y_label')
    t_label = tf.placeholder(tf.float32, shape=(BATCH_SIZE, y_axis, x_axis, 2), name='t_label')
    x_dc = tf.placeholder(tf.complex64, shape=(BATCH_SIZE, y_axis, x_axis, 1), name='x_kspace')
    mask_k = tf.placeholder(tf.complex64, shape=(BATCH_SIZE, y_axis, x_axis, 1), name='mask1')

    regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
    y_SDN1, y_SDN2, y_SDN3, y_SDN4, t_FDN, var = getModel(x, x_dc, mask_k, num_FDN, regularizer)
    total_loss = mae(y_SDN4, y_)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    next_batch_start = 0
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, os.path.join(model_path, 'model.ckpt-JTF3D'))
        loss = 0
        next_batch_start = 0
        data_path = '../../Processed_Data/'
                # if not os.path.exists("/data4/ly/3D_sim_generate/res%d"%int(k+1)):
                #     os.mkdir("/data4/ly/3D_sim_generate/res%d"%int(k+1))
        for m in range(2):
            var_m = np.zeros([10,z_axis,y_axis,x_axis],dtype=np.complex)
            pred = np.zeros([10,z_axis,y_axis,x_axis],dtype=np.complex)
            input_data, dc_t, mask_batch, factor, next_batch_start = load_Azi(data_path,
                                                                                int(nb_test_samples / nb_everygap),
                                                                                BATCH_SIZE,
                                                                                load_gap,
                                                                                next_batch_start,m,z_axis,y_axis,x_axis)
            for f in range(10):
                predict, a_var = sess.run([y_SDN4,var],
                                                    feed_dict={x: input_data, x_dc: dc_t, mask_k: mask_batch})
                x_real = predict[:, :, :, 0]
                x_imag = predict[:, :, :, 1]
                pred_c = x_real + 1j * x_imag
                pred[f,:,:,:] = pred_c
                x_real = a_var[:, :, :, 0]
                x_imag = a_var[:, :, :, 1]
                var_p = x_real + 1j * x_imag
                var_m[f,:,:,:] = var_p
            for q in range(z_axis):
                pred[:,q,:,:] = pred[:,q,:,:]*factor[0,q]
                var_m[:,q,:,:] = var_m[:,q,:,:]*factor[0,q]
            var_m = np.mean(var_m, axis=0)

            savemat("../../Rec_Data/Rec_mat/res%d.mat"%int(m+1),{'res': pred})
            savemat("../../Rec_Data/Rec_mat/ale%d.mat"%int(m+1), {'ale': var_m})




        print('Reconstruction complete!')





test()