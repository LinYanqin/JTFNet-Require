import tensorflow as tf
from model1 import getModel
from data import load_real
from scipy.io import savemat
import os

data_type = 'temp'
y_axis = 1
x_axis = 176
data_path = '/data1/ly/JTF-Require-Code/Matlab_Code/preprocess/GB1/'
BATCH_SIZE = 601

model_path = './model'

os.environ["CUDA_VISIBLE_DEVICES"]="3"
regularization_rate = 1e-7
def test():
    x = tf.placeholder(tf.float32, shape=(BATCH_SIZE, y_axis, x_axis, 2), name='x_input')
    y_ = tf.placeholder(tf.float32, shape=(BATCH_SIZE, y_axis, x_axis, 2), name='y_label')
    x_dc = tf.placeholder(tf.complex64, shape=(BATCH_SIZE, y_axis, x_axis, 1), name='x_kspace')
    mask_k = tf.placeholder(tf.complex64, shape=(BATCH_SIZE, y_axis, x_axis, 1), name='mask')

    y_SDN1, y_SDN2, y_SDN3, y_SDN4, t_FDN, var = getModel(x, x_dc, mask_k)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, os.path.join(model_path, 'model.ckpt-JTF'))
        loss = 0
        for k in range(1):
            for i in range(10):
                label_f_all, label_t_all, input_all, dc_t, mask = load_real(BATCH_SIZE, data_path, x_axis)
                predict, a_var = sess.run([y_SDN4, var],
                                               feed_dict={x: input_all, x_dc: dc_t, mask_k : mask})
                pred_c = predict[:,:,:,0] + 1j*predict[:,:,:,1]
                p_var = a_var[:,:,:,0] + 1j*a_var[:,:,:,1]

                savemat("/data1/ly/JTF-Require-Code/Matlab_Code/Post-processing/rec_" + data_type + "/res_" + data_type + "_%d.mat"%int(i), {'F': pred_c})
                savemat("/data1/ly/JTF-Require-Code/Matlab_Code/Post-processing/rec_" + data_type + "/ale_" + data_type + "_%d.mat"%int(i), {'V': p_var})
                loss += 0


        print('Reconstruct finish')





test()