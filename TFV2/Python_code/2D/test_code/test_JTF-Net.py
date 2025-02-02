import tensorflow.compat.v1 as tf
from model1 import getModel
from data import load_test
from scipy.io import savemat
import os
from preprocess import process_data
import time
tf.disable_v2_behavior()

#model_path = '../train_code/model_best'
model_path = './model_best'
num_to_rec = 10

#os.environ["CUDA_VISIBLE_DEVICES"]="3"
regularization_rate = 1e-7

def test():
    BATCH_SIZE = int(input("Please input the size of direct dimension:"))
    x_axis = int(input("Please input the size of indirect dimension:"))
    x = tf.placeholder(tf.float32, shape=(BATCH_SIZE, y_axis, x_axis, 2), name='x_input')
    x_dc = tf.placeholder(tf.complex64, shape=(BATCH_SIZE, y_axis, x_axis, 1), name='x_kspace')
    mask_k = tf.placeholder(tf.complex64, shape=(BATCH_SIZE, y_axis, x_axis, 1), name='mask')

    t_TDN1, t_TDN2, t_TDN3, t_TDN4, t_TDN5, t_TDN6, t_TDN7, t_TDN8, y_SDN1, y_SDN2,y_SDN3, y_SDN4, y_SDN5, y_SDN6, y_SDN7, y_SDN8, y_SDN9, y_SDN10, var = getModel(x, x_dc, mask_k)


    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, os.path.join(model_path, 'model.ckpt-0'))
        while True:
            input_type = input("Whether to change the SNR of the original spectrum(Y/N):")
            if input_type == "Y":
                noise_factor = float(input("please input noise factor:"))
                mask_path = input("please input mask path:")
                data_path = input("please input data path:")
                SNR = process_data(data_path, mask_path, BATCH_SIZE, x_axis,noise_factor)
                print("The SNR is %.8f" % SNR)
            elif input_type == "N":
                noise_factor = 0
                mask_path = input("please input mask path:")
                data_path = input("please input data path:")
                _ = process_data(data_path, mask_path, BATCH_SIZE, x_axis, noise_factor)
            loss = 0
            input_path = './temp_data/input/'
            rec_path = './temp_data/rec_temp/'
            if not os.path.exists(rec_path):
                os.makedirs(rec_path)
                print(f"Folder {rec_path} not exist, begin to build")
            else:
                print(f"Folder {rec_path} exist")

            input_all, dc_t, mask = load_test(BATCH_SIZE, input_path, y_axis, x_axis)
            _, _ = sess.run([y_SDN10, var],
                                      feed_dict={x: input_all, x_dc: dc_t, mask_k: mask})
            total_time = 0
            for i in range(num_to_rec):
                start_time = time.time()
                predict, a_var = sess.run([y_SDN10, var],
                                               feed_dict={x: input_all, x_dc: dc_t, mask_k : mask})
                end_time = time.time()
                total_time += (end_time - start_time)
                print("Single reconstruction time %.8f" % (end_time - start_time))
                pred_c = predict[:,:,:,0] + 1j*predict[:,:,:,1]
                p_var = a_var[:,:,:,0] + 1j*a_var[:,:,:,1]

                savemat(rec_path + "/res_temp_%d.mat"%int(i), {'F': pred_c})
                savemat(rec_path + "/ale_temp_%d.mat"%int(i), {'V': p_var})
                loss += 0
            print("Reconstruction finish! Total reconstructed time is %.8f" % total_time)




y_axis = 1
test()
