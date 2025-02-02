import tensorflow.compat.v1 as tf
import numpy as np
from model3D_1 import getModel
from losses import mae
from data import load_Azi
from scipy.io import savemat
import os
import time
from preprocess import preprocess_BMRB
from Postprocess import postprocess_data
tf.disable_v2_behavior()



#model_path = '../train_code/model_best'
model_path = './model_best'
num_to_rec = 10
os.environ["CUDA_VISIBLE_DEVICES"]="3"
def test():
    y_axis = int(input("Please input the size of the indirect dimension (y-axis):"))
    x_axis = int(input("Please input the size of the indirect dimension (z-axis):"))
    direct_dim = int(input("Please input the size of the direct dimension:"))
    nb_everygap = direct_dim
    nb_test_samples = direct_dim
    BATCH_SIZE = direct_dim
    x = tf.placeholder(tf.float32, shape=(BATCH_SIZE, x_axis, y_axis, 2), name='x_input')
    x_dc = tf.placeholder(tf.complex64, shape=(BATCH_SIZE, x_axis, y_axis, 1), name='x_kspace')
    mask_k = tf.placeholder(tf.complex64, shape=(BATCH_SIZE, x_axis, y_axis, 1), name='mask1')


    t_TDN1, t_TDN2, t_TDN3, t_TDN4, t_TDN5, t_TDN6, t_TDN7, t_TDN8, t_TDN9, t_TDN10, t_TDN11, t_TDN12, t_TDN13, t_TDN14, y_SDN1, y_SDN2,y_SDN3, y_SDN4, y_SDN5, y_SDN6, y_SDN7, y_SDN8, y_SDN9, y_SDN10, y_SDN11,y_SDN12, y_SDN13, y_SDN14, y_SDN15, y_SDN16,var = getModel(x, x_dc, mask_k)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    next_batch_start = 0
    with tf.Session() as sess:
        sess.run(init)
        num_iter = num_to_rec
        while True:
            input_type = input("Whether to change the SNR of the original spectrum(Y/N):")
            if input_type == "Y":
                noise_factor = float(input("Please input the noise factor:"))
                mask_path = input("Please input the mask path:")
                oridata_path = input("Please input the data path:")
                SNR = preprocess_BMRB(oridata_path, mask_path, y_axis, x_axis, direct_dim, noise_factor)
                print("The SNR is %.8f" % SNR)
            elif input_type == "N":
                noise_factor = 0
                mask_path = input("Please input the mask path:")
                oridata_path = input("Please input the data path:")
                _ = preprocess_BMRB(oridata_path, mask_path, y_axis, x_axis, direct_dim, noise_factor)
            else:
                print("Error input")
                break
            saver.restore(sess, os.path.join(model_path, 'model.ckpt-0'))
            print("Begin to reconstrut the NUS data...")

            loss = 0
            next_batch_start = 0
            data_path = './input_temp'
            process_time = 0
            for m in range(2):
                var_m = np.zeros([num_iter,direct_dim,x_axis,y_axis],dtype=np.complex64)
                pred = np.zeros([num_iter,direct_dim,x_axis,y_axis],dtype=np.complex64)
                input_data, dc_t, mask_batch, factor, next_batch_start = load_Azi(data_path,
                                                                                    int(nb_test_samples / nb_everygap),
                                                                                    BATCH_SIZE,
                                                                                    load_gap,
                                                                                    next_batch_start,m,direct_dim,y_axis,x_axis)
                _, _ = sess.run([y_SDN15, var],
                                          feed_dict={x: input_data, x_dc: dc_t, mask_k: mask_batch})
                start_time = time.time()
                for f in range(num_to_rec):
                    predict, a_var = sess.run([y_SDN15,var],
                                                        feed_dict={x: input_data, x_dc: dc_t, mask_k: mask_batch})
                    end_time = time.time()
                    x_real = predict[:, :, :, 0]
                    x_imag = predict[:, :, :, 1]
                    pred_c = x_real + 1j * x_imag
                    pred[f,:,:,:] = pred_c
                    x_real = a_var[:, :, :, 0]
                    x_imag = a_var[:, :, :, 1]
                    var_p = x_real + 1j * x_imag
                    var_m[f,:,:,:] = var_p

                process_time += (end_time-start_time)
                print('Finished reconstructed.')
                for q in range(direct_dim):
                    pred[:,q,:,:] = pred[:,q,:,:]*factor[0,q]
                    var_m[:,q,:,:] = var_m[:,q,:,:]*factor[0,q]
                var_m = np.mean(var_m, axis=0)
                folder_restemp = "./rec_temp"
                if not os.path.exists(folder_restemp):
                    os.makedirs(folder_restemp)
                    print(f"Folder {folder_restemp} is not exit，begin to bulit this folder")
                else:
                    print(f"Folder {folder_restemp} exit")

                savemat(folder_restemp + "/res%d.mat"%int(m+1),{'res': pred})
                savemat(folder_restemp + "/ale%d.mat"%int(m+1), {'ale': var_m})
            print("Post-process the reconstructed data, please wait...")
            postprocess_data(x_axis * 2, y_axis * 2, nb_test_samples, oridata_path, num_iter)


            print('The reconstructed time is %8f' % process_time)



load_gap = 1
test()
#/data1/ly/3D_REC/ok.ft
#/data4/ly/Matlab_project/generate_for_tfnum_3D/mask3D.txt
