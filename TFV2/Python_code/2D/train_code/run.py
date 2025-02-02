import tensorflow.compat.v1 as tf
import numpy as np
from model1 import getModel,real2complex
from losses import mae, mse
from data import load_batch
import os
import time
import random
tf.disable_v2_behavior()


EPOCHS = 300
BATCH_SIZE = 40
nb_train = 40000
nb_val = 4000
regularization_rate = 1e-7
data_path = '../../../Dataset/2D/train/'
data_path_val = '../../../Dataset/2D/val/'
model_save_path = './model_per'
bestmodel_save_path = './model_best'
model_name = 'model.ckpt'
lr_base = 0.001
lr_decay_rate = 0.95
loss={'batch':[], 'count':[], 'epoch':[]}
val_loss=[]
y_axis = 1
x_axis = 176
os.environ["CUDA_VISIBLE_DEVICES"]="6"


def complex2real(x):
    x_real = np.real(x)
    x_imag = np.imag(x)
    return np.concatenate([x_real,x_imag], axis=-1)

def real2complex_array(x):
    x = np.asarray(x)
    channel = x.shape[-1] // 2
    x_real = x[:,:,:,:channel]
    x_imag = x[:,:,:,channel:]
    return x_real + x_imag * 1j



def train():
    # nb_train = len(train_data)
    x = tf.placeholder(tf.float32,shape=(BATCH_SIZE,y_axis,x_axis,2),name='x_input')
    y_ = tf.placeholder(tf.float32,shape=(BATCH_SIZE,y_axis,x_axis,2),name='y_label')
    t_label = tf.placeholder(tf.float32,shape=(BATCH_SIZE,y_axis,x_axis,2),name='t_label')
    x_dc = tf.placeholder(tf.complex64,shape=(BATCH_SIZE,y_axis,x_axis,1),name='x_kspace')
    mask_k = tf.placeholder(tf.complex64,shape=(BATCH_SIZE,y_axis,x_axis,1),name='mask')


    t_TDN1, t_TDN2, t_TDN3, t_TDN4, t_TDN5, t_TDN6, t_TDN7, t_TDN8, y_SDN1, y_SDN2,y_SDN3, y_SDN4, y_SDN5, y_SDN6, y_SDN7, y_SDN8, y_SDN9, y_SDN10, var = getModel(x, x_dc, mask_k)
    global_step = tf.Variable(0.,trainable=False)
    with (tf.name_scope('mse_loss')):
        a =tf.reduce_mean(tf.multiply(tf.real(tf.exp(-real2complex(var))),tf.real(mse(y_, y_SDN10))))
        t_loss = mae(t_label, t_TDN1) + mae(t_label, t_TDN2) + mae(t_label, t_TDN3) + mae(t_label, t_TDN4) + mae(t_label, t_TDN5) + mae(t_label, t_TDN6) + mae(t_label, t_TDN7) + mae(t_label, t_TDN8)
        f_loss = mae(y_, y_SDN1) + mae(y_, y_SDN2) + mae(y_, y_SDN3) + mae(y_, y_SDN4) + mae(y_, y_SDN5) + mae(y_, y_SDN6) + mae(y_, y_SDN7) + mae(y_, y_SDN8) + mae(y_, y_SDN9) + mae(y_, y_SDN10)
        total_loss = 10*t_loss + f_loss + a + tf.reduce_mean(tf.square(var))
        val_loss = mae(y_, y_SDN10)
    lr = tf.train.exponential_decay(lr_base,
                                    global_step=global_step,
                                    decay_steps=20000,
                                    decay_rate=lr_decay_rate,
                                    staircase=False)
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(lr).minimize(total_loss,global_step=global_step)

    saver = tf.train.Saver(max_to_keep=40)
    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        # saver.restore(sess, os.path.join(model_save_path, 'model.ckpt-512500'))
        label_f_all, label_t_all, input_all, dc_t, mask = load_batch(nb_train, data_path, y_axis, x_axis)
        index = [i for i in range(nb_train)]
        random.shuffle(index)
        dc_t = dc_t[index]
        input_all = input_all[index]
        label_t_all = label_t_all[index]
        label_f_all = label_f_all[index]
        mask = mask[index]

        label_f_val, label_t_val, input_val, dc_val, mask_val = load_batch(nb_val, data_path_val, y_axis, x_axis)
        index_val = [i for i in range(nb_val)]
        random.shuffle(index_val)
        dc_val = dc_val[index_val]
        input_val = input_val[index_val]
        label_t_val = label_t_val[index_val]
        label_f_val = label_f_val[index_val]
        mask_val = mask_val[index_val]
        count_nochange = 0
        loss_val1 = 100.0
        start_time  = time.time()
        for i in range(EPOCHS):
            loss_sum = 0.0
            count_batch = 0

            nb_batches = int((nb_train) // BATCH_SIZE)
            for n_batch in range(nb_batches):
                input_data_batch = input_all[n_batch * BATCH_SIZE:(n_batch + 1) * BATCH_SIZE, :, :, :]
                label_f_batch = label_f_all[n_batch * BATCH_SIZE:(n_batch + 1) * BATCH_SIZE, :, :, :]
                label_t_batch = label_t_all[n_batch * BATCH_SIZE:(n_batch + 1) * BATCH_SIZE, :, :, :]
                dc_batch = dc_t[n_batch * BATCH_SIZE:(n_batch + 1) * BATCH_SIZE, :, :, :]
                mask_batch = mask[n_batch * BATCH_SIZE:(n_batch + 1) * BATCH_SIZE, :, :, :]
                _,loss_value, step = sess.run([train_step, total_loss, global_step],
                                              feed_dict={x: input_data_batch, y_: label_f_batch, t_label: label_t_batch, x_dc: dc_batch, mask_k : mask_batch})

                loss_sum += loss_value
                count_batch += 1
                ave_loss = loss_sum / count_batch
                loss['batch'].append(ave_loss)
                print ('Epoch %3d-batch %3d/%3d  training loss: %8f ' % (i+1, count_batch, nb_batches, ave_loss))
            print('Begin to val......')
            nb_vals = int((nb_val) // BATCH_SIZE)
            loss_val = 0.0


            for n_batch in range(nb_vals):
                input_data_batch = input_val[n_batch * BATCH_SIZE:(n_batch + 1) * BATCH_SIZE, :, :, :]
                label_f_batch = label_f_val[n_batch * BATCH_SIZE:(n_batch + 1) * BATCH_SIZE, :, :, :]
                label_t_batch = label_t_val[n_batch * BATCH_SIZE:(n_batch + 1) * BATCH_SIZE, :, :, :]
                dc_batch = dc_val[n_batch * BATCH_SIZE:(n_batch + 1) * BATCH_SIZE, :, :, :]
                mask_batch = mask_val[n_batch * BATCH_SIZE:(n_batch + 1) * BATCH_SIZE, :, :, :]
                loss_value= sess.run([val_loss],
                                               feed_dict={x: input_data_batch, y_: label_f_batch,
                                                          t_label: label_t_batch, x_dc: dc_batch,
                                                          mask_k: mask_batch})
                loss_val += loss_value[0]
            mean_val_loss = loss_val/nb_val

            if loss_val1>mean_val_loss:
                loss_val1 = mean_val_loss
                count_nochange = 0
                saver.save(sess, os.path.join(bestmodel_save_path, model_name), global_step=0)
            else:
                count_nochange += 1
            print('The mean val loss of %depoch is %8f'%(i+1, mean_val_loss))
            saver.save(sess, os.path.join(model_save_path,model_name), global_step=global_step)
            if count_nochange >= 10:
                end_time = time.time()
                print('Training data is %8f'%(end_time-start_time))
                print('Finish train')
                break
            # test every 5 epochs


def main(argv=None):
    train()

if __name__ == '__main__':
    main()