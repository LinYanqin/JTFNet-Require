import tensorflow.compat.v1 as tf
import numpy as np
from model3D_1 import getModel,real2complex
from losses import mae, mse
import os
from utils import load_file_list, natural_keys
from generator import data_generator, TEST_generator
import time
tf.disable_v2_behavior()

EPOCHS = 300
BATCH_SIZE = 4
nb_val = 400
nb_train_samples = 4000
model_save_path = './model_per'
bestmodel_save_path = './model_best'
model_name = 'model.ckpt'
lr_base = 0.001
lr_decay_rate = 0.95
loss={'batch':[], 'count':[], 'epoch':[]}
val_loss=[]
num_FDN = 1
num_SDN = 4
y_axis = 64
x_axis = 64
os.environ["CUDA_VISIBLE_DEVICES"]="4"

training_data_path = "../../../Dataset/3D/train/Input/"
training_gooddata_path = "../../../Dataset/3D/train/Label/"
training_mask_path = "../../../Dataset/3D//train/Mask/"
val_data_path = "../../../Dataset/3D/val/Input/"
val_gooddata_path = "../../../Dataset/3D/val/Label/"
val_mask_path = "../../../Dataset/3D/val/Mask/"

x_train = load_file_list(path=training_data_path,
                                      regx='.*.mat')
x_train.sort(key=natural_keys)

y_train = load_file_list(path=training_gooddata_path,
                                      regx='.*.mat')
y_train.sort(key=natural_keys)

mask_train = load_file_list(path=training_mask_path,
                                      regx='.*.mat')
mask_train.sort(key=natural_keys)
x_val = load_file_list(path=val_data_path,
                                      regx='.*.mat')
x_val.sort(key=natural_keys)

y_val = load_file_list(path=val_gooddata_path,
                                      regx='.*.mat')
y_val.sort(key=natural_keys)

mask_val = load_file_list(path=val_mask_path,
                                      regx='.*.mat')
mask_val.sort(key=natural_keys)
train_all_num = len(x_train)
f_xtrain = []
f_ytrain = []
f_masktrain = []
for i in range(train_all_num):
    f_xtrain.append(os.path.join(training_data_path, x_train[i]))
    f_ytrain.append(os.path.join(training_gooddata_path, y_train[i]))
    f_masktrain.append(os.path.join(training_mask_path, mask_train[i]))
val_all_num = len(x_val)
f_xval = []
f_yval = []
f_maskval = []
for i in range(val_all_num):
    f_xval.append(os.path.join(val_data_path, x_val[i]))
    f_yval.append(os.path.join(val_gooddata_path, y_val[i]))
    f_maskval.append(os.path.join(val_mask_path, mask_val[i]))

def train():
    # nb_train = len(train_data)
    x = tf.placeholder(tf.float32,shape=(BATCH_SIZE,y_axis,x_axis,2),name='x_input')
    y_ = tf.placeholder(tf.float32,shape=(BATCH_SIZE,y_axis,x_axis,2),name='y_label')
    t_label = tf.placeholder(tf.float32,shape=(BATCH_SIZE,y_axis,x_axis,2),name='t_label')
    x_dc = tf.placeholder(tf.complex64,shape=(BATCH_SIZE,y_axis,x_axis,1),name='x_kspace')
    mask_k = tf.placeholder(tf.complex64,shape=(BATCH_SIZE,y_axis,x_axis,1),name='mask1')

    t_TDN1, t_TDN2, t_TDN3, t_TDN4, t_TDN5, t_TDN6, t_TDN7, t_TDN8, t_TDN9, t_TDN10, t_TDN11, t_TDN12, t_TDN13, t_TDN14, y_SDN1, y_SDN2,y_SDN3, y_SDN4, y_SDN5, y_SDN6, y_SDN7, y_SDN8, y_SDN9, y_SDN10, y_SDN11,y_SDN12, y_SDN13, y_SDN14, y_SDN15, y_SDN16,var = getModel(x, x_dc, mask_k)
    global_step = tf.Variable(0.,trainable=False)
    with tf.name_scope('mse_loss'):
        a = tf.reduce_mean(tf.multiply(tf.real(tf.exp(-real2complex(var))), tf.real(mse(y_, y_SDN16))))
        t_loss = mae(t_label, t_TDN1) + mae(t_label, t_TDN2) + mae(t_label, t_TDN3) + mae(t_label, t_TDN4) + mae(
            t_label, t_TDN5) + mae(t_label, t_TDN6) + mae(t_label, t_TDN7) + mae(t_label, t_TDN8) + mae(t_label, t_TDN9) + mae(t_label, t_TDN10) + mae(t_label, t_TDN11) + mae(t_label, t_TDN12) + mae(t_label, t_TDN13) + mae(t_label, t_TDN14)
        f_loss = mae(y_, y_SDN1) + mae(y_, y_SDN2) + mae(y_, y_SDN3) + mae(y_, y_SDN4) + mae(y_, y_SDN5) + mae(y_, y_SDN6) + mae(
            y_, y_SDN7) + mae(y_, y_SDN8) + mae(y_, y_SDN9) + mae(y_, y_SDN10) + mae(y_, y_SDN11) + mae(y_, y_SDN12) + mae(y_, y_SDN13) + mae(y_, y_SDN14) + mae(y_, y_SDN15) + mae(y_, y_SDN16)
        total_loss = 10 * t_loss + f_loss + a + tf.reduce_mean(tf.square(var))
        val_loss = mae(y_, y_SDN16)
    lr = tf.train.exponential_decay(lr_base,
                                    global_step=global_step,
                                    decay_steps=20000,
                                    decay_rate=lr_decay_rate,
                                    staircase=False)
    lr =0.001
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(lr).minimize(total_loss,global_step=global_step)


    saver = tf.train.Saver(max_to_keep=40)
    with tf.Session() as sess:


        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        print('begin to load data......')
        count_nochange = 0
        loss_val1 = 100.0
        start_time = time.time()

        for i in range(EPOCHS):
            loss_sum = 0.0
            count_batch = 0
            batch_generator = data_generator(f_xtrain, f_ytrain, f_masktrain, y_axis, x_axis, BATCH_SIZE)
            nb_num = nb_train_samples / BATCH_SIZE
            nb_batches = int(np.ceil((nb_train_samples) // BATCH_SIZE))
            for n_batch in range(nb_batches):
                input_data_batch, label_f_batch ,label_t_batch,mask_batch,dc_batch = next(batch_generator)
                _,loss_value, step = sess.run([train_step, total_loss, global_step],
                                                      feed_dict={x: input_data_batch, y_: label_f_batch, t_label: label_t_batch, x_dc: dc_batch, mask_k : mask_batch})

                loss_sum += loss_value
                count_batch += 1
                ave_loss = loss_sum / count_batch
                loss['batch'].append(ave_loss)
                print('Epoch %3d-batch %3d/%3d  training loss: %8f ' % (i + 1, count_batch, nb_batches, ave_loss))
            print('Begin to val......')
            nb_vals = int((nb_val) // BATCH_SIZE)
            loss_val = 0.0
            val_generator = TEST_generator(f_xval, f_yval, f_maskval, y_axis, x_axis, BATCH_SIZE)
            for n_batch in range(nb_vals):
                input_data_batch ,label_f_batch,label_t_batch, mask_batch, dc_batch = next(val_generator)
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
            # test every


def main(argv=None):
    train()

if __name__ == '__main__':
    main()