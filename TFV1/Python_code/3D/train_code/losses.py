import tensorflow as tf
import numpy as np
from model3D_1 import real2complex


def mse(y_true, y_pred):
    y_pred_complex = real2complex(y_pred)
    y_true_complex = real2complex(y_true)
    diff = tf.square(tf.abs(y_pred_complex - y_true_complex))
    #loss = tf.reduce_mean(tf.reduce_mean(diff,[1,2,3]))
    return diff

def mae(y_true, y_pred):
    y_pred_complex = real2complex(y_pred)
    y_true_complex = real2complex(y_true)
    diff = tf.abs(y_pred_complex - y_true_complex)
    loss = tf.reduce_mean(tf.reduce_mean(diff,[1,2,3]))

    return loss

# def mae(y_true, y_pred):
#
#     diff0 = tf.abs(y_pred[:,:,:,0]- y_true[:,:,:,0])
#     diff1 = tf.abs(y_pred[:, :, :, 1] - y_true[:, :, :, 1])
#     loss0 = tf.reduce_mean(tf.reduce_mean(diff0,[1,2]))
#     loss1 = tf.reduce_mean(tf.reduce_mean(diff1, [1, 2]))
#
#     return loss0+loss1

