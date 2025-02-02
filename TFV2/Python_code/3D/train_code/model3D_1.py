import tensorflow.compat.v1   as tf
from utils  import ComplexInit
import numpy as np
tf.disable_v2_behavior()

def complex_conv2d(input,name,dilation_rate,kw,kh,n_out,sw,sh,activation=True):
    n_in = input.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel_init = ComplexInit(kernel_size=(kh,kw),
                                  input_dim=n_in,
                                  weight_dim=2,
                                  nb_filters=n_out,
                                  criterion='he')
        kernel = tf.get_variable(scope + 'weights',
                                 shape=[kh,kw,n_in,n_out],
                                 dtype=tf.float32,
                                 initializer=kernel_init)
        bias_init = tf.constant(0.0001,dtype=tf.float32,shape=[n_out*2])
        biases = tf.get_variable(scope+'biases', dtype=tf.float32, initializer=bias_init)

        conv = tf.nn.conv2d(input,kernel,strides=[1,sh,sw,1],padding='SAME',dilations=[1, dilation_rate, dilation_rate, 1])
        conv_bias = tf.nn.bias_add(conv,biases)
        if activation:
            act = tf.nn.leaky_relu(conv_bias)
            output = act

        else:
            output = conv_bias


        return output

def Conv_transpose(x, name,filter_size, in_filters, out_filters, fraction=2, padding="SAME"):
    with tf.variable_scope(name):
        n = filter_size * filter_size * out_filters
        kernel = tf.get_variable('filter', [filter_size, filter_size, out_filters, in_filters], tf.float32,
                                 initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
        size = tf.shape(x)
        output_shape = tf.stack([size[0], size[1] * fraction, size[2] * fraction, out_filters])
        x = tf.nn.conv2d_transpose(x, kernel, output_shape, [1, fraction, fraction, 1], padding)

        return x

def real2complex(x):
    channel = x.shape[-1] // 2
    if x.shape.ndims == 3:
        return tf.complex(x[:,:,:channel], x[:,:,channel:])
    elif x.shape.ndims == 4:
        return tf.complex(x[:,:,:,:channel], x[:,:,:,channel:])

def complex2real(x):
    x_real = np.real(x)
    x_imag = np.imag(x)
    return np.concatenate([x_real,x_imag], axis=-1)

def dc(generated, X_k, mask,lambda1=8000):
    gene_complex = real2complex(generated)
    gene_complex = tf.transpose(gene_complex, [0, 3, 1, 2])
    mask = tf.transpose(mask, [0, 3, 1, 2])
    X_k = tf.transpose(X_k, [0, 3, 1, 2])
    gene_fft = tf.ifft2d(gene_complex)
    out_fft = ((X_k*mask)*lambda1 + (gene_fft*mask))/(1+lambda1) + (gene_fft * (1.0 - mask))
    output_complex = tf.fft2d(out_fft)
    output_complex = tf.transpose(output_complex, [0, 2, 3, 1])
    output_real = tf.cast(tf.real(output_complex), dtype=tf.float32)
    output_imag = tf.cast(tf.imag(output_complex), dtype=tf.float32)
    output = tf.concat([output_real, output_imag], axis=-1)
    return output

def dc_tdomain(generated, X_k, mask,lambda1=8000):
    gene_t = real2complex(generated)
    out_fft = ((X_k*mask)*lambda1 + (gene_t*mask))/(1+lambda1) + (gene_t * (1.0 - mask))
    output_complex = out_fft
    output_real = tf.cast(tf.real(output_complex), dtype=tf.float32)
    output_imag = tf.cast(tf.imag(output_complex), dtype=tf.float32)
    output = tf.concat([output_real,output_imag], axis=-1)
    return output

def complex_real(x):
    output_real = tf.cast(tf.real(x), dtype=tf.float32)
    output_imag = tf.cast(tf.imag(x), dtype=tf.float32)
    output = tf.concat([output_real, output_imag], axis=-1)
    return output
#X_K是时域采样后的数据
def t2f(x):
    k_temp1 = real2complex(x)
    k_temp = tf.transpose(k_temp1, [0, 3, 1, 2])
    temp = tf.fft2d(k_temp)
    temp = tf.transpose(temp, [0, 2, 3, 1])
    temp_1 = complex_real(temp)
    return temp_1
def f2t(x):
    k_temp1 = real2complex(x)
    k_temp = tf.transpose(k_temp1, [0, 3, 1, 2])
    temp = tf.ifft2d(k_temp)
    temp = tf.transpose(temp, [0, 2, 3, 1])
    temp_1 = complex_real(temp)
    return temp_1

def t_module(nm, x, x_dc, mask):
    conv1_SDN0 = complex_conv2d(x, nm + 'Tconv1', 3, kw=3, kh=3, n_out=16, sw=2, sh=2, activation=True)
    conv1_dropout = tf.nn.dropout(conv1_SDN0, rate=1-0.9)
    conv2_SDN0 = complex_conv2d(conv1_dropout, nm + 'Tconv2', 2, kw=3, kh=3, n_out=32, sw=2, sh=2, activation=True)
    conv2_dropout = tf.nn.dropout(conv2_SDN0, rate=1-0.9)
    conv3_SDN0 = complex_conv2d(conv2_dropout, nm + 'Tconv3', 1, kw=3, kh=3, n_out=64, sw=2, sh=2, activation=True)
    conv3_dropout = tf.nn.dropout(conv3_SDN0, rate=1-0.9)
    deconv3_SDN0 = Conv_transpose(conv3_dropout, nm + 'Tdeconv3', filter_size=3, in_filters=128, out_filters=64)
    SDN0_up2 = tf.concat([deconv3_SDN0, conv2_SDN0], axis=3, name=nm + 'TDN0_up2')
    deconv2_SDN0 = Conv_transpose(SDN0_up2, nm + 'Tdeconv2', filter_size=3, in_filters=128, out_filters=32)
    SDN0_up1 = tf.concat([deconv2_SDN0, conv1_SDN0], axis=3, name=nm + 'TDN0_up1')
    deconv1_SDN0 = Conv_transpose(SDN0_up1, nm + 'Tdeconv1', filter_size=3, in_filters=64, out_filters=2)
    block_SDN0 = deconv1_SDN0 + x
    k_temp_real = dc_tdomain(block_SDN0, x_dc, mask)
    return k_temp_real

def f_module(nm, x, x_dc, mask):
    conv1_SDN1 = complex_conv2d(x, nm + 'Sconv1', 1, kw=3, kh=3, n_out=16, sw=2, sh=2, activation=True)
    conv1_dropout = tf.nn.dropout(conv1_SDN1, rate=1-0.9)
    conv2_SDN1 = complex_conv2d(conv1_dropout, nm + 'Sconv2', 1, kw=3, kh=3, n_out=32, sw=2, sh=2, activation=True)
    conv2_dropout = tf.nn.dropout(conv2_SDN1, rate=1-0.9)
    conv3_SDN1 = complex_conv2d(conv2_dropout, nm + 'Sconv3', 1, kw=3, kh=3, n_out=64, sw=2, sh=2, activation=True)
    conv3_dropout = tf.nn.dropout(conv3_SDN1, rate=1-0.9)
    deconv3_SDN1 = Conv_transpose(conv3_dropout, nm + 'Sdeconv3', filter_size=3, in_filters=128, out_filters=64)
    SDN1_up2 = tf.concat([deconv3_SDN1, conv2_SDN1], axis=3, name=nm + 'SDN1_up2')
    deconv2_SDN1 = Conv_transpose(SDN1_up2, nm + 'Sdeconv2', filter_size=3, in_filters=128, out_filters=32)
    SDN1_up1 = tf.concat([deconv2_SDN1, conv1_SDN1], axis=3, name= nm + 'SDN1_up1')
    deconv1_SDN1 = Conv_transpose(SDN1_up1, nm + 'Sdeconv1', filter_size=3, in_filters=64, out_filters=2)
    block_SDN1 = deconv1_SDN1 + x
    temp_SDN1 = dc(block_SDN1, x_dc, mask)
    return temp_SDN1


def getModel(x, x_dc, mask):

    T1_out = t_module('T1', x, x_dc, mask)
    F1_out = f_module('F1', t2f(x), x_dc, mask)
    input_t = (T1_out + f2t(F1_out)) / 2
    input_f = (F1_out + t2f(T1_out)) / 2
    T2_out = t_module('T2', input_t, x_dc, mask)
    F2_out = f_module('F2', input_f, x_dc, mask)
    input_t = (T2_out + f2t(F2_out)) / 2
    input_f = (F2_out + t2f(T2_out)) / 2
    T3_out = t_module('T3', input_t, x_dc, mask)
    F3_out = f_module('F3', input_f, x_dc, mask)
    input_t = (T3_out + f2t(F3_out)) / 2
    input_f = (F3_out + t2f(T3_out)) / 2
    T4_out = t_module('T4', input_t, x_dc, mask)
    F4_out = f_module('F4', input_f, x_dc, mask)
    input_t = (T4_out + f2t(F4_out)) / 2
    input_f = (F4_out + t2f(T4_out)) / 2
    T5_out = t_module('T5', input_t, x_dc, mask)
    F5_out = f_module('F5', input_f, x_dc, mask)
    input_t = (T5_out + f2t(F5_out)) / 2
    input_f = (F5_out + t2f(T5_out)) / 2
    T6_out = t_module('T6', input_t, x_dc, mask)
    F6_out = f_module('F6', input_f, x_dc, mask)
    input_t = (T6_out + f2t(F6_out)) / 2
    input_f = (F6_out + t2f(T6_out)) / 2
    T7_out = t_module('T7', input_t, x_dc, mask)
    F7_out = f_module('F7', input_f, x_dc, mask)
    input_t = (T7_out + f2t(F7_out)) / 2
    input_f = (F7_out + t2f(T7_out)) / 2
    T8_out = t_module('T8', input_t, x_dc, mask)
    F8_out = f_module('F8', input_f, x_dc, mask)
    input_t = (T8_out + f2t(F8_out)) / 2
    input_f = (F8_out + t2f(T8_out)) / 2
    T9_out = t_module('T9', input_t, x_dc, mask)
    F9_out = f_module('F9', input_f, x_dc, mask)
    input_t = (T9_out + f2t(F9_out)) / 2
    input_f = (F9_out + t2f(T9_out)) / 2
    T10_out = t_module('T10', input_t, x_dc, mask)
    F10_out = f_module('F10', input_f, x_dc, mask)
    input_t = (T10_out + f2t(F10_out)) / 2
    input_f = (F10_out + t2f(T10_out)) / 2
    T11_out = t_module('T11', input_t, x_dc, mask)
    F11_out = f_module('F11', input_f, x_dc, mask)
    input_t = (T11_out + f2t(F11_out)) / 2
    input_f = (F11_out + t2f(T11_out)) / 2
    T12_out = t_module('T12', input_t, x_dc, mask)
    F12_out = f_module('F12', input_f, x_dc, mask)
    input_t = (T12_out + f2t(F12_out)) / 2
    input_f = (F12_out + t2f(T12_out)) / 2
    T13_out = t_module('T13', input_t, x_dc, mask)
    F13_out = f_module('F13', input_f, x_dc, mask)
    input_t = (T13_out + f2t(F13_out)) / 2
    input_f = (F13_out + t2f(T13_out)) / 2
    T14_out = t_module('T14', input_t, x_dc, mask)
    F14_out = f_module('F14', input_f, x_dc, mask)
    input_f = (F14_out + t2f(T14_out)) / 2


    conv1_SDN10 = complex_conv2d(input_f, 'conv1_SDN10',1, kw=3, kh=3, n_out=16, sw=2, sh=2, activation=True)
    conv2_SDN10 = complex_conv2d(conv1_SDN10, 'conv2_SDN10', 1, kw=3, kh=3, n_out=32, sw=2, sh=2, activation=True)
    conv3_SDN10 = complex_conv2d(conv2_SDN10, 'conv3_SDN10', 1, kw=3, kh=3, n_out=64, sw=2, sh=2, activation=True)
    deconv3_SDN10 = Conv_transpose(conv3_SDN10, 'deconv3_SDN10', 3, in_filters=128, out_filters=64)
    SDN10_up2 = tf.concat([deconv3_SDN10, conv2_SDN10], axis=3, name='SDN10_up2')
    deconv2_SDN10 = Conv_transpose(SDN10_up2, 'deconv2_SDN10', 3, in_filters=128, out_filters=32)
    SDN10_up1 = tf.concat([deconv2_SDN10, conv1_SDN10], axis=3, name='SDN10_up1')
    deconv1_SDN10 = Conv_transpose(SDN10_up1, 'deconv1_SDN10', 3, in_filters=64, out_filters=2)
    block_SDN10 = deconv1_SDN10 + input_f
    temp_SDN10 = dc(block_SDN10, x_dc, mask)


    conv1_out1 = complex_conv2d(temp_SDN10, 'conv_final1', 1,kw=3, kh=3, n_out=32, sw=1, sh=1, activation=True)
    conv1_out2 = complex_conv2d(conv1_out1, 'conv_final2', 1,kw=3, kh=3, n_out=32, sw=1, sh=1, activation=True)
    conv1_out3 = complex_conv2d(conv1_out2, 'conv_final3', 1,kw=3, kh=3, n_out=16, sw=1, sh=1, activation=True)
    conv1_out4 = complex_conv2d(conv1_out3, 'conv_final4', 1,kw=3, kh=3, n_out=16, sw=1, sh=1, activation=True)
    conv1_out5 = complex_conv2d(conv1_out4, 'conv_final5', 1,kw=3, kh=3, n_out=1, sw=1, sh=1, activation=True)
    conv1_out5 = dc(conv1_out5, x_dc, mask)

    deconv3_SDN8_var = Conv_transpose(conv3_SDN10, 'deconv3_SDN8_var', filter_size=3, in_filters=128, out_filters=64)
    deconv2_SDN8_var = Conv_transpose(deconv3_SDN8_var, 'deconv2_SDN8_var', filter_size=3, in_filters=64, out_filters=32)
    var = Conv_transpose(deconv2_SDN8_var, 'deconv1_SDN8_var', filter_size=3, in_filters=32, out_filters=2)
    return T1_out, T2_out, T3_out, T4_out, T5_out, T6_out, T7_out, T8_out, T9_out, T10_out, T11_out, T12_out, T13_out, T14_out, F1_out, F2_out, F3_out, F4_out, F5_out, F6_out, F7_out, F8_out, F9_out, F10_out, F11_out, F12_out, F13_out, F14_out, temp_SDN10, conv1_out5, var
