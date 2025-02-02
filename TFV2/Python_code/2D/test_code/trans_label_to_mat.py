import nmrglue as ng
import scipy.io as sio
import numpy as np

data_path = "./label_path/"

dic, data = ng.pipe.read(data_path + "label.dat")
    #
print(data.ndim)
print(data.shape)
print(data.dtype)
    #
    #
data = np.fft.fft(data)
data = data / np.max(np.real(data))
sio.savemat(data_path+'/label2D.mat',{'label2D':data})