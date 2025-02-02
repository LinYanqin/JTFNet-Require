import numpy as np
import scipy.io as sio
import nmrglue as ng
import os
def postprocess_data(x_axis, y_axis, z_axis,data_path, num_iter):
    folder_restemp = "./rec_temp"
    ale_3D = np.zeros((x_axis, y_axis, z_axis))
    name_ale1 = folder_restemp + '/ale1.mat'
    name_res1 = folder_restemp + '/res1.mat'
    ale_data1 = sio.loadmat(name_ale1)['ale']
    res_data1 = sio.loadmat(name_res1)['res']
    ale_data1 = np.transpose(ale_data1, (1, 2, 0))
    res_data1 = np.transpose(res_data1, (2, 3, 1, 0))
    name_ale2 = folder_restemp + '/ale2.mat'
    name_res2 = folder_restemp + '/res2.mat'
    ale_data2 = sio.loadmat(name_ale2)['ale']
    res_data2 = sio.loadmat(name_res2)['res']
    ale_data2 = np.transpose(ale_data2, (1, 2, 0))
    res_data2 = np.transpose(res_data2, (2, 3, 1, 0))
    for i in range(num_iter):
        res_3D = np.zeros((x_axis, y_axis, z_axis))
        for k in range(z_axis):
            temp = np.fft.ifft2(np.fft.ifftshift(res_data1[:, :, k, i]))
            res_3D[::2, ::2, k] = np.real(temp)
            res_3D[::2, 1::2, k] = np.imag(temp)
            temp = np.fft.ifft2(np.fft.ifftshift(res_data2[:, :, k, i]))
            res_3D[1::2, ::2, k] = np.real(temp)
            res_3D[1::2, 1::2, k] = np.imag(temp)
        name_res = f'./rec_temp/res_full{i+1}.mat'
        sio.savemat(name_res, {'res_3D': res_3D})
    for k in range(z_axis):
        temp = np.fft.ifft2(np.fft.ifftshift(ale_data1[:, :, k]))
        ale_3D[::2, ::2, k] = np.real(temp)
        ale_3D[::2, 1::2, k] = np.imag(temp)
        temp = np.fft.ifft2(np.fft.ifftshift(ale_data2[:, :, k]))
        ale_3D[1::2, ::2, k] = np.real(temp)
        ale_3D[1::2, 1::2, k] = np.imag(temp)
    name_ale = folder_restemp + '/ale_full.mat'
    sio.savemat(name_ale, {'ale_3D': ale_3D})

    folder_rec = "./rec_data"
    if not os.path.exists(folder_rec):
        os.makedirs(folder_rec)
        print(f"Folder {folder_rec} is not exit，begin to bulit this folder")
    else:
        print(f"Folder {folder_rec} exit")
    dic, data = ng.pipe.read(data_path)

    print(data.ndim)
    print(data.shape)
    print(data.dtype)

    M = sio.loadmat(folder_restemp + '/ale_full.mat')
    Data = M['ale_3D']
    Data = np.array(Data,dtype='float32')
    ng.pipe.write(folder_rec + "/ale_full.dat", dic, Data, overwrite=True)

    for i in range(num_iter):
        M = sio.loadmat(folder_restemp + '/res_full%d.mat'%int(i+1))
        Data = M['res_3D']
        Data = np.array(Data, dtype='float32')
        ng.pipe.write(folder_rec + "/res_full%d.dat"%int(i+1), dic, Data, overwrite=True)

def postprocess_sim(x_axis, y_axis, z_axis,data_path,read_path,save_path, num):
    ale_3D = np.zeros((x_axis, y_axis, z_axis))
    name_ale1 = read_path + '/ale1.mat'
    name_res1 = read_path + '/res1.mat'
    ale_data1 = sio.loadmat(name_ale1)['ale']
    res_data1 = sio.loadmat(name_res1)['res']
    ale_data1 = np.transpose(ale_data1, (1, 2, 0))
    res_data1 = np.transpose(res_data1, (2, 3, 1, 0))
    name_ale2 = read_path + '/ale2.mat'
    name_res2 = read_path + '/res2.mat'
    ale_data2 = sio.loadmat(name_ale2)['ale']
    res_data2 = sio.loadmat(name_res2)['res']
    ale_data2 = np.transpose(ale_data2, (1, 2, 0))
    res_data2 = np.transpose(res_data2, (2, 3, 1, 0))
    for i in range(5):
        res_3D = np.zeros((x_axis, y_axis, z_axis))
        for k in range(z_axis):
            temp = np.fft.ifft2(np.fft.ifftshift(res_data1[:, :, k, i]))
            res_3D[::2, ::2, k] = np.real(temp)
            res_3D[::2, 1::2, k] = np.imag(temp)
            temp = np.fft.ifft2(np.fft.ifftshift(res_data2[:, :, k, i]))
            res_3D[1::2, ::2, k] = np.real(temp)
            res_3D[1::2, 1::2, k] = np.imag(temp)
        name_res = f'{read_path}/res_full{i+1}.mat'
        sio.savemat(name_res, {'res_3D': res_3D})
    for k in range(z_axis):
        temp = np.fft.ifft2(np.fft.ifftshift(ale_data1[:, :, k]))
        ale_3D[::2, ::2, k] = np.real(temp)
        ale_3D[::2, 1::2, k] = np.imag(temp)
        temp = np.fft.ifft2(np.fft.ifftshift(ale_data2[:, :, k]))
        ale_3D[1::2, ::2, k] = np.real(temp)
        ale_3D[1::2, 1::2, k] = np.imag(temp)
    name_ale = read_path + '/ale_full.mat'
    sio.savemat(name_ale, {'ale_3D': ale_3D})

    dic, data = ng.pipe.read(data_path)

    # print(data.ndim)
    # print(data.shape)
    # print(data.dtype)
    folder_path  = save_path + "rec_" + str(num+1)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    M = sio.loadmat(read_path + '/ale_full.mat')
    Data = M['ale_3D']
    Data = np.array(Data,dtype='float32')
    ng.pipe.write(folder_path + "/ale_full.dat", dic, Data, overwrite=True)

    for i in range(5):
        M = sio.loadmat(read_path + '/res_full%d.mat'%int(i+1))
        Data = M['res_3D']
        Data = np.array(Data, dtype='float32')
        ng.pipe.write(folder_path + "/res_full%d.dat"%int(i+1), dic, Data, overwrite=True)