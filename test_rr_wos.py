import cv2
import h5py
import torch
import numpy as np
import hdf5storage
import torch.nn as nn
from model.v10.bwnet import BWNet  ##################### 改成自己的模型 #################
import scipy.io as sio
import os


def get_edge(data):  # get high-frequency
    rs = np.zeros_like(data)
    if len(rs.shape) == 3:
        for i in range(data.shape[2]):
            rs[:, :, i] = data[:, :, i] - cv2.boxFilter(data[:, :, i], -1, (5, 5))
    else:
        rs = data - cv2.boxFilter(data, -1, (5, 5))
    return rs


def load_set(file_path):
    data = sio.loadmat(file_path)  # HxWxC=256x256x8
    ms = torch.from_numpy(data['ms'] / 2047.0)  # HxWxC = 256x256x8
    ms = ms.permute(2, 0, 1)
    lms = torch.from_numpy(data['lms'] / 2047.0)  # HxWxC = 256x256x8
    lms = lms.permute(2, 0, 1)
    pan = torch.from_numpy(data['pan'] / 2047.0)   # HxW = 256x256

    return ms, lms, pan


def load_set_hp(file_path):
    data = sio.loadmat(file_path)  # HxWxC=256x256x8
    ms = torch.from_numpy(data['ms'] / 2047.0).permute(2, 0, 1)  # CxHxW = 8x256x256
    ms_hp = torch.from_numpy(get_edge(data['ms'] / 2047.0)).permute(2, 0, 1)  # CxHxW= 8x64x64
    pan = torch.from_numpy(data['pan'] / 2047.0)  # HxW = 256x256
    pan_hp = torch.from_numpy(get_edge(data['pan'] / 2047.0))   # HxW = 256x256

    return ms, ms_hp, pan, pan_hp


def load_h5py(file_path):
    data = h5py.File(file_path)
    ms = data["ms"][...]  # W H C N
    ms = np.array(ms, dtype=np.float32) / 2047.
    ms = torch.from_numpy(ms)
    lms = data["lms"][...]  # W H C N
    lms = np.array(lms, dtype=np.float32) / 2047.
    lms = torch.from_numpy(lms)
    pan = data["pan"][...]  # W H C N
    pan = np.array(pan, dtype=np.float32) / 2047.
    pan = torch.from_numpy(pan)
    gt = data["gt"][...]  # W H C N
    gt = np.array(gt, dtype=np.float32) / 2047.
    gt = torch.from_numpy(gt)

    return ms, lms, pan, gt


ckpt = 'model/v10/1000.pth'  ##################### 改成自己的模型参数 #################


def test(file_path, save_path, cut=False):

    model = BWNet().cuda().eval()  ##################### 改成自己的模型 #################
    weight = torch.load(ckpt)
    model.load_state_dict(weight)

    ms, _, pan, gt = load_h5py(file_path)
    B, C, H, W = gt.shape

    if cut == False:
        for k in range(B):
            with torch.no_grad():
                # x1, x2, = ms, pan
                MS, PAN, GT = ms[k, :, :, :], pan[k, 0, :, :], gt[k, :, :, :]
                PAN = PAN.cuda().unsqueeze(0).unsqueeze(1).float()
                MS = MS.cuda().unsqueeze(0).float()
                GT = GT.cuda().unsqueeze(0).float()
                output = model(MS, PAN)
                output = torch.clamp(output, 0, 1)

                output = torch.squeeze(output).permute(1, 2, 0).cpu().detach().numpy() * 2047.  # HxWxC
                save_name = os.path.join(save_path, "output_mulExm_" + str(k) + ".mat")
                sio.savemat(save_name, {'sr': output})

    else:
        image_num = 20
        cut_size = 64  # 256
        ms_size = int(cut_size // 4)
        edge = 0  # must
        pad = 4

        for k in range(image_num):
            # load matlab
            # file_name = file_path + str(k+1) + '.mat'
            # _, ms, pan = load_set(file_name)
            # ms, _, pan = load_set(file_name)
            # ms, _, pan = load_h5py(file_path)

            with torch.no_grad():
                # x1, x2, = ms, pan
                x1, x2 = ms[k, :, :, :], pan[k, 0, :, :]
                x2 = x2.cuda().unsqueeze(dim=0).unsqueeze(dim=1).float()
                x1 = x1.cuda().unsqueeze(dim=0).float()
                _, _, H, W = x2.shape
                B, C, h, w = x1.shape

                x1_pad = torch.zeros(1, C, h + pad // 2 + edge // 4, w + pad // 2 + edge // 4).cuda()
                x2_pad = torch.zeros(1, 1, H + pad * 2 + edge, W + pad * 2 + edge).cuda()
                # x1_pad[:, :, pad // 4: h + pad // 4, pad // 4: w + pad // 4] = x1
                # x2_pad[:, :, pad: H + pad, pad: W + pad] = x2
                x1_pad = torch.nn.functional.pad(x1, (pad // 4, pad // 4, pad // 4, pad // 4), 'reflect')
                x2_pad = torch.nn.functional.pad(x2, (pad, pad, pad, pad), 'reflect')

                scale = int(H / cut_size)
                output = torch.zeros(B, C, H + edge, W + edge).cuda()
                for i in range(scale):
                    for j in range(scale):
                        PAN = x2_pad[:, :, i * cut_size: (i + 1) * cut_size + 2 * pad,
                              j * cut_size: (j + 1) * cut_size + 2 * pad]
                        MS = x1_pad[:, :, i * ms_size: (i + 1) * ms_size + pad // 2,
                             j * ms_size: (j + 1) * ms_size + pad // 2]
                        # _, _, _, sr = model(MS, PAN)
                        sr = model(MS, PAN)
                        sr = torch.clamp(sr, 0, 1)
                        output[:, :, i * cut_size:(i + 1) * cut_size, j * cut_size:(j + 1) * cut_size] = \
                            sr[:, :, pad: cut_size + pad, pad: cut_size + pad] * 2047.
                output = output[:, :, 0: H, 0: W]
                output = torch.squeeze(output).permute(1, 2, 0).cpu().detach().numpy()  # HxWxC
                save_name = os.path.join(save_path, "output_mulExm_" + str(k) + ".mat")
                sio.savemat(save_name, {'sr': output})




if __name__ == '__main__':
    # file_path = "D:/Study/pansharpening_new_data/reduced_examples_wv3/Test(HxWxC)_wv3_data"
    file_path = 'D:/Study/pansharpening_new_data/02-Test-toolbox-for-traditional-and-DL(Matlab)/1_TestData/PanCollection/test_wv2_multiExm1.h5'  ##################### 改成自己的地址 #################
    save_path = 'D:/Study/pansharpening_new_data/02-Test-toolbox-for-traditional-and-DL(Matlab)/2_DL_Result/WV2_Reduced/BWNET_New/results'  ##################### 改成自己的地址 #################
    test(file_path, save_path, cut=False)
