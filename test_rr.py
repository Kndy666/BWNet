import cv2
import h5py
import torch
import numpy as np
import hdf5storage
import torch.nn as nn
from model.v8.bwnet import BWNet
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


ckpt = 'model/v8/1000.pth'  # sota


def test(file_path):

    model = BWNet().cuda().eval()  # 1, 8, 32 or 1, 8, 16
    weight = torch.load(ckpt)
    model.load_state_dict(weight)

    ms, _, pan, gt = load_h5py(file_path)
    B, C, H, W = gt.shape

    for k in range(B):
        with torch.no_grad():
            # x1, x2, = ms, pan
            MS, PAN, GT = ms[k, :, :, :], pan[k, 0, :, :], gt[k, :, :, :]
            PAN = PAN.cuda().unsqueeze(0).unsqueeze(1).float()
            MS = MS.cuda().unsqueeze(0).float()
            GT = GT.cuda().unsqueeze(0).float()
            output0, output1, output2, output3 = model(MS, PAN)
            output = torch.cat([output0, output1, output2, output3], 0)
            output = torch.clamp(output, 0, 1)
            OUTPUT = torch.zeros(1, C, H, W).cuda()

            criterion = nn.L1Loss(size_average=True).cuda()

            print('==========================={}============================='.format(k+1))
            ################ original ###################
            total_loss = 0.
            for i in range(C):
                total_loss += criterion(output[3, i, :, :].unsqueeze(0).unsqueeze(0), GT[:, i, :, :].unsqueeze(0))
            print('w/o selection: {}'.format(total_loss))

            ################ selected ###################
            total_loss_selected = 0.
            for i in range(C):
                loss = torch.zeros(4)
                for j in range(4):
                    l1 = criterion(output[j, i, :, :].unsqueeze(0).unsqueeze(0), GT[:, i, :, :].unsqueeze(0))
                    loss[j] = l1
                print('output port: {}'.format(torch.argmin(loss)+1))
                total_loss_selected += loss[torch.argmin(loss)]
                OUTPUT[:, i, :, :] = output[torch.argmin(loss), i, :, :].unsqueeze(0)
            print('with selection: {}'.format(total_loss_selected))

            OUTPUT = output[3, :, :, :].unsqueeze(0)
            OUTPUT = torch.squeeze(OUTPUT).permute(1, 2, 0).cpu().detach().numpy() * 2047.  # HxWxC
            save_name = os.path.join(
                "D:/Study/pansharpening_new_data/02-Test-toolbox-for-traditional-and-DL(Matlab)/2_DL_Result/WV2_Reduced/BWNET_wos/results",
                "output_mulExm_" + str(k) + ".mat")
            sio.savemat(save_name, {'sr': OUTPUT})


if __name__ == '__main__':
    # file_path = "D:/Study/pansharpening_new_data/reduced_examples_wv3/Test(HxWxC)_wv3_data"
    file_path = 'D:/Study/pansharpening_new_data/02-Test-toolbox-for-traditional-and-DL(Matlab)/1_TestData/PanCollection/test_wv2_multiExm1.h5'
    test(file_path)
