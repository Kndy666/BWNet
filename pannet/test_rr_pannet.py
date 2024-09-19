import cv2
import h5py
import torch
import numpy as np
import hdf5storage
import torch.nn as nn
from PanNet import PanNet
from BWNet3 import BWNet
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


def get_edge_h5py(data):  # get high-frequency
    rs = np.zeros_like(data)
    N = data.shape[0]
    for i in range(N):
        if len(data.shape) == 3:
            rs[i, :, :] = data[i, :, :] - cv2.boxFilter(data[i, :, :], -1, (5, 5))
        else:
            rs[i, :, :, :] = data[i, :, :, :] - cv2.boxFilter(data[i, :, :, :], -1, (5, 5))
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
    lms = torch.from_numpy(data['lms'] / 2047.0)  # HxWxC = 256x256x8
    lms = lms.permute(2, 0, 1)

    return ms, ms_hp, pan, pan_hp, lms


def load_h5py(file_path):
    data = h5py.File(file_path)
    ms = data["ms"][...]
    ms = np.array(ms, dtype=np.float32) / 2047.
    ms = torch.from_numpy(ms)
    lms = data["lms"][...]
    lms = np.array(lms, dtype=np.float32) / 2047.
    lms = torch.from_numpy(lms)
    pan = data["pan"][...]
    pan = np.array(pan, dtype=np.float32) / 2047.
    pan = torch.from_numpy(pan)
    gt = data["gt"][...]
    gt = np.array(gt, dtype=np.float32) / 2047.
    gt = torch.from_numpy(gt)

    return ms, lms, pan, gt


def load_h5py_hp(file_path):
    data = h5py.File(file_path)
    ms = data["ms"][...]
    ms = np.array(ms.transpose(0, 2, 3, 1), dtype=np.float32) / 2047.
    ms_hp = torch.from_numpy(get_edge_h5py(ms).transpose(0, 3, 1, 2))
    ms = torch.from_numpy(ms)

    lms = data["lms"][...]
    lms = np.array(lms, dtype=np.float32) / 2047.
    lms = torch.from_numpy(lms)

    pan = data["pan"][...]
    pan = np.array(pan, dtype=np.float32) / 2047.
    pan_hp = np.squeeze(pan, axis=1)  # NxHxW
    pan_hp = np.expand_dims(get_edge_h5py(pan_hp), axis=1)  # Nx1xHxW
    pan_hp = torch.from_numpy(pan_hp)
    pan = torch.from_numpy(pan)

    gt = data["gt"][...]
    gt = np.array(gt, dtype=np.float32) / 2047.
    gt = torch.from_numpy(gt)

    return ms, ms_hp, lms, pan, pan_hp, gt


# ckpt = '950.pth'  # sota
ckpt = '950_BWNet.pth'

def test(file_path):

    # model = PanNet().cuda().eval()
    model = BWNet().cuda().eval()
    weight = torch.load(ckpt)
    model.load_state_dict(weight)

    _, ms_hp, lms, _, pan_hp, gt = load_h5py_hp(file_path)
    B = gt.shape[0]

    for k in range(B):
        with torch.no_grad():
            # x1, x2, = ms, pan
            MS, PAN, LMS = ms_hp[k, :, :, :], pan_hp[k, :, :, :], lms[k, :, :, :]
            PAN = PAN.cuda().unsqueeze(0).float()
            MS = MS.cuda().unsqueeze(0).float()
            LMS = LMS.cuda().unsqueeze(0).float()
            output = model(MS, PAN, LMS)
            output = torch.squeeze(output).permute(1, 2, 0).cpu().detach().numpy() * 2047.  # HxWxC
            save_name = os.path.join(
                "D:/Study/pansharpening_new_data/02-Test-toolbox-for-traditional-and-DL(Matlab)/2_DL_Result/WV2_Reduced/BWNet/results",
                "output_mulExm_" + str(k) + ".mat")
            sio.savemat(save_name, {'sr': output})


if __name__ == '__main__':
    # file_path = "D:/Study/pansharpening_new_data/reduced_examples_wv3/Test(HxWxC)_wv3_data"
    file_path = 'D:/Study/pansharpening_new_data/02-Test-toolbox-for-traditional-and-DL(Matlab)/1_TestData/PanCollection/test_wv2_multiExm1.h5'
    test(file_path)
