import cv2
import h5py
import torch
import numpy as np
import hdf5storage
from FusionNet import FusionNet
from BWNet5 import BWNet_FusionNet
import scipy.io as sio
import os


def get_edge(data):
    rs = np.zeros_like(data)
    N = data.shape[0]
    for i in range(N):
        if len(data.shape) == 3:
            rs[i, :, :] = data[i, :, :] - cv2.boxFilter(data[i, :, :], -1, (5, 5))
        else:
            rs[i, :, :, :] = data[i, :, :, :] - cv2.boxFilter(data[i, :, :, :], -1, (5, 5))
    return rs


def load_mat(file_path):
    data = sio.loadmat(file_path)  # HxWxC=256x256x8

    ms = torch.from_numpy(data['ms'] / 2047.0)  # HxWxC = 256x256x8
    ms = ms.permute(2, 0, 1)
    pan = torch.from_numpy(data['pan'] / 2047.0)   # HxW = 256x256

    return ms, pan


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

    return ms, lms, pan


def load_h5py_hp(file_path):
    data = h5py.File(file_path)

    ms1 = data["ms"][...]
    ms = np.array(ms1, dtype=np.float32) / 2047.
    ms = torch.from_numpy(ms)

    ms_hp = np.array(ms1.transpose(0, 2, 3, 1), dtype=np.float32) / 2047.  # NxHxWxC
    ms_hp = get_edge(ms_hp)
    ms_hp = torch.from_numpy(ms_hp).permute(0, 3, 1, 2)

    pan1 = data["pan"][...]  # W H C N
    pan = np.array(pan1, dtype=np.float32) / 2047.
    pan = torch.from_numpy(pan)

    pan_hp = np.array(pan1.transpose(0, 2, 3, 1), dtype=np.float32) / 2047.  # NxHxWx1
    pan_hp = np.squeeze(pan_hp, axis=3)  # NxHxW
    pan_hp = get_edge(pan_hp)  # NxHxW
    pan_hp = np.expand_dims(pan_hp, axis=3)  # NxHxWx1
    pan_hp = torch.from_numpy(pan_hp).permute(0, 3, 1, 2)

    lms = data["lms"][...]
    lms = np.array(lms, dtype=np.float32) / 2047.
    lms = torch.from_numpy(lms)

    return ms, ms_hp, pan, pan_hp, lms


ckpt = '350_BWNet.pth'


def test(file_path, save_path):

    model = BWNet_FusionNet().cuda().eval()
    weight = torch.load(ckpt)
    model.load_state_dict(weight)

    ms, ms_hp, pan, pan_hp, lms = load_h5py_hp(file_path)
    B = ms_hp.shape[0]

    for k in range(B):
        with torch.no_grad():
            # x1, x2, = ms, pan
            MS, PAN, LMS = ms[k, :, :, :], pan[k, :, :, :], lms[k, :, :, :]
            PAN = PAN.cuda().unsqueeze(0).float()
            MS = MS.cuda().unsqueeze(0).float()
            LMS = LMS.cuda().unsqueeze(0).float()

            output = model(LMS, PAN)
            output = torch.clamp(output, 0, 1)

            output = torch.squeeze(output).permute(1, 2, 0).cpu().detach().numpy() * 2047.  # HxWxC
            save_name = os.path.join(save_path, "output_mulExm_" + str(k) + ".mat")
            sio.savemat(save_name, {'sr': output})


if __name__ == '__main__':
    file_path = 'D:/Study/pansharpening_new_data/02-Test-toolbox-for-traditional-and-DL(Matlab)/1_TestData/PanCollection/test_wv3_OrigScale_multiExm1.h5'
    save_path = 'D:/Study/pansharpening_new_data/02-Test-toolbox-for-traditional-and-DL(Matlab)/2_DL_Result/WV3_Full/BWNet/results'
    test(file_path, save_path)
