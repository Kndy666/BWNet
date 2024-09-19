import torch.utils.data as data
import numpy as np
import torch
import h5py
import cv2


def get_edge(data):
    rs = np.zeros_like(data)
    N = data.shape[0]
    for i in range(N):
        if len(data.shape) == 3:
            rs[i, :, :] = data[i, :, :] - cv2.boxFilter(data[i, :, :], -1, (5, 5))
        else:
            rs[i, :, :, :] = data[i, :, :, :] - cv2.boxFilter(data[i, :, :, :], -1, (5, 5))
    return rs


class Dataset_Pro(data.Dataset):
    def __init__(self, file_path):
        super(Dataset_Pro, self).__init__()
        data = h5py.File(file_path)

        # tensor type:
        gt = data["gt"][...]
        gt = np.array(gt, dtype=np.float32) / 2047.
        self.gt = torch.from_numpy(gt)

        lms = data["lms"][...]
        lms = np.array(lms, dtype=np.float32) / 2047.
        self.lms = torch.from_numpy(lms)

        ms = data["ms"][...]  # NxCxHxW
        self.ms = torch.from_numpy(np.array(ms, dtype=np.float32) / 2047.)

        ms_hp = np.array(ms.transpose(0, 2, 3, 1), dtype=np.float32) / 2047.  # NxHxWxC
        ms_hp = get_edge(ms_hp)
        self.ms_hp = torch.from_numpy(ms_hp).permute(0, 3, 1, 2)

        pan = data['pan'][...]
        self.pan = torch.from_numpy(np.array(pan, dtype=np.float32) / 2047.)

        pan_hp = np.array(pan.transpose(0, 2, 3, 1), dtype=np.float32) / 2047.  # NxHxWx1
        pan_hp = np.squeeze(pan_hp, axis=3)  # NxHxW
        pan_hp = get_edge(pan_hp)  # NxHxW
        pan_hp = np.expand_dims(pan_hp, axis=3)  # NxHxWx1
        self.pan_hp = torch.from_numpy(pan_hp).permute(0, 3, 1, 2)  # Nx1xHxW:

    def __getitem__(self, index):
        return self.gt[index, :, :, :].float(), \
               self.lms[index, :, :, :].float(), \
               self.pan[index, :, :, :].float(), \
               self.pan_hp[index, :, :, :].float(), \
               self.ms[index, :, :, :].float(), \
               self.ms_hp[index, :, :, :].float()

    def __len__(self):
        return self.gt.shape[0]