import torch
import torch.nn as nn
import numpy as np

x = torch.from_numpy(np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]], dtype=np.float32))

a = torch.norm(x, p=2, dim=-1, keepdim=True)
a = a @ a.transpose(-2, -1)
print(a)