import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from einops import rearrange, repeat


a = torch.rand(1, 8, 8)
print(a)
a1 = a.softmax(dim=-1)
print(a1)
b = (a * 16)
print(b)
b = b.softmax(dim=-1)
print(b)

print('done')