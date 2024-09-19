import torch
import torch.nn as nn

a = torch.rand(4, 8, 32, 32)
b = torch.rand(4, 8, 32, 32)

c = [a, b]
print(torch.stack(c, 1).shape)