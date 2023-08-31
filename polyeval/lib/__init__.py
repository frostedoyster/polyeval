import os
import sysconfig
import torch

from .polyeval import reference_implementation

_HERE = os.path.realpath(os.path.dirname(__file__))

torch.ops.load_library(_HERE + '/polyeval_cc.so')

if torch.cuda.is_available():
    torch.ops.load_library(_HERE + '/polyeval_cuda.so')

def optimized_implementation(nu1_basis, indices, multipliers):

    if nu1_basis.device != indices.device:
        raise ValueError("All tensors must be on the same device")
    if nu1_basis.device != indices.device:
        raise ValueError("All tensors must be on the same device")
    if nu1_basis.dtype != multipliers.dtype:
        raise ValueError("The two float tensors must have the same dtype")

    if nu1_basis.is_cuda:
        result = torch.ops.polyeval_cu.polyeval(nu1_basis, indices.T.contiguous(), multipliers, 128, 2)
    else:
        result = torch.ops.polyeval_cc.polyeval(nu1_basis, indices, multipliers)

    return result
