import numpy as np
from scipy.sparse import csr_matrix
from typing import Union

DType = np.dtype
Number = Union[int, float, np.number]
DenseMatrix = np.ndarray
Matrix = Union[np.ndarray, csr_matrix]

# Add torch.Tensor and torch.dtype to matrix types if torch is available
try:
    import torch
    DTypes = Union[DType, torch.dtype]
    DenseMatrix = Union[DenseMatrix, torch.Tensor]
    Matrix = Union[Matrix, torch.Tensor]
    TORCH_FLOAT_TYPE = torch.float32
except ImportError:
    pass

# TODO: remove these hardcoded types
IND_TYPE = np.int32  # type of indices in CSR matrix
INT_TYPE = np.int32
FLOAT_TYPE = np.float32

