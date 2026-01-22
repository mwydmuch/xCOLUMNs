from typing import Callable, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix


DType = np.dtype
Number = Union[int, float, np.number]
DenseMatrix = np.ndarray
Matrix = Union[np.ndarray, csr_matrix]
CSRMatrixAsTuple = Tuple[np.ndarray, np.ndarray, np.ndarray]
DefaultIndDType = np.int32
DefaultDataDType = np.float32
DefaultAccDataDType = np.float64

# Add torch.Tensor and torch.dtype to matrix types if torch is available
TORCH_AVAILABLE = False
try:
    import torch

    DType = Union[DType, torch.dtype]
    DenseMatrix = Union[np.ndarray, torch.Tensor]
    Matrix = Union[np.ndarray, csr_matrix, torch.Tensor]
    DefaultTorchDataDType = torch.float32
    TORCH_AVAILABLE = True
except ImportError:
    pass
