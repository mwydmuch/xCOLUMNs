from typing import Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix


DType = np.dtype
Number = Union[int, float, np.number]
DenseMatrix = np.ndarray
Matrix = Union[np.ndarray, csr_matrix]
CSRMatrixAsTuple = Tuple[np.ndarray, np.ndarray, np.ndarray]
DefaultIndDType = np.int32
DefaultDataDType = np.float32
FLOAT_TYPE = np.float32

# Add torch.Tensor and torch.dtype to matrix types if torch is available
TORCH_AVAILABLE = False
try:
    import torch

    DType = Union[DType, torch.dtype]
    DenseMatrix = Union[DenseMatrix, torch.Tensor]
    Matrix = Union[Matrix, torch.Tensor]
    TORCH_FLOAT_TYPE = torch.float32
    TORCH_AVAILABLE = True
except ImportError:
    pass
