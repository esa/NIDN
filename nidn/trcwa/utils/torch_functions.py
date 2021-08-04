"""Utility functions to make torch operations compatible with numpy syntax."""
import torch


def torch_zeros(shape, dtype):
    """Implementation of torch.zeros that returns a complex tensor. Else falls back on torch.zeros.

    Args:
        shape (list): Shape vector for zeros tensor.
        dtype (dtype): Requested datatype.

    Returns:
        torch.tensor: Initialized tensor with real and imaginary part set to zero
    """
    if dtype == complex:
        real = torch.zeros(shape)
        imag = torch.zeros(shape)
        return torch.complex(real, imag)
    else:
        return torch.zeros(shape, dtype=dtype)


def torch_eye(size, dtype=None):
    """Implementation of torch.eye that returns a complex tensor. Else falls back on torch.eye.

    Args:
        shape (list): Shape vector for identity matrix.
        dtype (dtype): Requested datatype.

    Returns:
        torch.tensor: Initialized tensor with real part of the diagonal set to one and imaginary part of the diagonal set to zero, and zeros elsewhere
    """
    if dtype == complex:
        real = torch.eye(size)
        imag = torch.zeros([size, size])
        return torch.complex(real, imag)
    else:
        return torch.eye(size, dtype=dtype)


def torch_transpose(n_dimensional_tensor):
    """Custom transpose for torch as torch.transpose diverges from numpy, see https://github.com/pytorch/pytorch/issues/51280

    Args:
        n_dimensional_tensor (torch.tensor): Tensor to transpose.

    Returns:
        torch.tensor: Transposed tensor
    """
    # Construct reverse index (e.g. [3,2,1,0])
    indices = [
        len(n_dimensional_tensor.shape) - i - 1
        for i in range(len(n_dimensional_tensor.shape))
    ]
    return n_dimensional_tensor.permute(indices)


def torch_dot(a, b):
    """Computes a dot product in torch that is compatible with numpy style syntax.

    Args:
        a (torch.tensor): Tensor a in the dot product of a and b.
        b (torch.tensor): Tensor b in the dot product of a and b.

    Returns:
        torch.tensor: Dot product of a and b
    """
    if a.dtype != b.dtype:  # in case data types are different
        if a.dtype == torch.cfloat or a.dtype == torch.cdouble:
            b = b.type(a.dtype)
        elif b.dtype == torch.cfloat or b.dtype == torch.cdouble:
            a = a.type(b.dtype)
    # to be compatible with numpy broadcasting
    if len(a.shape) == 2 and len(b.shape) == 1:
        return torch.mv(a, b)
    if len(a.shape) > 1:
        return torch.mm(a, b)
    else:
        return torch.dot(a, b)
