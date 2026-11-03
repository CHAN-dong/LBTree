"""Sparse multiplication with a cached transpose for immutable loss inputs."""
import torch


class _FixedSparseMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix, transpose, vector):
        ctx.save_for_backward(transpose)
        return torch.sparse.mm(matrix, vector)

    @staticmethod
    def backward(ctx, gradient):
        (transpose,) = ctx.saved_tensors
        return None, None, torch.sparse.mm(transpose, gradient)


def fixed_sparse_mm(matrix, vector, transpose=None):
    if transpose is None or matrix.requires_grad:
        return torch.sparse.mm(matrix, vector)
    return _FixedSparseMM.apply(matrix, transpose, vector)
