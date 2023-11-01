# -*- coding: utf-8 -*-
from packaging import version
import torch
import torchvision.utils as vutils


def sinkhorn(dot, max_iter=100):
    """
    dot: n x in_size x out_size
    mask: n x in_size
    output: n x in_size x out_size
    """
    n, in_size, out_size = dot.shape
    K = dot
    # K: n x in_size x out_size
    u = K.new_ones((n, in_size))
    v = K.new_ones((n, out_size))
    a = float(out_size / in_size)
    for _ in range(max_iter):
        u = a / torch.bmm(K, v.view(n, out_size, 1)).view(n, in_size)
        v = 1. / torch.bmm(u.view(n, 1, in_size), K).view(n, out_size)
    K = u.view(n, in_size, 1) * (K * v.view(n, 1, out_size))
    return K

def OT(q, k, eps=1.0, max_iter=100, cost_type=None, mPA=None, num_patches=None):
    """Compute the weights using Sinkhorn OT
    q: n x in_size x in_dim
    k: m x out_size x in_dim (m: number of heads/ref)
    output: n x out_size x m x in_size
    """
    n, in_size, in_dim = q.shape
    m, out_size = k.shape[:-1]

    C = torch.einsum('bid,bod->bio', q, k)

    if cost_type == 'easy':
        K = 1 - C.clone()
    elif cost_type == 'hard':
        K = C.clone()
    elif cost_type == 'mixed':
        # print(f"mPA.shape[0] {mPA.shape[0]}, C.shape[0] {C.shape[0]}")
        if mPA.shape[0] == C.shape[0]:
            mPA = mPA.view(-1, 1, 1)
            K = (1-mPA) * C.clone() + mPA * (1-C.clone())
        else:
            # [3*8] =(reshape)> [3, 8] =(diag_embed)> [3, 8, 8] =(kron)> [3, X, X]. => * with C
            # print("shape unequal!")
            mPA = mPA.view(C.shape[0], -1)
            mPA = torch.diag_embed(mPA)
            # print("After diag, mPA shape", mPA.shape)
            mPA = torch.kron(mPA, torch.ones(num_patches, num_patches, device=mPA.device))
            # print("After kron, mPA shape", mPA.shape)
            # print("C shape", C.shape)
            K = (1-mPA) * C.clone() + mPA * (1-C.clone())

    npatches = q.size(1)
    mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
    diagonal = torch.eye(npatches, device=q.device, dtype=mask_dtype)[None, :, :]
    K.masked_fill_(diagonal, -10)


    # K: n x m x in_size x out_size
    K = K.reshape(-1, in_size, out_size)
    # K: nm x in_size x out_size

    K = torch.exp(K / eps)
    K = sinkhorn(K, max_iter=max_iter)
    # K: nm x in_size x out_size
    K = K.permute(0, 2, 1).contiguous()
    return K