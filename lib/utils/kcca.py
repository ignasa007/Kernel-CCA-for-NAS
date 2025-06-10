import torch
from torch.nn.functional import one_hot


def featurize_targets(targets, num_classes):

    # can ignore `num_classes` with one_hot because won't affect inner product
    return one_hot(targets, num_classes).float()

def kcca(K, L, reg_eps=1e-2, **lobpcg_kwargs):

    # assume K, L symmetric
    assert K.ndim == 2 and K.shape[0] == K.shape[1], K.shape
    assert K.shape == L.shape, (K.shape, L.shape)
    K.requires_grad = L.requires_grad = False

    # No-GPU implementation
    K = K.to(torch.device('cpu'))
    L = L.to(torch.device('cpu'))

    N = K.shape[0]
    H = torch.eye(N) - torch.ones((N, N)) / N
    K_tilde = H @ K @ H
    L_tilde = H @ L @ H
    
    A = torch.cat((
        torch.cat((torch.zeros((N, N)), K_tilde @ L_tilde), dim=1),
        torch.cat((L_tilde @ K_tilde, torch.zeros((N, N))), dim=1),
    ), dim=0)
    B = torch.cat((
        torch.cat((K_tilde @ K_tilde + N*reg_eps * K_tilde, torch.zeros((N, N))), dim=1),
        torch.cat((torch.zeros((N, N)), L_tilde @ L_tilde + N*reg_eps * L_tilde), dim=1),
    ), dim=0)

    gamma, _ = torch.lobpcg(A, B=B, k=1, largest=True, **lobpcg_kwargs)

    return gamma.item()