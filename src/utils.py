import torch

@torch.no_grad()
def modified_rv_coefficient(X_i, X_j):

    S_i = torch.matmul(X_i, X_i.t())
    S_j = torch.matmul(X_j, X_j.t())

    # Subtract the diagonal elements from each covariance matrix
    S_i_mod = S_i - torch.diag(torch.diag(S_i))
    S_j_mod = S_j - torch.diag(torch.diag(S_j))

    # Compute the trace of the product of the modified covariance matrices
    numerator = torch.trace(torch.mm(S_i_mod, S_j_mod))

    # Compute the denominator
    denominator = torch.sqrt(torch.trace(torch.mm(S_i_mod, S_i_mod)) * torch.trace(torch.mm(S_j_mod, S_j_mod)))

    # Compute the modified RV coefficient
    rv_mod = numerator / denominator

    return rv_mod
