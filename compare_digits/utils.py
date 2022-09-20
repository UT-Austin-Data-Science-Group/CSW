import torch
import math
import torch
from torch.distributions.kl import register_kl
def one_dimensional_Wasserstein(X_prod,Y_prod,p):
    X_prod = X_prod.view(X_prod.shape[0], -1)
    Y_prod = Y_prod.view(Y_prod.shape[0], -1)
    wasserstein_distance = torch.abs(
        (
                torch.sort(X_prod, dim=0)[0]
                - torch.sort(Y_prod, dim=0)[0]
        )
    )
    wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p), dim=0), 1.0 / p)
    wasserstein_distance = torch.pow(torch.pow(wasserstein_distance, p).mean(), 1.0 / p)
    return wasserstein_distance
def one_dimensional_Wasserstein_mixture(X_prod,Y_prod,k,p):
    L = X_prod.shape[1]
    inds = torch.arange(k)
    X_prod = X_prod.view(X_prod.shape[0], -1)
    Y_prod = Y_prod.view(Y_prod.shape[0], -1)
    X_prod_main = X_prod[:,inds]
    Y_prod_main = Y_prod[:, inds]
    X_main_sorted, X_main_indices = torch.sort(X_prod_main, dim=0)
    Y_main_sorted, Y_main_indices = torch.sort(Y_prod_main, dim=0)
    X_indices = X_main_indices.repeat((1,int(L/k)))
    Y_indices = Y_main_indices.repeat((1,int(L/k)))
    wasserstein_distance = torch.abs(
        (
                torch.gather(X_prod,0,X_indices)-
                torch.gather(Y_prod,0,Y_indices)
        )
    )
    wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p), dim=0), 1.0 / p)
    wasserstein_distance = torch.pow(torch.pow(wasserstein_distance, p).mean(), 1.0 / p)
    return wasserstein_distance


