import math
import torch
import numpy as np


# Construct covariance matrix
def construct_covariance(vectors, values):
    vectors = np.array(vectors).transpose()
    values = np.array(values)
    return vectors * values


# def multivariate_gaussian(z, mean, determinant, inverse_covariance_matrix):
def multivariate_gaussian(z, mean, determinant, inverse_cov_mat_diag):
    b_size = z.shape[0]
    z_flat = z.reshape(b_size, -1)
    k = z_flat.shape[1]
    log_p = -0.5 * (k * math.log(2 * math.pi) +
                    torch.diag((((z_flat - mean) @
                                 (inverse_cov_mat_diag.to(z.device) * torch.eye(inverse_cov_mat_diag.shape[0]).to(
                                     z.device)).unsqueeze(0)).reshape(b_size, -1) @
                                torch.transpose(z_flat - mean, 1, 0)).reshape(b_size, -1), 0)) - math.log(determinant)

    return log_p
