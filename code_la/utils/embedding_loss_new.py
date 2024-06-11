import scipy
import torch
import math
import numpy as np
import torch.nn.functional as F
from utils.pmath import dist, expmap0, artanh


def exp_map(z):
    return torch.div(torch.exp(z), torch.add(1, torch.exp(z)))


def mobius_transform(z):
    return torch.div(torch.add(1, z), torch.add(1, torch.mul(z, 1j)))


def arctanh(x):
    x = torch.clamp(x, -1+1e-7, 1-1e-7)
    return 0.5 * torch.log((1 + x) / (1 - x))


def geodesic_distance(z1, z2):
    z1 = torch.tensor(z1, dtype=torch.float32)
    z2 = torch.tensor(z2, dtype=torch.float32)
    return 2 * arctanh(torch.div(torch.abs(z1 - z2), 1 - torch.mul(torch.abs(z1), torch.abs(z2))))


def hungarian_matching_loss(x, y):
    x_flat = x.view(x.size(0), -1)
    y_flat = y.view(y.size(0), -1)

    mse_loss = F.mse_loss(x_flat, y_flat, reduction='none')
    mse_loss = torch.where(torch.isnan(mse_loss), torch.ones_like(mse_loss), mse_loss)
    #print(mse_loss)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(mse_loss)
    matching_loss = mse_loss[row_ind, col_ind].mean()

    return matching_loss

# def hungarian_matching_loss(x, y):
#     cost_matrix = torch.cdist(x.view(x.size(0), -1), y.view(y.size(0), -1))
#     cost_matrix = torch.where(torch.isnan(cost_matrix), torch.zeros_like(cost_matrix), cost_matrix)
#
#     row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
#     matching_loss = cost_matrix[row_ind, col_ind].mean()
#
#     return matching_loss


def disk_to_half_plane(z):
    """
    Transform from Poincaré disk model to Poincaré half-plane model.
    z: Tensor of shape (..., 2) representing points in the Poincaré disk.
    """
    # Converting to complex numbers
    z_complex = z[..., 0] + 1j * z[..., 1]

    # Apply transformation
    w = (z_complex + 1) / (z_complex - 1)

    # Splitting back into real and imaginary parts
    return torch.stack([w.real, w.imag], dim=-1)


def translate_to_half_plane(euclidean_embeddings):
    """
    Translate Euclidean embeddings to ensure they fall in y > 0 half-plane.
    euclidean_embeddings: Tensor of shape (..., 2) representing Euclidean points.
    """
    min_y = torch.min(euclidean_embeddings[..., 1])
    translation = 0.1 - min_y  # Ensure a small buffer above y=0
    euclidean_embeddings[..., 1] += translation
    return euclidean_embeddings


def hyperbolic_embedding_loss(hyperbolic_embeddings, euclidean_embeddings, bs):
    mu_h_l = disk_to_half_plane(hyperbolic_embeddings)
    mu_e_l = disk_to_half_plane(euclidean_embeddings)

    # Hyperbolic Loss
    x1, y1 = mu_h_l[0][0], mu_h_l[0][1]
    x2, y2 = mu_h_l[1][0], mu_h_l[1][1]
    x3, y3 = mu_h_l[2][0], mu_h_l[2][1]

    u, v, w = torch.tensor((x1, y1)), torch.tensor((x2, y2)), torch.tensor((x3, y3))

    c = ab_h = dist(u.cpu(), v.cpu())
    b = ca_h = dist(w.cpu(), u.cpu())
    a = bc_h = dist(v.cpu(), w.cpu())

    cosh_alpha = (torch.cosh(b) * torch.cosh(c) - torch.cosh(a)) / (torch.sinh(b) * torch.sinh(c))
    alpha = torch.acos(torch.clamp(cosh_alpha, -1, 1))

    cosh_gamma = (torch.cosh(a) * torch.cosh(b) - torch.cosh(c)) / (torch.sinh(a) * torch.sinh(b))
    gamma = torch.acos(torch.clamp(cosh_gamma, -1, 1))

    cosh_beta = (torch.cosh(a) * torch.cosh(c) - torch.cosh(b)) / (torch.sinh(a) * torch.sinh(c))
    beta = torch.acos(torch.clamp(cosh_beta, -1, 1))

    angles_h = torch.Tensor([alpha.item(), beta.item(), gamma.item()])
    torch.nan_to_num(angles_h, nan=torch.deg2rad(torch.tensor(180)).item())
    angles_h[angles_h == float('inf')] = 0


    # Euclidean Loss
    e_x1, e_y1 = mu_e_l[0][0], mu_e_l[0][1]
    e_x2, e_y2 = mu_e_l[1][0], mu_e_l[1][1]
    e_x3, e_y3 = mu_e_l[2][0], mu_e_l[2][1]

    u2, v2, w2 = torch.tensor((e_x1, e_y1)), torch.tensor((e_x2, e_y2)), torch.tensor((e_x3, e_y3))

    c1 = ab_h1 = dist(u2.cpu(), v2.cpu())
    b1 = ca_h1 = dist(w2.cpu(), u2.cpu())
    a1 = bc_h1 = dist(v2.cpu(), w2.cpu())

    cosh_alpha_e = (torch.cosh(b1) * torch.cosh(c1) - torch.cosh(a1)) / (torch.sinh(b1) * torch.sinh(c1))
    alpha_e = torch.acos(torch.clamp(cosh_alpha_e, -1, 1))

    cosh_gamma_e = (torch.cosh(a1) * torch.cosh(b1) - torch.cosh(c1)) / (torch.sinh(a1) * torch.sinh(b1))
    gamma_e = torch.acos(torch.clamp(cosh_gamma_e, -1, 1))

    cosh_beta_e = (torch.cosh(a1) * torch.cosh(c1) - torch.cosh(b1)) / (torch.sinh(a1) * torch.sinh(c1))
    beta_e = torch.acos(torch.clamp(cosh_beta_e, -1, 1))

    angles_e = torch.Tensor([alpha_e.item(), beta_e.item(), gamma_e.item()])
    torch.nan_to_num(angles_e, nan=torch.deg2rad(torch.tensor(180)).item())
    angles_e[angles_e == float('inf')] = 0

    loss = hungarian_matching_loss(angles_h, angles_e)

    return loss


def hyperbolic_embedding_recon_loss(hyperbolic_embeddings, euclidean_embeddings, bs):
    mu_h_l = disk_to_half_plane(hyperbolic_embeddings)
    mu_e_l = disk_to_half_plane(euclidean_embeddings)

    # Hyperbolic Loss
    x1, y1 = mu_h_l[0][0], mu_h_l[0][1]
    x2, y2 = mu_h_l[1][0], mu_h_l[1][1]
    x3, y3 = mu_h_l[2][0], mu_h_l[2][1]

    u, v, w = torch.tensor((x1, y1)), torch.tensor((x2, y2)), torch.tensor((x3, y3))

    hyp = torch.stack([u, v, w])

    # Euclidean Loss
    e_x1, e_y1 = mu_e_l[0][0], mu_e_l[0][1]
    e_x2, e_y2 = mu_e_l[1][0], mu_e_l[1][1]
    e_x3, e_y3 = mu_e_l[2][0], mu_e_l[2][1]

    u2, v2, w2 = torch.tensor((e_x1, e_y1)), torch.tensor((e_x2, e_y2)), torch.tensor((e_x3, e_y3))

    euc = torch.stack([u2, v2, w2])

    loss = hungarian_matching_loss(hyp, euc)

    return loss


def hyperbolic_embedding_loss_new(hyperbolic_embeddings, euclidean_embeddings, bs):
    mu_h_l = disk_to_half_plane(hyperbolic_embeddings)
    mu_e_l = disk_to_half_plane(euclidean_embeddings)

    # Hyperbolic Loss
    x1, y1 = mu_h_l[0][0], mu_h_l[0][1]
    x2, y2 = mu_h_l[1][0], mu_h_l[1][1]
    x3, y3 = mu_h_l[2][0], mu_h_l[2][1]

    u, v, w = torch.tensor((x1, y1)), torch.tensor((x2, y2)), torch.tensor((x3, y3))

    c = ab_h = dist(u.cpu(), v.cpu())
    b = ca_h = dist(w.cpu(), u.cpu())
    a = bc_h = dist(v.cpu(), w.cpu())

    cosh_alpha = (torch.cosh(b) * torch.cosh(c) - torch.cosh(a)) / (torch.sinh(b) * torch.sinh(c))
    alpha = torch.acos(torch.clamp(cosh_alpha, -1, 1))

    cosh_gamma = (torch.cosh(a) * torch.cosh(b) - torch.cosh(c)) / (torch.sinh(a) * torch.sinh(b))
    gamma = torch.acos(torch.clamp(cosh_gamma, -1, 1))

    cosh_beta = (torch.cosh(a) * torch.cosh(c) - torch.cosh(b)) / (torch.sinh(a) * torch.sinh(c))
    beta = torch.acos(torch.clamp(cosh_beta, -1, 1))

    angles_h = torch.Tensor([alpha.item(), beta.item(), gamma.item()])
    torch.nan_to_num(angles_h, nan=torch.deg2rad(torch.tensor(180)).item())
    angles_h[angles_h == float('inf')] = 0


    # Euclidean Loss
    e_x1, e_y1 = mu_e_l[0][0], mu_e_l[0][1]
    e_x2, e_y2 = mu_e_l[1][0], mu_e_l[1][1]
    e_x3, e_y3 = mu_e_l[2][0], mu_e_l[2][1]

    u2, v2, w2 = torch.tensor((e_x1, e_y1)), torch.tensor((e_x2, e_y2)), torch.tensor((e_x3, e_y3))

    c1 = ab_h1 = dist(u2.cpu(), v2.cpu())
    b1 = ca_h1 = dist(w2.cpu(), u2.cpu())
    a1 = bc_h1 = dist(v2.cpu(), w2.cpu())

    cosh_alpha_e = (torch.cosh(b1) * torch.cosh(c1) - torch.cosh(a1)) / (torch.sinh(b1) * torch.sinh(c1))
    alpha_e = torch.acos(torch.clamp(cosh_alpha_e, -1, 1))

    cosh_gamma_e = (torch.cosh(a1) * torch.cosh(b1) - torch.cosh(c1)) / (torch.sinh(a1) * torch.sinh(b1))
    gamma_e = torch.acos(torch.clamp(cosh_gamma_e, -1, 1))

    cosh_beta_e = (torch.cosh(a1) * torch.cosh(c1) - torch.cosh(b1)) / (torch.sinh(a1) * torch.sinh(c1))
    beta_e = torch.acos(torch.clamp(cosh_beta_e, -1, 1))

    angles_e = torch.Tensor([alpha_e.item(), beta_e.item(), gamma_e.item()])
    torch.nan_to_num(angles_e, nan=torch.deg2rad(torch.tensor(180)).item())
    angles_e[angles_e == float('inf')] = 0

    loss = F.mse_loss(angles_h, angles_e, reduction='none')

    return loss


def euclidean_embedding_loss(hyperbolic_embeddings, euclidean_embeddings, bs):
    mu_h_l = disk_to_half_plane(hyperbolic_embeddings)
    mu_e_l = translate_to_half_plane(euclidean_embeddings)

    e_c = ab_e = torch.sqrt(((mu_h_l[0].cpu() - mu_h_l[1].cpu()) ** 2).sum())
    e_b = ca_e = torch.sqrt(((mu_h_l[2].cpu() - mu_h_l[0].cpu()) ** 2).sum())
    e_a = bc_e = torch.sqrt(((mu_h_l[1].cpu() - mu_h_l[2].cpu()) ** 2).sum())

    e_cos_alpha = (-e_a.pow(2) + e_b.pow(2) + e_c.pow(2)) / (2 * e_b * e_c)
    e_alpha_h = torch.acos(e_cos_alpha)

    e_cos_gamma = (-e_c.pow(2) + e_b.pow(2) + e_a.pow(2)) / (2 * e_b * e_a)
    e_gamma_h = torch.acos(e_cos_gamma)

    e_cos_beta = (-e_b.pow(2) + e_c.pow(2) + e_a.pow(2)) / (2 * e_c * e_a)
    e_beta_h = torch.acos(e_cos_beta)

    e_angles_h = torch.Tensor([e_alpha_h.item(), e_beta_h.item(), e_gamma_h.item()])
    torch.nan_to_num(e_angles_h, nan=torch.deg2rad(torch.tensor(180)).item())
    e_angles_h[e_angles_h == float('inf')] = 0

    e_c1 = ab_e1 = torch.sqrt(((mu_e_l[0].cpu() - mu_e_l[1].cpu()) ** 2).sum())
    e_b1 = ca_e1 = torch.sqrt(((mu_e_l[2].cpu() - mu_e_l[0].cpu()) ** 2).sum())
    e_a1 = bc_e1 = torch.sqrt(((mu_e_l[1].cpu() - mu_e_l[2].cpu()) ** 2).sum())

    e_cos_alpha = (-e_a1.pow(2) + e_b1.pow(2) + e_c1.pow(2)) / (2 * e_b1 * e_c1)
    e_alpha_e = torch.acos(e_cos_alpha)

    e_cos_gamma = (-e_c1.pow(2) + e_b1.pow(2) + e_a1.pow(2)) / (2 * e_b1 * e_a1)
    e_gamma_e = torch.acos(e_cos_gamma)

    e_cos_beta = (-e_b1.pow(2) + e_c1.pow(2) + e_a1.pow(2)) / (2 * e_c1 * e_a1)
    e_beta_e = torch.acos(e_cos_beta)

    e_angles_e = torch.Tensor([e_alpha_e.item(), e_beta_e.item(), e_gamma_e.item()])
    torch.nan_to_num(e_angles_e, nan=torch.deg2rad(torch.tensor(180)).item())
    e_angles_e[e_angles_e == float('inf')] = 0

    loss = hungarian_matching_loss(e_angles_h, e_angles_e)

    return loss


def discrepancy_loss(hyperbolic_embeddings, euclidean_embeddings):
    # Normalize the embeddings
    hyperbolic_norm = torch.nn.functional.normalize(hyperbolic_embeddings, p=2, dim=1)
    euclidean_norm = torch.nn.functional.normalize(euclidean_embeddings, p=2, dim=1)

    # Calculate cosine similarity
    cosine_similarity = torch.sum(hyperbolic_norm * euclidean_norm, dim=1)

    # Maximize the difference (minimize similarity)
    loss = torch.mean(cosine_similarity)
    return loss



def discrepancy_loss_torch(hyperbolic_embeddings, euclidean_embeddings):
    # Normalize the embeddings
    hyperbolic_norm = F.normalize(hyperbolic_embeddings, p=2, dim=1)
    euclidean_norm = F.normalize(euclidean_embeddings, p=2, dim=1)

    # Calculate cosine similarity using torch function
    cosine_similarity = torch.cosine_similarity(hyperbolic_norm, euclidean_norm)

    # Maximize the difference (minimize similarity)
    loss = torch.mean(cosine_similarity)
    return loss


def discrepancy_loss_torch_2(hyperbolic_embeddings, euclidean_embeddings):
    # Normalize the embeddings
    hyperbolic_norm = F.normalize(hyperbolic_embeddings, p=2, dim=1)
    euclidean_norm = F.normalize(euclidean_embeddings, p=2, dim=1)

    # Calculate cosine similarity using torch function
    cosine_similarity = torch.cosine_similarity(hyperbolic_norm, euclidean_norm)

    # Maximize the difference (minimize similarity)
    loss = torch.mean(cosine_similarity)
    return loss



