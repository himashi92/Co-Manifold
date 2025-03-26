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


def discrepancy_loss_torch_center(hyperbolic_embeddings, euclidean_embeddings, center=None, epsilon=1e-8):
    """
    Calculates the discrepancy loss between hyperbolic and Euclidean embeddings
    with distances computed from a center.

    Args:
        hyperbolic_embeddings (torch.Tensor): Hyperbolic embeddings (batch_size, embedding_dim).
        euclidean_embeddings (torch.Tensor): Euclidean embeddings (batch_size, embedding_dim).
        center (torch.Tensor): The center coordinate (embedding_dim,). Default is zero vector.
        epsilon (float): Small value to prevent numerical issues.

    Returns:
        torch.Tensor: The calculated loss.
    """
    if center is None:
        center = torch.zeros_like(hyperbolic_embeddings[0])

    # Clip hyperbolic embeddings to avoid norm exceeding 1
    hyperbolic_norm = torch.linalg.norm(hyperbolic_embeddings, dim=1, keepdim=True)
    hyperbolic_embeddings = hyperbolic_embeddings / torch.clamp(hyperbolic_norm, max=1 - epsilon)

    # Hyperbolic distance from center
    hyperbolic_distance = torch.acosh(
        torch.clamp(
            1 + 2 * (torch.linalg.norm(hyperbolic_embeddings - center, dim=1) ** 2) /
            ((1 - hyperbolic_norm ** 2 + epsilon) * (1 - torch.linalg.norm(center) ** 2 + epsilon)),
            min=1 + epsilon
        )
    )

    # Euclidean distance from center
    euclidean_distance = torch.linalg.norm(euclidean_embeddings - center, dim=1)

    # Normalize embeddings
    hyperbolic_normalized = F.normalize(hyperbolic_embeddings, p=2, dim=1)
    euclidean_normalized = F.normalize(euclidean_embeddings, p=2, dim=1)

    # Cosine similarity
    cosine_similarity = torch.cosine_similarity(hyperbolic_normalized, euclidean_normalized, dim=1)

    # Loss: Combine cosine similarity and distance discrepancy
    distance_discrepancy = torch.abs(hyperbolic_distance - euclidean_distance)
    loss = torch.mean(cosine_similarity) + torch.mean(distance_discrepancy)

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



