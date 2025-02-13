import math

import torch
import torch.nn as nn
from torch.autograd import Variable


def Binary_dice_loss(predictive, target, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def kl_loss(inputs, targets, ep=1e-8):
    kl_loss = nn.KLDivLoss(reduction='mean')
    consist_loss = kl_loss(torch.log(inputs + ep), targets)
    return consist_loss


def soft_ce_loss(inputs, target, ep=1e-8):
    logprobs = torch.log(inputs + ep)
    return torch.mean(-(target[:, 0, ...] * logprobs[:, 0, ...] + target[:, 1, ...] * logprobs[:, 1, ...]))


def mse_loss(input1, input2):
    return torch.mean((input1 - input2) ** 2)


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersection = torch.sum(score * target)
        union = torch.sum(score * score) + torch.sum(target * target) + smooth
        loss = 1 - intersection / union
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


CE = torch.nn.BCELoss()
mse = torch.nn.MSELoss()


def loss_diff1(u_prediction_1, u_prediction_2):
    loss_a = 0.0

    for i in range(u_prediction_2.size(1)):
        loss_a = CE(u_prediction_1[:, i, ...].clamp(1e-8, 1 - 1e-7),
                    Variable(u_prediction_2[:, i, ...].float(), requires_grad=False))

    loss_diff_avg = loss_a.mean().item()
    return loss_diff_avg


def loss_diff2(u_prediction_1, u_prediction_2):
    loss_b = 0.0

    for i in range(u_prediction_2.size(1)):
        loss_b = CE(u_prediction_2[:, i, ...].clamp(1e-8, 1 - 1e-7),
                    Variable(u_prediction_1[:, i, ...], requires_grad=False))

    loss_diff_avg = loss_b.mean().item()
    return loss_diff_avg


def loss_mask(u_prediction_1, u_prediction_2, critic_segs, T_m):
    # u_prediction_1 from E
    # u_prediction_2 from H
    gen_mask = (critic_segs.squeeze(0) > T_m).float()
    loss_a = gen_mask * CE(u_prediction_1,
                           Variable(u_prediction_2.float(), requires_grad=False))

    loss_diff_avg = loss_a.mean()

    return loss_diff_avg


# _logcosh
def recon_loss(y_t, y_prime_t):
    ey_t = y_t - y_prime_t
    return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))


def hyperbolic_angle(u, v):
    dot = torch.dot(u, v)
    norm_u = torch.norm(u, p=2)
    norm_v = torch.norm(v, p=2)
    angle = torch.acosh(dot / (norm_u * norm_v))
    return angle


def loss_hyp_uncertainty(hyperbolic_pred, u_prediction_1, u_prediction_2, T_m=0.1):
    curvature = 0.3
    loss_a = 0.0
    for i in range(u_prediction_2.size(1)):
        confidence_map = torch.linalg.norm(hyperbolic_pred[:, i, ...], dim=0, ord=2)
        radius = 1.0 / math.sqrt(curvature)
        normalized_cm = confidence_map / radius
        gen_mask = (normalized_cm > T_m).float()
        loss_a = loss_a + gen_mask * CE(u_prediction_1[:, i, ...],
                                            Variable(u_prediction_2[:, i, ...].float(), requires_grad=False))

    loss_mask = loss_a.mean() / u_prediction_2.size(1)

    return loss_mask


def loss_hyp_uncertainty_rev(hyperbolic_pred, u_prediction_1, u_prediction_2, T_m=0.1, curvature=0.3, k=10.0):
    """
    Computes a loss based on hyperbolic predictions and uncertainty maps.

    Parameters:
    - hyperbolic_pred: Tensor of hyperbolic predictions (B, C, H, W, ...)
    - u_prediction_1: Tensor of logits (predicted by model1) (B, C, H, W, ...)
    - u_prediction_2: Tensor of ground truth or reference predictions (B, C, H, W, ...)
    - T_m: Threshold for confidence mask generation
    - curvature: Curvature of hyperbolic space
    - k: Sharpness parameter for sigmoid-based mask

    Returns:
    - loss_mask: Computed loss
    """

    # u_prediction_1 from H
    # u_prediction_2 from E
    # Define the radius of hyperbolic space
    radius = 1.0 / math.sqrt(curvature)
    loss_a = 0.0

    for i in range(u_prediction_2.size(1)):  # Iterate over each class
        # Compute confidence map based on hyperbolic norms
        confidence_map = torch.linalg.norm(hyperbolic_pred[:, i, ...], dim=0, ord=2)
        normalized_cm = confidence_map / radius

        # Generate differentiable confidence mask
        gen_mask = torch.sigmoid(k * (normalized_cm - T_m))

        # Compute cross-entropy loss for this class
        loss_a += gen_mask * CE(u_prediction_1[:, i, ...],
                                            Variable(u_prediction_2[:, i, ...].float(), requires_grad=False))

    # Normalize loss across classes and apply a mean
    loss_mask = loss_a.sum() / (gen_mask.sum() + 1e-8)  # Avoid division by zero

    return loss_mask


def loss_euc_uncertainty(u_prediction_1, u_prediction_2, critic_segs, T_m):
    gen_mask = (critic_segs.squeeze(0) > T_m).float()
    loss_a = gen_mask * CE(u_prediction_1,
                           Variable((u_prediction_2 > 0.5).float(), requires_grad=False))

    loss_diff_avg = loss_a.mean()

    return loss_diff_avg


def disc_loss(pred, target, target_zeroes, target_ones):
    real_loss1 = CE(target, target_ones.float())
    fake_loss1 = CE(pred, target_zeroes.float())

    loss = (1/2) * (real_loss1 + fake_loss1)

    return loss


def gen_loss(pred, target_ones):
    fake_loss1 = CE(pred, target_ones.float())

    loss = fake_loss1

    return loss