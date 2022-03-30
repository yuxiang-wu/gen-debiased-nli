import torch
import torch.nn as nn
import torch.nn.functional as F


def convert_2d_prob_to_3d(prob_dist):
    prob_dist = torch.cat([(prob_dist[:, 0] / 2.0).view(-1, 1),
                           prob_dist[:, 1].view(-1, 1),
                           (prob_dist[:, 0] / 2.0).view(-1, 1)], dim=1)
    return prob_dist


# Focal loss's implementation is adapted from
# https://github.com/zhoudaxia233/focal_loss_pytorch/blob/master/multi_class_focal_loss.py
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, size_average=True, ensemble_training=False, aggregate_ensemble="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.ensemble_training = ensemble_training
        self.aggregate_ensemble = aggregate_ensemble

    def compute_probs(self, inputs, targets):
        prob_dist = F.softmax(inputs, dim=1)
        pt = prob_dist.gather(1, targets)
        return pt

    def aggregate(self, p1, p2, operation):
        if self.aggregate_ensemble == "mean":
            result = (p1 + p2) / 2
            return result
        elif self.aggregate_ensemble == "multiply":
            result = p1 * p2
            return result
        else:
            assert NotImplementedError("Operation ", operation, "is not implemented.")

    def forward(self, inputs, targets, inputs_adv=None, second_inputs_adv=None):
        targets = targets.view(-1, 1)
        norm = 0.0
        pt = self.compute_probs(inputs, targets)
        pt_scale = self.compute_probs(inputs if inputs_adv is None else inputs_adv, targets)
        if self.ensemble_training:
            pt_scale_second = self.compute_probs(second_inputs_adv, targets)
            if self.aggregate_ensemble in ["mean", "multiply"]:
                pt_scale_total = self.aggregate(pt_scale, pt_scale_second, "mean")
                batch_loss = -self.alpha * (torch.pow((1 - pt_scale_total), self.gamma)) * torch.log(pt)
        else:
            batch_loss = -self.alpha * (torch.pow((1 - pt_scale), self.gamma)) * torch.log(pt)
        norm += self.alpha * (torch.pow((1 - pt_scale), self.gamma))

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss


class POELoss(nn.Module):
    """Implements the product of expert loss."""

    def __init__(self, size_average=True, ensemble_training=False, poe_alpha=1):
        super().__init__()
        self.size_average = size_average
        self.ensemble_training = ensemble_training
        self.poe_alpha = poe_alpha

    def compute_probs(self, inputs):
        prob_dist = F.softmax(inputs, dim=1)
        return prob_dist

    def forward(self, inputs, targets, inputs_adv, second_inputs_adv=None):
        targets = targets.view(-1, 1)
        pt = self.compute_probs(inputs)
        pt_adv = self.compute_probs(inputs_adv)
        if self.ensemble_training:
            pt_adv_second = self.compute_probs(second_inputs_adv)
            joint_pt = F.softmax((torch.log(pt) + torch.log(pt_adv) + torch.log(pt_adv_second)), dim=1)
        else:
            joint_pt = F.softmax((torch.log(pt) + self.poe_alpha * torch.log(pt_adv)), dim=1)
        joint_p = joint_pt.gather(1, targets)
        batch_loss = -torch.log(joint_p)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class RUBILoss(nn.Module):
    # Implements the RUBI loss.
    def __init__(self, num_labels, size_average=True):
        super().__init__()
        self.size_average = size_average
        self.num_labels = num_labels
        self.loss_fct = torch.nn.CrossEntropyLoss()

    def compute_probs(self, inputs):
        prob_dist = F.softmax(inputs, dim=1)
        return prob_dist

    def forward(self, inputs, targets, inputs_adv):
        inputs = inputs.view(-1, self.num_labels)
        inputs_adv = inputs_adv.view(-1, self.num_labels)

        targets = targets.view(-1)
        logits = inputs * torch.sigmoid(inputs_adv)

        logits = logits.view(-1, self.num_labels)
        loss = self.loss_fct(logits, targets)
        return loss


epsilon = 1e-8


def log(x):
    """
    We assume the given input is a probability and this is not over 1 or below 0.
    """
    return torch.log(torch.clamp(x, min=epsilon, max=1 - epsilon))
