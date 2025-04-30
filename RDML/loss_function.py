import torch
import torch.nn.functional as F
import torch.nn as nn


def kl_divergence(alpha, num_classes, device):
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device, useKL=True):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    if not useKL:
        return A

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div


def edl_digamma_loss(alpha, target, epoch_num, num_classes, annealing_step, device):
    loss = edl_loss(torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device)
    return torch.mean(loss)


# adversarial consistency loss
def get_ac_loss(evidences, device):
    num_views = len(evidences)
    batch_size, num_classes = evidences[0].shape[0], evidences[0].shape[1]
    p = torch.zeros((num_views, batch_size, num_classes)).to(device)
    u = torch.zeros((num_views, batch_size)).to(device)
    for v in range(num_views):
        alpha = evidences[v] + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        p[v] = alpha / S
        u[v] = torch.squeeze(num_classes / S)
    dc_sum = 0
    for i in range(num_views):
        pd = torch.sum(torch.abs(p - p[i]) / 2, dim=2) / (num_views - 1)  # (num_views, batch_size)
        # cc = (1 - u[i]) * (1 - u)  # (num_views, batch_size)
        cc = 1
        dc = pd * cc
        dc_sum = dc_sum + torch.sum(dc, dim=0)
    dc_sum = torch.mean(dc_sum)
    return dc_sum


# get evidential classification loss and adversarial consistency loss
def get_ea_loss(evidences, evidence_a, target, epoch_num, num_classes, annealing_step, gamma, device):
    target = F.one_hot(torch.tensor(target, dtype=int), num_classes)
    alpha_a = evidence_a + 1
    loss_acc = edl_digamma_loss(alpha_a, target, epoch_num, num_classes, annealing_step, device)
    for v in range(len(evidences)):
        alpha = evidences[v] + 1
        loss_acc += edl_digamma_loss(alpha, target, epoch_num, num_classes, annealing_step, device)
    loss_acc = loss_acc / (len(evidences) + 1)
    loss = loss_acc + gamma * get_ac_loss(evidences, device)
    return loss, loss_acc


def get_pred(out, labels):
    adv_labels = []
    for i in range(len(out)):
        pred = out[i].sort(dim=-1, descending=True)[1][:, 0]
        second_pred = out[i].sort(dim=-1, descending=True)[1][:, 1]
        adv_label = torch.where(pred == labels, second_pred, pred)
        adv_labels.append(adv_label)

    return adv_labels


# get disentanglement loss and recalibration loss
def get_dr_loss(adv_outputs, adv_r_outputs, adv_nr_outputs, adv_rec_outputs, targets, lam_sep, lam_rec, device):
    criterion = nn.CrossEntropyLoss(reduction='mean')
    adv_labels = get_pred(adv_outputs, targets)

    r_loss = torch.tensor(0.).to(device)
    for i in range(len(adv_r_outputs)):
        r_loss += lam_sep * criterion(adv_r_outputs[i], targets)
    r_loss /= len(adv_r_outputs)

    nr_loss = torch.tensor(0.).to(device)
    for i in range(len(adv_nr_outputs)):
        nr_loss += lam_sep * criterion(adv_nr_outputs[i], adv_labels[i])
    nr_loss /= len(adv_nr_outputs)
    dis_loss = r_loss + nr_loss

    rec_loss = torch.tensor(0.).to(device)
    for i in range(len(adv_rec_outputs)):
        rec_loss += lam_rec * criterion(adv_rec_outputs[i], targets)
    rec_loss /= len(adv_rec_outputs)

    return dis_loss, rec_loss
