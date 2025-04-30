import torch
import numpy as np


def pgd_attack_partial(model, evi_collector, views, Y, atk_num, eps=8. / 255., alpha=2. / 255., iters=10):
    rand = np.random.choice(len(views), atk_num, False).tolist()
    delta = []
    for i in range(len(rand)):
        perturbation = torch.nn.Parameter(torch.rand_like(views[rand[i]]) * eps * 2 - eps)
        delta.append(perturbation)
        views[rand[i]] = views[rand[i]] + perturbation
    for i in range(iters):
        model.zero_grad()
        _, _, _, _, evidence_a = model(views, evi_collector)
        loss_acc = get_cls_loss(evidence_a, Y)
        loss_acc.backward()
        for j in range(len(rand)):
            delta[j].data = delta[j].data + alpha * delta[j].grad.sign()
            delta[j].grad = None
            delta[j].data = torch.clamp(delta[j].data, min=-eps, max=eps)
            delta[j].data = torch.clamp(views[rand[j]] + delta[j].data, min=0, max=1) - views[rand[j]]
            views[rand[j]] = views[rand[j]] + delta[j]
    return [views[x].detach() for x in views]


def pgd_attack_partial_ec(model, views, Y, atk_num, eps=8. / 255., alpha=2. / 255., iters=10):
    rand = np.random.choice(len(views), atk_num, False).tolist()
    delta = []
    for i in range(len(rand)):
        perturbation = torch.nn.Parameter(torch.rand_like(views[rand[i]]) * eps * 2 - eps)
        delta.append(perturbation)
        views[rand[i]] = views[rand[i]] + perturbation
    for i in range(iters):
        model.zero_grad()
        _, evidence_a = model(views)
        loss_acc = get_cls_loss(evidence_a, Y)
        loss_acc.backward()
        for j in range(len(rand)):
            delta[j].data = delta[j].data + alpha * delta[j].grad.sign()
            delta[j].grad = None
            delta[j].data = torch.clamp(delta[j].data, min=-eps, max=eps)
            delta[j].data = torch.clamp(views[rand[j]] + delta[j].data, min=0, max=1) - views[rand[j]]
            views[rand[j]] = views[rand[j]] + delta[j]
    return [views[x].detach() for x in views]


def get_cls_loss(evidence_a, Y):
    function = torch.nn.CrossEntropyLoss()
    return function(evidence_a, Y.long())
