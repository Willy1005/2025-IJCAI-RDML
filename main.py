import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from data import PIE
from loss_function import get_ea_loss, get_dr_loss
from attack_function import pgd_attack_partial, pgd_attack_partial_ec
import argparse
from model import RDML, MultiEvidenceCollector
import numpy as np
import warnings
warnings.filterwarnings("ignore")

np.set_printoptions(precision=4, suppress=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cat_and_shuffle(x1, x2, y, device):
    x = dict()
    for i in range(len(x1)):
        x[i] = torch.cat((x1[i], x2[i]), dim=0)
    y = torch.cat((y, y), dim=0).to(device)
    indices = torch.randperm(y.size(0))
    for i in range(len(x)):
        x[i] = x[i][indices].to(device)
    y = y[indices]
    return x, y


def pretrain(args, dataset):
    num_samples = len(dataset)
    num_classes = dataset.num_classes
    num_views = dataset.num_views
    dims = dataset.dims
    index = np.arange(num_samples)
    np.random.shuffle(index)
    train_index = index[:int(0.8 * num_samples)]
    train_loader = DataLoader(Subset(dataset, train_index), batch_size=args.batch_size, shuffle=True)

    evi_collector = MultiEvidenceCollector(dims, num_classes)
    ec_optimizer = optim.Adam(evi_collector.parameters(), lr=args.pt_lr, weight_decay=1e-5)

    PATH = '{}//ec-{}-{}.pth'.format(args.ec_path, args.dataset, args.setting)

    evi_collector.to(device)
    evi_collector.train()
    print("Start pretraining...")
    for epoch in range(1, args.pt_epochs + 1):
        for x, y, indexes in train_loader:
            for v in range(num_views):
                x[v] = x[v].to(device)
            y = y.to(device)
            if args.setting == 'A':
                x_adv = pgd_attack_partial_ec(evi_collector, x, y, args.pt_atk_num, args.eps)
                x, y = cat_and_shuffle(x, x_adv, y, device)
            evidences, evidence_a = evi_collector(x)
            ec_loss, _ = get_ea_loss(evidences, evidence_a, y, epoch, num_classes, args.annealing_step, args.gamma, device)

            ec_optimizer.zero_grad()
            ec_loss.backward()
            ec_optimizer.step()

    print("Saving pretrained state dict at {}...".format(PATH))
    torch.save(evi_collector.state_dict(), PATH)


def main(args, dataset):
    num_samples = len(dataset)
    num_classes = dataset.num_classes
    num_views = dataset.num_views
    dims = dataset.dims
    index = np.arange(num_samples)
    np.random.shuffle(index)
    train_index, test_index = index[:int(0.8 * num_samples)], index[int(0.8 * num_samples):]

    train_loader = DataLoader(Subset(dataset, train_index), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_index), batch_size=args.batch_size, shuffle=False)

    model = RDML(dims, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    evi_collector = MultiEvidenceCollector(dims, num_classes)
    evi_collector.to(device)

    PATH = '{}//ec-{}-{}.pth'.format(args.ec_path, args.dataset, args.setting)
    evi_collector.load_state_dict(torch.load(PATH))
    # load parameters from pretrained evidence collector
    model.load_params(evi_collector)

    # freeze the evidence collector
    for p in evi_collector.parameters():
        p.requires_grad = False

    # training stage
    model.to(device)
    model.train()
    evi_collector.train()
    print("Start training...")
    for epoch in range(1, args.epochs + 1):
        for x, y, indexes in train_loader:
            for v in range(num_views):
                x[v] = x[v].to(device)
            y = y.to(device)
            # attack on random views
            if args.setting == 'A':
                x = pgd_attack_partial(model, evi_collector, x, y, args.atk_num, args.eps)
            outputs, r_outputs, nr_outputs, rec_outputs, evidence_a = model(x, evi_collector)

            # get disentanglement loss and recalibration loss
            dis_loss, rec_loss = get_dr_loss(outputs, r_outputs, nr_outputs, rec_outputs, y, args.lam_sep, args.lam_rec, device)
            # get evidential classification loss and adversarial consistency loss
            ea_loss, _ = get_ea_loss(outputs, evidence_a, y, epoch, num_classes, args.annealing_step, args.gamma, device)
            loss = ea_loss + args.alpha * dis_loss + args.beta * rec_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # evaluating stage
    model.eval()
    evi_collector.eval()
    print("Start evaluating...")
    num_correct, num_sample = 0, 0
    for x, y, indexes in test_loader:
        for v in range(num_views):
            x[v] = x[v].to(device)
        y = y.to(device)
        # attack on random views
        if args.setting == 'A':
            x = pgd_attack_partial(model, evi_collector, x, y, args.atk_num, args.eps)
        with torch.no_grad():
            _, _, _, _, evidence_a = model(x, evi_collector, True)
            _, y_pred = torch.max(evidence_a, dim=1)
            num_correct += (y_pred == y).sum().item()
            num_sample += y.shape[0]
    test_acc = num_correct / num_sample

    print('====> Dataset: {}, Setting: {}, Test acc: {:.4f}'.format(dataset.data_name, args.setting, test_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=200, metavar='N', help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to train')
    parser.add_argument('--pt_epochs', type=int, default=1000, metavar='N', help='number of epochs to pretrain')
    parser.add_argument('--annealing_step', type=int, default=50, metavar='N', help='gradually increase the value of lambda from 0 to 1')
    parser.add_argument('--pt_lr', type=float, default=0.003, metavar='LR', help='pretrain lr, default 0.003')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR', help='learning rate, default 0.003')
    parser.add_argument('--seed', type=int, default=200, metavar='SEED', help='random seed')
    parser.add_argument('--lam_sep', type=float, default=1, metavar='COE', help='coefficient')
    parser.add_argument('--lam_rec', type=float, default=1, metavar='COE', help='coefficient')
    parser.add_argument('--alpha', type=float, default=1, metavar='COE', help='coefficient')
    parser.add_argument('--beta', type=float, default=1, metavar='COE', help='coefficient')
    parser.add_argument('--gamma', type=float, default=1, metavar='COE', help='coefficient')
    parser.add_argument('--eps', type=float, default=8. / 255., metavar='COE', help='coefficient, default 8. / 255.')
    parser.add_argument('--ec_path', type=str, default='state_dict', metavar='PATH', help='state dict path')
    parser.add_argument('--dataset', type=str, default='P', metavar='DATA', help='dataset,'
                        'P -> PIE, S -> Scene, L -> Leaves, NW -> NUS-WIDE, M -> MSRCV5, F -> Fashion')
    parser.add_argument('--pt_atk_num', type=int, default=1, metavar='N', help='number of attacked views in pretraining')
    parser.add_argument('--atk_num', type=int, default=1, metavar='N', help='number of attacked views')
    parser.add_argument('--setting', type=str, default='N', metavar='SET', help='experiment setting, N -> normal, A -> adversarial')

    args = parser.parse_args()
    print(args)

    if args.dataset == "P":
        dataset = PIE()
        if args.setting == 'A':
            args.epochs = 500
            args.lr = 0.003
            args.pt_lr = 0.003

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # pretrain the evidence collector
    pretrain(args, dataset)
    # train and evaluate
    main(args, dataset)
