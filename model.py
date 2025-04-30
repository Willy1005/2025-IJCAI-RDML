import torch
import torch.nn as nn
import numpy as np


class EvidenceCollector(nn.Module):
    def __init__(self, dim, num_classes):
        super(EvidenceCollector, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, num_classes),
            nn.Softplus()
        )

    def forward(self, x):
        return self.net(x)


class MultiEvidenceCollector(nn.Module):
    def __init__(self, dims, num_classes):
        super(MultiEvidenceCollector, self).__init__()
        self.num_classes = num_classes
        self.nets = nn.ModuleList([nn.Sequential(nn.Linear(int(dims[i]), num_classes), nn.Softplus()) for i in range(len(dims))])

    def DS_Combin(self, alpha):
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """
        def DS_Combin_two(alpha1, alpha2):
            """
            :param alpha1: Dirichlet distribution parameters of view 1
            :param alpha2: Dirichlet distribution parameters of view 2
            :return: Combined Dirichlet distribution parameters
            """
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
                E[v] = alpha[v]-1
                b[v] = E[v]/(S[v].expand(E[v].shape))
                u[v] = self.num_classes/S[v]

            # b^0 @ b^(0+1)
            bb = torch.bmm(b[0].view(-1, self.num_classes, 1), b[1].view(-1, 1, self.num_classes))
            # b^0 * u^1
            uv1_expand = u[1].expand(b[0].shape)
            bu = torch.mul(b[0], uv1_expand)
            # b^1 * u^0
            uv_expand = u[0].expand(b[0].shape)
            ub = torch.mul(b[1], uv_expand)
            # calculate C
            bb_sum = torch.sum(bb, dim=(1, 2), out=None)
            bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            C = bb_sum - bb_diag

            # calculate b^a
            b_a = (torch.mul(b[0], b[1]) + bu + ub)/((1-C).view(-1, 1).expand(b[0].shape))
            # calculate u^a
            u_a = torch.mul(u[0], u[1])/((1-C).view(-1, 1).expand(u[0].shape))

            # calculate new S
            S_a = self.num_classes / u_a
            # calculate new e_k
            e_a = torch.mul(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a

        for v in range(len(alpha)-1):
            if v==0:
                alpha_a = DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a = DS_Combin_two(alpha_a, alpha[v+1])
        return alpha_a

    def forward(self, x, abf=False):
        evidences = dict()
        alpha = dict()
        for i in range(0, len(self.nets)):
            evidences[i] = self.nets[i](x[i])
            alpha[i] = evidences[i] + 1
        if abf:  # average belief fusion
            evidence_a = evidences[0]
            for i in range(1, len(self.nets)):
                evidence_a = (evidences[i] + evidence_a) / 2
        else:  # dempster’s combination rule
            alpha_a = self.DS_Combin(alpha)
            evidence_a = alpha_a - 1

        return evidences, evidence_a


class Separation(nn.Module):
    def __init__(self, dim, num_classes, tau=0.1):
        super(Separation, self).__init__()
        self.tau = tau
        self.sep_net = nn.Sequential(
            nn.Linear(num_classes, dim),
        )

    def forward(self, feat, evidence_model, is_eval=False):
        rob_map = evidence_model(feat)
        rob_map = self.sep_net(rob_map)

        mask = torch.nn.Sigmoid()(rob_map.unsqueeze(1))
        mask = GumbelSigmoid(tau=self.tau)(mask, is_eval=is_eval)
        mask = mask[:, 0]

        r_feat = feat * mask
        nr_feat = feat * (1 - mask)

        return r_feat, nr_feat, mask


class Recalibration(nn.Module):
    def __init__(self, dim):
        super(Recalibration, self).__init__()
        self.rec_net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Softplus()
        )

    def forward(self, nr_feat, mask):
        rec_units = self.rec_net(nr_feat)
        rec_units = rec_units * (1 - mask)
        # rec_feat = nr_feat + rec_units

        # return rec_feat
        return rec_units


class CE_linear(nn.Module):
    def __init__(self, dim, num_classes):
        super(CE_linear, self).__init__()
        self.ce_linear = nn.Sequential(
            nn.Linear(num_classes, dim),
            nn.Softplus()
        )

    def forward(self, att):
        return self.ce_linear(att)


class GumbelSigmoid(nn.Module):
    def __init__(self, tau=1.0):
        super(GumbelSigmoid, self).__init__()

        self.tau = tau
        self.softmax = nn.Softmax(dim=1)
        self.p_value = 1e-8

    def forward(self, x, is_eval=False):
        r = 1 - x

        x = (x + self.p_value).log()
        r = (r + self.p_value).log()

        if not is_eval:
            x_N = torch.rand_like(x)
            r_N = torch.rand_like(r)
        else:
            x_N = 0.5 * torch.ones_like(x)
            r_N = 0.5 * torch.ones_like(r)

        x_N = -1 * (x_N + self.p_value).log()
        r_N = -1 * (r_N + self.p_value).log()
        x_N = -1 * (x_N + self.p_value).log()
        r_N = -1 * (r_N + self.p_value).log()

        x = x + x_N
        x = x / (self.tau + self.p_value)
        r = r + r_N
        r = r / (self.tau + self.p_value)

        x = torch.cat((x, r), dim=1)
        x = self.softmax(x)

        return x


class RDML(nn.Module):
    def __init__(self, dims, num_classes=10, num_layers=1, tau=0.1):
        super(RDML, self).__init__()
        self.tau = tau
        self.num_views = len(dims)
        self.num_classes = num_classes
        dims = np.repeat(dims, num_layers, axis=1)
        self.separation = nn.ModuleList([Separation(int(dims[i]), self.num_classes, tau=self.tau) for i in range(self.num_views)])
        self.recalibration = nn.ModuleList([Recalibration(int(dims[i])) for i in range(self.num_views)])
        self.aux = nn.ModuleList([EvidenceCollector(int(dims[i]), self.num_classes) for i in range(self.num_views)])
        self.ce_fusion = nn.ModuleList([CE_linear(int(dims[i]), self.num_classes) for i in range(self.num_views)])
        self.EvidenceCollectors = nn.ModuleList([EvidenceCollector(int(dims[i]), self.num_classes) for i in range(self.num_views)])

    def load_params(self, ec):
        self.EvidenceCollectors.state_dict()['0.net.0.weight'].data.copy_(ec.state_dict()['nets.0.0.weight'].data)
        self.EvidenceCollectors.state_dict()['0.net.0.bias'].data.copy_(ec.state_dict()['nets.0.0.bias'].data)
        self.EvidenceCollectors.state_dict()['1.net.0.weight'].data.copy_(ec.state_dict()['nets.1.0.weight'].data)
        self.EvidenceCollectors.state_dict()['1.net.0.bias'].data.copy_(ec.state_dict()['nets.1.0.bias'].data)
        self.EvidenceCollectors.state_dict()['2.net.0.weight'].data.copy_(ec.state_dict()['nets.2.0.weight'].data)
        self.EvidenceCollectors.state_dict()['2.net.0.bias'].data.copy_(ec.state_dict()['nets.2.0.bias'].data)

    def DS_Combin(self, alpha):
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """
        def DS_Combin_two(alpha1, alpha2):
            """
            :param alpha1: Dirichlet distribution parameters of view 1
            :param alpha2: Dirichlet distribution parameters of view 2
            :return: Combined Dirichlet distribution parameters
            """
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
                E[v] = alpha[v]-1
                b[v] = E[v]/(S[v].expand(E[v].shape))
                u[v] = self.num_classes/S[v]

            # b^0 @ b^(0+1)
            bb = torch.bmm(b[0].view(-1, self.num_classes, 1), b[1].view(-1, 1, self.num_classes))
            # b^0 * u^1
            uv1_expand = u[1].expand(b[0].shape)
            bu = torch.mul(b[0], uv1_expand)
            # b^1 * u^0
            uv_expand = u[0].expand(b[0].shape)
            ub = torch.mul(b[1], uv_expand)
            # calculate C
            bb_sum = torch.sum(bb, dim=(1, 2), out=None)
            bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            C = bb_sum - bb_diag

            # calculate b^a
            b_a = (torch.mul(b[0], b[1]) + bu + ub)/((1-C).view(-1, 1).expand(b[0].shape))
            # calculate u^a
            u_a = torch.mul(u[0], u[1])/((1-C).view(-1, 1).expand(u[0].shape))

            # calculate new S
            S_a = self.num_classes / u_a
            # calculate new e_k
            e_a = torch.mul(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a

        for v in range(len(alpha)-1):
            if v==0:
                alpha_a = DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a = DS_Combin_two(alpha_a, alpha[v+1])
        return alpha_a

    def disentangle(self, x, v, evidence_model, is_eval=False):
        r_feat, nr_feat, mask = self.separation[v](x, evidence_model.nets[v], is_eval)
        r_out = self.aux[v](r_feat)
        nr_out = self.aux[v](nr_feat)
        return r_feat, nr_feat, r_out, nr_out, mask

    def repair(self, nr_feat, mask, v):
        rec_feat = self.recalibration[v](nr_feat, mask)
        rec_out = self.aux[v](rec_feat)
        return rec_feat, rec_out

    def CE_fusion(self, feat, evidence_model):
        # CE: classification evidence
        pretrained_ce = []
        for v in range(len(feat)):
            pretrained_ce.append(evidence_model.nets[v](feat[v]).unsqueeze(1))
        pretrained_ce = torch.cat(pretrained_ce, dim=1)
        att = nn.Softmax(dim=2)(pretrained_ce)
        for v in range(len(feat)):
            feat[v] = self.ce_fusion[v](att[:, v, :]).squeeze(1) * feat[v]
        out = []
        for v in range(len(feat)):
            out.append(self.EvidenceCollectors[v](feat[v]))

        return out

    def forward(self, x, ec, is_eval=False, abf=False):
        outputs = []
        r_outputs = []
        nr_outputs = []
        rec_outputs = []

        for v in range(self.num_views):
            r_feat, nr_feat, r_out, nr_out, mask = self.disentangle(x[v], v, ec, is_eval)
            rec_feat, rec_out = self.repair(nr_feat, mask, v)

            outputs.append(r_feat+rec_feat)
            r_outputs.append(r_out)
            nr_outputs.append(nr_out)
            rec_outputs.append(rec_out)

        outputs = self.CE_fusion(outputs, ec)

        if abf:  # average belief fusion
            evidence_a = outputs[0]
            for i in range(1, self.num_views):
                evidence_a = (outputs[i] + evidence_a) / 2
        else:  # dempster’s combination rule
            alpha = dict()
            for i in range(self.num_views):
                alpha[i] = outputs[i] + 1
            evidence_a = self.DS_Combin(alpha) - 1

        return outputs, r_outputs, nr_outputs, rec_outputs, evidence_a
