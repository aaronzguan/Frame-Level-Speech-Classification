import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class L2Norm(nn.Module):
    def forward(self, input, dim=1):
        return F.normalize(input, p=2, dim=dim)


class asoftmax(nn.Module):
    def __init__(self, class_num, args):
        super(asoftmax, self).__init__()
        self.in_features = args.feature_dim
        self.out_features = class_num
        self.w_norm = args.use_w_norm
        self.f_norm = args.use_f_norm
        self.m = args.m_1
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.Lambda = 1500.0
        self.CELoss = torch.nn.CrossEntropyLoss()
        self.weight = torch.nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.register_parameter('bias', None) # Biases are zero
        self.reset_parameters()  # weight initialization
        assert (self.w_norm == True and self.f_norm == False), 'Wrong implementation of A-Softmax loss.'
        assert self.m >= 1., 'margin m of asoftmax should >= 1.0'

    def forward(self, input, target):
        if self.w_norm:
            weight = L2Norm()(self.weight)
        else:
            weight = self.weight
        scores = F.linear(input, weight, self.bias)  # x @ weight.t() + bias(if any)
        index = torch.zeros_like(scores).scatter_(1, target, 1)  # the index of the target is 1, others are 0

        x_len = input.norm(dim=1)
        cos_theta = scores / (x_len.view(-1, 1).clamp(min=1e-12))  # cos_theta = a.b / |a|x|b|
        cos_theta = cos_theta.clamp(-1, 1)
        m_theta = self.m * torch.acos(cos_theta)  # acos returns arc cosine in radians
        k = (m_theta / 3.141592653589793).floor().detach()  # floor(), return the largest integer smaller or equal to
        cos_m_theta = torch.cos(m_theta)
        psi_theta = ((-1) ** k) * cos_m_theta - 2 * k
        psi_theta = psi_theta * x_len.view(-1, 1)  # ||x|| * psi_theta

        self.Lambda = max(self.LambdaMin, self.LambdaMax / (1 + 0.1 * self.it))
        self.it += 1
        scores_new = scores - scores * index / (1 + self.Lambda) + psi_theta * index / (1 + self.Lambda)

        _, label_pred = torch.max(scores_new, 1)
        correct = torch.sum(label_pred == target.view(-1)).item()

        return scores, self.CELoss(scores_new, target.view(-1)), correct

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)