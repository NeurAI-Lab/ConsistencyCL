# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *
import torch
from self_supervised.criterion import DINOLoss, NTXent
import sys
from torchvision import transforms


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay++.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    return parser


class CR(ContinualModel):
    NAME = 'cr'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(CR, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device)
        if self.args.pretext_task == 'dino':
            self.dino = DINOLoss(args.n_classes)
        if self.args.pretext_task == 'simclr':
            self.simclr = NTXent()
        if self.args.pretext_task == 'mae':
            self.l1_loss = torch.nn.L1Loss()

    def compute_pretext_task_loss(self, buf_outputs, buf_logits):

        # Alignment and uniform loss
        if self.args.pretext_task == 'align_uni':
            loss = self.args.align_weight * (buf_outputs - buf_logits.detach()).norm(p=2, dim=1).pow(2).mean() / 10 \
                    + self.args.uni_weight * (
                        torch.pdist(F.normalize(buf_outputs), p=2).pow(2).mul(-2).exp().mean().log())

        # Barlow twins
        elif self.args.pretext_task == 'barlow_twins':
            buf_outputs_norm = F.normalize(buf_outputs)
            buf_logits_norm = F.normalize(buf_logits)
            c = torch.mm(buf_outputs_norm.T, buf_logits_norm)
            c.div_(self.args.minibatch_size)
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(c).pow_(2).sum()
            loss = self.args.barlow_on_weight * on_diag + self.args.barlow_off_weight * off_diag

        # SimSiam loss
        elif self.args.pretext_task == 'simsiam':
            buf_outputs_norm = F.normalize(buf_outputs)
            buf_logits_norm = F.normalize(buf_logits)
            loss = -(buf_logits_norm * buf_outputs_norm).sum(dim=1).mean()

        # BYOL
        elif self.args.pretext_task == 'byol':
            buf_outputs_norm = F.normalize(buf_outputs)
            buf_logits_norm = F.normalize(buf_logits)
            loss = self.args.byol_weight * (2 - 2 * (buf_outputs_norm * buf_logits_norm).sum(dim=-1).mean())

        # Dino
        elif self.args.pretext_task == 'dino':
            loss = self.args.dino_weight * self.dino(F.normalize(buf_outputs), F.normalize(buf_logits))
        # Simclr
        elif self.args.pretext_task == 'simclr':
            loss = self.args.simclr_weight * self.simclr(F.normalize(buf_outputs), F.normalize(buf_logits))
        # Mutual information
        elif self.args.pretext_task == 'mi':
            EPS = sys.float_info.epsilon
            z = F.softmax(buf_outputs, dim=-1)
            zt = F.softmax(buf_logits, dim=-1)
            _, C = z.size()
            P_temp = (z.unsqueeze(2) * zt.unsqueeze(1)).sum(dim=0)
            P = ((P_temp + P_temp.t()) / 2) / P_temp.sum()
            P[(P < EPS).data] = EPS
            Pi = P.sum(dim=1).view(C, 1).expand(C, C).clone()
            Pj = P.sum(dim=0).view(1, C).expand(C, C).clone()
            Pi[(Pi < EPS).data] = EPS
            Pj[(Pj < EPS).data] = EPS
            loss = self.args.mi_weight * (P * (torch.log(Pi) + torch.log(Pj) - torch.log(P))).sum()
        # L1
        elif self.args.pretext_task == 'l1':
            loss = self.args.alpha * torch.pairwise_distance(buf_outputs, buf_logits, p=1).mean()
        # L2
        elif self.args.pretext_task == 'l2':
            loss = self.args.alpha * torch.pairwise_distance(buf_outputs, buf_logits, p=2).mean()
        # L_inf
        elif self.args.pretext_task == 'linf':
            loss = self.args.alpha * torch.pairwise_distance(buf_outputs, buf_logits, p=float('inf')).mean()
        # KL-Divergence
        elif self.args.pretext_task == 'kl':
            sim_logits = F.softmax(buf_logits)
            loss = self.args.alpha * F.kl_div(F.log_softmax(buf_outputs), sim_logits)
        # Mean squared error
        else:
            loss = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

        return loss

    def observe(self, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()
        outputs = self.net(inputs)
        # CE for current task samples
        loss = self.loss(outputs, labels)
        loss_1 = torch.tensor(0)
        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)

            # Pretext task
            loss_1 = self.compute_pretext_task_loss(buf_outputs, buf_logits)
            loss += loss_1

            # CE for buffered images
            buf_inputs_2, buf_labels_2, _ = self.buffer.get_data(
                self.args.batch_size, transform=self.transform)
            buf_outputs_2 = self.net(buf_inputs_2)
            loss += self.args.beta * self.loss(buf_outputs_2, buf_labels_2)

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=outputs.data)

        return loss.item(), loss_1.item()
