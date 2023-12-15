import optim
import torch
import re
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
from utils.prun import prun_model
from svd_agent.pruning_agent import PruningAgent
import pdb

class PruningSVDAgent(PruningAgent):
    def __init__(self, model, config, task_counts):
        super().__init__(model, config, task_counts)

        self.fea_in_hook = {}
        self.fea_in = defaultdict(dict)
        self.fea_in_count = defaultdict(int)
        self.cfg = []
        self.backbone = []

        self.drop_num = 0

        self.empFI = False
        self.svd_lr = self.config['model_lr']  # first task
        self.init_model_optimizer()

    def init_model_optimizer(self):
        # 可能要对应改原版！！
        if self.config['model_type'] == 'resnet':
            fea_params = [p for n, p in self.model.named_parameters(
            ) if not bool(re.match('last', n)) and 'bn' not in n and 'shortcut.1' not in n]
        elif self.config['model_type'] == 'preact_resnet':
            fea_params = [p for n, p in self.model.named_parameters(
            ) if not bool(re.match('last', n)) and 'bn' not in n and 'select' not in n]
        # All parameter for the classfier
        cls_params_all = list(
            p for n, p in self.model.named_children() if bool(re.match('last', n)))[0]
        # classfier's parameter for current task'
        cls_params = list(cls_params_all[self.task_count].parameters())
        bn_params = [p for n, p in self.model.named_parameters() if 'bn' in n or 'shortcut.1' in n]
        model_optimizer_arg = {'params': [{'params': fea_params, 'svd': True, 'lr': self.svd_lr,
                                           'thres': self.config['svd_thres']},
                                          {'params': cls_params, 'weight_decay': 0.0,
                                           'lr': self.config['head_lr']},
                                          {'params': bn_params, 'lr': self.config['bn_lr']}],
                               'lr': self.config['model_lr'],
                               'weight_decay': self.config['model_weight_decay']}
        if self.config['model_optimizer'] in ['SGD', 'RMSprop']:
            model_optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['model_optimizer'] in ['Rprop']:
            model_optimizer_arg.pop('weight_decay')
        elif self.config['model_optimizer'] in ['amsgrad']:
            if self.config['model_optimizer'] == 'amsgrad':
                model_optimizer_arg['amsgrad'] = True
            self.config['model_optimizer'] = 'Adam'

        self.model_optimizer = getattr(
            optim, self.config['model_optimizer'])(**model_optimizer_arg)
        self.model_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.model_optimizer,
                                                                    milestones=self.config['schedule_pruned'],
                                                                    gamma=self.config['gamma'])

    def train_task(self, backbone_model, train_loader, val_loader=None):
        # 1.Learn the parameters for current task
        self.train_model(train_loader, val_loader)
        # 2. get the cov matrix
        self.backbone.append(backbone_model)
        with torch.no_grad():
            fea_in = self.update_optim_transforms(train_loader)

        self.model.zero_grad()
        self.model_optimizer.zero_grad()

        return fea_in

    def update_optim_transforms(self, train_loader):
        modules = [m for n, m in self.model.named_modules() if hasattr(
            m, 'weight') and not bool(re.match('last', n))]
        handles = []
        for m in modules:
            handles.append(m.register_forward_hook(hook=self.compute_cov))

        for i, (inputs, target, task) in enumerate(train_loader):
            if self.config['gpu']:
                inputs = inputs.cuda()
            self.model.forward(inputs)
            # Reset the mask_idx
            self.mask_idx = -1

        # calculate do svd
        # do it in the backbone model
        # self.model_optimizer.get_eigens(self.fea_in)

        # calculate null space
        # do it in the backbone model
        # self.model_optimizer.get_transforms()

        for h in handles:
            h.remove()
        torch.cuda.empty_cache()

        return self.fea_in
