from .agent import Agent
import optim
import torch
import numpy as np
import re
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
from utils.prun import prun_model
from utils.utils import pruned_weight_to_original_model_conv
from svd_agent.pruning_svd_based import PruningSVDAgentAvg
from models.channel_selection import channel_selection
from utils.utils import find_module_name
from utils.utils import find_module_by_name
import pdb


class SVDAgent(Agent):
    def __init__(self, config):
        super().__init__(config)

        self.fea_in_hook = {}
        self.fea_in = defaultdict(dict)
        self.fea_in_count = defaultdict(int)

        self.drop_num = 0

        self.regularization_terms = {}
        self.reg_params = {n: p for n,
        p in self.model.named_parameters() if 'bn' in n or 'shortcut.1' in n}
        self.empFI = False
        self.svd_lr = self.config['model_lr']  # first task
        self.init_model_optimizer()
        self.bn_parameters = {}

        self.params_json = {p: n for n, p in self.model.named_parameters()}

    def init_model_optimizer(self):
        if self.config['model_type'] == 'resnet':
            fea_params = [p for n, p in self.model.named_parameters(
            ) if not bool(re.match('last', n)) and 'bn' not in n and 'shortcut.1' not in n]
        elif self.config['model_type'] == 'preact_resnet':
            fea_params = [p for n, p in self.model.named_parameters(
            ) if not bool(re.match('last', n)) and 'bn' not in n]
        # All parameter for the classfier
        cls_params_all = list(
            p for n, p in self.model.named_children() if bool(re.match('last', n)))[0]
        # classfier's parameter for current task'
        cls_params = list(cls_params_all[str(self.task_count + 1)].parameters())
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
                                                                    milestones=self.config['schedule'],
                                                                    gamma=self.config['gamma'])

    def train_task(self, train_loader, val_loader=None, task_number=None):
        # 1.Learn the parameters for current task
        self.train_model(train_loader, val_loader)

        # 2.pruning the parameters
        if self.config['pruning']:
            pruned_model, cfg, cfg_mask = prun_model(self.model, self.config['pruning_ratio'])

        # 3. generate an agent for refinement
        self.pruned_models.append(PruningSVDAgentAvg(pruned_model, self.config, task_number, cfg_mask))

        # if self.fea_in is not none, then we need to extract a subset of the fea_in for pruned network training
        # 需不需要clone一下
        if int(task_number) > 1:
            layer_id_in_cfg = -1
            for m in pruned_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    layer_id_in_cfg += 1
                elif isinstance(m, nn.Conv2d):
                    name = find_module_name(m, pruned_model)
                    module_in_backbone = find_module_by_name(self.model, name)
                    if layer_id_in_cfg == -1 or 'shortcut' in name:
                        self.pruned_models[-1].fea_in[m.weight] = self.fea_in[module_in_backbone.weight].clone()
                    else:
                        kernel_size = m.kernel_size
                        kernel_size_square = kernel_size[0] * kernel_size[0]
                        if torch.cuda.is_available():
                            self.pruned_models[-1].fea_in[m.weight] = torch.zeros(m.in_channels * kernel_size_square,
                                                                                  m.in_channels * kernel_size_square).cuda()
                        else:
                            self.pruned_models[-1].fea_in[m.weight] = torch.zeros(m.in_channels * kernel_size_square,
                                                                                  m.in_channels * kernel_size_square)
                        j = 0
                        for i in range(len(cfg_mask[layer_id_in_cfg])):
                            if cfg_mask[layer_id_in_cfg][i] == 0:
                                continue
                            else:
                                index_start_in_backbone = i * kernel_size_square
                                index_end_in_backbone = (i + 1) * kernel_size_square
                                index_start_in_pruned = j * kernel_size_square
                                index_end_in_pruned = (j + 1) * kernel_size_square
                                self.pruned_models[-1].fea_in[m.weight][index_start_in_pruned:index_end_in_pruned,
                                index_start_in_pruned:index_end_in_pruned] = self.fea_in[module_in_backbone.weight][
                                                                             index_start_in_backbone:index_end_in_backbone,
                                                                             index_start_in_backbone:index_end_in_backbone].clone()
                                j += 1

        # 4. refine the model and update the cov
        self.pruned_models[int(task_number) - 1].cfg = cfg
        # the cov is stored in the self.fea_in
        # After the first task, the pruned model's fea_in is linked to the backbone's fea_in
        if int(task_number) == 1:
            self.fea_in = self.pruned_models[int(task_number) - 1].train_task(self.model, train_loader, val_loader)
        else:
            self.pruned_models[int(task_number) - 1].model_optimizer.get_eigens_wo_grad(self.pruned_models[-1].fea_in)
            self.pruned_models[int(task_number) - 1].model_optimizer.get_transforms_wo_grad()
            self.pruned_models[-1].fea_in = self.fea_in
            fea_in = self.pruned_models[int(task_number) - 1].train_task(self.model, train_loader, val_loader)
            self.fea_in = fea_in

        self.task_count += 1

        if self.task_count < self.num_task or self.num_task is None:
            if self.reset_model_optimizer:  # Reset model optimizer before learning each task
                self.log('Classifier Optimizer is reset!')
                self.svd_lr = self.config['svd_lr']
                self.init_model_optimizer()

            # 计算svd
            self.model_optimizer.get_eigens(self.fea_in)
            # 获得零空间
            self.model_optimizer.get_transforms()

            self.model.zero_grad()

        # copy the parameters of pruned model to the original model
        old_modules = list(self.model.modules())
        new_modules = list(pruned_model.modules())
        layer_id_in_cfg = 0
        start_mask = torch.ones(3)
        end_mask = cfg_mask[layer_id_in_cfg]
        conv_count = 0

        for layer_id in range(len(new_modules)):
            # backbone model
            m0 = old_modules[layer_id]
            # pruned model
            m1 = new_modules[layer_id]
            if isinstance(m1, nn.BatchNorm2d):
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))

                if isinstance(new_modules[layer_id + 1], nn.ModuleDict):
                    # If the next layer is the classifier layer, then copy the weight exactly
                    m0.weight.data = m1.weight.data.clone()
                    m0.bias.data = m1.bias.data.clone()
                    m0.running_mean = m1.running_mean.clone()
                    m0.running_var = m1.running_var.clone()
                    continue

                if isinstance(new_modules[layer_id + 1], channel_selection):
                    # If the next layer is the channel selection layer, copy the weight and bias
                    # channel selection layer need not be reset
                    m0.weight.data = m1.weight.data.clone()
                    m0.bias.data = m1.bias.data.clone()
                    m0.running_mean = m1.running_mean.clone()
                    m0.running_var = m1.running_var.clone()

                    layer_id_in_cfg += 1
                    start_mask = end_mask.clone()
                    if layer_id_in_cfg < len(cfg_mask):
                        end_mask = cfg_mask[layer_id_in_cfg]
                else:
                    # reverse process of the prun.py
                    m0.weight.data[idx1.tolist()] = m1.weight.data.clone()
                    m0.bias.data[idx1.tolist()] = m1.bias.data.clone()
                    m0.running_mean[idx1.tolist()] = m1.running_mean.clone()
                    m0.running_var[idx1.tolist()] = m1.running_var.clone()

                    layer_id_in_cfg += 1
                    start_mask = end_mask.clone()
                    if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                        end_mask = cfg_mask[layer_id_in_cfg]
            elif isinstance(m1, nn.Conv2d):
                if conv_count == 0:
                    m0.weight.data = m1.weight.data.clone()
                    conv_count += 1
                    continue
                if isinstance(new_modules[layer_id - 1], channel_selection) or isinstance(new_modules[layer_id - 1],
                                                                                          nn.BatchNorm2d):
                    # This covers the convolutions in the residual block. The convolutions are either after the
                    # channel selection layer or after the batch normalization layer.
                    conv_count += 1
                    idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))

                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))

                    # If the current convolution is not the last convolution in the residual block, then we can
                    # change the number of output channels. Currently, we use `conv_count` to detect whether it
                    # is such convolution.
                    if conv_count % 2 != 1:
                        pruned_weight_to_original_model_conv(m1, m0, idx0, idx1)
                    else:
                        m0.weight.data[:, idx0.tolist(), :, :] = m1.weight.data.clone()
                    continue
                # We need to consider the case where there are down sampling convolutions.
                # For these convolutions, we just copy the weights.
                m0.weight.data = m1.weight.data.clone()
            elif isinstance(m1, nn.Linear):
                # do not prun for linear layers
                m0.weight.data = m1.weight.data.clone()
                m0.bias.data = m1.bias.data.clone()

        bn_params = []
        layer_id_in_cfg = 0
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                if layer_id_in_cfg == len(cfg_mask) - 1:
                    bn_params.append((m.weight.clone(), m.bias.clone(), m.running_mean.clone(),
                                      m.running_var.clone()))
                else:
                    mask = cfg_mask[layer_id_in_cfg]
                    weight = torch.mul(m.weight.data, mask)
                    bias = torch.mul(m.bias.data, mask)
                    bn_params.append((weight, bias, m.running_mean.clone(), m.running_var.clone()))
                    layer_id_in_cfg += 1

            #  Set the backbone model's pruned bn weights and bias to zero
            # layer_id_in_cfg = 0
            # for k, m in enumerate(self.model.modules()):
            #     if isinstance(m, nn.BatchNorm2d):
            #         mask = cfg_mask[layer_id_in_cfg]
            #         m.weight.data.mul_(mask)
            #         m.bias.data.mul_(mask)
            #         layer_id_in_cfg += 1
            #         if layer_id_in_cfg == len(cfg_mask)-1:
            #             break

            # save all the bn parameters
            # bn_params = []
            # for m in self.model.modules():
            #     if isinstance(m, torch.nn.BatchNorm2d):
            #         bn_params.append((m.weight, m.bias))

        self.bn_parameters[str(task_number)] = bn_params

            # if self.reg_params:
            #     if len(self.regularization_terms) == 0:
            #         self.regularization_terms = {'importance': defaultdict(
            #             list), 'task_param': defaultdict(list)}
            #     importance = self.calculate_importance(train_loader)
            #     for n, p in self.reg_params.items():
            #         self.regularization_terms['importance'][n].append(
            #             importance[n].unsqueeze(0))
            #         self.regularization_terms['task_param'][n].append(
            #             p.unsqueeze(0).clone().detach())
            # Use a new slot to store the task-specific information

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

        self.model_optimizer.get_eigens(self.fea_in)

        self.model_optimizer.get_transforms()
        for h in handles:
            h.remove()
        torch.cuda.empty_cache()

    def change_bn_parameters(self, task_number):
        bn_index = 0
        original_bn_params = []
        bn_params = self.bn_parameters[str(task_number)]
        for m in self.model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                original_bn_params.append((m.weight.data.clone(), m.bias.data.clone(), m.running_mean.clone(),
                                           m.running_var.clone()))
                weight, bias, mean, var = bn_params[bn_index]
                m.weight.data = weight
                m.bias.data = bias
                m.running_mean = mean
                m.running_var = var
                bn_index += 1
        return original_bn_params

    def recover_bn_parameters(self, original_bn_params):
        bn_index = 0
        for m in self.model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                weight, bias, mean, var = original_bn_params[bn_index]
                m.weight.data = weight
                m.bias.data = bias
                m.running_mean = mean
                m.running_var = var
                bn_index += 1

    # def calculate_importance(self, dataloader):
    #     self.log('computing EWC')
    #     importance = {}
    #     for n, p in self.reg_params.items():
    #         importance[n] = p.clone().detach().fill_(0)
    #
    #     mode = self.model.training
    #     self.model.eval()
    #     for _, (inputs, targets, task) in enumerate(dataloader):
    #         if self.config['gpu']:
    #             inputs = inputs.cuda()
    #             targets = targets.cuda()
    #
    #         output = self.model.forward(inputs)
    #
    #         if self.empFI:
    #             ind = targets
    #         else:
    #             task_name = task[0] if self.multihead else 'ALL'
    #             pred = output[task_name] if not isinstance(self.valid_out_dim, int) else output[task_name][:,
    #                                                                                      :self.valid_out_dim]
    #             ind = pred.max(1)[1].flatten()
    #
    #         loss = self.criterion(output, ind, task, regularization=False)
    #         self.model.zero_grad()
    #         loss.backward()
    #
    #         for n, p in importance.items():
    #             if self.reg_params[n].grad is not None:
    #                 p += ((self.reg_params[n].grad ** 2)
    #                       * len(inputs) / len(dataloader))
    #
    #     return importance

    # def reg_loss(self):
    #     self.reg_step += 1
    #     reg_loss = 0
    #     for n, p in self.reg_params.items():
    #         importance = torch.cat(
    #             self.regularization_terms['importance'][n], dim=0)
    #         old_params = torch.cat(
    #             self.regularization_terms['task_param'][n], dim=0)
    #         new_params = p.unsqueeze(0).expand(old_params.shape)
    #         reg_loss += (importance * (new_params - old_params) ** 2).sum()
    #
    #     self.summarywritter.add_scalar(
    #         'reg_loss', reg_loss, self.reg_step)
    #     return reg_loss
