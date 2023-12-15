import numpy as np
import torch
import torch.nn as nn
from models.channel_selection import channel_selection
from models.preact_resnet import PreActResNet, PreActBlock
from types import MethodType


def prun_model(source_model, prun_ratio):
    # get bn weights dimensions
    total_number_of_bn_weights = 0
    for m in source_model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total_number_of_bn_weights += m.weight.data.shape[0]

    aggregated_bn_weights = torch.zeros(total_number_of_bn_weights)
    index = 0
    # collect all bn weights in aggregated_bn_weights
    for m in source_model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            aggregated_bn_weights[index:(index + size)] = m.weight.data.abs().clone()
            index += size

    sorted_bn_weights, indices = torch.sort(aggregated_bn_weights)
    threshold_index = int(total_number_of_bn_weights * prun_ratio / 100)
    threshold = sorted_bn_weights[threshold_index]

    pruned = 0
    # how many channels are not pruned
    cfg = []
    # mask for each layer (1 for not pruned, 0 for pruned)
    cfg_mask = []

    for k, m in enumerate(source_model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(threshold).float()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            # print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            # format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')

    print('====================== Refine Pruned Network =======================')

    pruned_model = PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=10, cfg=cfg)
    pruned_model.last = nn.ModuleDict()

    for i in range(1, 11):
        pruned_model.last[str(i)] = nn.Linear(512, 10, bias=True)

    # Replace the task-dependent function
    pruned_model.logits = MethodType(new_logits, pruned_model)

    if torch.cuda.is_available():
        pruned_model.cuda()

    old_modules = list(source_model.modules())
    new_modules = list(pruned_model.modules())
    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    conv_count = 0

    # 思路一，根据新模型的名字，找到旧模型中对应的层，然后将旧模型中的参数复制到新模型中，但是这样新模型和旧模型结构不一样，需要重新写一个训练函数（凉了）。
    # 思路二，生成一个和旧模型层次结构完全一样的新模型(因为旧模型最后有10个全连接层)，这样就可以直接将旧模型的参数复制到新模型中，不需要重新写训练函数（用的这个）。
    for layer_id in range(len(new_modules)):
        m0 = old_modules[layer_id]
        m1 = new_modules[layer_id]
        if isinstance(m1, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))

            if isinstance(new_modules[layer_id + 1], nn.ModuleDict):
                continue

            if isinstance(new_modules[layer_id + 1], channel_selection):
                # If the next layer is the channel selection layer, then the current batchnorm 2d layer won't be pruned.
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()

                # We need to set the channel selection layer.
                m2 = new_modules[layer_id + 1]
                m2.indexes.data.zero_()
                m2.indexes.data[idx1.tolist()] = 1.0

                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):
                    end_mask = cfg_mask[layer_id_in_cfg]
            else:
                m1.weight.data = m0.weight.data[idx1.tolist()].clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                m1.running_mean = m0.running_mean[idx1.tolist()].clone()
                m1.running_var = m0.running_var[idx1.tolist()].clone()
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                    end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m1, nn.Conv2d):
            if conv_count == 0:
                m1.weight.data = m0.weight.data.clone()
                conv_count += 1
                continue
            if isinstance(new_modules[layer_id - 1], channel_selection) or isinstance(new_modules[layer_id - 1],
                                                                                      nn.BatchNorm2d):
                # This covers the convolutions in the residual block.
                # The convolutions are either after the channel selection layer or after the batch normalization layer.
                conv_count += 1
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()

                # If the current convolution is not the last convolution in the residual block, then we can change the
                # number of output channels. Currently, we use `conv_count` to detect whether it is such convolution.
                if conv_count % 2 != 1:
                    w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()
                continue

            # We need to consider the case where there are downsampling convolutions.
            # For these convolutions, we just copy the weights.
            m1.weight.data = m0.weight.data.clone()
        elif isinstance(m1, nn.Linear):
            # do not prun for linear layers
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()

    return pruned_model, cfg, cfg_mask


# Redefine the task-dependent function
def new_logits(self, x):
    outputs = {}
    for task, func in self.last.items():
        outputs[task] = func(x)
    return outputs
