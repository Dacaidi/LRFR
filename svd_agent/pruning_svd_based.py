from svd_agent.pruning_svd_agent import PruningSVDAgent
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import find_module_name
from utils.utils import find_module_by_name


class PruningSVDAgentAvg(PruningSVDAgent):
    def __init__(self, model, config, task_counts, mask):
        super().__init__(model, config, task_counts)
        self.mask_idx = -1
        self.mask = mask

    def compute_cov(self, module, fea_in, fea_out):
        if isinstance(module, nn.Linear):
            self.update_cov(torch.mean(fea_in[0], 0, True), module.weight)

        if isinstance(module, nn.BatchNorm2d):
            self.mask_idx += 1

        elif isinstance(module, nn.Conv2d):
            # shortcut现在的版本没有经过select层，如果经过select层出来之后shortcut也可以有更小的rank，暂时没考虑
            name = find_module_name(module, self.model)
            module_in_backbone = find_module_by_name(self.backbone[0], name)

            kernel_size = module.kernel_size
            stride = module.stride
            padding = module.padding
            # calculate how many slides in the input feature map
            slides = (fea_in[0].shape[-1] - kernel_size[0] + 2 * padding[0]) // stride[0] + 1
            kernels_for_this_layer = module_in_backbone.in_channels * kernel_size[0] * kernel_size[1]

            # initialize the plain feature map according to the backbone network size
            fea_in_plain = torch.zeros(slides * slides, kernels_for_this_layer)
            if torch.cuda.is_available():
                fea_in_plain = fea_in_plain.cuda()

            fea_in_ = F.unfold(torch.mean(fea_in[0], 0, True), kernel_size=kernel_size, padding=padding, stride=stride)
            fea_in_ = fea_in_.permute(0, 2, 1)
            fea_in_ = fea_in_.reshape(-1, fea_in_.shape[-1])

            if 'shortcut' in name:
                self.update_cov(fea_in_, module_in_backbone.weight)
            elif self.mask_idx > -1 and 'shortcut' not in name:
                # fit the aggregated feature map into the plain feature map in appropriate positions
                for i in range(len(self.mask[self.mask_idx])):
                    if self.mask[self.mask_idx][i] == 0:
                        continue
                    else:
                        kernel_size_square = kernel_size[0] * kernel_size[0]
                        fea_in_plain[:, i * kernel_size_square:((i+1) * kernel_size_square)] = \
                            fea_in_[:, 0: kernel_size_square]
                        fea_in_ = fea_in_[:, kernel_size_square:]
                self.update_cov(fea_in_plain, module_in_backbone.weight)
            else:
                self.update_cov(fea_in_, module_in_backbone.weight)

        torch.cuda.empty_cache()
        return None

    def update_cov(self, fea_in, k):
        cov = torch.mm(fea_in.transpose(0, 1), fea_in)
        if len(self.fea_in[k]) == 0:
            self.fea_in[k] = cov
        else:
            self.fea_in[k] = self.fea_in[k] + cov

# def svd_based(config):
#     return SVDAgentAvg(config)
