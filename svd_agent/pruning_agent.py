import time
import torch
import torch.nn as nn
from types import MethodType
from tensorboardX import SummaryWriter
from datetime import datetime

from utils.metric import accumulate_acc, AverageMeter, Timer
from utils.utils import count_parameter, factory
import optim


class PruningAgent(nn.Module):
    def __init__(self, model, agent_config, task_counts):
        '''
        :param agent_config (dict): lr=float,momentum=float,weight_decay=float,
                                    schedule=[int],  # The last number in the list is the end of epoch
                                    model_type=str,model_name=str,out_dim={task:dim},model_weights=str
                                    force_single_head=bool
                                    print_freq=int
                                    gpuid=[int]
        '''
        super().__init__()
        self.log = print if agent_config['print_freq'] > 0 else lambda \
                *args: None  # Use a void function to replace the print
        self.config = agent_config
        # self.log(agent_config)
        # If out_dim is a dict, there is a list of tasks. The model will have a head for each task.
        self.model = model
        self.task_count = task_counts
        self.multihead = True if len(
            self.config['out_dim']) > 1 else False  # A convenience flag to indicate multi-head/task

        self.criterion_fn = nn.CrossEntropyLoss()

        self.clf_param_num = count_parameter(self.model)
        self.summarywritter = SummaryWriter(
            './log/' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        if self.config['gpu']:
            self.model = self.model.cuda()
            self.criterion_fn = self.criterion_fn.cuda()
        self.log('#param of model:{}'.format(self.clf_param_num))

    def init_model_optimizer(self):
        model_optimizer_arg = {'params': self.model.parameters(),
                               'lr': self.config['model_lr'],
                               'weight_decay': self.config['model_weight_decay']}
        if self.config['model_optimizer'] in ['SGD', 'RMSprop']:
            model_optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['model_optimizer'] in ['Rprop']:
            model_optimizer_arg.pop('weight_decay')
        elif self.config['model_optimizer'] in ['amsgrad', 'Adam']:
            if self.config['model_optimizer'] == 'amsgrad':
                model_optimizer_arg['amsgrad'] = True
            self.config['model_optimizer'] = 'Adam'

        self.model_optimizer = getattr(
            optim, self.config['model_optimizer'])(**model_optimizer_arg)
        self.model_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.model_optimizer,
                                                                    milestones=self.config['schedule'],
                                                                    gamma=0.1)

    def train_task(self, train_loader, val_loader=None):
        raise NotImplementedError

    def train_epoch(self, train_loader, epoch, count_cls_step):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()

        end = time.time()
        for i, (inputs, target, task) in enumerate(train_loader):
            # print("*"*100)
            # print(inputs.mean())
            count_cls_step += 1
            data_time.update(time.time() - end)  # measure data loading time

            if self.config['gpu']:
                inputs = inputs.cuda()
                target = target.cuda()
            output = self.model.forward(inputs)

            loss = self.criterion(output, target, task)

            acc = accumulate_acc(output, target, task, acc)

            self.model_optimizer.zero_grad()

            loss.backward()

            self.model_optimizer.step()

            # self.model_scheduler.step(epoch)

            batch_time.update(time.time() - end)
            end = time.time()

            losses.update(loss, inputs.size(0))

            if ((self.config['print_freq'] > 0) and (i % self.config['print_freq'] == 0)) or (i + 1) == len(
                    train_loader):
                self.log('[{0}/{1}]\t'
                         '{batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                         '{data_time.val:.4f} ({data_time.avg:.4f})\t'
                         '{loss.val:.3f} ({loss.avg:.3f})\t'
                         '{acc.val:.2f} ({acc.avg:.2f})'.format(
                    i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, acc=acc))
        self.log(' * Train Acc {acc.avg:.3f}'.format(acc=acc))

        return losses.avg, acc.avg

    def train_model(self, train_loader, val_loader=None):
        count_cls_step = 0

        for epoch in range(self.config['schedule_pruned'][-1]):

            # Config the model and optimizer
            self.log('Epoch:{0}'.format(epoch))
            self.model.train()
            # self.hypermodel.eval()

            for param_group in self.model_optimizer.param_groups:
                self.log('LR:', param_group['lr'])

            self.log('Itr\t\tTime\t\t  Data\t\t  Loss\t\tAcc')
            # end = time.time()
            losses, acc = self.train_epoch(train_loader, epoch, count_cls_step)
            self.model_scheduler.step()
            # print('one epoch time: {}'.format(time.time() - end))
            # Evaluate the performance of current task
            if val_loader is not None:
                self.validation(val_loader)

    def validation(self, dataloader):
        # This function doesn't distinguish tasks.
        batch_timer = Timer()
        val_acc = AverageMeter()
        losses = AverageMeter()
        batch_timer.tic()

        # self.hypermodel.eval()
        self.model.eval()

        for i, (inputs, target, task) in enumerate(dataloader):

            with torch.no_grad():
                if self.config['gpu']:
                    inputs = inputs.cuda()
                    target = target.cuda()

                output = self.model.forward(inputs)
                loss = self.criterion(output, target, task)
            losses.update(loss, inputs.size(0))
            for t in output.keys():
                output[t] = output[t].detach()
            # Summarize the performance of all tasks, or 1 task, depends on dataloader.
            # Calculated by total number of data.
            val_acc = accumulate_acc(output, target, task, val_acc)

        self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
                 .format(acc=val_acc, time=batch_timer.toc()))
        self.log(' * Val loss {loss.avg:.3f}, Total time {time:.2f}'
                 .format(loss=losses, time=batch_timer.toc()))
        return val_acc.avg

    def criterion(self, preds, targets, tasks):
        loss = self.cross_entropy(preds, targets, tasks)
        return loss

    def cross_entropy(self, preds, targets, tasks):
        # The inputs and targets could come from single task or a mix of tasks
        # The network always makes the predictions with all its heads
        # The criterion will match the head and task to calculate the loss
        if self.multihead:
            loss = 0
            for t, t_preds in preds.items():
                # The index of inputs that matched specific task
                inds = [i for i in range(len(tasks)) if tasks[i] == t]
                if len(inds) > 0:
                    t_preds = t_preds[inds]
                    t_target = targets[inds]
                    # restore the loss from average
                    loss += self.criterion_fn(t_preds, t_target) * len(inds)
            # Average the total loss by the mini-batch size
            loss /= len(targets)
        else:
            pred = preds['All']
            if isinstance(self.valid_out_dim,
                          int):  # (Not 'ALL') Mask out the outputs of unseen classes for incremental class scenario
                pred = preds['All'][:, :self.valid_out_dim]
            loss = self.criterion_fn(pred, targets)
        return loss

    def add_valid_output_dim(self, dim=0):
        # This function is kind of ad-hoc, but it is the simplest way to support incremental class learning
        self.log('Incremental class: Old valid output dimension:',
                 self.valid_out_dim)
        if self.valid_out_dim == 'ALL':
            self.valid_out_dim = 0  # Initialize it with zero
        self.valid_out_dim += dim
        self.log('Incremental class: New Valid output dimension:',
                 self.valid_out_dim)
        return self.valid_out_dim
