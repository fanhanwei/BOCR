# Code for "AMC: AutoML for Model Compression and Acceleration on Mobile Devices"
# Yihui He*, Ji Lin*, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han
# {jilin, songhan}@mit.edu

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import AverageMeter, accuracy, prGreen
from lib.data import get_split_dataset
import math
from env.rewards import *
import numpy as np
import copy
import gc


class ChannelPruningEnv:
    """
    Env for channel pruning search
    """
    def __init__(self, model, checkpoint, data, preserve_ratio, args, n_data_worker=4,
                 batch_size=256, export_model=False, use_new_input=False):
        # default setting
        self.prunable_layer_types = [torch.nn.modules.conv.Conv2d, torch.nn.modules.linear.Linear]

        # save options
        self.model = model
        self.checkpoint = checkpoint
        self.n_data_worker = n_data_worker
        self.batch_size = batch_size
        self.data_type = data
        self.preserve_ratio = preserve_ratio

        # options from args
        self.args = args
        self.lbound = args.lbound
        self.rbound = args.rbound

        self.use_real_val = args.use_real_val

        self.n_calibration_batches = args.n_calibration_batches
        self.n_points_per_layer = args.n_points_per_layer
        self.channel_round = args.channel_round
        self.acc_metric = args.acc_metric
        self.data_root = args.data_root

        self.export_model = export_model
        self.use_new_input = use_new_input

        # sanity check
        assert self.preserve_ratio > self.lbound, 'Error! You can make achieve preserve_ratio smaller than lbound!'

        # prepare data
        self._init_data()

        # build indexs
        self._build_index()
        self.n_prunable_layer = len(self.prunable_idx)

        # extract information for preparing
        self._extract_layer_information()
        self.all_layers = sorted(self.layer_info_dict.keys())

        # build embedding (static part)
        self._build_state_embedding()

        # build reward
        self.reset()  # restore weight
        self.org_acc1, self.org_acc5 = self._validate(self.val_loader, self.model)
        print('=> original acc top1: {:.3f}%, top5: {:.3f}%'.format(self.org_acc1, self.org_acc5))
        self.org_model_size = sum(self.wsize_list)
        print('=> original weight size: {:.4f} M param'.format(self.org_model_size * 1. / 1e6))
        self.org_flops = sum(self.flops_list)
        print('=> FLOPs:')
        print([self.layer_info_dict[idx]['flops']/1e6 for idx in self.all_layers])

        print('=> original FLOPs: {:.4f} M'.format(self.org_flops * 1. / 1e6))
        self.prunable_flop = sum([self.layer_info_dict[_]['flops'] for _ in self.prunable_idx])
        self.buffer_flop = sum([self.layer_info_dict[_]['flops'] for _ in self.buffer_idx])
        self.branch_flop = sum([self.layer_info_dict[_]['flops'] for _ in self.branch_idx])

        self.expected_preserve_computation = self.preserve_ratio * self.org_flops

        self.reward = eval(args.reward)

        #self.best_reward = -math.inf
        self.best_top1 = -math.inf
        self.best_top5 = -math.inf
        self.best_strategy = None
        self.best_d_prime_list = None

        self.org_w_size = sum(self.wsize_list)

    def reset(self):
        gc.collect()
        torch.cuda.empty_cache()
        # restore env by loading the checkpoint
        self.model.load_state_dict(self.checkpoint)
        self.cur_ind = 0
        self.strategy = []  # pruning strategy
        self.d_prime_list = []
        self.strategy_dict = copy.deepcopy(self.min_strategy_dict)
        # reset layer embeddings
        self.layer_embedding[:, -1] = 1.
        self.layer_embedding[:, -2] = 0.
        self.layer_embedding[:, -3] = 0.
        obs = self.layer_embedding[0].copy()
        obs[-2] = sum(self.wsize_list[1:]) * 1. / sum(self.wsize_list)
        self.extract_time = 0
        self.fit_time = 0
        self.val_time = 0
        # for share index
        self.visited = [False] * len(self.prunable_idx)
        self.index_buffer = {}
        return obs

    def set_export_path(self, path):
        self.export_path = path

    def prune_kernel(self, op_idx, preserve_ratio, preserve_idx=None):
        '''Return the real ratio'''
        m_list = list(self.model.modules())
        op = m_list[op_idx]
        assert (preserve_ratio <= 1.)

        if preserve_ratio == 1:  # do not prune
            return 1., op.weight.size(1), None  # TODO: should be a full index
            # n, c, h, w = op.weight.size()
            # mask = np.ones([c], dtype=bool)

        def format_rank(x):#四舍五入取整
            rank = int(np.around(x))
            return max(rank, 1)

        n, c = op.weight.size(0), op.weight.size(1)
        d_prime = format_rank(c * preserve_ratio)
        d_prime = int(np.ceil(d_prime * 1. / self.channel_round) * self.channel_round)#ceil向上取整
        if d_prime > c:
            d_prime = int(np.floor(c * 1. / self.channel_round) * self.channel_round)#floor向下取整

        extract_t1 = time.time()
        X = self.layer_info_dict[op_idx]['input_feat'].cuda()  # input after pruning of previous ops
        Y = self.layer_info_dict[op_idx]['output_feat'].cuda()  # fixed output from original model
        weight = op.weight.data#.cpu().numpy()
        # conv [C_out, C_in, ksize, ksize]
        # fc [C_out, C_in]
        op_type = 'Conv2D'
        if len(weight.shape) == 2:
            op_type = 'Linear'
            weight = weight[:, :, None, None]
        extract_t2 = time.time()
        self.extract_time += extract_t2 - extract_t1
        fit_t1 = time.time()

        if preserve_idx is None:  # not provided, generate new
            importance = weight.abs().sum((0, 2, 3))
            sorted_idx = torch.argsort(-importance)  # sum magnitude along C_in, sort descend
            preserve_idx = sorted_idx[:d_prime]  # to preserve index
        assert len(preserve_idx) == d_prime
        _mask = np.zeros(weight.shape[1], bool)
        mask = torch.tensor(_mask)
        mask[preserve_idx] = True
        _mask = mask.cpu().numpy()

        # reconstruct, X, Y <= [N, C]
        masked_X = X[:, mask]
        from lib.utils import least_square_sklearn
        if weight.shape[2] == 1:  # 1x1 conv or fc
            rec_weight, _ = torch.lstsq(input=Y, A=masked_X)
            rec_weight = rec_weight[:masked_X.shape[1]].t()
            rec_weight = rec_weight.reshape(-1, 1, 1, d_prime)  # (C_out, K_h, K_w, C_in')
            rec_weight = rec_weight.permute(0, 3, 1, 2)  # (C_out, C_in', K_h, K_w)
        else:
            #############################################################################################
            #support 3*3
            #############################################################################################
            X_fit = masked_X.reshape(masked_X.shape[0], -1)
            rec_weight, _ = torch.lstsq(input=Y, A=X_fit)
            rec_weight = rec_weight[:X_fit.shape[1]].t()
            rec_weight = rec_weight.reshape(-1, d_prime, 3, 3)  # (C_out, C_in', K_h, K_w)

        if not self.export_model:  # pad, pseudo compress
            rec_weight_pad = torch.zeros_like(weight)#np.zeros_like
            rec_weight_pad[:, mask, :, :] = rec_weight
            rec_weight = rec_weight_pad

        if op_type == 'Linear':
            rec_weight = rec_weight.squeeze()
            assert len(rec_weight.shape) == 2
        fit_t2 = time.time()
        self.fit_time += fit_t2 - fit_t1
        # now assign
        op.weight.data = rec_weight
        action = np.sum(_mask) * 1. / len(_mask)  # calculate the ratio
        if self.export_model:  # prune previous buffer ops
            prev_idx = self.all_layers[self.all_layers.index(op_idx) - 1]
            print('layers to be compressed: {} --> {}'.format(prev_idx, op_idx))
            for idx in range(prev_idx, op_idx):
                m = m_list[idx]
                if type(m) == nn.Conv2d:  
                    m.weight.data = m.weight.data[mask, :, :, :]
                    if m.groups == m.in_channels:# depthwise
                        m.groups = int(np.sum(_mask))
                elif type(m) == nn.BatchNorm2d:
                    m.weight.data = m.weight.data[mask]
                    m.bias.data = m.bias.data[mask]
                    m.running_mean.data = m.running_mean.data[mask]
                    m.running_var.data = m.running_var.data[mask]
        return action, d_prime, preserve_idx

    def _is_final_layer(self):
        return self.cur_ind == len(self.prunable_idx) - 1

    def _action_wall(self, action):
        assert len(self.strategy) == self.cur_ind

        action = float(action)
        action = np.clip(action, 0, 1)

        other_comp = 0
        this_comp = 0
        self.cur_buffer_flop = self._get_buffer_flops(self.prunable_idx[self.cur_ind])
        for i, idx in enumerate(self.prunable_idx):
            flop = self.layer_info_dict[idx]['flops']
            buffer_flop = self._get_buffer_flops(idx)

            if i == self.cur_ind - 1:
                if self.cur_buffer_flop == 0:
                    this_comp += flop * self.strategy_dict[idx][0]
                    other_comp += buffer_flop * self.strategy_dict[idx][0]
                else:
                    other_comp += (flop + buffer_flop) * self.strategy_dict[idx][0]
            elif i == self.cur_ind:
                this_comp += flop * self.strategy_dict[idx][1]
                # also add buffer here (influenced by ratio)
                this_comp += buffer_flop
            else:
                other_comp += flop * self.strategy_dict[idx][0] * self.strategy_dict[idx][1]
                # add buffer
                other_comp += buffer_flop * self.strategy_dict[idx][0]  # only consider input reduction
        other_comp += self.branch_flop
        self.expected_min_preserve = other_comp + this_comp * action
        max_preserve_ratio = (self.expected_preserve_computation - other_comp) * 1. / this_comp

        action = np.minimum(action, max_preserve_ratio)
        action = np.maximum(action, self.strategy_dict[self.prunable_idx[self.cur_ind]][0])  # impossible (should be)

        return action

    def _get_buffer_flops(self, idx):
        buffer_idx = self.buffer_dict[idx]
        buffer_flop = sum([self.layer_info_dict[_]['flops'] for _ in buffer_idx])
        return buffer_flop

    def _cur_flops(self):
        flops = 0
        for i, idx in enumerate(self.prunable_idx):
            c, n = self.strategy_dict[idx]  # input, output pruning ratio
            flops += self.layer_info_dict[idx]['flops'] * c * n
            # add buffer computation
            flops += self._get_buffer_flops(idx) * c  # downsample do not prune
        return flops + self.branch_flop

    def _cur_reduced(self):
        # return the reduced weight
        reduced = self.org_flops - self._cur_flops()
        return reduced

    def _init_data(self):
        # split the train set into train + val
        # for CIFAR, split 5k for val
        # for ImageNet, split 3k for val
        val_size = 5000 if 'cifar' in self.data_type else 3000
        self.train_loader, self.val_loader, n_class = get_split_dataset(self.data_type, self.batch_size,
                                                                        self.n_data_worker, val_size,
                                                                        data_root=self.data_root,
                                                                        use_real_val=self.use_real_val,
                                                                        shuffle=False)  # same sampling
        if self.use_real_val:  # use the real val set for eval, which is actually wrong
            print('*** USE REAL VALIDATION SET!')

    def _build_index(self):
        self.prunable_idx = []
        self.prunable_ops = []
        self.branch_idx = []
        self.layer_type_dict = {}
        self.strategy_dict = {}
        self.buffer_dict = {}
        this_buffer_list = []
        self.org_channels = []
        # build index and the min strategy dict
        for i, (name, m) in enumerate(self.model.named_modules()):
            if type(m) in self.prunable_layer_types:
                if type(m) == nn.Conv2d and 'downsample' in name:  #downsample conv in shortcut, do not prune
                    self.branch_idx.append(i)
                elif 'conv1' in name and name != 'conv1':
                    this_buffer_list.append(i)
                else:  # really prunable
                    self.prunable_idx.append(i)
                    self.prunable_ops.append(m)
                    self.layer_type_dict[i] = type(m)
                    self.buffer_dict[i] = this_buffer_list
                    this_buffer_list = []  # empty
                    self.org_channels.append(m.in_channels if type(m) == nn.Conv2d else m.in_features)

                    self.strategy_dict[i] = [self.lbound, 1.]

        self.strategy_dict[self.prunable_idx[0]][0] = 1  # modify the input
        self.strategy_dict[self.prunable_idx[-1]][1] = 1  # modify the output
        self.min_strategy_dict = copy.deepcopy(self.strategy_dict)
        self.buffer_idx = []
        for k, v in self.buffer_dict.items():
            self.buffer_idx += v

        print('=> Prunable layer idx: {}'.format(self.prunable_idx))
        print('=> length Prunable layer idx: {}'.format(len(self.prunable_idx)))
        print('=> Branch layer idx: {}'.format(self.branch_idx))
        print('=> Buffer layer idx: {}'.format(self.buffer_idx))
        print('=> Buffer dict: {}'.format(self.buffer_dict))
        print('=> Initial min strategy dict: {}'.format(self.min_strategy_dict))

        # added for supporting residual connections during pruning
        self.visited = [False] * len(self.prunable_idx)
        self.index_buffer = {}

    def _extract_layer_information(self):
        m_list = list(self.model.modules())

        self.data_saver = []
        self.layer_info_dict = dict()
        self.wsize_list = []
        self.flops_list = []
        self.k_out_list = []

        from lib.utils import measure_layer_for_pruning

        # extend the forward fn to record layer info
        def new_forward(m):
            def lambda_forward(x):
                m.input_feat = x.clone()
                measure_layer_for_pruning(m, x)
                y = m.old_forward(x)
                m.output_feat = y.clone()
                return y

            return lambda_forward

        for idx in self.prunable_idx + self.buffer_idx + self.branch_idx:  # get all
            m = m_list[idx]
            m.old_forward = m.forward
            m.forward = new_forward(m)

        # now let the image flow
        print('=> Extracting information...')
        with torch.no_grad():
            for i_b, (input, target) in enumerate(self.train_loader):  # use image from train set
                if i_b == self.n_calibration_batches:
                    break
                # print('num of info extraction:  ', i_b+1)
                self.data_saver.append((input.clone(), target.clone()))
                input_var = torch.autograd.Variable(input).cuda()

                # inference and collect stats
                _ = self.model(input_var)

                if i_b == 0:  # first batch
                    for idx in self.prunable_idx + self.buffer_idx +self.branch_idx:
                        self.layer_info_dict[idx] = dict()
                        self.layer_info_dict[idx]['params'] = m_list[idx].params
                        self.layer_info_dict[idx]['flops'] = m_list[idx].flops
                        self.wsize_list.append(m_list[idx].params)
                        self.flops_list.append(m_list[idx].flops)
                        self.k_out_list.append(m_list[idx].k_out)
                for idx in self.prunable_idx:
                    f_in_np = m_list[idx].input_feat.data#.cpu().numpy()
                    f_out_np = m_list[idx].output_feat.data#.cpu().numpy()
                    #print('\nidx: {}, input: {}, output: {}'.format(idx, f_in_np.shape, f_out_np.shape))
                    if len(f_in_np.shape) == 4:  # conv
                        if self.prunable_idx.index(idx) == 0:  # first conv
                            f_in2save, f_out2save = None, None
                        else:  
                            feat_out_H = f_out_np.shape[2]
                            feat_out_W = f_out_np.shape[3]
                            randx = np.random.randint(0, feat_out_H - 0, self.n_points_per_layer)
                            randy = np.random.randint(0, feat_out_W - 0, self.n_points_per_layer)
                            #print(randx, randy)
                            # input: [N, C, H, W]
                            self.layer_info_dict[idx][(i_b, 'randx')] = randx.copy()
                            self.layer_info_dict[idx][(i_b, 'randy')] = randy.copy()

                            f_out2save = f_out_np[:, :, randx, randy].permute(0, 2, 1) \
                                .reshape(self.batch_size * self.n_points_per_layer, -1)

                            if m_list[idx].weight.size(3) > 1:  # 3x3 conv
                                ########################################################################
                                #      3*3 conv special extract
                                ########################################################################
                                kh, kw = m_list[idx].kernel_size[0], m_list[idx].kernel_size[1]
                                dh, dw = m_list[idx].stride[0], m_list[idx].stride[1]

                                # Pad tensor to get the same output
                                x = F.pad(f_in_np, (1, 1, 1, 1))

                                # get all image windows of size (kh, kw) and stride (dh, dw)
                                patches = x.unfold(2, kh, dh).unfold(3, kw, dw)
                                #print(patches.shape)  # [n, c, h_out, w_out, k, k]
                                # Permute so that channels are next to patch dimension
                                patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # [n, h_out, w_out, c, k, k]
                                # View as [batch_size, height, width, channels*kh*kw]
                                #patches = patches.view(*patches.size()[:3], -1)

                                f_in2save = patches[:, randx, randy].reshape(self.batch_size * self.n_points_per_layer \
                                    , *patches.size()[3:])

                            else: # 1x1 conv
                                #print("1x1")
                                f_in2save = f_in_np[:, :, randx, randy].permute(0, 2, 1) \
                                    .reshape(self.batch_size * self.n_points_per_layer, -1)
                            
                    else:
                        assert len(f_in_np.shape) == 2
                        f_in2save = f_in_np
                        f_out2save = f_out_np
                    if self.prunable_idx.index(idx) != 0:
                        if 'input_feat' not in self.layer_info_dict[idx]:
                            self.layer_info_dict[idx]['input_feat'] = f_in2save.cpu()
                            self.layer_info_dict[idx]['output_feat'] = f_out2save.cpu()
                        else:
                            self.layer_info_dict[idx]['input_feat'] = torch.cat(
                                (self.layer_info_dict[idx]['input_feat'], f_in2save.cpu()),0)
                            self.layer_info_dict[idx]['output_feat'] = torch.cat(
                                (self.layer_info_dict[idx]['output_feat'], f_out2save.cpu()),0)
                        #print('length of layer_info_dict',len(self.layer_info_dict[idx]['input_feat']))
        
        self.importance_list = []
        for op_idx in self.prunable_idx[1:-1]:
            op = m_list[op_idx]
            weight = op.weight.data
            self.importance_list.append(abs(weight).sum((0, 2, 3)))

    def _regenerate_input_feature(self):
        # only re-generate the input feature
        m_list = list(self.model.modules())

        # delete old features
        for k, v in self.layer_info_dict.items():
            if 'input_feat' in v:
                v.pop('input_feat')

        # now let the image flow
        print('=> Regenerate features...')

        with torch.no_grad():
            for i_b, (input, target) in enumerate(self.data_saver):
                input_var = torch.autograd.Variable(input).cuda()

                # inference and collect stats
                _ = self.model(input_var)

                for idx in self.prunable_idx:
                    f_in_np = m_list[idx].input_feat.data.cpu().numpy()
                    if len(f_in_np.shape) == 4:  # conv
                        if self.prunable_idx.index(idx) == 0:  # first conv
                            f_in2save = None
                        else:
                            randx = self.layer_info_dict[idx][(i_b, 'randx')]
                            randy = self.layer_info_dict[idx][(i_b, 'randy')]
                            f_in2save = f_in_np[:, :, randx, randy].copy().transpose(0, 2, 1)\
                                .reshape(self.batch_size * self.n_points_per_layer, -1)
                    else:  # fc
                        assert len(f_in_np.shape) == 2
                        f_in2save = f_in_np.copy()
                    if 'input_feat' not in self.layer_info_dict[idx]:
                        self.layer_info_dict[idx]['input_feat'] = f_in2save
                    else:
                        self.layer_info_dict[idx]['input_feat'] = np.vstack(
                            (self.layer_info_dict[idx]['input_feat'], f_in2save))

    def _build_state_embedding(self):
        # build the static part of the state embedding
        layer_embedding = []
        module_list = list(self.model.modules())
        for i, ind in enumerate(self.prunable_idx):
            m = module_list[ind]
            this_state = []
            if type(m) == nn.Conv2d:
                this_state.append(i)  # index
                this_state.append(0)  # layer type, 0 for conv
                this_state.append(m.in_channels)  # in channels
                this_state.append(m.out_channels)  # out channels
                this_state.append(m.stride[0])  # stride
                this_state.append(m.kernel_size[0])  # kernel size
                this_state.append(np.prod(m.weight.size()))  # weight size
            elif type(m) == nn.Linear:
                this_state.append(i)  # index
                this_state.append(1)  # layer type, 1 for fc
                this_state.append(m.in_features)  # in channels
                this_state.append(m.out_features)  # out channels
                this_state.append(0)  # stride
                this_state.append(1)  # kernel size
                this_state.append(np.prod(m.weight.size()))  # weight size

            # this 3 features need to be changed later
            this_state.append(0.)  # reduced
            this_state.append(0.)  # rest
            this_state.append(1.)  # a_{t-1}
            layer_embedding.append(np.array(this_state))

        # normalize the state
        layer_embedding = np.array(layer_embedding, 'float')
        print('=> shape of embedding (n_layer * n_dim): {}'.format(layer_embedding.shape))
        self.layer_embedding_ori = layer_embedding.copy()
        assert len(layer_embedding.shape) == 2, layer_embedding.shape
        for i in range(layer_embedding.shape[1]):
            fmin = min(layer_embedding[:, i])
            fmax = max(layer_embedding[:, i])
#            layer_embedding[:, i] = layer_embedding[:, i] / fmax
            if fmax - fmin > 0:
                layer_embedding[:, i] = (layer_embedding[:, i] - fmin) / (fmax - fmin)

        self.layer_embedding = layer_embedding

    def _validate(self, val_loader, model, verbose=False):
        '''
        Validate the performance on validation set
        :param val_loader:
        :param model:
        :param verbose:
        :return:
        '''
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        criterion = nn.CrossEntropyLoss().cuda()
        # switch to evaluate mode
        model.eval()
        end = time.time()

        t1 = time.time()
        with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
                target = target.cuda(non_blocking=True)
                input_var = torch.autograd.Variable(input).cuda()
                target_var = torch.autograd.Variable(target).cuda()

                # compute output
                output = model(input_var)
                loss = criterion(output, target_var)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(prec1.item(), input.size(0))
                top5.update(prec5.item(), input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
        t2 = time.time()
        if verbose:
            print('* Test loss: %.3f    top1: %.3f    top5: %.3f    time: %.3f' %
                  (losses.avg, top1.avg, top5.avg, t2 - t1))
        return top1.avg, top5.avg

    def step_BO(self, action):
        # Pseudo prune and get the corresponding statistics. The real pruning happens till the end of all pseudo pruning
        action = self._action_wall(action)  # percentage to preserve
        preserve_idx = None

        # prune and update action
        action, d_prime, preserve_idx = self.prune_kernel(self.prunable_idx[self.cur_ind], action, preserve_idx)

        if self.export_model:  # export checkpoint
            print('# Pruning {}: ratio: {}, d_prime: {}'.format(self.cur_ind, action, d_prime))

        self.strategy.append(action)  # save action to strategy
        self.d_prime_list.append(d_prime)

        self.strategy_dict[self.prunable_idx[self.cur_ind]][0] = action
        if self.cur_ind > 0 and self.cur_buffer_flop == 0:
            self.strategy_dict[self.prunable_idx[self.cur_ind - 1]][1] = action

        best_flag = False
        # all the actions are made
        if self._is_final_layer():
            assert len(self.strategy) == len(self.prunable_idx)
            current_flops = self._cur_flops()
            acc_t1 = time.time()
            # adaptive_BN(self.train_loader, self.model)
            acc1, acc5 = self._validate(self.val_loader, self.model)
            acc_t2 = time.time()
            self.val_time = acc_t2 - acc_t1
            compress_ratio = current_flops * 1. / self.org_flops
            info_set = {'compress_ratio': compress_ratio, 'acc1': acc1, 'acc5': acc5, 'strategy': self.strategy.copy(), 'flops': current_flops * 1. / 1e6}
            print('Actual strategy: ', self.strategy)
            if self.acc_metric == 'acc1':
                acc = acc1
            elif self.acc_metric == 'acc5':
                acc = acc5
            else:
                raise NotImplementedError
            reward = self.reward(self, acc, current_flops)

            if acc1 > self.best_top1 or acc5 > self.best_top5:
                best_flag = True
                if acc1 > self.best_top1:
                    self.best_top1 = acc1
                if acc5 > self.best_top5:
                    self.best_top5 = acc5
                self.best_strategy = self.strategy.copy()
                self.best_d_prime_list = self.d_prime_list.copy()
                prGreen('New best top1: {:.4f}, top5: {:.4f}, flops: {:.4f}'.format(acc1, acc5, current_flops * 1. / 1e6))
                prGreen('New best policy: {}'.format(self.best_strategy))
                prGreen('New best d primes: {}'.format(self.best_d_prime_list))

            done = True
            if self.export_model:  # export state dict
                torch.save(self.model.state_dict(), self.export_path)
                return None, None, None, None
            return reward, done, info_set, best_flag

        info_set = None
        reward = 0
        done = False
        self.visited[self.cur_ind] = True  # set to visited
        self.cur_ind += 1  # the index of next layer
        # build next state (in-place modify)
        self.layer_embedding[self.cur_ind][-3] = self._cur_reduced() * 1. / self.org_flops  # reduced
        self.layer_embedding[self.cur_ind][-2] = sum(self.flops_list[self.cur_ind + 1:]) * 1. / self.org_flops  # rest
        self.layer_embedding[self.cur_ind][-1] = self.strategy[-1]  # last action

        return reward, done, info_set, best_flag