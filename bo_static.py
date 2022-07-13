import os
import numpy as np
np.set_printoptions(linewidth=400, suppress=True)
import random
import argparse
import pandas as pd
from copy import deepcopy
import torch
torch.backends.cudnn.deterministic = True

from lib.utils import get_output_folder
from lib.BO import BayesianAgent

from tensorboardX import SummaryWriter
import time

def parse_args():
    parser = argparse.ArgumentParser(description='AMC search script')

    parser.add_argument('--job', default='train', type=str, help='support option: train/export')
    parser.add_argument('--suffix', default=None, type=str, help='suffix to help you remember what experiment you ran')
    # env
    parser.add_argument('--model', default='mobilenet', type=str, help='model to prune')
    parser.add_argument('--dataset', default='imagenet', type=str, help='dataset to use (cifar/imagenet)')
    parser.add_argument('--data_root', default=None, type=str, help='dataset path')
    parser.add_argument('--preserve_ratio', default=0.5, type=float, help='preserve ratio of the model')
    parser.add_argument('--lbound', default=0.2, type=float, help='minimum preserve ratio')
    parser.add_argument('--rbound', default=1., type=float, help='maximum preserve ratio')
    parser.add_argument('--acc_metric', default='acc1', type=str, help='use acc1 or acc5')
    parser.add_argument('--reward', default='acc_reward', type=str, help='Setting the reward')
    parser.add_argument('--use_real_val', dest='use_real_val', action='store_true')
    parser.add_argument('--ckpt_path', default=None, type=str, help='manual path of checkpoint')
    parser.add_argument('--n_calibration_batches', default=60, type=int,
                        help='n_calibration_batches')
    parser.add_argument('--n_points_per_layer', default=10, type=int,
                        help='method to prune (fg/cp for fine-grained and channel pruning)')
    parser.add_argument('--channel_round', default=1, type=int, help='Round channel to multiple of channel_round')
    # training
    parser.add_argument('--gpu_idx', default="0", type=str, help='choose which gpu to use')
    parser.add_argument('--max_iter', default=390, type=int, help='')
    parser.add_argument('--initial_points', default=10, type=int, help='')
    parser.add_argument('--output', default='./logs', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--seed', default=None, type=int, help='random seed to set')
    parser.add_argument('--n_gpu', default=1, type=int, help='number of gpu to use')
    parser.add_argument('--n_worker', default=6, type=int, help='number of data loader worker')
    parser.add_argument('--data_bsize', default=50, type=int, help='number of data batch size')
    # export
    parser.add_argument('--ratios', default=None, type=str, help='ratios for pruning')
    parser.add_argument('--channels', default=None, type=str, help='channels after pruning')
    parser.add_argument('--export_path', default=None, type=str, help='path for exporting models')
    parser.add_argument('--use_new_input', dest='use_new_input', action='store_true', help='use new input feature')
    #clustering
    parser.add_argument('--static_cluster', action='store_true', default=False, help='use static layer clustering')
    parser.add_argument('--simlarity', default='EU', type=str, help='choose similarity measure: EU(Euclidean distance between distributions), JS(Jensen-Shannon divergence between distributions), ST(structure)')
    parser.add_argument('--n_clusters', default=6, type=int, help='')

    return parser.parse_args()


def get_model_and_checkpoint(model, dataset, checkpoint_path, n_gpu=1):
    if model == 'mobilenet' and dataset == 'imagenet':
        from models.mobilenet import MobileNet
        net = MobileNet(n_class=1000)
    elif model == 'mobilenetv2' and dataset == 'imagenet':
        from models.mobilenet_v2 import MobileNetV2
        net = MobileNetV2(n_class=1000)
    elif model == 'resnet56' and dataset == 'cifar10':
        from models.resnet56 import resnet56
        net = resnet56()
    else:
        raise NotImplementedError
    sd = torch.load(checkpoint_path)
    if 'state_dict' in sd:  # a checkpoint but not a state_dict
        sd = sd['state_dict']
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    net.load_state_dict(sd)
    net = net.cuda()
    if n_gpu > 1:
        net = torch.nn.DataParallel(net, range(n_gpu))

    return net, deepcopy(net.state_dict())

def train(env, args):
    allstate = env.layer_embedding_ori
    feature_names = ['layer', 'type', 'c_in', 'c_out', 'stride', 'k', 'params','reduced', 'rest', 'a_next']
    states = pd.DataFrame(allstate, columns=feature_names)
    states = states.loc[:,['c_in','c_out','k','params']]
    states['flops'] = np.array(env.flops_list[:len(allstate)])
    states['k_out'] = np.array(env.k_out_list[:len(allstate)])
    states['quot'] = np.array(states['c_in'] / states['c_out']).reshape(len(states),1)
    if args.model == 'mobilenetv2':
        states = states.drop([0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,35],axis=0)
        feature = np.array((states/states.max()).values, 'float')
        print(feature, file=open(os.path.join(args.output, 'features.txt'), 'a'))
    else:
        states = states.drop([0,len(states)-1],axis=0)
        feature = np.array((states/states.max()).values, 'float')
        print(feature, file=open(os.path.join(args.output, 'features.txt'), 'a'))
    agent = BayesianAgent(env=env, features=feature, args=args, tfwriter=tfwriter)
    best_epoch, best_info = agent.bayesianOptimize()
    print('Training ends')
    print('Best epoch: {}, info {}'.format(best_epoch, best_info))
    print('Best epoch: {}, info {}'.format(best_epoch, best_info), file=open(os.path.join(args.output, 'final.txt'), 'a'))


def export_model(env, args):
    assert args.ratios is not None or args.channels is not None, 'Please provide a valid ratio list or pruned channels'
    assert args.export_path is not None, 'Please provide a valid export path'
    env.set_export_path(args.export_path)

    print('=> Original model channels: {}'.format(env.org_channels))
    if args.ratios:
        ratios = args.ratios.split(',')
        ratios = [float(r) for r in ratios]
        if args.model == 'mobilenetv2':
            for redundant_idx in [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32]:
                ratios.insert(redundant_idx, 1)
        assert  len(ratios) == len(env.org_channels)
        channels = [int(r * c) for r, c in zip(ratios, env.org_channels)]
    else:
        channels = args.channels.split(',')
        channels = [int(r) for r in channels]
        ratios = [c2 / c1 for c2, c1 in zip(channels, env.org_channels)]
    print('=> Pruning with ratios: {}'.format(ratios))
    print('=> Channels after pruning: {}'.format(channels))

    for r in ratios:
        env.step_BO(r)

    return


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
    print('seed: {}, Layer clustering: {}, Metric: {}'.format(args.seed, args.static_cluster, args.acc_metric))

    if args.seed is None:
        args.seed=random.randint(0,3000)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model, checkpoint = get_model_and_checkpoint(args.model, args.dataset, checkpoint_path=args.ckpt_path,
                                                 n_gpu=args.n_gpu)

    if 'mobilenet' in args.model:
        from env.channel_pruning_env import ChannelPruningEnv
    elif 'resnet' in args.model:
        from env.channel_pruning_env_resnet import ChannelPruningEnv
    else:
        raise RuntimeError('Model Not Implemented: {}'.format(args.model))
    
    env = ChannelPruningEnv(model, checkpoint, args.dataset,
                            preserve_ratio=1. if args.job == 'export' else args.preserve_ratio,
                            n_data_worker=args.n_worker, batch_size=args.data_bsize,
                            args=args, export_model=args.job == 'export', use_new_input=args.use_new_input)

    if args.job == 'train':
        # build folder and logs
        base_folder_name = 'Static_{}_{}'.format(args.seed,args.model)#
        if args.suffix is not None:
            base_folder_name = base_folder_name + '_' + args.suffix
        args.output = get_output_folder(args.output, base_folder_name)
        print('=> Saving logs to {}'.format(args.output))
        tfwriter = SummaryWriter(logdir=args.output)
        print('=> Output path: {}...'.format(args.output))
        print('seed: {}, static_cluster: {}, Metric: {}'.format(args.seed, args.static_cluster, args.acc_metric), \
         file=open(os.path.join(args.output, 'final.txt'), 'a'))
        begin=time.time()
        train(env, args)
        print('training time', time.time()-begin )

    elif args.job == 'export':
        export_model(env, args)
    else:
        raise RuntimeError('Undefined job {}'.format(args.job))
