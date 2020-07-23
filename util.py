import argparse
import numpy as np
import torch
import torch.nn as nn
import random
from random import randint
from pytorch_lightning.loggers import LightningLoggerBase
from torch.utils.tensorboard import SummaryWriter
import os


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='floating_mnist_quadrant', help='dataset name')
    parser.add_argument('--num_workers', type=int, default=0, help='num workers')
    parser.add_argument('--distributed_backend', type=str, default=None, help='distributed backend')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0], help='gpus')
    parser.add_argument('--sanity_steps', type=int, default=0, help='overfit multiplier')
    parser.add_argument('--resume', type=str, default=None, help='resume checkpoint')
    parser.add_argument('--coords', type=str, default=None, help='coordinates')
    parser.add_argument('--val_check_percent', type=float, default=1.0, help='percentage of val checked')
    parser.add_argument('--val_check_interval', type=float, default=1.0, help='check val every fraction of epoch')
    parser.add_argument('--max_epoch', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--save_epoch', type=int, default=5, help='save every nth epoch')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate, default=0.001')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--experiment', type=str, default='fast_dev', help='experiment directory')
    parser.add_argument('--seed', type=int, default=-1, help='random seed')

    args = parser.parse_args()

    if args.seed == -1:
        args.seed = randint(0, 999)
    print("Seed: ", args.seed)
    np.random.seed(seed=args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if len(args.gpu) > 1:
        args.distributed_backend = 'ddp'

    if args.val_check_interval > 1:
        args.val_check_interval = int(args.val_check_interval)

    return args


class NestedTensorboardLogger(LightningLoggerBase):

    @property
    def experiment(self):
        return self.experiment_root

    @property
    def save_dir(self):
        return self._save_dir

    def __init__(self, save_dir, name, **kwargs):
        super().__init__()
        self._save_dir = save_dir
        self._name = name
        self.experiment_root = None
        self.experiment_train = None
        self.experiment_val = None
        self.kwargs = kwargs
        self.setup_experiments()

    def setup_experiments(self):
        root_dir = os.path.join(self.save_dir, self.name)
        os.makedirs(root_dir, exist_ok=True)
        train_dir = os.path.join(self.save_dir, self.name, "train")
        val_dir = os.path.join(self.save_dir, self.name, "val")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        self.experiment_root = SummaryWriter(log_dir=root_dir, **self.kwargs)
        self.experiment_val = SummaryWriter(log_dir=val_dir, **self.kwargs)
        self.experiment_train = SummaryWriter(log_dir=train_dir, **self.kwargs)

    def log_hyperparams(self, params):
        for k in params:
            if params[k] is None:
                params[k] = "None"
            if type(params[k]) == list:
                params[k] = torch.LongTensor(params[k])
        self.experiment_root.add_hparams(hparam_dict=dict(params), metric_dict={})

    def log_train_loss(self, global_step, error):
        self.experiment_train.add_scalar('loss', error, global_step)

    def log_val_loss(self, global_step, loss):
        self.experiment_val.add_scalar('loss', loss, global_step)

    def log_metrics(self, metrics, step):
        return

    def save(self):
        self.experiment_train._get_file_writer().flush()
        self.experiment_val._get_file_writer().flush()
        self.experiment_root._get_file_writer().flush()

    def finalize(self, status):
        self.save()

    @property
    def version(self):
        return self._name

    @property
    def name(self):
        return self._name

################################################################################
## Embedder code from:                                                        ##
## https://github.com/yenchenlin/nerf-pytorch/blob/master/run_nerf_helpers.py ##
################################################################################

PI = 3.141592

# Positional encoding (section 5.1 of the NERF paper)
# note that the embedder takes inputs in the range [-1,1]
class Embedder_NERF:
    def __init__(self, input_dims=3, include_input=True, max_freq_log2=10 - 1, num_freqs=10, log_sampling=True, periodic_fns=[torch.sin, torch.cos]):
        self.input_dims = input_dims
        self.include_input = include_input
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling
        self.periodic_fns = periodic_fns

        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.input_dims
        out_dim = 0
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.max_freq_log2
        N_freqs = self.num_freqs

        if self.log_sampling:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], 1)


def get_embedder_nerf(multires, input_dims=3, i=0):
    if i == -1:
        return nn.Identity(), input_dims

    embed_kwargs = {
        'input_dims': input_dims,
        'include_input': True,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder_NERF(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(PI * x)
    return embed, embedder_obj.out_dim


################################################################################
################################################################################
################################################################################


class Embedder_DISCRETIZATION:
    def __init__(self, input_dims=3, include_input=True, num_discretization=10, basis=2):
        self.input_dims = input_dims
        self.include_input = include_input
        self.num_discretization = num_discretization
        self.basis = basis
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.input_dims
        out_dim = 0
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += d

        for discr in range(self.num_discretization):
            embed_fns.append(lambda x, discr=discr: torch.floor(torch.frac(x * (self.basis ** discr)) * self.basis) / (self.basis - 1.0))
            out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], 1)


def get_embedder_dicretization(multires, basis=2, input_dims=3, i=0):
    if i == -1:
        return nn.Identity(), input_dims

    embed_kwargs = {
        'input_dims': input_dims,
        'include_input': True,
        'num_discretization': multires,
        'basis': basis,
    }

    embedder_obj = Embedder_DISCRETIZATION(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(0.5 * x + 0.5)
    return embed, embedder_obj.out_dim

