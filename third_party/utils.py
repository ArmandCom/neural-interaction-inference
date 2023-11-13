# Code from https://github.com/yilundu/ebm_code_release_pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import random
import cv2
import subprocess
import numpy as np

from PIL import Image

from torch.nn.utils import spectral_norm
from torch.autograd import Variable
from torchvision import transforms
import matplotlib.pyplot as plt
import math
from numpy import linalg as la
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

stats_nba = [112.17005, -10.52223, 93.0180, -94.27939]
stats_charged = [4.9999, -4.9999, 6.5547, -6.95569]
stats_springs = [4.9999, -4.9999, 1.5167, -1.5588]

from matplotlib.colors import LinearSegmentedColormap

def get_trajectory_figure(state, b_idx, lims=None, highlight_nodes=None, node_thickness=None,
                          args=None, grads=None, grad_id=None, mark_point=None, stats=None, ref_point=None):
    fig = plt.figure()
    axes = plt.gca()
    if isinstance(state, torch.Tensor):
        state = state.detach().cpu().numpy()
    sz_pt = [80] * state.shape[1]
    lw = [1.5] * state.shape[1]
    if node_thickness is not None:
        sz_pt = (node_thickness - node_thickness.min()) / (node_thickness.max() - node_thickness.min())
        sz_pt = 80 + sz_pt * 170
        labels = [str(node_thickness[i])[:4] for i in range(node_thickness.shape[0])]
    else:
        labels = [-1] * state.shape[1]

    alpha = 0.7

    try:
        state = np.transpose(state[b_idx], (1, 2, 0))
    except:
        plt.close(); return None, None
    loc = state[:, :2][None]
    if state.shape[1] == 4:
        vel = state[:, 2:][None]

    if lims is not None and len(lims) == 4:
        axes.set_xlim([lims[0], lims[1]])
        axes.set_ylim([lims[2], lims[3]])

    colors = sns.color_palette(as_cmap=True)
    color_arange = np.arange(loc.shape[1])
    # construct cmap
    palettes = \
        [sns.color_palette("viridis", as_cmap=True),
         sns.color_palette("crest_r", as_cmap=True),
         sns.color_palette("rocket", as_cmap=True),
         sns.color_palette("light:" + colors[2], as_cmap=True),
         sns.color_palette("light:" + colors[3], as_cmap=True)]
    # [palette.reverse() for palette in palettes]
    if loc.shape[-1] == 11:
        sz_pt[0] = 40
        cmap = [LinearSegmentedColormap.from_list(
            "Custom", [(255 / 255, 165 * 1.1 / 255, 0), (255 / 255, 165 * 1.1 / 255, 0)],
            N=loc.shape[1])]  # ['Greys_r']
        cmap.extend([palettes[3]] * 5)  # 1
        cmap.extend([palettes[4]] * 5)  # 2

    elif node_thickness is not None:
        cmap = [palettes[2]]  # Note: hardcoded for the two specific examples
        cmap.extend([palettes[1]] * 5)
    else:
        cmap = []
        for i in range(loc.shape[-1]):
            palette = sns.color_palette("light:" + colors[i], as_cmap=True)

            cmap.append(palette)

    if stats is not None:
        loc_max, loc_min, vel_max, vel_min = stats
        loc = ((loc + 1) * (loc_max - loc_min)) / 2 + loc_min
        vel = ((vel + 1) * (vel_max - vel_min)) / 2 + vel_min
        if ref_point is not None:
            ref_point = [((ref_point_i + 1) * (loc_max - loc_min)) / 2 + loc_min for ref_point_i in ref_point]

    if ref_point is not None:
        plt.scatter(ref_point[0], ref_point[1], s=sz_pt[0] * 2, c='r')

    plt.grid(color='gray', linestyle='--', linewidth=0.5)

    for i in range(loc.shape[-1]):
        acc_pos = loc[0, 0:1, 0, i], loc[0, 0:1, 1, i]
        vels = vel[0, :, 0, i], vel[0, :, 1, i]
        for t in range(loc.shape[1] - 1):
            acc_pos = np.concatenate([acc_pos[0], acc_pos[0][t:t + 1] + vels[0][t:t + 1]]), \
                np.concatenate([acc_pos[1], acc_pos[1][t:t + 1] + vels[1][t:t + 1]])
        plt.scatter(acc_pos[0], acc_pos[1], s=sz_pt[i], c=color_arange, cmap=cmap[i], alpha=alpha, label=labels[i])
        if mark_point is not None:
            plt.scatter(acc_pos[0][mark_point], acc_pos[1][mark_point], s=sz_pt[0], c='k')

    if grads is not None:
        grads = np.transpose(grads[b_idx], (1, 2, 0))[None]
        for i in range(1, loc.shape[-1]):
            if grad_id is not None:
                if i != grad_id: continue
            num_fixed_timesteps = 5
            Q = plt.quiver(loc[0, num_fixed_timesteps:, 0, i], loc[0, num_fixed_timesteps:, 1, i],
                           grads[0, num_fixed_timesteps:, 0, i], grads[0, num_fixed_timesteps:, 0, i],
                           [(grads[0, num_fixed_timesteps:, 0, i] ** 2 + grads[0, num_fixed_timesteps:, 1,
                                                                         i] ** 2) ** 0.5],
                           angles='xy', width=0.003 * (lw[0]))
    plt.axis('off')
    hfont = {'fontname': 'Times New Roman'}
    if labels[0] != -1:
        axes.legend(prop={'size': 8})
    axes.tick_params(axis='both', labelsize=15)

    return plt, fig


def swish(x):
    return x * torch.sigmoid(x)


def my_softmax(input, axis=1):
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input)
    return soft_max_1d.transpose(axis, 0)


def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from Gumbel(0, 1)

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).float()
    return - torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    NOTE: from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Draw a sample from the Gumbel-Softmax distribution

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise)
    return my_softmax(y / tau, axis=-1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    NOTE: from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes

    Constraints:
    - this implementation only works on batch_size x num_features tensor for now

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


def linear_annealing(device, step, start_step, end_step, start_value, end_value):
    """
    Linear annealing

    :param x: original value. Only for getting device
    :param step: current global step
    :param start_step: when to start changing value
    :param end_step: when to stop changing value
    :param start_value: initial value
    :param end_value: final value
    :return:
    """
    if device is not None:
        if step <= start_step:
            x = torch.tensor(start_value, device=device)
        elif start_step < step < end_step:
            slope = (end_value - start_value) / (end_step - start_step)
            x = torch.tensor(start_value + slope * (step - start_step), device=device)
        else:
            x = torch.tensor(end_value, device=device)
    else:
        if step <= start_step:
            x = start_value
        elif start_step < step < end_step:
            slope = (end_value - start_value) / (end_step - start_step)
            x = start_value + slope * (step - start_step)
        else:
            x = end_value

    return x


class GaussianBlur(object):
    def __init__(self, min=0.1, max=2.0, kernel_size=9):
        self.min = min
        self.max = max
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample


def init_distributed_mode(params):
    """
    Handle single and multi-GPU / multi-node / SLURM jobs.
    Initialize the following variables:
        - n_nodes
        - node_id
        - local_rank
        - global_rank
        - world_size
    """
    SLURM_VARIABLES = [
        'SLURM_JOB_ID',
        'SLURM_JOB_NODELIST', 'SLURM_JOB_NUM_NODES', 'SLURM_NTASKS', 'SLURM_TASKS_PER_NODE',
        'SLURM_MEM_PER_NODE', 'SLURM_MEM_PER_CPU',
        'SLURM_NODEID', 'SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_TASK_PID'
    ]
    PREFIX = "%i - " % int(os.environ['SLURM_PROCID'])
    for name in SLURM_VARIABLES:
        value = os.environ.get(name, None)
        print(PREFIX + "%s: %s" % (name, str(value)))
    # number of nodes / node ID
    params.nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
    params.node_rank = int(os.environ['SLURM_NODEID'])
    # define master address and master port
    hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']])
    params.master_addr = hostnames.split()[0].decode('utf-8')


def visualize_trajectories(state, state_gen, edges, savedir=None, b_idx=0):

    if state is not None:
        plt, fig = get_trajectory_figure(state, b_idx)
        if not savedir:
            plt.show()
        else:
            fig.savefig(savedir + '_gt.png', dpi=fig.dpi); plt.close()

    if state_gen is not None:
        plt2, fig2 = get_trajectory_figure(state_gen, b_idx)
        if not savedir:
            plt2.show()
        else:
            fig2.savefig(savedir + '_gen.png', dpi=fig.dpi); plt2.close()


def gaussian_kernel(a, b):
    # Modification for allowing batch
    bs = a.shape[0]
    dim1_1, dim1_2 = a.shape[1], b.shape[1]
    depth = a.shape[2]
    a = a.reshape(bs, dim1_1, 1, depth)
    b = b.reshape(bs, 1, dim1_2, depth)
    a_core = a.expand(bs, dim1_1, dim1_2, depth)
    b_core = b.expand(bs, dim1_1, dim1_2, depth)
    numerator = (a_core - b_core).pow(3).mean(3) / depth
    return torch.exp(-numerator)


# Implemented from: https://github.com/Saswatm123/MMD-VAE
def batch_MMD(a, b):
    return gaussian_kernel(a, a).mean() + gaussian_kernel(b, b).mean() - 2 * gaussian_kernel(a, b).mean()


def MMD(latent):
    ls = latent.shape[1]
    randperm = (torch.arange(ls) + torch.randint(1, ls, (1,))) % ls
    latent = latent.permute(1, 0, 2)
    return batch_MMD(latent[randperm], latent)


def augment_trajectories(locvel, rotation=None):
    if rotation is not None:
        if rotation == 'random':
            rotation = random.random() * math.pi * 2
        else:
            raise NotImplementedError
        locvel = rotate_with_vel(points=locvel, angle=rotation)
    return locvel


def rotate(origin, points, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = points[..., 0:1], points[..., 1:2]

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    return torch.cat([qx, qy], dim=-1)


def rotate_with_vel(points, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """

    locs = torch.view_as_complex(points[..., :2])
    locpol = torch.polar(locs.abs(), locs.angle() + angle)
    locs = torch.view_as_real(locpol)

    if points.shape[-1] > 2:
        vels = torch.view_as_complex(points[..., 2:])
        velpol = torch.polar(vels.abs(), vels.angle() + angle)
        vels = torch.view_as_real(velpol)
        feat = torch.cat([locs, vels], dim=-1)
    else:
        feat = locs
    return feat


def normalize_trajectories(state, augment=False, normalize=True):
    '''
    state: [BS, NO, T, XY VxVy]
    '''

    if augment:
        state = augment_trajectories(state, 'random')

    if normalize:
        loc, vel = state[..., :2], state[..., 2:]
        ## Instance normalization
        loc_max = torch.amax(loc, dim=(1, 2, 3), keepdim=True)
        loc_min = torch.amin(loc, dim=(1, 2, 3), keepdim=True)
        # vel_max = torch.amax(vel, dim=(1,2,3), keepdim=True)
        # vel_min = torch.amin(vel, dim=(1,2,3), keepdim=True)

        ## Batch normalization
        # loc_max = loc.max()
        # loc_min = loc.min()
        # vel_max = vel.max()
        # vel_min = vel.min() #(dim=-2, keepdims=True)[0]

        # Normalize to [-1, 1]
        loc = (loc - loc_min) * 2 / (loc_max - loc_min) - 1

        if state.shape[-1] > 2:
            vel = vel * 2 / (loc_max - loc_min)
            state = torch.cat([loc, vel], dim=-1)
        else:
            state = loc
    return state

def accumulate_traj(states):
    loc, vel = states[..., :2], states[..., 2:]
    acc_pos = loc[:, :, 0:1]
    for t in range(loc.shape[2] - 1):
        acc_pos = torch.cat([acc_pos, acc_pos[:, :, t:t + 1] + vel[:, :, t:t + 1]], dim=2)
    return torch.cat([acc_pos, vel], dim=-1)


def get_rel_pairs(rel_send, rel_rec):
    ne, nn = rel_send.shape[1:]
    a = np.arange(ne)
    rel = (rel_send + rel_rec).detach().cpu().numpy()[0]
    unique_values = np.unique(rel, axis=0)
    group_list = []
    for value in unique_values:
        this_group = []
        for i in range(ne):
            if all(rel[i] == value):
                this_group.append(a[i])
        group_list.append(this_group)
    return group_list


def save_rel_matrices(model, rel_rec, rel_send):
    if isinstance(model, nn.DataParallel):
        if model.module.rel_rec is None:
            model.module.rel_rec, model.module.rel_send = rel_rec[0:1], rel_send[0:1]
    elif model.rel_rec is None:
        model.rel_rec, model.rel_send = rel_rec[0:1], rel_send[0:1]


def gaussian_fn(M, std):
    n = torch.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    return (w / w.sum())  # .double()


gkern = None


def smooth_trajectory(x, kernel_size, std, interp_size=100):
    x = x.permute(0, 1, 3, 2)
    traj_shape = x.shape
    x = x.flatten(0, 2)
    x_in = F.interpolate(x[:, None, :], interp_size, mode='linear')
    global gkern
    if gkern is None:
        gkern = gaussian_fn(kernel_size, std).to(x_in.device)
    x_in_sm = F.conv1d(x_in, weight=gkern[None, None], padding=kernel_size // 2)  # padding? double?
    x_sm = F.interpolate(x_in_sm, traj_shape[-1], mode='linear')

    # visualize
    # import matplotlib.pyplot as plt
    # plt.close();plt.plot(x_sm[0,0].cpu().detach().numpy());plt.plot(x[0].cpu().detach().numpy()); plt.show()
    return x_sm.reshape(traj_shape).permute(0, 1, 3, 2)


def get_model_grad_norm(models):
    parameters = [p for p in models[0].parameters() if p.grad is not None and p.requires_grad]
    if len(parameters) == 0:
        total_norm = 0.0
    else:
        device = parameters[0].grad.device
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()).to(device) for p in parameters]), 2.0).item()
    return total_norm


def get_model_grad_max(models):
    parameters = [abs(p.grad.detach()).max() for p in models[0].parameters() if p.grad is not None and p.requires_grad]
    return max(parameters)


def create_masks(FLAGS, dev):
    if FLAGS.masking_type == 'random':
        mask = torch.randint(2, (FLAGS.batch_size, FLAGS.components)).to(dev)
    elif FLAGS.masking_type == 'ones':
        mask = torch.ones((FLAGS.batch_size, FLAGS.components)).to(dev)
    elif FLAGS.masking_type == 'by_receiver':
        mask = torch.ones((FLAGS.batch_size, FLAGS.components)).to(dev)
        node_ids = torch.randint(FLAGS.n_objects, (FLAGS.batch_size,))
        sel_edges = (FLAGS.n_objects - 1) * node_ids
        for n in range(FLAGS.n_objects - 1):
            mask[torch.arange(0, FLAGS.batch_size), sel_edges + n] = 0
    else:
        raise NotImplementedError
    return mask