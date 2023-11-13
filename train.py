import torch
import time
import numpy as np
import torch.nn.functional as F
import os
import shutil
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from dataset import ChargedParticles, SpringsParticles
from models import UnConditional, NodeGraphEBM, EdgeGraphEBM, NodeGraphEBM_2Streams, EdgeGraphEBM_noF
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os.path as osp
import argparse
from third_party.utils import get_trajectory_figure, \
    linear_annealing, \
    normalize_trajectories
from third_party.utils import save_rel_matrices, get_model_grad_norm, get_model_grad_max
import random

homedir = '/data/Armand/EBM/'
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

"""Parse input arguments"""
parser = argparse.ArgumentParser(description='Train EBM model')

# Data
parser.add_argument('--dataset', default='charged', type=str, help='Dataset to use')
parser.add_argument('--new_dataset', default='', type=str, help='New Dataset to use (not the one used for training) for out of distribution detection')
parser.add_argument('--data_workers', default=4, type=int, help='Number of different data workers to load data in parallel')

# Logging flags
parser.add_argument('--logdir', default='/data/Armand/EBM/cachedir', type=str, help='location where log of experiments will be stored')
parser.add_argument('--logname', default='', type=str, help='name of logs')
parser.add_argument('--exp', default='', type=str, help='name of experiments')
parser.add_argument('--log_interval', default=500, type=int, help='log outputs every so many batches')
parser.add_argument('--save_interval', default=2000, type=int, help='save outputs every so many batches')

# arguments for NRI springs/charged dataset
parser.add_argument('--n_objects', default=3, type=int, help='Dataset to use (intphys or others or imagenet or cubes)')
parser.add_argument('--sequence_length', type=int, default=5000, help='Length of trajectory.')
parser.add_argument('--sample_freq', type=int, default=100, help='How often to sample the trajectory.')
parser.add_argument('--noise_var', type=float, default=0.0, help='Variance of the noise if present.')
parser.add_argument('--interaction_strength', type=float, default=1., help='Size of the box')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')

# Training
parser.add_argument('--train', action='store_true', help='whether or not to train')
parser.add_argument('--test_manipulate', action='store_true', help='test model properties')
parser.add_argument('--cuda', default=True, action='store_true', help='whether to use cuda or not')
parser.add_argument('--port', default=6010, type=int, help='Port for distributed')
parser.add_argument('--resume_iter', default=0, type=int, help='iteration to resume training')
parser.add_argument('--resume_name', default='', type=str, help='name of the model to resume')
parser.add_argument('--batch_size', default=64, type=int, help='size of batch of input to use')
parser.add_argument('--num_epoch', default=500, type=int, help='number of epochs of training to run')
parser.add_argument('--lr', default=3e-4, type=float, help='learning rate for training')
parser.add_argument('--forecast', default=-1, type=int, help='forecast N steps in the future (the encoder only sees the previous). -1 invalidates forecasting')
parser.add_argument('--cd_and_ae', action='store_true', help='if set to True we use L2 loss and Contrastive Divergence')
parser.add_argument('--pred_only', action='store_true', help='Fix points and predict after the K used for training.')
parser.add_argument('--no_mask', action='store_true', help='No partition of the edges, all processed in one graph.')
parser.add_argument('--masking_type', default='random', type=str, help='type of masking in training: {ones, random, by_receiver}')
parser.add_argument('--num_fixed_timesteps', default=5, type=int, help='constraints')
parser.add_argument('--num_timesteps', default=70, type=int, help='total timesteps')
parser.add_argument('--num_steps', default=6, type=int, help='Initial Steps of gradient descent for training')
parser.add_argument('--num_steps_test', default=6, type=int, help='Steps of gradient descent for training')
parser.add_argument('--step_lr', default=500.0, type=float, help='step size of latents')
parser.add_argument('--step_lr_decay_factor', default=1.0, type=float, help='step size of latents')
parser.add_argument('--noise_coef', default=0.0, type=float, help='step size of latents')
parser.add_argument('--noise_decay_factor', default=1.0, type=float, help='step size of latents')
parser.add_argument('--ns_iteration_end', default=300000, type=int, help='training iteration where the number of sampling steps reach their max.')
parser.add_argument('--num_steps_end', default=-1, type=int, help='number of sampling steps at the end of training')
parser.add_argument('--sample', action='store_true', help='generate negative samples through Langevin')
parser.add_argument('--decoder', action='store_true', help='decoder for model')
parser.add_argument('--mse_all_steps', action='store_true', help='', default=False)
parser.add_argument('--scheduler_plateau', action='store_true', help='', default=False)

# Test
parser.add_argument('--compute_energy', action='store_true', help='Get energies')
parser.add_argument('--new_energy', default='', type=str, help='do we add a new energy term?')
parser.add_argument('--new_energy_magnitude', default=0, type=float, help='magnitude of the new energy term')
parser.add_argument('--reference_point', default=(0,0), type=tuple, help='reference point for new energy')
parser.add_argument('--avoid_area', default=(0,0), type=tuple, help='magnitude of the new energy term')

# Model specific settings
parser.add_argument('--ensembles', default=2, type=int, help='use an ensemble of models')
parser.add_argument('--model_name', default='Node', type=str, help='model name: Options in code')
parser.add_argument('--component_weight', action='store_true', help='optimize for weights of the components also')
parser.add_argument('--dropout', default=0.0, type=float, help='dropout probability')
parser.add_argument('--factor_encoder', action='store_true', help='if we use message passing in the encoder')
parser.add_argument('--normalize_data_latent', action='store_true', help='if we normalize data before encoding the latents')
parser.add_argument('--obj_id_embedding', action='store_true', help='add object identifier')
parser.add_argument('--latent_ln', action='store_true', help='layernorm in the latent')
parser.add_argument('--num_additional', default=0, type=int, help='Number of additional components to add')
parser.add_argument('--additional_model', action='store_true', help='Extra unconditional model')

# Model Dims
parser.add_argument('--filter_dim', default=256, type=int, help='number of filters to use')
parser.add_argument('--input_dim', default=4, type=int, help='dimension of an object')
parser.add_argument('--latent_hidden_dim', default=256, type=int, help='hidden dimension of the latent')
parser.add_argument('--latent_dim', default=64, type=int, help='dimension of the latent')
parser.add_argument('--obj_id_dim', default=6, type=int, help='size of the object id embedding')

## Other training Options
parser.add_argument('--kl', action='store_true', help='Whether we compute the KL component of the CD loss')
parser.add_argument('--kl_coeff', default=0.2, type=float, help='Coefficient multiplying the KL term in the loss')
parser.add_argument('--sm', action='store_true', help='Whether we compute the smoothness component of the CD loss')
parser.add_argument('--sm_coeff', default=1, type=float, help='Coefficient multiplying the Smoothness term in the loss')
parser.add_argument('--spectral_norm', action='store_true', help='Spectral normalization in ebm')
parser.add_argument('--sample_ema', action='store_true', help='If set to True, we sample from the ema model')
parser.add_argument('--entropy_nn', action='store_true', help='If set to True, we add an entropy component to the loss')

# Distributed training hyperparameters
parser.add_argument('--nodes', default=1, type=int, help='number of nodes for training')
parser.add_argument('--gpus', default=1, type=int, help='number of gpus per nodes')
parser.add_argument('--node_rank', default=0, type=int, help='rank of node')
parser.add_argument('--gpu_rank', default=0, type=int, help='number of gpus per nodes')


def init_model(FLAGS, device, dataset):

    if FLAGS.model_name == 'Node':
        modelname = NodeGraphEBM
    elif FLAGS.model_name == 'Edge':
        modelname = EdgeGraphEBM
    elif FLAGS.model_name == 'Edge_noFactor':
        modelname = EdgeGraphEBM_noF
    elif FLAGS.model_name == 'Node_2S':
        modelname = NodeGraphEBM_2Streams
    else: raise NotImplementedError

    model = modelname(FLAGS, dataset).to(device)
    models = [model for i in range(FLAGS.ensembles)]

    if FLAGS.additional_model:
        models.append(UnConditional(FLAGS, dataset).to(device))

    optimizers = [Adam(model.parameters(), lr=FLAGS.lr) for model in models] # Note: From CVR , betas=(0.5, 0.99)

    try:
        if FLAGS.scheduler_plateau:
            schedulers = [ReduceLROnPlateau(optimizer, factor=0.7, patience=20000 // FLAGS.save_interval) for optimizer in optimizers] # Note: It will only be called every save_interval
        else:
            schedulers = [StepLR(optimizer, step_size=100000, gamma=0.5) for optimizer in
                          optimizers]
    except:
        schedulers = [StepLR(optimizer, step_size=100000, gamma=0.5) for optimizer in
                      optimizers]
    return models, optimizers, schedulers

def forward_pass_models(models, feat_in, latent, FLAGS):
    latent_ii, mask = latent
    energy = 0
    splits = 1 if FLAGS.no_mask else 2
    for ii in range(splits):
        for iii in range(FLAGS.ensembles//splits):
            if ii == 1:     curr_latent = (latent_ii[..., iii, :], 1 - mask)
            else:           curr_latent = (latent_ii[..., iii, :],     mask)
            if FLAGS.no_mask:
                energy = models[iii].forward(feat_in, curr_latent) + energy
            else: energy = models[ii + 2*iii].forward(feat_in, curr_latent) + energy
    if FLAGS.additional_model:
        energy = models[-1].forward(feat_in) + energy
    if FLAGS.new_energy != '':
        energy = new_constraint(FLAGS.new_energy, FLAGS.new_energy_magnitude, feat_in, energy, FLAGS)
    return energy


def new_constraint(type, magnitude, feat_neg, energy, FLAGS):
    energy_add = 0

    loc = feat_neg[..., :2]
    vel = feat_neg[..., 2:]

    if FLAGS.dataset == 'charged':
        vel_max, vel_min = 6.523, -6.417  # Hardcoded: Statistics for unnormalizing
        vel = (vel + 1) * (vel_max - vel_min) / 2 + vel_min
        vel =  vel / 50
        vloc = torch.cumsum(torch.cat([loc[..., :1, :], vel], dim=-2), dim=-2)

    batch = loc.shape[0]

    loc_x = loc[..., :1]
    loc_y = loc[..., 1:2]
    vel_x = vel[..., :1]
    vel_y = vel[..., 1:2]

    loc_mod = (loc_x**2 + loc_y**2)**(0.5)
    vel_mod = (vel_x**2 + vel_y**2)**(0.5)
    vel_abs_sum = (vel_x.abs() + vel_y.abs())
    acc_mod = ((vel_x[..., 1:, :] - vel_x[..., :-1, :])**2 +
               (vel_y[..., 1:, :] - vel_y[..., :-1, :])**2)**(0.5)

    if type == 'attraction':
        # # Example 2, attraction with the raw angle.
        # reference_point = (0.7, -0.4) #paper figure
        # reference_point = (-0.7, -1) #additional
        vector_to_ref = torch.cat([(loc_x-FLAGS.reference_point[0]), (loc_y-FLAGS.reference_point[1])], dim= -1)
        inner_product = (vector_to_ref*vel).sum(dim=-1)
        a_norm = vector_to_ref.pow(2).sum(dim=-1).pow(0.5)
        b_norm = vel.pow(2).sum(dim=-1).pow(0.5)

        cos = inner_product / (2 * a_norm * b_norm)
        angle = torch.acos(cos)
        angle = torch.minimum(angle, 2*torch.pi - angle)
        energy_add = 3e-4 * (angle).mean(2)

    elif type == 'avoid_area' and FLAGS.new_energy_magnitude != 0:
        area = (0, 1)
        eps = 1e-5
        vloc_x = vloc[..., :1]
        # check if value is in range area
        in_range = (vloc_x > area[0]) * (vloc_x < area[1])
        # Option 1:
        energy_get_out = (vloc_x[in_range] - area[0]).abs() #* mag[None, None, :, None]
        # Option 2:
        # energy_get_out = in_range.sum()
        # energy_get_out = (vloc_x[in_range] - area[0]).abs() * (vloc_x[in_range] - area[1]).abs(
        energy_get_away = - (vloc_x[torch.logical_not(in_range)] - area[0]).abs()

        energy_add = energy_get_out.sum() #+ energy_get_away.sum()

    elif type == 'attract_to_center' and FLAGS.new_energy_magnitude != 0:

        center = (0, 0)
        eps = 1e-5
        vloc_x = vloc[..., :1]
        vloc_y = vloc[..., 1:2]
        # check if value is in range area
        # Option 1:
        energy_attract_x = (vloc_x - center[0]).abs() #* mag[None, None, :, None]
        energy_attract_y = (vloc_y - center[1]).abs()
        energy_add = energy_attract_x.sum() + energy_attract_y.sum()

    elif type == 'velocity' and FLAGS.new_energy_magnitude != 0:
        energy_add = - vel_mod.mean(3).mean(2).sum()

    else: raise NotImplementedError
    energy = energy + magnitude * energy_add
    return energy

def  gen_trajectories(latent, FLAGS, models, models_ema, feat_neg, feat, num_steps, sample=False, create_graph=True, idx=None, training_step=0, fixed_mask=None):

    num_fixed_timesteps = FLAGS.num_fixed_timesteps

    ## Step
    step_lr = FLAGS.step_lr

    ## Noise
    noise_coef = FLAGS.noise_coef
    ini_noise_coef = noise_coef

    feat_negs = [feat_neg]
    feat_neg = feat_neg[:, :, num_fixed_timesteps:]

    feat_noise = torch.randn_like(feat_neg).detach()

    feat_neg.requires_grad_(requires_grad=True) # noise [b, n_o, T, f]

    # Num steps linear annealing
    if not sample:
        if FLAGS.num_steps_end != -1 and training_step > 0:
            num_steps = int(linear_annealing(None, training_step, start_step=20000, end_step=FLAGS.ns_iteration_end, start_value=num_steps, end_value=FLAGS.num_steps_end))

    for i in range(num_steps):
        feat_noise.normal_()

        if FLAGS.num_fixed_timesteps > 0:
            feat_in = torch.cat([feat[:, :, :num_fixed_timesteps], feat_neg], dim=2)

        ## Add Noise
        if FLAGS.noise_decay_factor != 1.0:
            noise_coef = linear_annealing(None, i, start_step=0, end_step=num_steps-1, start_value=ini_noise_coef, end_value=FLAGS.step_lr_decay_factor * ini_noise_coef)
        if noise_coef > 0:
            feat_neg = feat_neg + noise_coef * feat_noise

        # Compute energy
        energy = forward_pass_models(models, feat_in, latent, FLAGS)

        # Get grad for current optimization iteration.
        feat_grad, = torch.autograd.grad([energy.sum()], [feat_neg], create_graph=create_graph)

        # Clip grad if needed
        # feat_grad = torch.clamp(feat_grad, min=-0.5, max=0.5)

        feat_neg = feat_neg - step_lr * feat_grad # GD computation
        feat_neg = torch.clamp(feat_neg, -1.5, 1.5)

        if FLAGS.num_fixed_timesteps > 0:
            feat_out = torch.cat([feat[:, :, :num_fixed_timesteps], feat_neg], dim=2)
        else: feat_out = feat_neg

        feat_negs.append(feat_out)
        feat_neg = feat_neg.detach()
        feat_neg.requires_grad_()  # Q: Why detaching and reataching? Is this to avoid backprop through this step?

    return feat_out, feat_negs, energy, feat_grad

def test_manipulate(train_dataloader, models, models_ema, FLAGS, step=0, save = False, logger = None):
    if FLAGS.cuda:
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    counter = 0

    single_pass = input('Single pass? (y: Yes): ')
    save = input('Save? (y: Yes, n: No): ')
    # save = 'y'; single_pass = 'y'
    print('Begin test:')
    [model.eval() for model in models]
    energies = []
    edge_list = []
    all_points = 0
    points_in_avoid_area = [0,0]
    for (feat, edges), (rel_rec, rel_send), idx in train_dataloader:

        feat = feat.to(dev)
        edges = edges.to(dev)
        rel_rec = rel_rec.to(dev)
        rel_send = rel_send.to(dev)
        bs = feat.size(0)
        feat_copy = feat.clone()
        [save_rel_matrices(model, rel_rec, rel_send) for model in models]

        b_idx = 0
        b_idx_ref = 2

        rw_pair = None
        affected_nodes = None
        fixed_mask = None

        ### SELECT BY PAIRS ###
        # pair_id = 2
        # pairs = get_rel_pairs(rel_send, rel_rec)
        # rw_pair = pairs[pair_id]
        # affected_nodes = (rel_rec + rel_send)[0, rw_pair].mean(0).clamp_(min=0, max=1).data.cpu().numpy()

        ### SELECT BY NODES ###
        # node_id = 1
        # rw_pair = range((FLAGS.n_objects - 1)*node_id, (FLAGS.n_objects - 1)*(node_id + 1))
        # affected_nodes = torch.zeros(FLAGS.n_objects).to(dev)
        # affected_nodes[node_id] = 1

        ### Switch examples ###
        # rw_pair = torch.arange(FLAGS.components).to(dev)

        ### Mask definition
        mask = torch.ones(FLAGS.components).to(dev)
        if rw_pair is not None:
            mask[rw_pair] = 0
        # mask = mask * 0
        # mask[rw_pair] = 1

        if FLAGS.forecast is not -1:
            feat_enc = feat[:, :, :-FLAGS.forecast]
        else: feat_enc = feat
        feat_enc = normalize_trajectories(feat_enc, augment=False, normalize=FLAGS.normalize_data_latent)
        latent = models[0].embed_latent(feat_enc, rel_rec, rel_send, edges=edges)
        if isinstance(latent, tuple):
            latent, weights = latent

        # Latent codes manipulation.
        if rw_pair is not None:
            latent[:, rw_pair] = latent[b_idx_ref:b_idx_ref+1, rw_pair]

        factors = [[1, 1]]
        ori_latent = latent
        lims = None
        counter += feat.shape[0]
        for factor_id, factor in enumerate(factors):
            latent = torch.cat([ori_latent[..., :1, :] * factor[0],
                                ori_latent[..., 1:, :] * factor[1]], dim=-2)

            latent = (latent, mask)

            if FLAGS.pred_only:
                assert FLAGS.forecast > -1
                feat = feat[:, :, -FLAGS.forecast:]

            feat_neg = torch.rand_like(feat) * 2 - 1

            feat_neg, feat_negs, energy_neg, feat_grad = gen_trajectories(latent, FLAGS, models, models_ema, feat_neg, feat, FLAGS.num_steps_test, FLAGS.sample,
                                                                                   create_graph=False, fixed_mask=fixed_mask)

            if FLAGS.dataset == 'charged' and FLAGS.new_energy != '':

                # Unnormalize data
                vel_max, vel_min = 6.523, -6.417
                loc_max, loc_min = 5., -5.

                for ii, in_feats in enumerate([feat_neg[..., 1:, :], feat[..., 1:, :]]):
                    in_feats[..., 2:] = (in_feats[..., 2:] + 1) * (vel_max - vel_min) / 2 + vel_min
                    in_feats[..., 2:] = in_feats[..., 2:] / (10 * (loc_max - loc_min) / 2)  # * 100
                    if ii == 0:
                        plot_newe_feats = in_feats
                    if FLAGS.new_energy == 'avoid_area':
                        area = (0,1)

                        vloc = torch.cumsum(torch.cat([in_feats[..., :1, :2], in_feats[..., :-1, 2:]], dim=-2), dim=-2)
                        vloc_x = vloc[..., :1]
                        in_range = (vloc_x > area[0]) * (vloc_x < area[1])
                        points_in_avoid_area[ii] += in_range.sum()

                    if FLAGS.new_energy == 'attract_to_center':
                        c = (0, 0)

                        vloc = torch.cumsum(torch.cat([in_feats[..., :1, :2], in_feats[..., :-1, 2:]], dim=-2), dim=-2)
                        vloc_x = vloc[..., :1]
                        vloc_y = vloc[..., 1:]
                        vloc_mod = torch.sqrt((vloc_x-c[0]) ** 2 + (vloc_y-c[1]) ** 2)
                        points_in_avoid_area[ii] += vloc_mod.sum()

                    if FLAGS.new_energy == 'velocity':

                        points_in_avoid_area[ii] += in_feats[..., :-1, 2:].mean(-1).mean(-1).sum()

                all_points += in_feats.shape[0] * in_feats.shape[1] * (in_feats.shape[2] -1)

            elif not FLAGS.dataset == 'charged' and FLAGS.new_energy != '':
                raise NotImplementedError

            if save == 'y':

                limpos = limneg = 1
                if False:
                    lims = [-limneg, limpos]
                elif lims is None and len(factors)>1: lims = [feat_negs[-1][b_idx][..., :2].min().detach().cpu().numpy(),
                                          feat_negs[-1][b_idx][..., :2].max().detach().cpu().numpy()]

                for i_plt in range(len(feat_negs)):
                    logger.add_figure('test_manip_gen_rec', get_trajectory_figure(feat_negs[i_plt], lims=lims, b_idx=b_idx, highlight_nodes = affected_nodes, args=FLAGS)[1], step + i_plt + 100*factor_id)
                logger.add_figure('test_manip_gen', get_trajectory_figure(feat_negs[-1], lims=lims, b_idx=b_idx, highlight_nodes = affected_nodes, args=FLAGS)[1], step + 100*factor_id)
                logger.add_figure('test_manip_gt', get_trajectory_figure(feat, b_idx=b_idx, lims=lims, args=FLAGS)[1], step + 100*factor_id)
                logger.add_figure('test_manip_gt_ref', get_trajectory_figure(feat, b_idx=b_idx_ref, lims=lim, args=FLAGS)[1], step + 100*factor_id)
                print('Plotted.')

                if FLAGS.new_energy != '':
                    plt, fig = get_trajectory_figure(plot_newe_feats, lims=[-1,1], b_idx=b_idx,
                                                     args=FLAGS)
                    plt.show()
                    fig.savefig('temp.png', dpi=fig.dpi)
                    savedir = os.path.join('/',*(logger.log_dir.split('/')[:-3]),'results/new_constrain_2/')
                    inp = input('Next: n; or Save newconstrain plot with name: ')
                    if inp is not 'n':
                        savedir_i = savedir + inp + '_new_constrain_'
                        if FLAGS.reference_point is not None:
                            np.save(savedir_i + 'reference_point.npy', np.array(reference_point))
                        np.save(savedir_i + 'feats.npy', feat[b_idx:b_idx+1].detach().cpu().numpy())
                        np.save(savedir_i + 'feat_negs.npy', torch.stack(feat_negs)[:, b_idx:b_idx+1].detach().cpu().numpy())
                        np.save(savedir_i + 'feat_neg_last.npy',  feat_negs[-1][b_idx:b_idx+1].detach().cpu().numpy())
                        print('results saved')
                    else: continue

                else:
                    savedir = os.path.join('/',*(logger.log_dir.split('/')[:-3]),'results/pred_rec_examples/')
                    inp = input('Next: n; or Save generated sample with name: ')
                    if inp is not 'n':
                        if FLAGS.pred_only:
                            savedir += inp + '_pred_'+FLAGS.dataset+'_'
                        else: savedir += inp + '_rec-pred_'+FLAGS.dataset+'_'
                        np.save(savedir + 'gt_traj_all.npy', feat_copy[b_idx:b_idx+1].detach().cpu().numpy())
                        np.save(savedir + 'gt_traj_pred.npy', feat[b_idx:b_idx+1].detach().cpu().numpy())
                        np.save(savedir + 'pred_all_samples.npy', torch.stack(feat_negs)[:, b_idx:b_idx+1].detach().cpu().numpy())
                        print('All saved in dir {}'.format(savedir))
        if single_pass == 'y':
            break
        if counter >= 1000:
            print('Points; ', counter)
            break

    if all_points > 0: # This means that a new constraint has been added
        if FLAGS.new_energy == 'avoid_area':
            print('For {}: Percentage of points in area for prediction:{:.1f}%, and GT:{:.1f}%'.format(FLAGS.dataset,100 * points_in_avoid_area[0]/all_points,
                                                                                                               100 * points_in_avoid_area[1]/all_points))
        elif FLAGS.new_energy == 'velocity':
            print('For {}: Velocity for prediction:{:.2e}, and GT:{:.2e}'.format(FLAGS.dataset, points_in_avoid_area[0]/all_points,
                                                                                                                points_in_avoid_area[1]/all_points))
        elif FLAGS.new_energy == 'attract_to_center':
            print('For {}: Distance from center for prediction:{:.2e}, and GT:{:.2e}'.format(FLAGS.dataset, points_in_avoid_area[0]/all_points,
                                                                                                                points_in_avoid_area[1]/all_points))

    print('Test done.')
    exit()

def test(train_dataloader, models, models_ema, FLAGS, step=0, save = False):
    if FLAGS.cuda:
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    mse_20, mse_10, mse_1 = [], [], []
    [model.eval() for model in models]
    for (feat, edges), (rel_rec, rel_send), idx in train_dataloader:
        bs = feat.shape[0]
        feat = feat.to(dev)
        edges = edges.to(dev)

        if step == FLAGS.resume_iter:
            [save_rel_matrices(model, rel_rec.to(dev), rel_send.to(dev)) for model in models]
            step += 1

        if FLAGS.masking_type == 'random':
            mask = torch.randint(2, (bs, FLAGS.components)).to(dev)
        elif FLAGS.masking_type == 'ones':
            mask = torch.ones((bs, FLAGS.components)).to(dev)
        elif FLAGS.masking_type == 'by_receiver':
            mask = torch.ones((bs, FLAGS.components)).to(dev)
            node_ids = torch.randint(FLAGS.n_objects, (bs,))
            sel_edges = (FLAGS.n_objects - 1)*node_ids
            for n in range(FLAGS.n_objects-1):
                mask[torch.arange(0, bs), sel_edges+n] = 0
        else: raise NotImplementedError

        if FLAGS.forecast is not -1:
            feat_enc = feat[:, :, :-FLAGS.forecast]
        else: feat_enc = feat

        feat_enc = normalize_trajectories(feat_enc, augment=False, normalize=FLAGS.normalize_data_latent)

        with torch.no_grad():
            latent = models[0].embed_latent(feat_enc, rel_rec, rel_send, edges=edges)
            if isinstance(latent, tuple):
                latent, weights = latent

        if FLAGS.pred_only:
            feat = feat[: ,:, -FLAGS.forecast:]

        feat_neg = torch.rand_like(feat) * 2 - 1

        latent = (latent, mask)

        feat_neg, feat_negs, energy_neg, feat_grad = gen_trajectories(latent, FLAGS, models, models_ema, feat_neg, feat, FLAGS.num_steps_test, FLAGS.sample,
                                                                                create_graph=False)

        # Change for experiments with prediction horizon different than 20
        mse_20.append(F.mse_loss(feat_neg[:, :, -1],
                              feat[:, :,  -1]))

        mse_10.append(F.mse_loss(feat_neg[:, :, -11],
                              feat[:, :,  -11]))

        mse_1.append(F.mse_loss(feat_neg[:, :, -20],
                                 feat[:, :,  -20]))

        if not save:
            break

    print('Mean MSE for predicted timesteps save {}: {}/{}/{} '.format(int(save), sum(mse_1) / len(mse_1), sum(mse_10) / len(mse_10), sum(mse_20) / len(mse_20)))
    [model.train() for model in models]

    # Validation loss in 20th timetep
    val_loss = sum(mse_20)/len(mse_20)
    return val_loss

def train(train_dataloader, test_dataloader, logger, models, models_ema, optimizers, schedulers, FLAGS, mask, logdir, rank_idx, replay_buffer=None):

    it = FLAGS.resume_iter
    losses, l2_losses = [], []
    best_model_loss = np.inf

    [optimizer.zero_grad() for optimizer in optimizers]

    dev = torch.device("cuda")

    start_time = time.perf_counter()
    for epoch in range(FLAGS.num_epoch):
        print('Epoch {}\n'.format(epoch))

        for (feat, edges), (rel_rec, rel_send), idx in train_dataloader:
            print(feat.shape)

            bs = feat.shape[0]
            loss = 0.0
            feat = feat.to(dev)

            if it == FLAGS.resume_iter: [save_rel_matrices(model, rel_rec.to(dev), rel_send.to(dev)) for model in models]
            if FLAGS.forecast is not -1:
                feat_enc = feat[:, :, :-FLAGS.forecast]
            else: feat_enc = feat

            feat_enc = normalize_trajectories(feat_enc, augment=False, normalize=FLAGS.normalize_data_latent)

            # select model randomly
            rand_idx = torch.randint(len(models), (1,))
            latent = models[rand_idx].embed_latent(feat_enc, rel_rec, rel_send, edges=edges)

            if FLAGS.pred_only:
                assert FLAGS.forecast > -1
                feat = feat[: ,:, -FLAGS.forecast:]

            latent_norm = latent.norm()

            # Initialization of trajectory as uniform noise
            feat_neg = torch.rand_like(feat) * 2 - 1


            if FLAGS.masking_type == 'random':
                mask = torch.randint(2, (bs, FLAGS.components)).to(dev)
            elif FLAGS.masking_type == 'ones':
                mask = torch.ones((bs, FLAGS.components)).to(dev)
            elif FLAGS.masking_type == 'by_receiver':
                mask = torch.ones((bs, FLAGS.components)).to(dev)
                node_ids = torch.randint(FLAGS.n_objects, (bs,))
                sel_edges = (FLAGS.n_objects - 1)*node_ids
                for n in range(FLAGS.n_objects-1):
                    mask[torch.arange(0, bs), sel_edges+n] = 0
            else: raise NotImplementedError

            latent = (latent, mask)

            feat_neg, feat_negs, energy_neg, feat_grad = gen_trajectories(latent, FLAGS, models, models_ema, feat_neg, feat, FLAGS.num_steps, sample=False, training_step=it)
            feat_negs = torch.stack(feat_negs, dim=1)

            if FLAGS.mse_all_steps:
                # Option 1: Supervise all steps
                feat_negs = feat_negs[:, 1:]
                pows = torch.arange(feat_negs.shape[1]) + 1
                base = torch.ones_like(pows) * 2
                weight = base ** pows
                weight = weight / weight[-1]
                feat_loss = sum([weight[i]*
                                      (torch.pow(feat_negs[:, i] - feat, 2).mean()) for i in range(len(weight))])
            else:
                # Option 2: only supervise last
                feat_loss = torch.pow(feat_neg - feat, 2).mean()


            loss = loss + feat_loss

            if FLAGS.cd_and_ae:

                energy_pos = forward_pass_models(models, feat, latent, FLAGS)
                energy_neg = forward_pass_models(models, feat_neg.detach(), latent, FLAGS)
                ml_loss = energy_pos.mean() - energy_neg.mean()
                pow_energy_rec = (torch.pow(energy_pos, 2).mean() + torch.pow(energy_neg, 2).mean())
                loss = loss + 1e-4 * (ml_loss + pow_energy_rec)

            loss_sm = torch.zeros(1).mean()
            loss_kl = torch.zeros(1)
            loss.backward()

            if it > 30000:
                [torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1) for model in models]
            if it % FLAGS.log_interval == 0:
                model_grad_norm = get_model_grad_norm(models)
                model_grad_max = get_model_grad_max(models)

            [optimizer.step() for optimizer in optimizers]
            [optimizer.zero_grad() for optimizer in optimizers]
            try:
                if not FLAGS.scheduler_plateau:
                    [scheduler.step() for scheduler in schedulers]
            except: [scheduler.step() for scheduler in schedulers]

            losses.append(loss.item())
            l2_losses.append(feat_loss.item())
            if it % FLAGS.log_interval == 0:

                grad_norm = torch.norm(feat_grad)

                avg_loss = sum(losses) / len(losses)
                avg_feat_loss = sum(l2_losses) / len(l2_losses)
                losses, l2_losses = [], []

                kvs = {}
                kvs['loss'] = avg_loss

                energy_neg_mean = energy_neg.mean().item()
                energy_neg_std = energy_neg.std().item()

                kvs['aa-energy_neg_mean'] = energy_neg_mean
                kvs['energy_neg_std'] = energy_neg_std

                try:
                    if FLAGS.scheduler_plateau:
                        kvs['LR'] = optimizers[0].param_groups[0]['lr']
                    else: kvs['LR'] = schedulers[0].get_last_lr()[0]
                except: kvs['LR'] = schedulers[0].get_last_lr()[0]

                kvs['aaa-L2_loss'] = avg_feat_loss
                kvs['latent norm'] = latent_norm.item()

                if FLAGS.kl:
                    kvs['kl_loss'] = loss_kl.mean().item()

                if FLAGS.sm:
                    kvs['sm_loss'] = loss_sm.item()

                kvs['bb-max_abs_grad'] = torch.abs(feat_grad).max()
                kvs['bb-norm_grad'] = grad_norm
                kvs['bb-norm_grad_model'] = model_grad_norm
                kvs['bb-max_abs_grad_model'] = model_grad_max

                string = "It {} ".format(it)

                for k, v in kvs.items():
                    if k in ['aaa-L2_loss', 'aaa-ml_loss', 'kl_loss', 'LR']:
                        string += "%s: %.6f  " % (k,v)
                    logger.add_scalar(k, v, it)

                limpos = limneg = 1; fixed_box_debug=True
                if fixed_box_debug:
                    lims = [-limneg, limpos]
                else: lims = None
                if it % 500 == 0:
                    for i_plt in range(feat_negs.shape[1]):
                        logger.add_figure('gen', get_trajectory_figure(feat_negs[:, i_plt], lims=lims, b_idx=0, args=FLAGS)[1], it + i_plt)
                else: logger.add_figure('gen', get_trajectory_figure(feat_neg, lims=lims, b_idx=0, args=FLAGS)[1], it)

                logger.add_figure('gt', get_trajectory_figure(feat, lims=lims, b_idx=0, args=FLAGS)[1], it)
                _ = test(test_dataloader, models, models_ema, FLAGS, step=it, save=False)

                string += 'Time: %.1fs' % (time.perf_counter()-start_time)
                print(string)
                start_time = time.perf_counter()

            if it == FLAGS.resume_iter:
                shutil.copy(sys.argv[0], logdir + '/train_EBM_saved.py')
                shutil.copy('models.py', logdir + '/models_EBM_saved.py')
                shutil.copy('encoder_models.py', logdir + '/enc_models_EBM_saved.py')
                shutil.copy('dataset.py', logdir + '/dataset_saved.py')

            if it % FLAGS.save_interval == 0:

                model_path = osp.join(logdir, "model_{}.pth".format(it))

                ckpt = {'FLAGS': FLAGS}

                for i in range(len(models)):
                    ckpt['model_state_dict_{}'.format(i)] = models[i].state_dict()

                for i in range(len(optimizers)):
                    ckpt['optimizer_state_dict_{}'.format(i)] = optimizers[i].state_dict()

                for i in range(len(optimizers)):
                    ckpt['scheduler_state_dict_{}'.format(i)] = schedulers[i].state_dict()

                torch.save(ckpt, model_path)
                print("Saving model in directory {}".format(model_path))
                print('run test')

                val_loss = test(test_dataloader, models, models_ema, FLAGS, step=it, save=True)

                try:
                    if FLAGS.scheduler_plateau:
                        [scheduler.step(val_loss) for scheduler in schedulers]
                except:  pass
                if val_loss < best_model_loss:
                    torch.save(ckpt, osp.join(logdir, "model_best.pth"))
                    best_model_loss = val_loss
                    print('Best model so far! in epoch {}.'.format(epoch))
                # [scheduler.step(val_loss) for scheduler in schedulers] # If on plateau


                print('Test at step %d done!' % it)
                exp_name = logger.log_dir.split('/')
                print('Experiment: ' + exp_name[-2] + '/' + exp_name[-1])

            it += 1

def main_single(rank, FLAGS):
    rank_idx = FLAGS.node_rank * FLAGS.gpus + rank
    world_size = FLAGS.nodes * FLAGS.gpus
    if not os.path.exists('result/%s' % FLAGS.exp):
        try:
            os.makedirs('result/%s' % FLAGS.exp)
        except:
            pass

    if FLAGS.new_dataset != '':
        FLAGS.dataset = FLAGS.new_dataset
    if FLAGS.dataset == 'springs':
        dataset = SpringsParticles(FLAGS, 'train')
        valid_dataset = SpringsParticles(FLAGS, 'valid')
        test_dataset = SpringsParticles(FLAGS, 'test')
    elif FLAGS.dataset == 'charged':
        dataset = ChargedParticles(FLAGS, 'train')
        valid_dataset = ChargedParticles(FLAGS, 'valid')
        test_dataset = ChargedParticles(FLAGS, 'test')
    else:
        raise NotImplementedError
    FLAGS.timesteps = FLAGS.num_timesteps

    shuffle = True

    if world_size > 1:
        group = dist.init_process_group(backend='nccl', init_method='tcp://localhost:'+str(FLAGS.port), world_size=world_size, rank=rank_idx, group_name="default")
    torch.cuda.set_device(rank)
    device = torch.device('cuda')

    branch_folder = 'experiments_icml'
    if FLAGS.logname == 'debug':
        logdir = osp.join(FLAGS.logdir, FLAGS.exp, FLAGS.logname)
    else:
        logdir = osp.join(FLAGS.logdir, branch_folder, FLAGS.exp,
                            'NO' +str(FLAGS.n_objects)
                          + '_BS' + str(FLAGS.batch_size)
                          + '_S-LR' + str(FLAGS.step_lr)
                          + '_NS' + str(FLAGS.num_steps)
                          + '_NSEnd' + str(FLAGS.num_steps_end)
                          + 'at{}'.format(str(int(FLAGS.ns_iteration_end/1000)) + 'k' if FLAGS.num_steps_end > 0 else '')
                          + '_LR' + str(FLAGS.lr)
                          + '_LDim' + str(FLAGS.latent_dim)
                          + '{}'.format('LN' if FLAGS.latent_ln else '')
                          + '_SN' + str(int(FLAGS.spectral_norm))
                          + '_CDAE' + str(int(FLAGS.cd_and_ae))
                          + '_Mod' + str(FLAGS.model_name)
                          + '_NMod{}'.format(str(FLAGS.ensembles) if FLAGS.no_mask else str(FLAGS.ensembles//2))
                          + '{}'.format('+1' if FLAGS.additional_model else '')
                          + '_Mask-' + str(FLAGS.masking_type)
                          + '_NoM' + str(int(FLAGS.no_mask))
                          + '_FE' + str(int(FLAGS.factor_encoder))
                          + '_NDL' + str(int(FLAGS.normalize_data_latent))
                          + '_SeqL' + str(int(FLAGS.num_timesteps))
                          + '_FSeqL' + str(int(FLAGS.num_fixed_timesteps))
                          + '_FC' + str(FLAGS.forecast)
                          + '{}'.format('Only' if FLAGS.pred_only else '')
                          )
        if FLAGS.logname != '':
            logdir += '_' + FLAGS.logname
    FLAGS_OLD = FLAGS

    if FLAGS.resume_iter != 0:
        if FLAGS.resume_name is not '':
            logdir = osp.join(FLAGS.logdir, branch_folder, FLAGS.exp, FLAGS.resume_name)
        if FLAGS.resume_iter == -1:
            model_path = osp.join(logdir, "model_best.pth")
        else:
            model_path = osp.join(logdir, "model_{}.pth".format(FLAGS.resume_iter))
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        FLAGS = checkpoint['FLAGS']

        ## The following flags will be loaded from the saved model.
        FLAGS.normalize_data_latent = FLAGS_OLD.normalize_data_latent
        FLAGS.factor_encoder = FLAGS_OLD.factor_encoder
        FLAGS.ensembles = FLAGS_OLD.ensembles
        FLAGS.n_objects = FLAGS_OLD.n_objects
        FLAGS.components = FLAGS_OLD.components

        FLAGS.resume_iter = FLAGS_OLD.resume_iter
        FLAGS.save_interval = FLAGS_OLD.save_interval
        FLAGS.nodes = FLAGS_OLD.nodes
        FLAGS.gpus = FLAGS_OLD.gpus
        FLAGS.node_rank = FLAGS_OLD.node_rank
        FLAGS.train = FLAGS_OLD.train
        FLAGS.batch_size = FLAGS_OLD.batch_size
        FLAGS.decoder = FLAGS_OLD.decoder
        FLAGS.test_manipulate = FLAGS_OLD.test_manipulate
        FLAGS.lr = FLAGS_OLD.lr
        FLAGS.exp = FLAGS_OLD.exp
        FLAGS.step_lr = FLAGS_OLD.step_lr
        FLAGS.num_steps = FLAGS_OLD.num_steps
        FLAGS.num_steps_test = FLAGS_OLD.num_steps_test
        FLAGS.ns_iteration_end = FLAGS_OLD.ns_iteration_end
        FLAGS.num_steps_end = FLAGS_OLD.num_steps_end

        FLAGS.new_energy = FLAGS_OLD.new_energy
        FLAGS.new_energy_magnitude = FLAGS_OLD.new_energy_magnitude

        FLAGS.compute_energy = FLAGS_OLD.compute_energy
        FLAGS.pred_only = FLAGS_OLD.pred_only
        FLAGS.model_name = FLAGS_OLD.model_name
        FLAGS.train = FLAGS_OLD.train

        print('Flag list: \n', FLAGS)

        models, optimizers, schedulers = init_model(FLAGS, device, dataset)

        state_dict = models[0].state_dict()

        models_ema = None
        for i, (model, optimizer, scheduler) in enumerate(zip(models, optimizers, schedulers)):
            model.load_state_dict(checkpoint['model_state_dict_{}'.format(i)], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict_{}'.format(i)])
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict_{}'.format(i)])
            except:
                pass

    else:
        models, optimizers, schedulers = init_model(FLAGS, device, dataset)
        models_ema = None

    # if FLAGS.gpus > 1:
    #     sync_model(models)

    test_batch_size = 4 if FLAGS.dataset == 'horizons' else 20 # Make sure length of testset is divisible by batch
    if FLAGS.train:
        train_dataloader = DataLoader(dataset, num_workers=FLAGS.data_workers, batch_size=FLAGS.batch_size, shuffle=shuffle, pin_memory=False)
    test_manipulate_dataloader = DataLoader(test_dataset, num_workers=FLAGS.data_workers, batch_size=FLAGS.batch_size, shuffle=False, pin_memory=False, drop_last=True)
    test_dataloader = DataLoader(test_dataset, num_workers=FLAGS.data_workers, batch_size=test_batch_size, shuffle=False, pin_memory=False, drop_last=False)
    valid_dataloader = DataLoader(valid_dataset, num_workers=FLAGS.data_workers, batch_size=test_batch_size, shuffle=False, pin_memory=False, drop_last=False)

    logger = SummaryWriter(logdir)
    it = FLAGS.resume_iter

    mask = None
    if FLAGS.train:
        models = [model.train() for model in models]
    else:
        models = [model.eval() for model in models]

    if FLAGS.train:
        train(train_dataloader, valid_dataloader, logger, models, models_ema, optimizers, schedulers, FLAGS, mask, logdir, rank_idx, replay_buffer)

    elif FLAGS.test_manipulate:
        test_manipulate(test_manipulate_dataloader, models, models_ema, FLAGS, step=FLAGS.resume_iter, save=True, logger=logger)
    else:
        test(test_dataloader, models, models_ema, FLAGS, step=FLAGS.resume_iter, save=True)

def main():

    FLAGS = parser.parse_args()
    if FLAGS.no_mask: FLAGS.masking_type = 'ones'

    FLAGS.components = FLAGS.n_objects ** 2 - FLAGS.n_objects
    FLAGS.sample = True

    # if FLAGS.compute_energy and not FLAGS.dataset == 'nba' and not FLAGS.dataset == 'horizons':
    #     FLAGS.exp = FLAGS.exp + 'springs'
    # else:
    FLAGS.exp = FLAGS.exp + FLAGS.dataset
    logdir = osp.join(FLAGS.logdir, 'experiments', FLAGS.exp)

    if not osp.exists(logdir):
        os.makedirs(logdir)

    if FLAGS.gpus > 1:
        mp.spawn(main_single, nprocs=FLAGS.gpus, args=(FLAGS,))
    else:
        main_single(FLAGS.gpu_rank, FLAGS)


if __name__ == "__main__":
    main()
