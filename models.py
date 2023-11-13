import math
import torch.nn.functional as F
import torch.nn as nn
import torch

import warnings
from torch.nn.utils import spectral_norm as sn
from encoder_models import CNNMultipleLatentEncoder

warnings.filterwarnings("ignore")

def swish(x):
    return x * torch.sigmoid(x)

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        torch.nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

### Main Models
class NodeGraphEBM(nn.Module):
    def __init__(self, args, dataset):
        super(NodeGraphEBM, self).__init__()
        do_prob = args.dropout
        self.dropout_prob = do_prob

        filter_dim = args.filter_dim
        self.filter_dim = filter_dim
        latent_dim = args.latent_dim
        spectral_norm = args.spectral_norm

        self.factor = True
        self.num_time_instances = 3
        self.stride = 1
        state_dim = args.input_dim
        kernel_size = 5
        self.skip_con = True #True
        # if self.skip_con:
        #     print('Swithced NodeGraphEBM_CNN skipcon to {}, set to False if mixing. And check stride!'.format (self.skip_con))
        self.obj_id_embedding = args.obj_id_embedding
        if args.obj_id_embedding:
            state_dim += args.obj_id_dim
            self.obj_embedding = NodeID(args)

        self.cnn = CNNBlock(state_dim * 2, filter_dim, filter_dim, do_prob, kernel_size=kernel_size, spectral_norm=spectral_norm)
        if self.skip_con:
            self.skip_cnn = CNNBlock(state_dim, filter_dim, filter_dim, do_prob, kernel_size=kernel_size, stride=2, spectral_norm=spectral_norm)
            filter_dim_mlp1 = filter_dim * 2
        else: filter_dim_mlp1 = filter_dim
        self.mlp1_trans = MLPBlock(filter_dim, filter_dim, filter_dim, do_prob, spectral_norm=spectral_norm)


        self.mlp1 = MLPBlock(filter_dim_mlp1, filter_dim, filter_dim, do_prob, spectral_norm=spectral_norm)

        if spectral_norm:
            self.energy_map_cnn = sn(nn.Linear(filter_dim, 1))
            self.energy_map_trans = sn(nn.Linear(filter_dim, 1))
            self.mlp_encode = sn(nn.Linear(filter_dim * self.num_time_instances, filter_dim))
            self.mlp3 = sn(nn.Linear(filter_dim * 3, filter_dim))
        else:
            self.energy_map_cnn = nn.Linear(filter_dim, 1)
            self.energy_map_trans = nn.Linear(filter_dim, 1)
            self.mlp_encode = nn.Linear(filter_dim * self.num_time_instances, filter_dim)
            self.mlp3 = nn.Linear(filter_dim * 3, filter_dim)

        self.layer_cnn_encode = CondResBlock1d(filters=filter_dim, latent_dim=latent_dim, kernel_size=kernel_size, downsample=True, spectral_norm=spectral_norm)
        self.layer1_cnn = CondResBlock1d(filters=filter_dim, latent_dim=latent_dim, kernel_size=kernel_size, downsample=True, spectral_norm=spectral_norm)
        self.layer1 = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)

        self.layer1_trans = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)
        self.layer2_trans = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)

        if self.factor:
            print("Using factor graph CNN encoder.")
        else:
            print("Using CNN encoder.")

        self.init_weights()

        n_latents = args.ensembles if args.no_mask else args.ensembles//2
        self.latent_encoder = CNNMultipleLatentEncoder(args, state_dim, args.latent_hidden_dim, args.latent_dim,
                                               n_latents=n_latents, do_prob=args.dropout, factor=True)

        self.rel_rec = None
        self.rel_send = None
        self.ones_mask = None

    def embed_latent(self, traj, rel_rec, rel_send, edges=None):
        if self.rel_rec is None and self.rel_send is None:
            self.rel_rec, self.rel_send = rel_rec[0:1], rel_send[0:1]
        if rel_rec.device != traj.device:
            rel_rec, rel_send = self.rel_rec, self.rel_send
        if self.obj_id_embedding:
            traj = self.obj_embedding(traj)
        return self.latent_encoder(traj, rel_rec, rel_send, true_edges=edges)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)

    def node2edge_temporal(self, inputs, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.

        x = inputs.reshape(inputs.size(0), inputs.size(1), -1)
        receivers = torch.matmul(rel_rec, x)
        receivers = receivers.reshape(inputs.size(0) * receivers.size(1),
                                      inputs.size(2), inputs.size(3))
        receivers = receivers.transpose(2, 1)

        senders = torch.matmul(rel_send, x)
        senders = senders.reshape(inputs.size(0) * senders.size(1),
                                  inputs.size(2),
                                  inputs.size(3))
        senders = senders.transpose(2, 1)

        # receivers and senders have shape:
        # [num_sims * num_edges, num_dims, num_timesteps]
        edges = torch.cat([senders, receivers], dim=1)
        return edges

    def edge2node_temporal(self, x, rel_rec, rel_send):
        bs, nr, feat, T = x.shape
        incoming = torch.matmul(rel_rec.transpose(2, 1), x.flatten(-2,-1)).reshape(bs, -1, feat, T)
        return incoming / incoming.size(1)

    def edge2node(self, x, rel_rec, rel_send):
        incoming = torch.matmul(rel_rec.transpose(2, 1), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def forward(self, inputs, latent, rel_rec=None, rel_send=None):
        if rel_rec is None and rel_send is None:
            rel_rec = self.rel_rec
            rel_send = self.rel_send
        BS, NO, T, ND = inputs.shape
        NR = NO * (NO - 1)

        if self.obj_id_embedding:
            inputs = self.obj_embedding(inputs)
            ND = inputs.shape[-1]

        if isinstance(latent, tuple):
            latent, mask = latent
            if len(mask.shape) < 2:
                mask = mask[None]
        else:
            if self.ones_mask is None:
                self.ones_mask = torch.ones((1, 1)).to(inputs.device)
            mask = self.ones_mask

        if latent is not None:
            latent = latent * mask[..., None]

        # Process skip connection
        if self.skip_con:
            x_skip = swish(self.skip_cnn(inputs.flatten(0,1).permute(0, 2, 1))).mean(-1).reshape(BS, NO, -1)

        # Input has shape: [num_sims, num_atoms, num_timesteps, num_dims]
        edges = self.node2edge_temporal(inputs, rel_rec, rel_send) #[N, 4, T] --> [R, 8, T] # Marshalling

        # ### Note: Overall Trajectory
        edges_cnn = swish(self.cnn(edges))  # [R, 8, T] --> [R, F, T'] # CNN layers
        x = self.layer_cnn_encode(edges_cnn, latent=latent) # [R, F, T'] --> [R, F, T'']
        x = self.layer1_cnn(x, latent=latent) # [R, F, T'] --> [R, F, T'']

        # Join all edges with the edge of interest
        x = x.mean(-1)  # [R, F, T'''] --> [R, F] # Temporal Avg pool
        x = x.reshape(BS, NR, x.size(-1))

        x = self.edge2node(x, rel_rec, rel_send) # [R, F] --> [N, F] # marshalling

        if self.skip_con:
            x = torch.cat((x, x_skip), dim=2)

        x = swish(self.mlp1(x)) # [N, F] --> [N, F]
        energy_cnn = self.energy_map_cnn(x).squeeze(-1) # [R, F] --> [R, 1] # Project features to scalar
        energy = energy_cnn

        # ### Note: Pairwise / Transitions
        edges_unfold = edges_cnn.unfold(-1, self.num_time_instances, self.stride)
        _, _, NC, NTI = edges_unfold.shape
        edges_unfold = edges_unfold.reshape(BS, NR, -1, NC, NTI).permute(0, 3, 1, 2, 4)

        x = swish(self.mlp_encode(edges_unfold.reshape(BS, NC*NR, -1))).reshape(BS, NC, NR, -1)  # [R, 8 * NTI] --> [R, F] # CNN layers
        # # Unconditional pass for the rest of edges
        x = self.layer1_trans(x, latent)
        x = self.layer2_trans(x, latent)

        # Join all edges with the edge of interest
        x = x.reshape(BS*NC, NR, x.size(-1))
        x = self.edge2node(x, rel_rec, rel_send) # [R, F] --> [N, F] # marshalling
        x = swish(self.mlp1_trans(x)) # [N, F] --> [N, F] # TODO share mlps?

        energy_trans = self.energy_map_trans(x.reshape(BS, NC, NO, x.size(-1))).squeeze(-1).mean(1) # [R, F] --> [R, 1] # Project features to scalar
        energy = energy + energy_trans

        return energy

class UnConditional(nn.Module):
    def __init__(self, args, dataset):
        super(UnConditional, self).__init__()
        do_prob = args.dropout
        self.dropout_prob = do_prob

        filter_dim = args.filter_dim
        self.filter_dim = filter_dim
        latent_dim = args.latent_dim
        spectral_norm = args.spectral_norm

        self.factor = True
        self.num_time_instances = 3
        self.stride = 1
        state_dim = args.input_dim
        kernel_size = 5
        self.skip_con = True

        self.obj_id_embedding = args.obj_id_embedding
        if args.obj_id_embedding:
            state_dim += args.obj_id_dim
            self.obj_embedding = NodeID(args)

        self.cnn = CNNBlock(state_dim * 2, filter_dim, filter_dim, do_prob, kernel_size=kernel_size, spectral_norm=spectral_norm)
        if self.skip_con:
            self.skip_cnn = CNNBlock(state_dim, filter_dim, filter_dim, do_prob, kernel_size=kernel_size, stride=2, spectral_norm=spectral_norm)
            filter_dim_mlp1 = filter_dim * 2
        else: filter_dim_mlp1 = filter_dim
        self.mlp1_trans = MLPBlock(filter_dim, filter_dim, filter_dim, do_prob, spectral_norm=spectral_norm)


        self.mlp1 = MLPBlock(filter_dim_mlp1, filter_dim, filter_dim, do_prob, spectral_norm=spectral_norm)

        if spectral_norm:
            self.energy_map_cnn = sn(nn.Linear(filter_dim, 1))
            self.energy_map_trans = sn(nn.Linear(filter_dim, 1))
            self.mlp_encode = sn(nn.Linear(filter_dim * self.num_time_instances, filter_dim))
        else:
            self.energy_map_cnn = nn.Linear(filter_dim, 1)
            self.energy_map_trans = nn.Linear(filter_dim, 1)
            self.mlp_encode = nn.Linear(filter_dim * self.num_time_instances, filter_dim)


        if self.factor:
            print("Using factor graph CNN encoder.")
        else:
            print("Using CNN encoder.")

        self.init_weights()

        self.latent_encoder = CNNMultipleLatentEncoder(args, state_dim, args.latent_hidden_dim, args.latent_dim,
                                                       n_latents=args.ensembles//2, do_prob=args.dropout, factor=True)

        self.rel_rec = None
        self.rel_send = None
        self.ones_mask = None

    def embed_latent(self, traj, rel_rec, rel_send, edges=None):
        if self.rel_rec is None and self.rel_send is None:
            self.rel_rec, self.rel_send = rel_rec[0:1], rel_send[0:1]
        if self.obj_id_embedding:
            traj = self.obj_embedding(traj)

        return self.latent_encoder(traj, self.rel_rec, self.rel_send, true_edges=edges)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)

    def node2edge_temporal(self, inputs, rel_rec, rel_send):

        x = inputs.reshape(inputs.size(0), inputs.size(1), -1)

        receivers = torch.matmul(rel_rec, x)
        receivers = receivers.reshape(inputs.size(0) * receivers.size(1),
                                      inputs.size(2), inputs.size(3))
        receivers = receivers.transpose(2, 1)

        senders = torch.matmul(rel_send, x)
        senders = senders.reshape(inputs.size(0) * senders.size(1),
                                  inputs.size(2),
                                  inputs.size(3))
        senders = senders.transpose(2, 1)

        # receivers and senders have shape:
        # [num_sims * num_edges, num_dims, num_timesteps]
        edges = torch.cat([senders, receivers], dim=1)
        return edges

    def edge2node_temporal(self, x, rel_rec, rel_send):
        bs, nr, feat, T = x.shape
        incoming = torch.matmul(rel_rec.transpose(2, 1), x.flatten(-2,-1)).reshape(bs, -1, feat, T)
        return incoming / incoming.size(1)

    def edge2node(self, x, rel_rec, rel_send):
        incoming = torch.matmul(rel_rec.transpose(2, 1), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def forward(self, inputs):

        rel_rec = self.rel_rec
        rel_send = self.rel_send
        BS, NO, T, ND = inputs.shape
        NR = NO * (NO - 1)
        if self.obj_id_embedding:
            inputs = self.obj_embedding(inputs)
            ND = inputs.shape[-1]

        # Process skip connection
        if self.skip_con:
            x_skip = swish(self.skip_cnn(inputs.flatten(0,1).permute(0, 2, 1))).mean(-1).reshape(BS, NO, -1)

        # Input has shape: [num_sims, num_atoms, num_timesteps, num_dims]
        edges = self.node2edge_temporal(inputs, rel_rec, rel_send) #[N, 4, T] --> [R, 8, T] # Marshalling
        edges_cnn = swish(self.cnn(edges))  # [R, 8, T] --> [R, F, T'] # CNN layers

        # Join all edges with the edge of interest
        x = edges_cnn.mean(-1)
        x = x.reshape(BS, NR, x.size(-1))
        x = self.edge2node(x, rel_rec, rel_send) # [R, F] --> [N, F] # marshalling

        if self.skip_con:
            x = torch.cat((x, x_skip), dim=2)

        x = swish(self.mlp1(x)) # [N, F] --> [N, F]
        energy_cnn = self.energy_map_cnn(x).squeeze(-1) # [R, F] --> [R, 1] # Project features to scalar
        energy = energy_cnn

        # ### Note: Pairwise / Transitions ### Unused at this moment
        edges_unfold = edges_cnn.unfold(-1, self.num_time_instances, self.stride)
        _, _, NC, NTI = edges_unfold.shape
        edges_unfold = edges_unfold.reshape(BS, NR, -1, NC, NTI).permute(0, 3, 1, 2, 4)

        x = swish(self.mlp_encode(edges_unfold.reshape(BS, NC*NR, -1))).reshape(BS, NC, NR, -1)  # [R, 8 * NTI] --> [R, F] # CNN layers
        x = x.reshape(BS*NC, NR, x.size(-1))
        x = self.edge2node(x, rel_rec, rel_send) # [R, F] --> [N, F] # marshalling
        x = swish(self.mlp1_trans(x)) # [N, F] --> [N, F] # TODO share mlps?

        energy_trans = self.energy_map_trans(x.reshape(BS, NC, NO, x.size(-1))).squeeze(-1).mean(1) # [R, F] --> [R, 1] # Project features to scalar
        energy = energy + energy_trans

        return energy


#### Other Models for experimentation
class EdgeGraphEBM(nn.Module):
    def __init__(self, args, dataset):
        super(EdgeGraphEBM, self).__init__()
        do_prob = args.dropout
        self.dropout_prob = do_prob

        filter_dim = args.filter_dim
        self.filter_dim = filter_dim
        latent_dim = args.latent_dim
        spectral_norm = args.spectral_norm

        self.factor = True
        self.num_time_instances = 3
        self.stride = 1
        state_dim = args.input_dim

        self.obj_id_embedding = args.obj_id_embedding
        if args.obj_id_embedding:
            state_dim += args.obj_id_dim
            self.obj_embedding = NodeID(args)

        self.cnn = CNNBlock(state_dim * 2, filter_dim, filter_dim, do_prob, kernel_size=5, spectral_norm=spectral_norm)

        self.mlp1_trans = MLPBlock(filter_dim * self.num_time_instances, filter_dim, filter_dim, do_prob, spectral_norm=spectral_norm)

        if spectral_norm:
            self.energy_map_cnn = sn(nn.Linear(filter_dim, 1))
            self.energy_map_trans = sn(nn.Linear(filter_dim, 1))
            self.mlp2 = sn(nn.Linear(filter_dim, filter_dim))
            self.mlp3 = sn(nn.Linear(filter_dim * 3, filter_dim))
            self.mlp2_trans = sn(nn.Linear(filter_dim, filter_dim))
            self.mlp3_trans = sn(nn.Linear(filter_dim * 3, filter_dim))
        else:
            self.energy_map_cnn = nn.Linear(filter_dim, 1)
            self.energy_map_trans = nn.Linear(filter_dim, 1)
            self.mlp2 = nn.Linear(filter_dim, filter_dim)
            self.mlp3 = nn.Linear(filter_dim * 3, filter_dim)
            self.mlp2_trans = nn.Linear(filter_dim, filter_dim)
            self.mlp3_trans = nn.Linear(filter_dim * 3, filter_dim)

        self.layer_cnn_encode = CondResBlock1d(filters=filter_dim, latent_dim=latent_dim, kernel_size=5, rescale=False, spectral_norm=spectral_norm)
        self.layer1_cnn = CondResBlock1d(filters=filter_dim, latent_dim=latent_dim, downsample=False, rescale=False, spectral_norm=spectral_norm)
        self.layer1 = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)
        self.layer2 = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)
        # self.layer3 = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)

        self.layer1_trans = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)
        self.layer2_trans = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)
        self.layer3_trans = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)
        self.layer4_trans = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)

        if self.factor:
            print("Using factor graph CNN encoder.")
        else:
            print("Using CNN encoder.")

        self.init_weights()

        n_latents = args.ensembles if args.no_mask else args.ensembles//2
        self.latent_encoder = CNNMultipleLatentEncoder(args, state_dim, args.latent_hidden_dim, args.latent_dim,
                                                       n_latents=n_latents, do_prob=args.dropout, factor=True)

        self.rel_rec = None
        self.rel_send = None
        self.ones_mask = None

    def embed_latent(self, traj, rel_rec, rel_send, edges=None):
        if self.rel_rec is None and self.rel_send is None:
            self.rel_rec, self.rel_send = rel_rec[0:1], rel_send[0:1]
        if self.obj_id_embedding:
            traj = self.obj_embedding(traj)

        return self.latent_encoder(traj, self.rel_rec, self.rel_send, true_edges=edges)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)

    def node2edge_temporal(self, inputs, rel_rec, rel_send):

        x = inputs.reshape(inputs.size(0), inputs.size(1), -1)
        receivers = torch.matmul(rel_rec, x)
        receivers = receivers.reshape(inputs.size(0) * receivers.size(1),
                                      inputs.size(2), inputs.size(3))
        receivers = receivers.transpose(2, 1)

        senders = torch.matmul(rel_send, x)
        senders = senders.reshape(inputs.size(0) * senders.size(1),
                                  inputs.size(2),
                                  inputs.size(3))
        senders = senders.transpose(2, 1)

        # receivers and senders have shape:
        # [num_sims * num_edges, num_dims, num_timesteps]
        edges = torch.cat([senders, receivers], dim=1)
        return edges

    def edge2node(self, x, rel_rec, rel_send):
        incoming = torch.matmul(rel_rec.transpose(2, 1), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def forward(self, inputs, latent):

        rel_rec = self.rel_rec
        rel_send = self.rel_send
        BS, NO, T, ND = inputs.shape
        NR = NO * (NO - 1)

        if self.obj_id_embedding:
            inputs = self.obj_embedding(inputs)
            ND = inputs.shape[-1]

        if isinstance(latent, tuple):
            latent, mask = latent
            if len(mask.shape) < 2:
                mask = mask[None]
        else:
            if self.ones_mask is None:
                self.ones_mask = torch.ones((1, 1)).to(inputs.device)
            mask = self.ones_mask

        if latent is not None:
            latent = latent * mask[..., None]

        # Input has shape: [num_sims, num_atoms, num_timesteps, num_dims]
        edges = self.node2edge_temporal(inputs, rel_rec, rel_send) #[N, 4, T] --> [R, 8, T] # Marshalling
        edges_cnn = swish(self.cnn(edges))  # [R, 8, T] --> [R, F, T'] # CNN layers

        ### Note: CNN features ###
        # Unconditional pass for the rest of edges
        x = self.layer_cnn_encode(edges_cnn, latent=latent) # [R, F, T'] --> [R, F, T'']
        x = self.layer1_cnn(x, latent=latent) # [R, F, T'] --> [R, F, T'']

        # Join all edges with the edge of interest
        x = x.mean(-1) # [R, F, T'''] --> [R, F] # Temporal Avg pool
        x = x.reshape(BS, NR, x.size(-1))

        # Mask before message passing
        # x = x * mask[:, :, None]

        # Skip connection
        x_skip = x

        x = self.edge2node(x, rel_rec, rel_send) # [R, F] --> [N, F] # marshalling
        x = swish(self.mlp2(x)) # [N, F] --> [N, F]

        x = self.node2edge(x, rel_rec, rel_send) # [N, F] --> [R, 2F] # marshalling
        x = torch.cat((x, x_skip), dim=2)  # [R, 2F] --> [R, 3F] # Skip connection
        x = swish(self.mlp3(x)) # [R, 3F] --> [R, F]

        x = self.layer1(x, latent) # [R, F] --> [R, F] # Conditioning layer
        out_cnn = self.layer2(x, latent) # [R, F] --> [R, F] # Conditioning layer
        # out_cnn = self.layer3(x, latent) # [R, F] --> [R, F] # Conditioning layer

        energy_cnn = self.energy_map_cnn(out_cnn).squeeze(-1) # [R, F] --> [R, 1] # Project features to scalar

        ### Note: Pairwise / Transitions ###
        edges_unfold = edges_cnn.unfold(-1, self.num_time_instances, self.stride)
        _, _, NC, NTI = edges_unfold.shape
        edges_unfold = edges_unfold.reshape(BS, NR, -1, NC, NTI).permute(0, 3, 1, 2, 4)

        x = swish(self.mlp1_trans(edges_unfold.reshape(BS, NC*NR, -1))).reshape(BS, NC, NR, -1)  # [R, 8 * NTI] --> [R, F] # CNN layers

        # # Unconditional pass for the rest of edges
        x = self.layer1_trans(x, latent)
        x = self.layer2_trans(x, latent)


        # Join all edges with the edge of interest
        x = x.reshape(BS*NC, NR, x.size(-1))

        x_skip = x

        x = self.edge2node(x, rel_rec, rel_send) # [R, F] --> [N, F] # marshalling
        x = swish(self.mlp2_trans(x)) # [N, F] --> [N, F] # TODO share mlps?

        x = self.node2edge(x, rel_rec, rel_send) # [N, F] --> [R, 2F] # marshalling
        x = torch.cat((x, x_skip), dim=2)  # [R, 2F] --> [R, 3F] # Skip connection
        x = swish(self.mlp3_trans(x)) # [R, 3F] --> [R, F]

        x = self.layer1(x.reshape(BS, NC, NR, -1), latent=latent) # [R, F, T'] --> [R, F, T'']
        x = self.layer2(x, latent=latent) # [R, F, T'] --> [R, F, T'']
        # x = self.layer3(x, latent=latent) # [R, F] --> [R, F] # Conditioning layer

        energy_trans = self.energy_map_trans(x).squeeze(-1).mean(1) # [R, F] --> [R, 1] # Project features to scalar

        energy = torch.stack([energy_cnn * 0.01, energy_trans]) * mask[None]

        return energy

class NodeGraphEBM_2Streams(nn.Module):
    def __init__(self, args, dataset):
        super(NodeGraphEBM_2Streams, self).__init__()
        do_prob = args.dropout
        self.dropout_prob = do_prob

        filter_dim = args.filter_dim
        self.filter_dim = filter_dim
        latent_dim = args.latent_dim
        spectral_norm = args.spectral_norm

        self.factor = True
        self.num_time_instances = 3
        self.stride = 1
        state_dim = args.input_dim
        kernel_size = 5
        self.skip_con = False
        self.obj_id_embedding = args.obj_id_embedding
        if args.obj_id_embedding:
            state_dim += args.obj_id_dim
            self.obj_embedding = NodeID(args)

        # self.cnn = CNNBlock(state_dim * 2, filter_dim, filter_dim, do_prob, kernel_size=1, spectral_norm=spectral_norm)
        if self.skip_con:
            self.skip_cnn = CNNBlock(state_dim, filter_dim, filter_dim, do_prob, kernel_size=kernel_size, stride=2, spectral_norm=spectral_norm)
            filter_dim_mlp1 = filter_dim * 2
        else: filter_dim_mlp1 = filter_dim
        self.mlp1_trans = MLPBlock(filter_dim_mlp1, filter_dim, filter_dim, do_prob, spectral_norm=spectral_norm)
        self.mlp1 = MLPBlock(filter_dim_mlp1, filter_dim, filter_dim, do_prob, spectral_norm=spectral_norm)

        if spectral_norm:
            self.cnn = sn(nn.Conv1d(state_dim * 2, filter_dim, kernel_size=1))
            self.energy_map_cnn = sn(nn.Linear(filter_dim, 1))
            self.energy_map_trans = sn(nn.Linear(filter_dim, 1))
            self.mlp_encode = sn(nn.Linear(filter_dim * self.num_time_instances, filter_dim))
            self.mlp3 = sn(nn.Linear(filter_dim * 3, filter_dim))
        else:
            self.cnn = nn.Conv1d(state_dim * 2, filter_dim, kernel_size=1)
            self.energy_map_cnn = nn.Linear(filter_dim, 1)
            self.energy_map_trans = nn.Linear(filter_dim, 1)
            self.mlp_encode = nn.Linear(filter_dim * self.num_time_instances, filter_dim)
            self.mlp3 = nn.Linear(filter_dim * 3, filter_dim)
        self.layer_cnn_encode = CondResBlock1d(filters=filter_dim, latent_dim=latent_dim, kernel_size=kernel_size, downsample=True, spectral_norm=spectral_norm)
        self.layer1_cnn = CondResBlock1d(filters=filter_dim, latent_dim=latent_dim, kernel_size=kernel_size, downsample=True, spectral_norm=spectral_norm)
        self.layer1 = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)
        self.layer_uncond = CNNBlock(filter_dim, filter_dim, filter_dim, do_prob, kernel_size=kernel_size, stride=2, spectral_norm=spectral_norm)

        self.layer1_trans = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)
        self.layer2_trans = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)
        self.layer1_trans_uncond = MLPBlock(filter_dim, filter_dim, filter_dim, do_prob, spectral_norm=spectral_norm)

        if self.factor:
            print("Using factor graph CNN encoder.")
        else:
            print("Using CNN encoder.")

        self.init_weights()

        n_latents = args.ensembles if args.no_mask else args.ensembles//2

        self.latent_encoder = CNNMultipleLatentEncoder(args, state_dim, args.latent_hidden_dim, args.latent_dim,
                                                       n_latents=n_latents, do_prob=args.dropout, factor=True)

        self.rel_rec = None
        self.rel_send = None
        self.ones_mask = None

    def embed_latent(self, traj, rel_rec, rel_send, edges=None):
        if self.rel_rec is None and self.rel_send is None:
            self.rel_rec, self.rel_send = rel_rec[0:1], rel_send[0:1]
        if self.obj_id_embedding:
            traj = self.obj_embedding(traj)
        return self.latent_encoder(traj, self.rel_rec, self.rel_send, true_edges=edges)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)

    def node2edge_temporal(self, inputs, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.

        x = inputs.reshape(inputs.size(0), inputs.size(1), -1)

        receivers = torch.matmul(rel_rec, x)
        receivers = receivers.reshape(inputs.size(0) * receivers.size(1),
                                      inputs.size(2), inputs.size(3))
        receivers = receivers.transpose(2, 1)

        senders = torch.matmul(rel_send, x)
        senders = senders.reshape(inputs.size(0) * senders.size(1),
                                  inputs.size(2),
                                  inputs.size(3))
        senders = senders.transpose(2, 1)

        # receivers and senders have shape:
        # [num_sims * num_edges, num_dims, num_timesteps]
        edges = torch.cat([senders, receivers], dim=1)
        return edges

    def edge2node_temporal(self, x, rel_rec, rel_send):
        bs, nr, feat, T = x.shape
        incoming = torch.matmul(rel_rec.transpose(2, 1), x.flatten(-2,-1)).reshape(bs, -1, feat, T)
        return incoming / incoming.size(1)

    def edge2node(self, x, rel_rec, rel_send):
        incoming = torch.matmul(rel_rec.transpose(2, 1), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def forward(self, inputs, latent):

        rel_rec = self.rel_rec
        rel_send = self.rel_send
        BS, NO, T, ND = inputs.shape
        NR = NO * (NO - 1)

        if self.obj_id_embedding:
            inputs = self.obj_embedding(inputs)
            ND = inputs.shape[-1]

        if isinstance(latent, tuple):
            latent, mask = latent
            if len(mask.shape) < 2:
                mask = mask[None]
        else:
            if self.ones_mask is None:
                self.ones_mask = torch.ones((1, 1)).to(inputs.device)
            mask = self.ones_mask

        energy = 0
        # # Input has shape: [num_sims, num_atoms, num_timesteps, num_dims]
        edges = self.node2edge_temporal(inputs, rel_rec, rel_send) #[N, 4, T] --> [R, 8, T] # Marshalling
        edges_cnn = swish(self.cnn(edges))  # [R, 8, T] --> [R, F, T'] # CNN layers
        x = self.layer_cnn_encode(edges_cnn, latent=latent) # [R, F, T'] --> [R, F, T'']
        cond = self.layer1_cnn(x, latent=latent).mean(-1) # [R, F, T'] --> [R, F, T'']
        uncond = self.layer_uncond(edges_cnn).mean(-1)

        # # Join all edges with the edge of interest
        cond = cond.reshape(BS, NR, cond.size(-1))
        uncond = uncond.reshape(BS, NR, uncond.size(-1))

        # # Mask before message passing
        x = cond * mask[:, :, None] + uncond * (1-mask[:, :, None])
        x = self.edge2node(x, rel_rec, rel_send) # [R, F] --> [N, F] # marshalling

        energy_cnn = self.energy_map_cnn(x).squeeze(-1) # [R, F] --> [R, 1] # Project features to scalar
        energy = energy + energy_cnn

        ### Note: Pairwise / Transitions ### Unused at this moment
        edges_unfold_1 = edges_cnn.unfold(-1, self.num_time_instances, self.stride)
        _, _, NC, NTI = edges_unfold_1.shape
        edges_unfold = edges_unfold_1.reshape(BS, NR, -1, NC, NTI).permute(0, 3, 1, 2, 4)

        edges = swish(self.mlp_encode(edges_unfold.reshape(BS, NC*NR, -1))).reshape(BS, NC, NR, -1)  # [R, 8 * NTI] --> [R, F] # CNN layers
        # # Unconditional pass for the rest of edges
        x = self.layer1_trans(edges, latent)
        cond = self.layer2_trans(x, latent)
        uncond = self.layer1_trans_uncond(edges)

        # Join all edges with the edge of interest
        cond = cond.reshape(BS, NC, NR, cond.size(-1))
        uncond = uncond.reshape(BS, NC, NR, uncond.size(-1))

        x = (cond * mask[:, None, :, None]) + (uncond * (1 - mask[:, None, :, None]))
        x = x.reshape(BS*NC, NR, -1)
        x = self.edge2node(x, rel_rec, rel_send) # [R, F] --> [N, F] # marshalling
        x = swish(self.mlp1_trans(x)) # [N, F] --> [N, F] # TODO share mlps?

        energy_trans = self.energy_map_trans(x.reshape(BS, NC, NO, x.size(-1))).squeeze(-1).mean(1) # [R, F] --> [R, 1] # Project features to scalar
        energy = energy + energy_trans

        return energy

## No factorization
class EdgeGraphEBM_noF(nn.Module):
    def __init__(self, args, dataset):
        super(EdgeGraphEBM_noF, self).__init__()
        do_prob = args.dropout
        self.dropout_prob = do_prob

        filter_dim = args.filter_dim
        self.filter_dim = filter_dim
        latent_dim = args.latent_dim
        spectral_norm = args.spectral_norm

        self.factor = True
        self.num_time_instances = 3
        self.stride = 1
        state_dim = args.input_dim

        self.obj_id_embedding = args.obj_id_embedding
        if args.obj_id_embedding:
            state_dim += args.obj_id_dim
            self.obj_embedding = NodeID(args)

        self.cnn = CNNBlock(state_dim * 2, filter_dim, filter_dim, do_prob, spectral_norm=spectral_norm)

        self.mlp1_trans = MLPBlock(filter_dim * self.num_time_instances, filter_dim, filter_dim, do_prob, spectral_norm=spectral_norm)
        # self.mlp2 = MLPBlock(filter_dim, filter_dim, filter_dim, do_prob, spectral_norm=spectral_norm)
        # self.mlp3 = MLPBlock(filter_dim * 3, filter_dim, filter_dim, do_prob, spectral_norm=spectral_norm)

        if spectral_norm:
            self.energy_map_cnn = sn(nn.Linear(filter_dim, 1))
            self.energy_map_trans = sn(nn.Linear(filter_dim, 1))
            self.mlp2 = sn(nn.Linear(filter_dim, filter_dim))
        else:
            self.energy_map_cnn = nn.Linear(filter_dim, 1)
            self.energy_map_trans = nn.Linear(filter_dim, 1)
            self.mlp2 = nn.Linear(filter_dim, filter_dim)

        self.layer_cnn_encode = CondResBlock1d(filters=filter_dim, latent_dim=latent_dim, rescale=False, spectral_norm=spectral_norm)
        self.layer1_cnn = CondResBlock1d(filters=filter_dim, latent_dim=latent_dim, downsample=False, rescale=False, spectral_norm=spectral_norm)
        self.layer1 = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)

        self.layer1_trans = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)
        self.layer2_trans = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)
        self.layer3_trans = CondMLPResBlock1d(filters=filter_dim, latent_dim=latent_dim, spectral_norm=spectral_norm)

        if self.factor:
            print("Using factor graph CNN encoder.")
        else:
            print("Using CNN encoder.")

        self.init_weights()


        n_latents = args.ensembles if args.no_mask else args.ensembles//2

        self.latent_encoder = CNNMultipleLatentEncoder(state_dim, args.latent_hidden_dim, args.latent_dim,
                                                       n_latents=n_latents, do_prob=args.dropout, factor=True)

        self.rel_rec = None
        self.rel_send = None
        self.ones_mask = None


    def embed_latent(self, traj, rel_rec, rel_send, edges=None):
        if self.rel_rec is None and self.rel_send is None:
            self.rel_rec, self.rel_send = rel_rec[0:1], rel_send[0:1]

        if self.obj_id_embedding:
            traj = self.obj_embedding(traj)

        return self.latent_encoder(traj, self.rel_rec, self.rel_send, true_edges=edges)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)

    def node2edge_temporal(self, inputs, rel_rec, rel_send):

        x = inputs.reshape(inputs.size(0), inputs.size(1), -1)

        receivers = torch.matmul(rel_rec, x)
        receivers = receivers.reshape(inputs.size(0) * receivers.size(1),
                                      inputs.size(2), inputs.size(3))
        receivers = receivers.transpose(2, 1)

        senders = torch.matmul(rel_send, x)
        senders = senders.reshape(inputs.size(0) * senders.size(1),
                                  inputs.size(2),
                                  inputs.size(3))
        senders = senders.transpose(2, 1)

        # receivers and senders have shape:
        # [num_sims * num_edges, num_dims, num_timesteps]
        edges = torch.cat([senders, receivers], dim=1)
        return edges

    def edge2node(self, x, rel_rec, rel_send):
        incoming = torch.matmul(rel_rec.transpose(2, 1), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def forward(self, inputs, latent):

        rel_rec = self.rel_rec
        rel_send = self.rel_send
        BS, NO, T, ND = inputs.shape
        NR = NO * (NO - 1)

        if self.obj_id_embedding:
            inputs = self.obj_embedding(inputs)
            ND = inputs.shape[-1]

        if isinstance(latent, tuple):
            latent, mask = latent
            if len(mask.shape) < 2:
                mask = mask[None]
        else:
            if self.ones_mask is None:
                self.ones_mask = torch.ones((1, 1)).to(inputs.device)
            mask = self.ones_mask

        # Input has shape: [num_sims, num_atoms, num_timesteps, num_dims]
        edges = self.node2edge_temporal(inputs, rel_rec, rel_send) #[N, 4, T] --> [R, 8, T] # Marshalling
        edges_cnn = swish(self.cnn(edges))  # [R, 8, T] --> [R, F, T'] # CNN layers

        ### Note: CNN features ###
        # Unconditional pass for the rest of edges
        x = self.layer_cnn_encode(edges_cnn, latent=latent) # [R, F, T'] --> [R, F, T'']
        x = self.layer1_cnn(x, latent=latent) # [R, F, T'] --> [R, F, T'']

        # Join all edges with the edge of interest
        x = x.mean(-1) # [R, F, T'''] --> [R, F] # Temporal Avg pool

        x = self.mlp2(x)
        x = x.reshape(BS, NR, x.size(-1))
        x = self.layer1(x, latent) # [R, F] --> [R, F] # Conditioning layer
        out_cnn = x * mask[:, :, None]
        energy_cnn = self.energy_map_cnn(out_cnn).squeeze(-1) # [R, F] --> [R, 1] # Project features to scalar

        ### Note: Pairwise / Transitions ###
        edges_unfold = edges_cnn.unfold(-1, self.num_time_instances, self.stride)
        _, _, NC, NTI = edges_unfold.shape
        edges_unfold = edges_unfold.reshape(BS, NR, -1, NC, NTI).permute(0, 3, 1, 2, 4)

        x = swish(self.mlp1_trans(edges_unfold.reshape(BS, NC*NR, -1))).reshape(BS, NC, NR, -1)  # [R, 8 * NTI] --> [R, F] # CNN layers

        # # Unconditional pass for the rest of edges
        x = self.layer1_trans(x, latent)
        x = self.layer2_trans(x, latent)
        x = self.layer3_trans(x, latent)
        x = x * mask[:, None, :, None]
        energy_trans = self.energy_map_trans(x).squeeze(-1).mean(1) # [R, F] --> [R, 1] # Project features to scalar

        energy = torch.stack([energy_cnn, energy_trans])

        return energy

### Identifier to each node
class NodeID(nn.Module):
    def __init__(self, args):
        super(NodeID, self).__init__()
        self.obj_id_dim = args.obj_id_dim
        self.args = args
        # dim_id = int(np.ceil(np.log2(args.n_objects)))
        self.index_embedding = nn.Embedding(args.n_objects, args.obj_id_dim)
        # obj_ids = [torch.tensor(np.array(list(np.binary_repr(i, width=dim_id)), dtype=int)) for i in range(args.n_objects)]
        # obj_ids = torch.stack(obj_ids)
        obj_ids = torch.arange(args.n_objects)
        if self.args.dataset == 'nba' and self.obj_id_dim == 1:
            obj_ids[:1] = -1
            obj_ids[1:6] = 0
            obj_ids[6:] = 1
        self.register_buffer('obj_ids', obj_ids, persistent=False)

    def forward(self, states):
        if self.args.dataset == 'nba' and self.obj_id_dim == 1:
            embeddings = self.obj_ids[None, :, None, None].expand(*states.shape[:-1], self.obj_id_dim)
        else:
            embeddings = self.index_embedding(self.obj_ids)[None, :, None].expand(*states.shape[:-1], self.obj_id_dim)
        return torch.cat([states, embeddings], dim=-1)

### CNN and MLP modules
class CondResBlock1d(nn.Module):
    def __init__(self, downsample=True, rescale=False, filters=64, latent_dim=64, im_size=64, kernel_size=5, latent_grid=False, spectral_norm=False):
        super(CondResBlock1d, self).__init__()

        self.filters = filters
        self.latent_dim = latent_dim
        self.im_size = im_size
        self.downsample = downsample
        self.latent_grid = latent_grid

        # TODO: Why not used?
        if filters <= 128: # Q: Why this condition?
            self.bn1 = nn.InstanceNorm1d(filters, affine=False)
        else:
            self.bn1 = nn.GroupNorm(32, filters, affine=False)

        if spectral_norm:
            self.conv1 = sn(nn.Conv1d(filters, filters, kernel_size=kernel_size, stride=1, padding=kernel_size//2))
            self.conv2 = sn(nn.Conv1d(filters, filters, kernel_size=kernel_size, stride=1, padding=kernel_size//2))
        else:
            self.conv1 = nn.Conv1d(filters, filters, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            self.conv2 = nn.Conv1d(filters, filters, kernel_size=kernel_size, stride=1, padding=kernel_size//2)

        if filters <= 128:
            self.bn2 = nn.InstanceNorm1d(filters, affine=False)
        else:
            self.bn2 = nn.GroupNorm(32, filters, affine=False)

        # TODO: conv1?
        torch.nn.init.normal_(self.conv2.weight, mean=0.0, std=1e-5)

        # Upscale to an mask of image
        self.latent_fc1 = nn.Linear(latent_dim, 2*filters)
        self.latent_fc2 = nn.Linear(latent_dim, 2*filters)

        # Upscale to mask of image
        if downsample:
            if rescale:
                self.conv_downsample = nn.Conv1d(filters, 2 * filters, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            else:
                self.conv_downsample = nn.Conv1d(filters, filters, kernel_size=kernel_size, stride=1, padding=kernel_size//2)

            self.avg_pool = nn.AvgPool1d(3, stride=2, padding=0)

    def forward(self, x, latent):

        # TODO: How to properly condition the different objects. Equally? or should we concatenate them.
        x_orig = x

        if latent is not None:
            latent_1 = self.latent_fc1(latent)
            latent_2 = self.latent_fc2(latent)

            gain = latent_1[:, :, :self.filters]
            bias = latent_1[:, :, self.filters:]

            gain2 = latent_2[:, :, :self.filters]
            bias2 = latent_2[:, :, self.filters:]


            x = self.conv1(x)
            x = x.reshape(latent.size(0), -1, *x.shape[-2:])
            x = gain[..., None] * x + bias[..., None]
            x = swish(x)
            x = x.flatten(0,1)
            x = self.conv2(x)
            x = x.reshape(latent.size(0), -1, *x.shape[-2:])
            x = gain2[..., None] * x + bias2[..., None]
            x = swish(x)
            x = x.flatten(0,1)

        else:
            x = self.conv1(x)
            x = swish(x)
            x = self.conv2(x)
            x = swish(x)

        x_out = x_orig + x

        if self.downsample:
            x_out = swish(self.conv_downsample(x_out))
            x_out = self.avg_pool(x_out)

        return x_out

class CondMLPResBlock1d(nn.Module):
    def __init__(self, filters=64, latent_dim=64, im_size=64, latent_grid=False, spectral_norm=False):
        super(CondMLPResBlock1d, self).__init__()

        self.filters = filters
        self.latent_dim = latent_dim
        self.im_size = im_size
        self.latent_grid = latent_grid

        if spectral_norm:
            self.fc1 = sn(nn.Linear(filters, filters))
            self.fc2 = sn(nn.Linear(filters, filters))
        else:
            self.fc1 = nn.Linear(filters, filters)
            self.fc2 = nn.Linear(filters, filters)

        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=1e-5)
        torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=1e-5)

        # Upscale to an mask of image
        # TODO: Spectral norm?
        self.latent_fc1 = nn.Linear(latent_dim, 2*filters)
        self.latent_fc2 = nn.Linear(latent_dim, 2*filters)


    def forward(self, x, latent):

        # TODO: How to properly condition the different objects. Equally? or should we concatenate them.
        x_orig = x

        if latent is not None:
            latent_1 = self.latent_fc1(latent)
            latent_2 = self.latent_fc2(latent)

            gain = latent_1[:, :, :self.filters]
            bias = latent_1[:, :, self.filters:]

            gain2 = latent_2[:, :, :self.filters]
            bias2 = latent_2[:, :, self.filters:]

            if len(x.shape) > 3:
                x = self.fc1(x)
                x = gain[:, None] * x + bias[:, None]
                x = swish(x)


                x = self.fc2(x)
                x = gain2[:, None] * x + bias2[:, None]
                x = swish(x)
            else:
                x = self.fc1(x)
                x = gain * x + bias
                x = swish(x)


                x = self.fc2(x)
                x = gain2 * x + bias2
                x = swish(x)
        else:
            x = self.fc1(x)
            x = swish(x)

            x = self.fc2(x)
            x = swish(x)

        x_out = x_orig + x

        return x_out

class CNNBlock(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0., kernel_size = 5, stride=1, spectral_norm = False):
        super(CNNBlock, self).__init__()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=None, padding=0,
                                 dilation=1, return_indices=False,
                                 ceil_mode=False)

        if spectral_norm:
            self.conv1 = sn(nn.Conv1d(n_in, n_hid, kernel_size=kernel_size, stride=stride, padding=0))
            self.conv2 = sn(nn.Conv1d(n_hid, n_hid, kernel_size=kernel_size, stride=stride, padding=0))
            self.conv_predict = sn(nn.Conv1d(n_hid, n_out, kernel_size=1))
            self.conv_attention = sn(nn.Conv1d(n_hid, 1, kernel_size=1))
        else:
            self.conv1 = nn.Conv1d(n_in, n_hid, kernel_size=kernel_size, stride=stride, padding=0)
            self.conv2 = nn.Conv1d(n_hid, n_hid, kernel_size=kernel_size, stride=stride, padding=0)
            self.conv_predict = nn.Conv1d(n_hid, n_out, kernel_size=1)
            self.conv_attention = nn.Conv1d(n_hid, 1, kernel_size=1)

        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        # Input shape: [num_sims * num_edges, num_dims, num_timesteps]

        x = swish(self.conv1(inputs))

        pred = self.conv_predict(x)

        return pred

class MLPBlock(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0., spectral_norm = False):
        super(MLPBlock, self).__init__()
        if spectral_norm:
            self.fc1 = sn(nn.Linear(n_in, n_hid))
            self.fc2 = sn(nn.Linear(n_hid, n_out))
        else:
            self.fc1 = nn.Linear(n_in, n_hid)
            self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = swish(self.fc1(inputs)) # TODO: Switch to swish? (or only in ebm)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.fc2(x)
        return x

