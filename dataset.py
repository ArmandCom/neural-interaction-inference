import numpy as np
import torch.utils.data as data
import torch
import cv2

root = './data/datasets/'

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


class SpringsParticles(data.Dataset):
    def __init__(self, args, split):
        self.args = args
        self.n_objects = args.n_objects
        suffix = '_springs_strong'
        print('Springs Dataset (S)')
        suffix += str(self.n_objects)
        feat, edges, stats = self._load_data(suffix=suffix, split=split)

        assert self.n_objects == feat.shape[1]
        self.length = feat.shape[0]
        self.timesteps = args.num_timesteps

        self.feat, self.edges = feat[:, :, :self.timesteps], edges

        off_diag = np.ones([args.n_objects, args.n_objects]) - np.eye(args.n_objects)
        self.rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        self.rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing
        """
        return (torch.tensor(self.feat[index]), torch.tensor(self.edges[index])), \
               (torch.tensor(self.rel_rec), torch.tensor(self.rel_send)), index

    def __len__(self):
        """Return the total number of trajectories in the dataset."""
        return self.length

    def _load_data(self, batch_size=1, suffix='', split='train'):

        loc_train = np.load(root + 'loc_train' + suffix + '.npy')
        vel_train = np.load(root + 'vel_train' + suffix + '.npy')
        edges_train = np.load(root + 'edges_train' + suffix + '.npy')

        loc_valid = np.load(root + 'loc_valid' + suffix + '.npy')
        vel_valid = np.load(root + 'vel_valid' + suffix + '.npy')
        edges_valid = np.load(root + 'edges_valid' + suffix + '.npy')

        loc_test = np.load(root + 'loc_test' + suffix + '.npy')
        vel_test = np.load(root + 'vel_test' + suffix + '.npy')
        edges_test = np.load(root + 'edges_test' + suffix + '.npy')

        # [num_samples, num_timesteps, num_dims, num_atoms]
        num_atoms = loc_train.shape[3]

        # NRI stats
        loc_max = loc_train[:, :49].max()
        loc_min = loc_train[:, :49].min()
        vel_max = vel_train[:, :49].max()
        vel_min = vel_train[:, :49].min()

        # Normalize to [-1, 1]
        loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
        vel_train = (vel_train - vel_min) * 2 / (vel_max - vel_min) - 1

        loc_valid = (loc_valid - loc_min) * 2 / (loc_max - loc_min) - 1
        vel_valid = (vel_valid - vel_min) * 2 / (vel_max - vel_min) - 1

        loc_test = (loc_test - loc_min) * 2 / (loc_max - loc_min) - 1
        vel_test = (vel_test - vel_min) * 2 / (vel_max - vel_min) - 1

        # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
        loc_train = np.transpose(loc_train, [0, 3, 1, 2])
        vel_train = np.transpose(vel_train, [0, 3, 1, 2])
        feat_train = np.concatenate([loc_train, vel_train], axis=3)
        edges_train = np.reshape(edges_train, [-1, num_atoms ** 2])
        edges_train = np.array((edges_train + 1) / 2, dtype=np.int64)

        loc_valid = np.transpose(loc_valid, [0, 3, 1, 2])
        vel_valid = np.transpose(vel_valid, [0, 3, 1, 2])
        feat_valid = np.concatenate([loc_valid, vel_valid], axis=3)
        edges_valid = np.reshape(edges_valid, [-1, num_atoms ** 2])
        edges_valid = np.array((edges_valid + 1) / 2, dtype=np.int64)

        loc_test = np.transpose(loc_test, [0, 3, 1, 2])
        vel_test = np.transpose(vel_test, [0, 3, 1, 2])
        feat_test = np.concatenate([loc_test, vel_test], axis=3)
        edges_test = np.reshape(edges_test, [-1, num_atoms ** 2])
        edges_test = np.array((edges_test + 1) / 2, dtype=np.int64)

        feat_train = torch.FloatTensor(feat_train)
        edges_train = torch.LongTensor(edges_train)
        feat_valid = torch.FloatTensor(feat_valid)
        edges_valid = torch.LongTensor(edges_valid)
        feat_test = torch.FloatTensor(feat_test)
        edges_test = torch.LongTensor(edges_test)

        # Exclude self edges
        off_diag_idx = np.ravel_multi_index(
            np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
            [num_atoms, num_atoms])
        edges_train = edges_train[:, off_diag_idx]
        edges_valid = edges_valid[:, off_diag_idx]
        edges_test = edges_test[:, off_diag_idx]

        if split == 'train':
            feat = feat_train
            edges = edges_train
        elif split == 'valid':
            feat = feat_valid
            edges = edges_valid
        elif split == 'test':
            feat = feat_test
            edges = edges_test
        else: raise NotImplementedError
        return feat, edges, (loc_max, loc_min, vel_max, vel_min)

class ChargedParticles(data.Dataset):
    def __init__(self, args, split):
        self.args = args
        self.n_objects = args.n_objects
        suffix = '_charged'+str(self.n_objects)
        print('Charged Dataset')
        feat, edges, stats = self._load_data(suffix=suffix, split=split)
        assert self.n_objects == feat.shape[1]
        self.length = feat.shape[0]
        self.timesteps = args.num_timesteps

        self.feat, self.edges = feat[:, :, :self.timesteps], edges

        off_diag = np.ones([args.n_objects, args.n_objects]) - np.eye(args.n_objects)
        self.rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        self.rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing
        """

        return (torch.tensor(self.feat[index]), torch.tensor(self.edges[index])), \
               (torch.tensor(self.rel_rec), torch.tensor(self.rel_send)), index

    def __len__(self):
        """Return the total number of trajectories in the dataset."""
        return self.length

    def _load_data(self, batch_size=1, suffix='', split='train'):
        loc_train = np.load(root + 'loc_train' + suffix + '.npy')
        vel_train = np.load(root + 'vel_train' + suffix + '.npy')
        edges_train = np.load(root + 'edges_train' + suffix + '.npy')

        loc_valid = np.load(root + 'loc_valid' + suffix + '.npy')
        vel_valid = np.load(root + 'vel_valid' + suffix + '.npy')
        edges_valid = np.load(root + 'edges_valid' + suffix + '.npy')

        loc_test = np.load(root + 'loc_test' + suffix + '.npy')
        vel_test = np.load(root + 'vel_test' + suffix + '.npy')
        edges_test = np.load(root + 'edges_test' + suffix + '.npy')

        # [num_samples, num_timesteps, num_dims, num_atoms]
        num_atoms = loc_train.shape[3]

        # NRI stats
        loc_max = loc_train[:, :49].max()
        loc_min = loc_train[:, :49].min()
        vel_max = vel_train[:, :49].max()
        vel_min = vel_train[:, :49].min()

        # Normalize to [-1, 1]
        loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
        vel_train = (vel_train - vel_min) * 2 / (vel_max - vel_min) - 1

        loc_valid = (loc_valid - loc_min) * 2 / (loc_max - loc_min) - 1
        vel_valid = (vel_valid - vel_min) * 2 / (vel_max - vel_min) - 1

        loc_test = (loc_test - loc_min) * 2 / (loc_max - loc_min) - 1
        vel_test = (vel_test - vel_min) * 2 / (vel_max - vel_min) - 1

        # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
        loc_train = np.transpose(loc_train, [0, 3, 1, 2])
        vel_train = np.transpose(vel_train, [0, 3, 1, 2])
        feat_train = np.concatenate([loc_train, vel_train], axis=3)
        edges_train = np.reshape(edges_train, [-1, num_atoms ** 2])
        edges_train = np.array((edges_train + 1) / 2, dtype=np.int64)

        loc_valid = np.transpose(loc_valid, [0, 3, 1, 2])
        vel_valid = np.transpose(vel_valid, [0, 3, 1, 2])
        feat_valid = np.concatenate([loc_valid, vel_valid], axis=3)
        edges_valid = np.reshape(edges_valid, [-1, num_atoms ** 2])
        edges_valid = np.array((edges_valid + 1) / 2, dtype=np.int64)

        loc_test = np.transpose(loc_test, [0, 3, 1, 2])
        vel_test = np.transpose(vel_test, [0, 3, 1, 2])
        feat_test = np.concatenate([loc_test, vel_test], axis=3)
        edges_test = np.reshape(edges_test, [-1, num_atoms ** 2])
        edges_test = np.array((edges_test + 1) / 2, dtype=np.int64)

        feat_train = torch.FloatTensor(feat_train)
        edges_train = torch.LongTensor(edges_train)
        feat_valid = torch.FloatTensor(feat_valid)
        edges_valid = torch.LongTensor(edges_valid)
        feat_test = torch.FloatTensor(feat_test)
        edges_test = torch.LongTensor(edges_test)

        # Exclude self edges
        off_diag_idx = np.ravel_multi_index(
            np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
            [num_atoms, num_atoms])
        edges_train = edges_train[:, off_diag_idx]
        edges_valid = edges_valid[:, off_diag_idx]
        edges_test = edges_test[:, off_diag_idx]

        if split == 'train':
            feat = feat_train
            edges = edges_train
        elif split == 'valid':
            feat = feat_valid
            edges = edges_valid
        elif split == 'test':
            feat = feat_test
            edges = edges_test
        else: raise NotImplementedError
        return feat, edges, (loc_max, loc_min, vel_max, vel_min)

class ChargedSpringsParticles(data.Dataset):
    def __init__(self, args, split):
        self.args = args
        self.n_objects = args.n_objects

        suffix = '_charged-springs'+str(self.n_objects)

        feat, edges, stats = self._load_data(suffix=suffix, split=split)
        assert self.n_objects == feat.shape[1]
        self.length = feat.shape[0]
        self.timesteps = args.num_timesteps

        # Generate off-diagonal interaction graph
        self.feat, self.edges = feat[:, :, :self.timesteps], edges


        off_diag = np.ones([args.n_objects, args.n_objects]) - np.eye(args.n_objects)
        self.rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        self.rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing
        """
        return (torch.tensor(self.feat[index]), torch.tensor(self.edges[index])), \
               (torch.tensor(self.rel_rec), torch.tensor(self.rel_send)), index

    def __len__(self):
        """Return the total number of trajectories in the dataset."""
        return self.length

    def _load_data(self, batch_size=1, suffix='', split='train'):
        loc_train = np.load('/data/Armand/NRI/loc_train' + suffix + '.npy')
        vel_train = np.load('/data/Armand/NRI/vel_train' + suffix + '.npy')
        edges_train = np.load('/data/Armand/NRI/edges_train' + suffix + '.npy')

        loc_valid = np.load('/data/Armand/NRI/loc_valid' + suffix + '.npy')
        vel_valid = np.load('/data/Armand/NRI/vel_valid' + suffix + '.npy')
        edges_valid = np.load('/data/Armand/NRI/edges_valid' + suffix + '.npy')

        loc_test = np.load('/data/Armand/NRI/loc_test' + suffix + '.npy')
        vel_test = np.load('/data/Armand/NRI/vel_test' + suffix + '.npy')
        edges_test = np.load('/data/Armand/NRI/edges_test' + suffix + '.npy')

        # [num_samples, num_timesteps, num_dims, num_atoms]
        num_atoms = loc_train.shape[3]

        # NRI stats
        loc_max = loc_train[:, :49].max()
        loc_min = loc_train[:, :49].min()
        vel_max = vel_train[:, :49].max()
        vel_min = vel_train[:, :49].min()

        # Normalize to [-1, 1]
        loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
        vel_train = (vel_train - vel_min) * 2 / (vel_max - vel_min) - 1

        loc_valid = (loc_valid - loc_min) * 2 / (loc_max - loc_min) - 1
        vel_valid = (vel_valid - vel_min) * 2 / (vel_max - vel_min) - 1

        loc_test = (loc_test - loc_min) * 2 / (loc_max - loc_min) - 1
        vel_test = (vel_test - vel_min) * 2 / (vel_max - vel_min) - 1

        # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
        loc_train = np.transpose(loc_train, [0, 3, 1, 2])
        vel_train = np.transpose(vel_train, [0, 3, 1, 2])
        feat_train = np.concatenate([loc_train, vel_train], axis=3)
        # edges_train = np.reshape(edges_train, [-1, num_atoms ** 2])
        # edges_train = np.array((edges_train + 1) / 2, dtype=np.int64)

        loc_valid = np.transpose(loc_valid, [0, 3, 1, 2])
        vel_valid = np.transpose(vel_valid, [0, 3, 1, 2])
        feat_valid = np.concatenate([loc_valid, vel_valid], axis=3)
        # edges_valid = np.reshape(edges_valid, [-1, num_atoms ** 2])
        # edges_valid = np.array((edges_valid + 1) / 2, dtype=np.int64)

        loc_test = np.transpose(loc_test, [0, 3, 1, 2])
        vel_test = np.transpose(vel_test, [0, 3, 1, 2])
        feat_test = np.concatenate([loc_test, vel_test], axis=3)
        # edges_test = np.reshape(edges_test, [-1, num_atoms ** 2])
        # edges_test = np.array((edges_test + 1) / 2, dtype=np.int64)

        feat_train = torch.FloatTensor(feat_train)
        edges_train = torch.LongTensor(edges_train)
        feat_valid = torch.FloatTensor(feat_valid)
        edges_valid = torch.LongTensor(edges_valid)
        feat_test = torch.FloatTensor(feat_test)
        edges_test = torch.LongTensor(edges_test)

        if split == 'train':
            feat = feat_train
            edges = edges_train
        elif split == 'valid':
            feat = feat_valid
            edges = edges_valid
        elif split == 'test':
            feat = feat_test
            edges = edges_test
        else: raise NotImplementedError
        return feat, edges, (loc_max, loc_min, vel_max, vel_min)


if __name__ == "__main__":
    exit()
