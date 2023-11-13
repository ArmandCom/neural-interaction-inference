from synthetic_sim import ChargedParticlesSim, SpringSim, ChargedSpringsParticlesSim
import time
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--simulation', type=str, default='charged',
                    help='What simulation to generate.')
parser.add_argument('--datadir', type=str, default='./datasets/',
                    help='Where to dave the data.')
parser.add_argument('--num-train', type=int, default=50000,
                    help='Number of training simulations to generate.')
parser.add_argument('--num-valid', type=int, default=10000,
                    help='Number of validation simulations to generate.')
parser.add_argument('--num-test', type=int, default=10000,
                    help='Number of test simulations to generate.')
parser.add_argument('--length', type=int, default=10000,
                    help='Length of trajectory.')
parser.add_argument('--length-test', type=int, default=10000,
                    help='Length of test set trajectory.')
parser.add_argument('--sample-freq', type=int, default=100,
                    help='How often to sample the trajectory.')
parser.add_argument('--n-balls', type=int, default=5,
                    help='Number of balls in the simulation.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed.')

args = parser.parse_args()

if args.simulation == 'springs':
    sim = SpringSim(noise_var=0.0, n_balls=args.n_balls)
elif args.simulation == 'springs-medium':
    sim = SpringSim(noise_var=0.0, n_balls=args.n_balls,
                    interaction_strength=.5)
    suffix = '_springs_medium'
elif args.simulation == 'springs-strong':
    sim = SpringSim(noise_var=0.0, n_balls=args.n_balls,
                    interaction_strength=5)
    suffix = '_springs_strong'
elif args.simulation == 'charged-springs-strong':
    sim = ChargedSpringsParticlesSim(noise_var=0.0, n_balls=args.n_balls,
                                     interaction_strength_springs=5, interaction_strength_charged=1.,
                                     # box_size=float('inf')
                                     )
    suffix = '_charged-springs-strong-s1'
elif args.simulation == 'charged-springs-medium':
    sim = ChargedSpringsParticlesSim(noise_var=0.0, n_balls=args.n_balls,
                                     interaction_strength_springs=.5, interaction_strength_charged=1.,
                                     # box_size=float('inf')
                                     )
    suffix = '_charged-springs-medium-s1'
elif args.simulation == 'charged':
    sim = ChargedParticlesSim(noise_var=0.0, n_balls=args.n_balls)
    suffix = '_charged'
else:
    raise ValueError('Simulation {} not implemented'.format(args.simulation))

suffix += str(args.n_balls)
np.random.seed(args.seed)

print(suffix)


def generate_dataset(num_sims, length, sample_freq):
    loc_all = list()
    vel_all = list()
    edges_all = list()

    for i in range(num_sims):
        t = time.time()
        loc, vel, edges = sim.sample_trajectory(T=length,
                                                sample_freq=sample_freq)
        if i % 100 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))
        loc_all.append(loc)
        vel_all.append(vel)
        edges_all.append(edges)

    loc_all = np.stack(loc_all)
    vel_all = np.stack(vel_all)
    edges_all = np.stack(edges_all)

    return loc_all, vel_all, edges_all

print("Generating {} training simulations".format(args.num_train))
loc_train, vel_train, edges_train = generate_dataset(args.num_train,
                                                     args.length,
                                                     args.sample_freq)

print("Generating {} validation simulations".format(args.num_valid))
loc_valid, vel_valid, edges_valid = generate_dataset(args.num_valid,
                                                     args.length,
                                                     args.sample_freq)

print("Generating {} test simulations".format(args.num_test))
loc_test, vel_test, edges_test = generate_dataset(args.num_test,
                                                  args.length_test,
                                                  args.sample_freq)

datadir = args.datadir

print('Saving in: {}'.format(datadir + 'loc-vel_train-valid-test' + suffix + '.npy'))

np.save(datadir + 'loc_train' + suffix + '.npy', loc_train)
np.save(datadir + 'vel_train' + suffix + '.npy', vel_train)
np.save(datadir + 'edges_train' + suffix + '.npy', edges_train)

np.save(datadir + 'loc_valid' + suffix + '.npy', loc_valid)
np.save(datadir + 'vel_valid' + suffix + '.npy', vel_valid)
np.save(datadir + 'edges_valid' + suffix + '.npy', edges_valid)

np.save(datadir + 'loc_test' + suffix + '.npy', loc_test)
np.save(datadir + 'vel_test' + suffix + '.npy', vel_test)
np.save(datadir + 'edges_test' + suffix + '.npy', edges_test)
