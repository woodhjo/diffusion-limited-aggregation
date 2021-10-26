# -*- coding: utf-8 -*-
"""
Box Attractor

Simulation of 2D Diffusion Limited Aggregation where particles are released
at the centre of a lattice structure and the initial condition is a sticky
box boundary.

@author: woodhjo
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy
from numba import typeof, int64
from numba.experimental import jitclass

spec = [('max_particles', int64), ('nbhood', typeof('neumann')),
        ('lattice_width', int64), ('lattice_height', int64),
        ('lattice', typeof(np.zeros((20, 20)))),
        ('sticky_lattice', typeof(np.zeros((20, 20)))),
        ('particles', typeof(np.zeros((20, 2)))),
        ('sites', typeof([(0, 1), (0, -1), (1, 0), (-1, 0)])),
        ('directions', typeof([(0, 1), (0, -1), (1, 0), (-1, 0)]))]

@jitclass(spec)
class BoxDLA:
    def __init__(self, max_particles, lattice_width, lattice_height,
                 nbhood='neumann'):
        self.max_particles = int(max_particles)
        self.lattice_width = int(lattice_width)
        self.lattice_height = int(lattice_height)
        self.nbhood = nbhood

        # Make lattice structure
        self.lattice = np.zeros((self.lattice_height, self.lattice_width))
        self.sticky_lattice = np.zeros((self.lattice_height,
                                        self.lattice_width))
        self.particles = np.zeros((self.max_particles, 2))*np.nan

        # Particle movements and neighbouring sites
        self.directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1),
                           (-1, 1), (-1, -1)]
        if self.nbhood == 'neumann':
            self.sites = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        else:
            self.sites = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1),
                          (-1, 1), (-1, -1)]

        # Box seed and update arrays
        self.lattice[0, :], self.lattice[-1, :], self.lattice[:, 0], \
            self.lattice[:, -1] = 1, 1, 1, 1

        self.sticky_lattice[1, 1:-1], self.sticky_lattice[-2, 1:-1], \
            self.sticky_lattice[1:-1, 1], self.sticky_lattice[1:-1, -2] = \
            1, 1, 1, 1

    def _spawn(self):
        return [self.lattice_height//2, self.lattice_width//2]

    def _is_occupied(self, point):
        # Check if current site is occupied
        return (self.lattice[point[0], point[1]] != 0)

    def _is_sticky(self, point):
        # Check if current site is sticky
        return (self.sticky_lattice[point[0], point[1]] == 1)

    def _update_aggregate(self, point, val):
        # Update arrays given a point
        self.lattice[point[0], point[1]] = val
        self.particles[val - 1] = point
        self.sticky_lattice[point[0], point[1]] = 0

        for s in self.sites:
            row = point[0] + s[0]
            col = point[1] + s[1]
            # Set unoccupied neighbouring sites as sticky
            if self.lattice[row, col] == 0:
                self.sticky_lattice[row, col] = 1

    def make_aggregate(self):
        N = 0  # number of particles in aggregate
        while N < self.max_particles:
            point = self._spawn()
            if self._is_occupied(point):
                print('Structure has reached centre of lattice.')
                break
            while True:  # Random walk
                choice = np.random.choice(len(self.directions))
                (drow, dcol) = self.directions[choice]
                point[0] += drow
                point[1] += dcol
                if self._is_occupied(point):
                    # Particle 'bounces' if it moves onto an occupied site
                    point[0] -= drow
                    point[1] -= dcol
                if self._is_sticky(point):
                    N += 1
                    self._update_aggregate(point, N)
                    break
        # Select part of array
        self.particles = self.particles[:N]

# %%
model = BoxDLA(20000, 500, 500, 'moore')
model.make_aggregate()
cs = copy.copy(cm.get_cmap('plasma'))
cs.set_under('k')
fig = plt.imshow(model.lattice, interpolation='none', cmap=cs, vmin=0.5)