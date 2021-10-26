# -*- coding: utf-8 -*-
"""
Diffusion Limited Aggregation with Line Seed
--------------------------------------------
A simple model for diffusion limited aggregation, where the maximum 'height'
of the aggregate is supplied by the user.

Instead of a point seed, this uses a line seed at the bottom edge of the
lattice.

Section 1: DLA class
Section 2: Create image of mineral dendrites
Section 3: Create general aggregate

author: woodhjo
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import copy
from numba import typeof, int64
from numba.experimental import jitclass


spec = [('agg_height', int64), ('nbhood', typeof('neumann')),
        ('sticking_prob', typeof(0.5)), ('lattice_height', int64),
        ('lattice_width', int64), ('lattice', typeof(np.zeros((10, 10)))),
        ('sticky_lattice', typeof(np.zeros((10, 10)))),
        ('particles', typeof(np.zeros((10, 2)))),
        ('spawn_height', int64), ('kill_height', int64), ('max_height', int64),
        ('sites', typeof([(1, 0), (-1, 0), (0, 1), (0, -1)])),
        ('directions', typeof([(1, 0), (-1, 0), (0, 1), (0, -1)]))]


@jitclass(spec)
class LineDLA:
    def __init__(self, agg_height, lattice_width, nbhood='neumann',
                 sticking_prob=1):
        # Check validity of user input
        self._check_input(agg_height, lattice_width, nbhood, sticking_prob)

        self.agg_height = int(agg_height)
        self.lattice_width = int(lattice_width) + 2  # includes padding
        self.nbhood = nbhood
        self.sticking_prob = float(sticking_prob)

        # Create lattice structures
        self.lattice_height = 2*self.agg_height

        self.lattice = np.zeros((self.lattice_height, self.lattice_width))
        self.sticky_lattice = np.zeros((self.lattice_height,
                                        self.lattice_width))

        # Initial bounds
        self.spawn_height = 5
        self.kill_height = 10
        self.max_height = 0

        # Set direcions of movement and neighbouring cells
        if self.sticking_prob != 1:
            self.directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        else:
            self.directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1),
                               (1, -1), (-1, 1), (-1, -1)]

        if self.nbhood == 'neumann':
            self.sites = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        elif self.nbhood == 'moore':
            self.sites = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1),
                          (1, -1), (-1, 1), (-1, -1)]

        # Array to contain particle positions
        self.particles = np.zeros((self.agg_height*self.lattice_width, 2)) \
            * np.nan

        # Create seed and update arrays
        self.lattice[0, 1:self.lattice_width - 1] = 1
        self.sticky_lattice[1, 1:self.lattice_width - 1] = 1

    def _check_input(self, agg_height, lattice_width, nbhood, sticking_prob):
        if int(agg_height) < 10:
            raise ValueError('Please enter a larger height.')
        elif int(lattice_width < 10):
            raise ValueError('Please enter a larger width.')
        elif (nbhood != 'moore') and (nbhood != 'neumann'):
            raise ValueError('Neighbourhood not recognised. Accepted input: '
                             '"neumann", "moore"')
        elif float(sticking_prob) > 1 or float(sticking_prob) <= 0:
            raise ValueError('Sticking probability must be within (0, 1]')

    def _spawn(self):
        # Spawn particle somewhere along the spawn height
        rand_col = np.random.randint(1, self.lattice_width - 1)
        start_point = [self.spawn_height, rand_col]
        return start_point

    def _is_dead(self, point):
        # Check if particle has wandered into kill zone
        return not((1 <= point[0] < self.kill_height) and
                   (1 <= point[1] < self.lattice_width - 1))

    def _is_occupied(self, point):
        # Check that current site isn't occupied
        return (self.lattice[point[0], point[1]] != 0)

    def _is_sticky(self, point):
        # Check current site is sticky
        return (self.sticky_lattice[point[0], point[1]] == 1)

    def _will_stick(self):
        # Determines if particle will stick
        return (np.random.random() < self.sticking_prob)

    def _update_aggregate(self, point, val):
        # Update arrays given a point
        self.lattice[point[0], point[1]] = val
        self.sticky_lattice[point[0], point[1]] = 0
        self.particles[val-1] = point
        for n in self.sites:
            row = point[0] + n[0]
            col = point[1] + n[1]
            if self.lattice[row, col] == 0:
                self.sticky_lattice[row, col] = 1

        # Update height bounds if aggregate has grown in height
        if point[0] > self.max_height:
            self.max_height = point[0]
            self.spawn_height = point[0] + 5
            self.kill_height = min(self.spawn_height*2,
                                   self.lattice_height - 2)

    def make_aggregate(self):
        N = 0  # number of particles
        while self.max_height < self.agg_height:
            point = self._spawn()
            while True:  # Random walk
                choice = np.random.choice(len(self.directions))
                (drow, dcol) = self.directions[choice]
                point[0] += drow
                point[1] += dcol
                if self._is_dead(point):
                    break
                elif self._is_occupied(point):
                    # Bounce particle if it moves onto an occupied site
                    point[0] -= drow
                    point[1] -= dcol
                elif self._is_sticky(point) and self._will_stick():
                    N += 1
                    self._update_aggregate(point, N)
                    break
        self.particles = self.particles[:N]


# %% SECTION 2: Create image of mineral dendrites

# Import limestone image
limestone = mpimg.imread('limestone.png')
height, width = np.shape(limestone)[:2]

# Make aggregate
model = LineDLA(height, width, nbhood='moore')
model.make_aggregate()


# Plot aggregate over limestone texture
fig = plt.figure()
ax = fig.add_subplot(xlim=(0, width-1), ylim=(0, height), xlabel='$x$',
                     ylabel='$y$')

cs = copy.copy(cm.get_cmap('Greys'))
cs.set_under('k', alpha=0)
cs.set_over('#353028')

ax.imshow(limestone, origin='lower')
ax.imshow(model.lattice, origin='lower', vmin=0.5, vmax=0.75, cmap=cs)

# %% SECTION 3: Create aggregate from line seed

# User input: CHANGEABLE
agg_height = 200
lattice_width = 500
nbhood = 'neumann'
sticking_prob = 1


# Make aggegate
model2 = LineDLA(agg_height, lattice_width, nbhood, sticking_prob)
model2.make_aggregate()

cs2 = copy.copy(cm.get_cmap('RdBu_r'))
cs2.set_under('k')

fig2 = plt.figure()
ax2 = fig2.add_subplot(xlabel='$x$', ylabel='$y$', xlim=(1, lattice_width-1),
                       ylim=(0, agg_height*1.1))
ax2.imshow(model2.lattice, origin='lower', vmin=0.5, cmap=cs2,
           interpolation='none')
