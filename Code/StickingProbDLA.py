# -*- coding: utf-8 -*-
"""
2D Diffusion Limited Aggregation
--------------------------------
A simple model for diffusion limited aggregation, where the maximum bounding
radius of the aggregate is supplied by the user.

The probability that a particle sticks is supplied by the user.
Note: unlike in MainDLA, particles only move NSEW to avoid slipping through
diagonal gaps

Section 1: DLA class
Section 2: Create a simple aggregate and display it

@author: woodhjo
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy
from numba.experimental import jitclass
from numba import typeof, int64
from scipy.spatial import cKDTree
from matplotlib.ticker import StrMethodFormatter, NullFormatter

# To use with jitclass
spec = [('agg_radius', int64), ('nbhood', typeof('neumann')),
        ('lattice_radius', int64), ('lattice', typeof(np.zeros((20, 20)))),
        ('sticky_lattice', typeof(np.zeros((20, 20)))),
        ('particles', typeof(np.zeros((20, 2)))),
        ('spawn_radius', typeof(45.5)),
        ('kill_radius', typeof(45.5)),
        ('max_radius', typeof(45.5)),
        ('sites', typeof([(0, 1), (0, -1), (1, 0), (-1, 0)])),
        ('directions', typeof([(0, 1), (0, -1), (1, 0), (-1, 0)])),
        ('seed', typeof([1, 2])),
        ('lattice_size', int64),
        ('sticking_prob', typeof(0.5))]


@jitclass(spec)
class StickyDLA:
    def __init__(self, agg_radius, nbhood, sticking_prob=1):
        # Check validity of input
        self._check_input(agg_radius, nbhood, sticking_prob)

        self.agg_radius = int(agg_radius)
        self.nbhood = nbhood
        self.sticking_prob = float(sticking_prob)

        # Lattice 'radius' double the aggregate radius
        self.lattice_size = 4*self.agg_radius

        # Setup lattice structures
        self.lattice = np.zeros((self.lattice_size, self.lattice_size))
        self.sticky_lattice = np.zeros((self.lattice_size, self.lattice_size))

        # Use (2*agg_radius)^2 as max number of expected particles
        self.particles = np.zeros(((2*self.agg_radius)**2, 2))*np.nan

        # Initial radii
        self.spawn_radius = 5.0
        self.kill_radius = 10.0
        self.max_radius = 0.0

        # Set where sticky sites are
        if self.nbhood == 'neumann':
            self.sites = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        else:
            self.sites = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1),
                          (-1, 1), (-1, -1)]

        # Particle movement directions
        self.directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        # Create seed and update arrays
        self.seed = [self.lattice_size//2, self.lattice_size//2]
        self.lattice[self.seed[0], self.seed[1]] = 1
        self.particles[0] = self.seed
        for s in self.sites:
            self.sticky_lattice[self.seed[0] + s[0], self.seed[1] + s[1]] = 1

    def _check_input(self, agg_radius, nbhood, sticking_prob):
        if int(agg_radius) < 5:
            raise ValueError('Please enter a larger radius.')
        elif (nbhood != 'moore') and (nbhood != 'neumann'):
            raise ValueError('Neighbourhood not recognised. Accepted values: '
                             '"moore", "neumann"')
        elif float(sticking_prob) <= 0 or float(sticking_prob) > 1:
            raise ValueError('Sticking probability must be in (0, 1].')

    def _spawn(self):
        # Spawn particle somewhere along spawn radius
        rand_angle = np.random.random()*2*np.pi
        start_point = [int(self.seed[0] +
                           self.spawn_radius*np.cos(rand_angle)),
                       int(self.seed[1] +
                           self.spawn_radius*np.sin(rand_angle))]
        return start_point

    def _is_dead(self, point):
        # Check if particle has wandered into kill zone
        r2 = (point[0] - self.seed[0])**2 + (point[1] - self.seed[1])**2
        return (r2 >= self.kill_radius**2)

    def _is_occupied(self, point):
        # Check if current site is occupied
        return (self.lattice[point[0], point[1]] != 0)

    def _is_sticky(self, point):
        # Check if current site is sticky
        return (self.sticky_lattice[point[0], point[1]] == 1)

    def _will_stick(self):
        # Determine if particle will stick
        return (np.random.random() < self.sticking_prob)

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

        # Update radii bounds if aggregate has grown in size
        r2 = (point[0] - self.seed[0])**2 + (point[1] - self.seed[1])**2
        if r2 > self.max_radius**2:
            r = np.sqrt(r2)
            self.max_radius = r
            self.spawn_radius = r + 5
            self.kill_radius = min(self.spawn_radius*2,
                                   self.lattice_size//2 - 2)

    def make_aggregate(self):
        N = 1  # number of particles in aggregate
        while self.max_radius < self.agg_radius:
            point = self._spawn()
            while True:  # Random walk
                choice = np.random.choice(len(self.directions))
                (drow, dcol) = self.directions[choice]
                point[0] += drow
                point[1] += dcol
                if self._is_dead(point):
                    break
                elif self._is_occupied(point):
                    # Particle 'bounces' if it moves onto an occupied site
                    point[0] -= drow
                    point[1] -= dcol
                elif self._is_sticky(point) and self._will_stick():
                    N += 1
                    self._update_aggregate(point, N)
                    break
        # Select part of array
        self.particles = self.particles[:N]


# %% SECTION 2: Simple plot of an aggregate

# User input: CHANGEABLE
agg_radius = 200
nbhood = 'moore'
sticking_prob = 0.1

# Make aggregate
model = StickyDLA(agg_radius, nbhood, sticking_prob)
model.make_aggregate()

# Adjustments to show only part of image
min_bounds = int(model.lattice_size//2 - 1.05*agg_radius)
max_bounds = int(model.lattice_size//2 + 1.05*agg_radius)
lattice_plot = copy.copy(model.lattice[min_bounds:max_bounds+1,
                                       min_bounds:max_bounds+1])

left = -(np.shape(lattice_plot)[0]//2)
right = (np.shape(lattice_plot)[0]-1)+left
bottom = left
top = right

# Plot
cs = copy.copy(cm.get_cmap('RdBu_r'))
cs.set_under('k')

fig = plt.figure()
ax = fig.add_subplot(xlabel='$x$', ylabel='$y$')
ax.imshow(lattice_plot, cmap=cs, vmin=0.5, origin='lower',
          interpolation='none', extent=[left, right, bottom, top])
