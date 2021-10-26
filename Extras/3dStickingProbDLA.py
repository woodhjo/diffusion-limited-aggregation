# -*- coding: utf-8 -*-
"""
3D Diffusion Limited Aggregation
--------------------------------
Simulation of diffusion limited aggregation in three dimensions.

Note: this works a bit differently to the other programmes by requesting both
the maximum number of particles and maximum size of the structure.

The aggregate will stop growing once one of these has been reached. The reason
this is different is to try to avoid extremely large arrays being produced.

Section 1: DLA class
Section 2: Create an aggregate and display it

@author: woodhjo
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numba.experimental import jitclass
from numba import typeof, int64
from scipy.spatial import cKDTree
from matplotlib.ticker import NullFormatter, StrMethodFormatter

# To use with jitclass
spec = [('agg_radius', int64), ('lattice_size', int64),
        ('max_particles', int64),
        ('lattice', typeof(np.zeros((20, 20, 20)))),
        ('sticky_lattice', typeof(np.zeros((20, 20, 20)))),
        ('particles', typeof(np.zeros((20, 3)))),
        ('directions', typeof([(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)])),
        ('spawn_radius', typeof(0.5)),
        ('kill_radius', typeof(0.5)),
        ('max_radius', typeof(0.5)),
        ('seed', typeof([1, 2, 3])),
        ('sticking_prob', typeof(0.5))]


@jitclass(spec)
class StickyDLA3D:
    def __init__(self, max_particles, agg_radius, sticking_prob=1):
        # Check validity of input
        self._check_input(max_particles, agg_radius, sticking_prob)

        self.agg_radius = int(agg_radius)
        self.max_particles = int(max_particles)
        self.sticking_prob = float(sticking_prob)

        # Set up lattice structure
        self.lattice_size = 4*self.agg_radius
        self.lattice = np.zeros((self.lattice_size, self.lattice_size,
                                 self.lattice_size))
        self.sticky_lattice = np.zeros((self.lattice_size, self.lattice_size,
                                        self.lattice_size))

        # Set up array to contain particle positions
        self.particles = np.zeros((self.max_particles, 3))*np.nan

        # Initial radii
        self.spawn_radius = 5.0
        self.kill_radius = 10.0
        self.max_radius = 0.0

        # Set particle directions and sites to check
        self.directions = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 0, -1),
                           (0, -1, 0), (-1, 0, 0), (0, 1, 1), (0, -1, -1),
                           (0, -1, 1), (0, 1, -1), (1, 0, 1), (-1, 0, -1),
                           (1, 0, -1), (-1, 0, 1), (1, 1, 0), (-1, -1, 0),
                           (1, -1, 0), (-1, 1, 0), (1, 1, 1), (-1, -1, -1),
                           (1, -1, -1), (1, 1, -1), (-1, 1, 1), (1, -1, 1),
                           (-1, 1, -1), (-1, -1, 1)]

        # Make seed and update arrays
        self.seed = [self.lattice_size//2, self.lattice_size//2,
                     self.lattice_size//2]
        self.lattice[self.seed[0], self.seed[1], self.seed[2]] = 1
        self.particles[0] = self.seed
        for d in self.directions:
            self.sticky_lattice[self.seed[0] + d[0], self.seed[1] + d[1],
                                self.seed[2] + d[2]] = 1

    def _check_input(self, max_particles, agg_radius, sticking_prob):
        if int(max_particles) < 2:
            raise ValueError('Please enter a larger number of particles.')
        elif int(agg_radius) < 5:
            raise ValueError('Please enter a larger radius.')
        elif float(sticking_prob) <= 0 or float(sticking_prob) > 1:
            raise ValueError('Sticking probability must be in (0, 1].')

    def _spawn(self):
        # Spawn a particle at a random point along the spawn radius.
        # This uses a specific method to avoid bias towards the poles
        # Ref https://mathworld.wolfram.com/SpherePointPicking.html
        rand_theta = np.random.random()*2*np.pi
        rand_phi = np.arccos(2*np.random.random() - 1)
        start_point = [int(self.seed[0] +
                           self.spawn_radius * np.cos(rand_theta) *
                           np.sin(rand_phi)),
                       int(self.seed[1] +
                           self.spawn_radius * np.sin(rand_theta) *
                           np.sin(rand_phi)),
                       int(self.seed[2] +
                           self.spawn_radius * np.cos(rand_phi))]
        return start_point

    def _is_dead(self, point):
        # Check if particle has wandered into kill zone
        r2 = (point[0] - self.seed[0])**2 + \
            (point[1] - self.seed[1])**2 + \
            (point[2] - self.seed[2])**2
        return (r2 >= self.kill_radius**2)

    def _is_occupied(self, point):
        # Check if current cell is occupied
        return (self.lattice[point[0], point[1], point[2]] != 0)

    def _is_sticky(self, point):
        # Check if current cell is sticky
        return (self.sticky_lattice[point[0], point[1], point[2]] == 1)

    def _will_stick(self):
        return (np.random.random() < self.sticking_prob)

    def _update_aggregate(self, point, val):
        # Update arrays given a point
        self.lattice[point[0], point[1], point[2]] = val
        self.particles[val - 1] = point
        self.sticky_lattice[point[0], point[1], point[2]] = 0
        for d in self.directions:
            idx1 = point[0] + d[0]
            idx2 = point[1] + d[1]
            idx3 = point[2] + d[2]
            # Set unoccupied neighbouring sites as sticky
            if self.lattice[idx1, idx2, idx3] == 0:
                self.sticky_lattice[idx1, idx2, idx3] = 1

        # Increase radii bounds if aggregate has grown
        r2 = (point[0] - self.seed[0])**2 + \
            (point[1] - self.seed[1])**2 + \
            (point[2] - self.seed[2])**2
        if r2 >= self.max_radius**2:
            r = np.sqrt(r2)
            self.max_radius = r
            self.spawn_radius = r + 5
            self.kill_radius = min(self.spawn_radius*2,
                                   self.lattice_size//2 - 2)

    def make_aggregate(self):
        N = 1  # number of particles in aggregate
        # Generate random walkers until radius has been reached or maximum
        # num. of particles has been reached.
        while self.max_radius < self.agg_radius and N < self.max_particles:
            point = self._spawn()
            while True:
                choice = np.random.choice(len(self.directions))
                (dx, dy, dz) = self.directions[choice]
                point[0] += dx
                point[1] += dy
                point[2] += dz
                if self._is_dead(point):
                    break
                elif self._is_occupied(point):
                    # Particle 'bounces' if it moves onto an occupied site
                    point[0] -= dx
                    point[1] -= dy
                    point[2] -= dz
                elif self._is_sticky(point) and self._will_stick():
                    N += 1
                    self._update_aggregate(point, N)
                    break
        self.particles = self.particles[:N]


# %% SECTION 2:  Simple plot of an aggregate

# User input: CHANGEABLE
max_particles = 50000
agg_radius = 50
sticking_prob = 0.01


# Make aggregate
model = StickyDLA3D(max_particles, agg_radius, sticking_prob)
model.make_aggregate()
points = model.particles

# Setup plot
min_bounds = model.lattice_size//2 - int(model.max_radius)
max_bounds = model.lattice_size//2 + int(model.max_radius)

fig = plt.figure(figsize=(8, 8), facecolor='black')
ax = fig.add_subplot(projection='3d', facecolor='black',
                     xlim=(min_bounds, max_bounds),
                     ylim=(min_bounds, max_bounds),
                     zlim=(min_bounds, max_bounds))

# Formatting
ax.set_xlabel('$x$', color='white')
ax.set_ylabel('$y$', color='white')
ax.set_zlabel('$z$', color='white')
[i.set_color("white") for i in plt.gca().get_xticklabels()]
[i.set_color("white") for i in plt.gca().get_yticklabels()]
[i.set_color("white") for i in plt.gca().get_zticklabels()]
ax.w_xaxis.pane.fill = False
ax.w_yaxis.pane.fill = False
ax.w_zaxis.pane.fill = False

# Plot
cs = cm.Reds(np.arange(0, np.shape(points)[0])/np.shape(points)[0])
ax.scatter(points[:, 0], points[:, 1], points[:, 2], 'o', color=cs, s=1,
           alpha=1)
