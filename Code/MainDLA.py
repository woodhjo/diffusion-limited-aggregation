# -*- coding: utf-8 -*-
"""
2D Diffusion Limited Aggregation
--------------------------------
A simple model for diffusion limited aggregation, where the maximum bounding
radius of the aggregate is supplied by the user.

Particles always stick when they encounter an adjacent occupied site in either
the Von Neumann neighbourhood or the Moore neighbourhood.


Section 1: DLA class
Section 2: Create a simple aggregate and display it
Section 3: Create multiple aggregates and plot N-r graph
Sectin 4: Analyse result (find gradient) from section 3
Section 5: Plot the C-r graph for the aggregates created in section 3

@author: woodhjo
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy
from numba.experimental import jitclass
from numba import typeof, int64
from scipy.spatial import cKDTree
from scipy.optimize import leastsq
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
        ('lattice_size', int64)]


@jitclass(spec)
class DLA:
    def __init__(self, agg_radius, nbhood):
        # Check validity of input
        self._check_input(agg_radius, nbhood)

        self.agg_radius = int(agg_radius)
        self.nbhood = nbhood

        # Lattice 'radius' double aggregate radius (to reduce biases)
        self.lattice_size = 4*self.agg_radius

        # Set up lattice structures
        # Sticky lattice stores where sticky sites are
        self.lattice = np.zeros((self.lattice_size, self.lattice_size))
        self.sticky_lattice = np.zeros((self.lattice_size, self.lattice_size))

        # Use 2agg_radius^2 as MAX number of expected particles
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
        self.directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1),
                           (-1, 1), (-1, -1)]

        # Create central seed and update arrays
        self.seed = [self.lattice_size//2, self.lattice_size//2]
        self.lattice[self.seed[0], self.seed[1]] = 1
        self.particles[0] = self.seed
        for s in self.sites:
            self.sticky_lattice[self.seed[0] + s[0], self.seed[1] + s[1]] = 1

    def _check_input(self, agg_radius, nbhood):
        # Check validity of user input
        if int(agg_radius) < 5:
            raise ValueError('Please enter a larger radius.')
        elif (nbhood != 'moore') and (nbhood != 'neumann'):
            raise ValueError('Neighbourhood not recognised. Accepted values: '
                             '"moore", "neumann"')

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
                elif self._is_sticky(point):
                    N += 1
                    self._update_aggregate(point, N)
                    break
        # Select part of array
        self.particles = self.particles[:N]


# %% SECTION 2: Simple plot of an aggregate

# User input: CHANGEABLE
agg_radius = 200
nbhood = 'moore'


# Make aggregate
model = DLA(agg_radius, nbhood)
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


# %% SECTION 3: Plot N-r graph for set of results

# Input: CHANGEABLE
agg_radius = 200
nbhood = 'moore'
trials = 10  # number of aggregates to make
measurements = 101  # how many measurements to take along radius of aggregate

# Setup for data
lattice_size = 4*agg_radius
results = np.zeros((trials, measurements))
R = np.linspace(0, agg_radius, measurements)

# Calculate number of particles within radius r in R
for i in range(trials):
    model = DLA(agg_radius, nbhood)
    model.make_aggregate()

    tree = cKDTree(model.particles)
    for n, r in enumerate(R):
        num = tree.query_ball_point([lattice_size//2, lattice_size//2],
                                    r, return_length=True, n_jobs=-1)
        results[i, n] = num
    print(f'Finished trial {i + 1}')

# Calculate mean and standard error
mean = results.mean(axis=0)
sterror = results.std(axis=0)/np.sqrt(trials)

# Plot results
fig2 = plt.figure()
ax2 = fig2.add_subplot(xlabel='$r$', ylabel='$N(r)$',
                       xlim=(R[1], R[measurements-1]),
                       ylim=(mean[1]*0.5, mean[measurements-1]*2))
plt.loglog(R, mean, '.-')

# %% SECTION 4: Analyse results from above

# Specify over which points range the gradient must be taken: CHANGEABLE
minim = 10
maxim = 55

# This section uses the code from the site below to fit a
# line using errors.
# https://scipy-cookbook.readthedocs.io/items/FittingData.html

# Errors are considered small and ignore a constant prefactor
# Ref: http://faculty.washington.edu/stuve/uwess/log_error.pdf
logerror = sterror/mean


def powerlaw(x, a, b):
    return a * x**b


def linefit(p, x):
    return p[0]*x + p[1]


def errfit(p, x, y, err):
    return (y - linefit(p, x)) / err


p = [1, 1]
fitting_data = leastsq(errfit, p, args=(np.log10(R[minim:maxim]),
                                        np.log10(mean[minim:maxim]),
                                        logerror[minim:maxim]), full_output=1)

m, c = fitting_data[0]
covar = fitting_data[1]
merror = np.sqrt(covar[0][0])

print(f'The gradient of the N-r graph fitted over r = {R[minim]} to '
      f'r = {R[maxim-1]} is {m:.3f} +/- {merror:.3f}.')


# Final plot
fig3 = plt.figure()
ax3 = fig3.add_subplot(xlabel='$r$', ylabel='$N(r)$', xscale='log',
                       yscale='log',
                       xlim=(R[1], R[measurements-1]),
                       ylim=(mean[1]*0.5, mean[measurements-1]*2))
ax3.plot(R, powerlaw(R, 10**c, m), label='regression line')
ax3.plot(R, mean, '.', color='darkblue', label='data')
ax3.legend(loc='best')

# Formatting
ax3.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
ax3.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
ax3.xaxis.set_minor_formatter(NullFormatter())
ax3.yaxis.set_minor_formatter(NullFormatter())

# %% SECTION 5: Plot C-r graph for aggregates

# Set up tree for all sites in lattice structure
num_sites = np.zeros(measurements)
X, Y = np.mgrid[0:lattice_size, 0:lattice_size]
tree2 = cKDTree(list(zip(X.ravel(), Y.ravel())))

# Find number of sites within radius r of the lattice centre
for n, r in enumerate(R):
    num_sites[n] = tree2.query_ball_point([lattice_size//2, lattice_size//2],
                                          r, return_length=True, n_jobs=-1)

# Measure average density
density = np.divide(results, num_sites)
density_mean = density.mean(axis=0)

# Plot graph
fig4 = plt.figure()
ax4 = fig4.add_subplot(xlabel='$r$', ylabel='$C(r)$', xscale='log',
                       yscale='log',
                       xlim=(R[1], R[measurements-1]))
ax4.plot(R, density_mean, '.-')
