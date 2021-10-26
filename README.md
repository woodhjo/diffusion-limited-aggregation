# diffusion-limited-aggregation

Code to simulate the diffusion limited aggregation of particles under different conditions including sticking probabilities, different seeds and restrictions on the movement of the particles (NSEW or NSEW + NW NE SW SE). This was my final project for a module I took in my third year in university.

A general algorithm is:
- Place some starting particle (seed) in the centre of a lattice/grid.
- Release another particle at a random point on the edge of the grid, or at a radius r from the seed.
- Let the particle perform a random walk.
- If the particle encounters the seed, it sticks. Otherwise, if it wanders outside a certain distance from the seed or takes too long to stick, it is removed.
- Repeat steps 2-4 above to form an aggregate.

This results in aggregates with dendritic structures being formed.

MainDLA.py
------------------
A simple model for diffusion limited aggregation, where the maximum bounding radius of the aggregate is supplied by the user. Particles always stick when they encounter an adjacent occupied site in either the Von Neumann neighbourhood or the Moore neighbourhood. This also produces N(number of particles)-r(radius) plot as well as a C-r plot which acts to measure density in terms of the number of occupied sites within radius r of the seed site.

![moore_growth](https://user-images.githubusercontent.com/92552830/138931537-52f3d207-31b5-48ea-86bb-f5968670fadc.gif)

StickingProbDLA.py
-----------------
As above, but now with an additional property which sets the probability that a particle sticks when encountering another particle.
Note: unlike in MainDLA, particles only move NSEW to avoid slipping through diagonal gaps

![StickingProb](https://user-images.githubusercontent.com/92552830/138930618-3cbd14af-67f7-48ae-b209-bb022987e479.png)

3dDLA.py
----------------
Simulation of diffusion limited aggregation in three dimensions.
Note: this works a bit differently to the other programmes by requesting both
the maximum number of particles and maximum size of the structure. The aggregate will stop growing once one of these has been reached.

![3ddla](https://user-images.githubusercontent.com/92552830/138931228-5bb1c224-482e-4635-8a93-67ed5b1756cf.gif)

LineSeedDLA.py
-----------------
A simple model for diffusion limited aggregation, where the maximum 'height' of the aggregate is supplied by the user. Instead of a point seed, this uses a line seed at the bottom edge of the lattice.

![Dendrites](https://user-images.githubusercontent.com/92552830/138931662-c4543a5e-6bc6-4673-a05b-355214ceff04.png)


There are also some other files under Extras, including one that uses the edges of the lattice as a box seed, adding sticking probability to the 3D aggregate and a more realistic sticking probability for the 2D aggregate which takes into account the number of adjacent particles rather than having a set probability.
