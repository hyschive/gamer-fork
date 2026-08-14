"""
This file generates a toy vector potential for import into GAMER using the
OPT__INIT_BFIELD_BYVECPOT parameter. It does the following:

1. Generates a uniform coordinate grid
2. Defines a vector potential on the coordinate grid
3. Saves the coordinate grid and the vector potential to an HDF5 file

The units of the vector potential and the coordinate arrays should be the same
as those used in GAMER. So:

* coordinates are in UNIT_L
* vector potential components are in UNIT_B*UNIT_L = sqrt(4*pi*UNIT_P)*UNIT_L
  where UNIT_P = UNIT_M/UNIT_L/UNIT_T**2

The file should also be named "B_IC" for GAMER to recognize it.

It requires NumPy, h5py, and HDF5 to be installed.
"""

import re
import os
import h5py
import numpy as np


## User-defined parameters.
NLEVEL    = 2   # number of grid in each dimension will be NX0_TOT * 2^NLEVEL
num_chunk = 64  # number of chunk for computing the vector potential sequentially


## Setups
regex_num = r"\s*([-+]?\d+\.?\d*[eE]?[-+]?\d*)"  # regular expression of all numeric numbers
ColIdx_IC = {"tovstar_short": (0, 2, 3)}         # column index of radius, density, and pressure in the input profile

## Retrieve the runtime parameters in Input__Parameter
par = open("../Input__Parameter").read()

# Simulation scale (assume the number of base-level cells are the same in each direction)
BOX_SIZE = re.findall(r"BOX_SIZE"  + regex_num, par)
NX0_TOT  = re.findall(r"NX0_TOT_X" + regex_num, par)

BOX_SIZE = float(BOX_SIZE[0])
NX0_TOT  = int(NX0_TOT[0])

# unit system
UNIT_L = re.findall(r"UNIT_L" + regex_num, par)
UNIT_M = re.findall(r"UNIT_M" + regex_num, par)
UNIT_T = re.findall(r"UNIT_T" + regex_num, par)

UNIT_L = float(UNIT_L[0])
UNIT_M = float(UNIT_M[0])
UNIT_T = float(UNIT_T[0])
UNIT_P = UNIT_M / UNIT_L / UNIT_T**2  # use energy density unit same as that in GAMER code
UNIT_B = np.sqrt(4 * np.pi * UNIT_P)
UNIT_A = UNIT_B * UNIT_L


## Retrieve the runtime parameters in Input__TestProb
par_testprob = open("../Input__TestProb").read()

# input profile
CCSN_Prof_File = re.findall(r"CCSN_Prof_File\s*(\S*)\s", par_testprob)
CCSN_Prof_File = CCSN_Prof_File[0]

# parameters for magnetic field
CCSN_Mag = re.findall(r"CCSN_Mag" + regex_num, par_testprob)
CCSN_Mag = int(CCSN_Mag[0])

CCSN_Mag_B0 = re.findall(r"CCSN_Mag_B0" + regex_num, par_testprob)
CCSN_Mag_B0 = float(CCSN_Mag_B0[0]) / UNIT_B

if CCSN_Mag == 0:
    CCSN_Mag_np = re.findall(r"CCSN_Mag_np" + regex_num, par_testprob)
    CCSN_Mag_np = float(CCSN_Mag_np[0])

    ColIdx = ColIdx_IC.get(os.path.basename(CCSN_Prof_File), (0, 2, 6))
else:
    CCSN_Mag_R0 = re.findall(r"CCSN_Mag_R0" + regex_num, par_testprob)
    CCSN_Mag_R0 = float(CCSN_Mag_R0[0]) / UNIT_L


# Print used parameters
print("=" * 13 + " User-defined parameters " + "=" * 13)
print("NLEVEL                         : {:d}    ".format(NLEVEL))
print("=" * 13 + " Parameters adopted from Input__Parameter and Input__TestProb " + "=" * 13)
print("BOX_SIZE    in Input__Parameter: {:13.7e}".format(BOX_SIZE))
print("NX0_TOT     in Input__Parameter: {:13d}  ".format(NX0_TOT))
print("UNIT_L      in Input__Parameter: {:13.7e}".format(UNIT_L))
print("UNIT_M      in Input__Parameter: {:13.7e}".format(UNIT_M))
print("UNIT_T      in Input__Parameter: {:13.7e}".format(UNIT_T))
print("CCSN_Mag    in Input__TestProb : {:13d}  ".format(CCSN_Mag))
print("CCSN_Mag_B0 in Input__TestProb : {:13.7e}".format(CCSN_Mag_B0 * UNIT_B))
if CCSN_Mag == 0:
    print("CCSN_Mag_np in Input__TestProb : {:13.7e}".format(CCSN_Mag_np))
else:
    print("CCSN_Mag_R0 in Input__TestProb : {:13.7e}".format(CCSN_Mag_R0 * UNIT_L))


### Load initial profiles of density and pressure
if CCSN_Mag == 0:
    fn = os.path.join("..", CCSN_Prof_File)
    print("Loading intial profile in {}".format(fn))
    radius, dens, pres = np.genfromtxt(fn, usecols = ColIdx, unpack = True)
    radius /= UNIT_L

    # functions for interpolation
    interp_pres = lambda r: np.interp(r, radius, pres)
    interp_dens = lambda r: np.interp(r, radius, dens)

    # central density and pressure
    rho_c  = interp_dens(0.0)
    pres_c = interp_pres(0.0)


# Number of cells along each dimension of the input grid.
# This is somewhat arbitrary, but should be chosen in
# such a way as to adequately resolve the vector potential.

ddims = np.array([NX0_TOT * 2**NLEVEL]*3, dtype='int')

# Left edge and right edge coordinates of the desired
# simulation domain which will be used in GAMER.

le = np.zeros(3)
re = np.ones(3) * BOX_SIZE
ce = 0.5 * (le + re)

# Since we need to take derivatives of the vector potential
# to get the magnetic field on the simulation domain, the
# input grid must be extended a bit beyond this boundary.
# We therefore add a buffer of three cells on each side.
# (Three cells are necessary to solve some corner cases
# resulting from round-off errors.)

delta = (re-le)/ddims
ddims += 6
le -= 3.0*delta
re += 3.0*delta

# Construct the grid cell edge coordinates

x = np.linspace(le[0], re[0], ddims[0]+1)
y = np.linspace(le[1], re[1], ddims[1]+1)
z = np.linspace(le[2], re[2], ddims[2]+1)

# Find the grid cell midpoints

x = 0.5*(x[1:]+x[:-1])
y = 0.5*(y[1:]+y[:-1])
z = 0.5*(z[1:]+z[:-1])


# Write the ICs to an HDF5 file

f = h5py.File("../B_IC", "w")

# Write coordinate arrays

f.create_dataset("x", data=x)
f.create_dataset("y", data=y)
f.create_dataset("z", data=z)


# Functions for generating vector potential which depends on all three coordinates
if CCSN_Mag == 0:  # Liu+ 2008
    func_Ax = lambda xx, yy, zz, factor_dens, factor_pres: -yy * CCSN_Mag_B0 * factor_dens * factor_pres
    func_Ay = lambda xx, yy, zz, factor_dens, factor_pres:  xx * CCSN_Mag_B0 * factor_dens * factor_pres
    func_Az = lambda xx, yy, zz, factor_dens, factor_pres:  zz * 0.0
else:              # Suwa+ 2007
    func_Ax = lambda xx, yy, zz, rr: -yy * 0.5 * CCSN_Mag_B0 / (1.0 + (rr / CCSN_Mag_R0)**3)
    func_Ay = lambda xx, yy, zz, rr:  xx * 0.5 * CCSN_Mag_B0 / (1.0 + (rr / CCSN_Mag_R0)**3)
    func_Az = lambda xx, yy, zz, rr:  zz * 0.0

func_A = {"x": func_Ax,
          "y": func_Ay,
          "z": func_Az }


# Use the 1-D coordinate arrays to consruct 2D coordinate arrays
# that we will use to compute an analytic vector potential
xx, yy = np.meshgrid(x - ce[0], y - ce[1], sparse=False, indexing='ij')
xx     = xx[:, :, np.newaxis]
yy     = yy[:, :, np.newaxis]
varpi2 = xx * xx + yy * yy


# Here we construct the vector potential sequentially to save memory
for coord in "xyz":
    # Create vector potential arrays for writing
    dset_shape = x.size, y.size, z.size
    dset = f.create_dataset("magnetic_vector_potential_{}".format(coord), dset_shape, dtype = np.float64)

    func = func_A[coord]

    # Loop over z to compute the vector potential sequentially
    for idx_beg in range(0, z.size, num_chunk):
        idx_end = min(idx_beg + num_chunk, z.size)

        zz = z[np.newaxis, np.newaxis, idx_beg:idx_end] - ce[2]
        rr = np.sqrt(varpi2 + zz * zz)

        if CCSN_Mag == 0:
            factor_dens = (1.0 - interp_dens(rr) / rho_c)**CCSN_Mag_np
            factor_pres = interp_pres(rr) / pres_c

            A = func(xx, yy, zz, factor_dens, factor_pres)
        else:
            A = func(xx, yy, zz, rr)

        dset[:, :, idx_beg:idx_end] = A

    f.flush()


# Close the file

f.close()
