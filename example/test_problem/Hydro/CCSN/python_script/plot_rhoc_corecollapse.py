#! /usr/bin/env python
#
# -*- coding: utf-8 -*-
#
#  Purpose:
#    Compare the evolution of central density between FLASH and GAMER
#    for Core Collapse test problem
#

import matplotlib
matplotlib.use("Agg")

import re
import numpy as np
import matplotlib.pyplot as plt


# reference solution
RefSol = "../ReferenceSolution/CoreCollapse/flash_3D_s20_SFHo_nonup.dat"

### load data
## FLASH
t_flash, rhoc_flash = np.genfromtxt(RefSol, usecols = [0, 16], unpack = True)

## GAMER
fn_gamer = "../Record__CentralQuant"
t_gamer, rhoc_gamer = np.genfromtxt(fn_gamer, usecols = [0, 5], unpack = True)


### plot the evolution of central density
fig, ax = plt.subplots()
alpha = 0.8

ax.plot(t_gamer * 1e3, rhoc_gamer, label = "GAMER", alpha = alpha)
ax.plot(t_flash * 1e3, rhoc_flash, label = "FLASH", alpha = alpha)

ax.set_xlabel(r"$t$ [ms]")
ax.set_ylabel(r"Central density [g cm$^{-3}$]")
ax.set_yscale('log')
ax.legend(loc = "upper left")

fig.tight_layout()
plt.savefig("Rhoc_CoreCollapse.png", dpi = 200)

plt.close(fig)
