import os
import re
import numpy as np
import matplotlib.pylab as plt

# plt parameters
plt.rcParams["font.size"] = 24
plt.rcParams["lines.linewidth"] = 2.0
plt.rcParams["lines.marker"] = "o"
plt.rcParams["lines.markersize"] = 10.0
plt.rcParams["lines.markerfacecolor"] = "none"

# retrieve the unit system in GAMER
reg_pattern = r"\s*([-+]?\d+\.?\d*[eE]?[-+]?\d*)"
parfile = os.path.join("Record__Note")
par     = open(parfile).read()

UNIT_M    = re.findall(r"UNIT_M" + reg_pattern, par)
UNIT_L    = re.findall(r"UNIT_L" + reg_pattern, par)
UNIT_T    = re.findall(r"UNIT_T" + reg_pattern, par)
NX0_TOT_X = re.findall(r"NX0_TOT\[0\]\s+(\d+)", par)

UNIT_M    = float(UNIT_M[0])
UNIT_L    = float(UNIT_L[0])
UNIT_T    = float(UNIT_T[0])
NX0_TOT_X = int(NX0_TOT_X[0])

UNIT_V    = UNIT_L/UNIT_T
UNIT_D    = UNIT_M/UNIT_L**3
UNIT_P    = UNIT_M/UNIT_L/UNIT_T**2

# target GAMER file
DATA_ID   = 10
data      = np.loadtxt("Xline_y0.000_z0.000_%06d"%DATA_ID)

# plot GAMER data
fig, axes = plt.subplots(2, 2, figsize=(24, 18))
axes = axes.flatten()

X    = data[:, 3]
DENS = data[:, 6]
ENER = data[:,10]
VX   = data[:, 7]/DENS
YE   = data[:,13]

for ax in axes:
    ax.set_xlabel("X [cm]")
axes[0].plot(X*UNIT_L, DENS*UNIT_D)
axes[0].set_yscale("log")
axes[0].set_ylabel(r"Density [$\rm{g/cm^3}$]")
axes[1].plot(X*UNIT_L, VX*UNIT_V)
axes[1].set_yscale("log")
axes[1].set_ylabel("Velocity [g/cm]")
axes[2].plot(X*UNIT_L, ENER/DENS*UNIT_V**2)
axes[2].set_ylabel(r"Specific Energy [$\rm{cm^2/s^2}$]" )
axes[2].set_yscale("log")
axes[3].plot(X*UNIT_L, YE/DENS, label="GAMER ({:04d})".format(NX0_TOT_X))
axes[3].set_ylabel("Ye [1]")

# target reference files
ref_dir   = "ReferenceSolution/Nuclear/CCSN_Hydro/"
ref_0256  = ref_dir + "flash_n0256"
ref_1024  = ref_dir + "flash_n1024"
ref_4096  = ref_dir + "flash_n4096"
ref       = ref_0256, ref_1024, ref_4096
ref_label = "FLASH (0256)", "FLASH (1024)", "FLASH (4096)"

# plot reference data
for idx, ref_data in enumerate(ref):
    data = np.loadtxt(ref_data)
    X    = data[:,0]
    DENS = data[:,1]
    ENER = data[:,3]
    VX   = data[:,8]
    YE   = data[:,9]

    axes[0].plot(X, DENS)
    axes[1].plot(X, VX  )
    axes[2].plot(X, ENER)
    axes[3].plot(X, YE, label=ref_label[idx] )

plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig("Fig__Riemann_Hydro_Nuclear_%06d.png"%DATA_ID)
plt.close()
