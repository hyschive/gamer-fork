import numpy as np
import matplotlib.pylab as plt


gamer_file = 'Xline_y0.500_z0.500_000005'

def get_data(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
        name  = lines[0].split()[1:]
        lines = lines[1:]
        for i in range(len(lines)):
            lines[i] = lines[i].split()
            for j in range(len(lines[i])):
                lines[i][j] = float(lines[i][j])
    data = np.array(lines)
    return data

#['i', 'j', 'k', 'x', 'y', 'z', 'Dens', 'MomX', 'MomY', 'MomZ', 'Engy', 'CRay', 'MagX', 'MagY', 'MagZ', 'MagEngy']

gamer_data = get_data(gamer_file)
gamer_x    = gamer_data[:, 3]
gamer_rho  = gamer_data[:, 6]
gamer_vx   = gamer_data[:, 7] / gamer_data[:, 6]
gamer_engy = gamer_data[:, 10]
gamer_cray = gamer_data[:, 11]

fig, ax = plt.subplots(2, 2, figsize = (8, 8))

ax[0, 0].set(ylabel = 'rho')
ax[0, 0].plot(gamer_x, gamer_rho, 'o', ms = 3)

ax[0, 1].set(ylabel = 'v')
ax[0, 1].plot(gamer_x, gamer_vx, 'o', ms = 3)

ax[1, 0].set(ylabel = 'P/(gamma-1)')
ax[1, 0].plot(gamer_x, gamer_engy - gamer_cray, 'o', ms = 3)
ax[1, 0].ticklabel_format(axis = 'y', style = 'sci', scilimits = (4, 5))

ax[1, 1].set(ylabel = 'e_{cr}')
ax[1, 1].plot(gamer_x, gamer_cray, 'o', ms = 3)
ax[1, 1].ticklabel_format(axis = 'y', style = 'sci', scilimits = (4, 5))




plt.show()
