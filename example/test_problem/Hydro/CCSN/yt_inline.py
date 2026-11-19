# script for extracting slice data of various quantities on the XY, YZ, and XZ planes

import os
import yt
import yt_libyt
import numpy as np

yt.enable_parallelism()
yt.mylog.setLevel("ERROR")


# user setting
direction_dict = {
    "x": ("YZ", 0),
    "y": ("XZ", 1),
    "z": ("XY", 2)
}

field_list = [
    ("gamer", "CCMagX"),
    ("gamer", "CCMagY"),
    ("gamer", "CCMagZ"),
    ("gamer", "Dens"),
    ("gamer", "MomX"),
    ("gamer", "MomY"),
    ("gamer", "MomZ"),
    ("gamer", "Engy"),
    ("gamer", "Entr"),
    ("gamer", "Pres"),
    ("gamer", "Temp"),
    ("gas",   "specific_thermal_energy"),
    ("gas",   "ye")
]

direction_list = "xyz"
path_fnout  = "Data_Plt"
output_step = 2

os.makedirs(path_fnout, exist_ok = True)


# main routine
def yt_inline():
    # get data
    ds       = yt_libyt.libytDataset()
    unit_len = ds.length_unit.v
    step     = ds.parameters["step"]

    # dump data every "output_step" steps
    if step % output_step:
        return

    # add the Ye field
    def _ye(field, data):
        return yt.YTArray(data["Ye"].v / data["Dens"].v, "dimensionless")

    ds.add_field(("gas", "ye"), function = _ye,
                 units = "dimensionless", sampling_type = "cell")

    # read the PNS center from the last row of Record__CentralQuant
    pns_coord = np.genfromtxt("Record__CentralQuant", usecols = [11, 12, 13])  # in cm

    if len(pns_coord.shape) != 1:
        pns_coord = pns_coord[-1]

    pns_coord = [coord / unit_len  for coord in pns_coord]  # in code units

    # extract and save slices along the assigned coordinate axis
    for d in direction_list:
        fnout_label, axis = direction_dict[d]
        center_axis = pns_coord[axis]

        slc = ds.slice(axis = d, coord = center_axis, center = pns_coord)

        fnout = "Data_Plt_{}_{:06d}".format(fnout_label, step)
        fnout = os.path.join(path_fnout, fnout)
        slc.save_as_dataset(filename = fnout, fields = field_list)


def yt_inline_inputArg(fields):
    pass
