The procedures of this LSS simulation includes:

(a) Using CAMB/axionCAMB to produce transfer function (the input file planck_2018.ini/planck_2018_axion.ini for CAMB/axionCAMB is attached for reference)
# Note
  1. CAMB will automatically compute \sigma_8 based on the given cosomology parameters. If you find the given \sigma_8 is not the desired value, it can be tuned by changing the As (scalar_amp in the input file .ini). Say if one wants to change the original \simga_8 = 0.8119112  with As = 2.100549e-9 , to new value \simga_8 = 0.818 , one just changes the new As to (0.818/0.8119112)^2*2.100549e-9 = 2.132173e-09 , then CAMB will give new \simga_8 = 0.818. This is due to power spectrum is proportional to As*f(k) .

(b) Run the script "./set_velocities_zero.py F(T)" to produce the CAMB(axionCAMB) transfer function: planck_2018_transfer_out_no_vel.dat(planck_2018_axion_transfer_out_no_vel.dat) needed by MUSIC.

(c) Modifying the cosmology paramters consistent with CAMB/axionCAMB .ini files, as well as the input file parameters in ics_example.conf (transfer = camb_file; transfer_file = planck_2018_axion_transfer_out_no_vel.dat/planck_2018_axion_transfer_out_no_vel.dat; format = generic; filename = planck_2018_tf_no_vel.hdf5/planck_2018_axion_tf_no_vel.hdf5). Then run the MUSIC: "./MUSIC ics_example.conf".

(d) Run the script "./make_umic_from_hdf5.py S(D) hdf5_filename" to produce UM_IC file for single/double precision GAMER.

(e) Run the GAMER.
