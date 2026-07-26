The procedures of this LSS simulation includes:
---------------------------------------------------------------------------------------------------------------
    Flow chart
         if not axion transfer function           if axion transfer function
           |                                           |
           | "planck_2018.ini"                         | "planck_2018_axion.ini"
           |                                           |
           v                                           v
         [(a) CAMB]                                  [(a) axionCAMB]
         [camb planck_2018.ini]                      [./camb planck_2018_axion.ini]
           |                                           |
           | "planck_2018_transfer_out.dat"            | "planck_2018_axion_transfer_out.dat"
           |                                           |
           |                                           |
           ------->  [(b) set_velocities_zero.py] <-----
                     [python3 ./set_velocities_zero.py F planck_2018_trnasfer_out.dat planck_2018_transfer_out_no_vel.dat (CAMB)]
                     [python3 ./set_velocities_zero.py T planck_2018_axion_trnasfer_out.dat planck_2018_axion_transfer_out_no_vel.dat (axionCAMB)]
                                 |
                                 | "planck_2018_transfer_out_no_vel.dat" (CAMB)
                                 | "planck_2018_axion_transfer_out_no_vel.dat" (axionCAMB)
                                 |
       "ics_example.conf"        v
      ----------------------> [(c) MUSIC]
                              [./MUSIC ics_example.conf]
                                 |
                                 | "planck_2018_tf_no_vel.hdf5" (CAMB)
                                 | "planck_2018_axion_tf_no_vel.hdf5" (axionCAMB)
                                 |
      "ics_example.conf"         v
      --------------> [(d) make_umic_from_hdf5.py]
                      [python3 ./make_umic_from_hdf5.py S(D) planck_2018_tf_no_vel.hdf (CAMB)]
                      [python3 ./make_umic_from_hdf5.py S(D) planck_2018_axion_tf_no_vel.hdf (axionCAMB)]
                                 |
                                 | "UM_IC"
                                 |
                                 v
                              [(e) GAMER]

---------------------------------------------------------------------------------------------------------------

(a) After successful installation of CAMB/axionCAMB (see Notes for installation), use CAMB/axionCAMB to to produce transfer function.
    The input file planck_2018.ini/planck_2018_axion.ini for CAMB/axionCAMB is attached for reference.
    The expected output files are:
        planck_2018_trnasfer_out.dat       -> for CAMB
        planck_2018_axion_trnasfer_out.dat -> for axionCAMB

(b) Run the script: "python3 ./set_velocities_zero.py F(T) planck_2018_trnasfer_out.dat(planck_2018_axion_trnasfer_out.dat) planck_2018_transfer_out_no_vel.dat(planck_2018_axion_transfer_out_no_vel.dat)".
    The script reads-in the output files in step(a), and produce the transfer function requireded by MUSIC.
    The boolean F/T suggests whether axion-physics is included in the input transfer function files.
    The expected output files are:
        planck_2018_transfer_out_no_vel.dat       -> for CAMB
        planck_2018_axion_transfer_out_no_vel.dat -> for axionCAMB

(c) Modifying the cosmology paramters to be consistent with CAMB/axionCAMB .ini files, as well as the input file parameters in MUSIC input file ics_example.conf:
        transfer      = camb_file
        transfer_file = planck_2018_transfer_out_no_vel.dat/planck_2018_axion_transfer_out_no_vel.dat
        format        = generic
        filename      = planck_2018_tf_no_vel.hdf5/planck_2018_axion_tf_no_vel.hdf5)
    Install (see Notes for installation) and run the MUSIC: "./MUSIC ics_example.conf".

(d) Run the script "python3 ./make_umic_from_hdf5.py S(D) planck_2018_axion_tf_no_vel.hdf5" to produce UM_IC file for single(double) precision UM_IC for FDM simulation (or "python3 ./make_umic_from_hdf5.py S(D) planck_2018_tf_no_vel.hdf5" for CDM simulation).

(e) Run the GAMER.

# Notes
  1. CAMB will automatically compute \sigma_8 based on the given cosomology parameters.
     If you find the given \sigma_8 is not the desired value, it can be tuned by changing the As (scalar_amp in the input file .ini).
     Say if one wants to change the original \sigma_8 = 0.8119112  with As = 2.100549e-9 , to new value \sigma_8 = 0.818 , one just changes the new As to (0.818/0.8119112)^2*2.100549e-9 = 2.132173e-09 , then CAMB will give new \sigma_8 = 0.818.
     This is due to power spectrum is proportional to As*f(k) .
  2. CAMB (Code for Anisotropies in the Microwave Background)
         GitHub page:                    https://github.com/cmbant/CAMB
         Install procedures:             https://github.com/cmbant/CAMB#description-and-installation
  3. axionCAMB(modified version of CAMB to include axion physics)
         GitHub page:                    https://github.com/dgrin1/axionCAMB
         Install procedures:             https://github.com/dgrin1/axionCAMB?tab=readme-ov-file#compiling-axioncamb
  4. MUSIC(MUlti-Scale Initial Conditions for cosmological simulations)
         Bitbucket page:                 https://bitbucket.org/ohahn/music/src/master/
         Website (for user's guide pdf): https://www-n.oca.eu/ohahn/MUSIC/
         Install procedures:             modified the path and parameters in MUSIC-repo-provided Makefile, then make
                                         or, can refer to the section: "Building MUSIC" in the Bitbucket page

