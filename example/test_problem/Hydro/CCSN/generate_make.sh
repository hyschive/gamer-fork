# This script should run in the same directory as configure.py

PYTHON=python3

${PYTHON} configure.py --machine=eureka_intel --mpi=true --hdf5=true --fftw=FFTW3 --gpu=true --gpu_arch=TURING --debug=false \
                       --model=HYDRO --flu_scheme=MHM_RP --slope=PPM --mhd=false --gravity=true --unsplit_gravity=false \
                       --eos=NUCLEAR --nuc_table=NUC_TABLE_MODE_TEMP --nuc_solver=NUC_EOS_SOLVER_DIRECT --neutrino=LIGHTBULB --grep=true
