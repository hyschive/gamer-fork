# This script should run in the same directory as configure.py

PYTHON=python3

${PYTHON} configure.py --mpi=true --hdf5=true --fftw=FFTW3 --gpu=true --gpu_regcount_flu=255 --debug=false \
                       --model=HYDRO --flu_scheme=MHM_RP --slope=PPM --mhd=false --gravity=true --unsplit_gravity=false \
                       --eos=NUCLEAR --nuc_table=TEMP --nuc_solver=ORIG --neutrino=LIGHTBULB --grep=true "$@"
