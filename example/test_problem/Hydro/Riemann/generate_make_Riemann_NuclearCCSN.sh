# This script should run in the same directory as configure.py

PYTHON=python3

${PYTHON} configure.py --model=HYDRO --eos=NUCLEAR --flux=HLLC --gpu_regcount_flu=255 \
                       --nuc_table=TEMP --nuc_solver=ORIG \
                       --flu_scheme=MHM --slope=PPM --hdf5=true "$@"
