# This script should run in the same directory as configure.py

PYTHON=python3

${PYTHON} configure.py --machine=eureka_intel --model=HYDRO --eos=NUCLEAR --flux=HLLC --nuc_table=NUC_TABLE_MODE_TEMP --nuc_solver=NUC_EOS_SOLVER_ORIG --flu_scheme=MHM --slope=PPM --hdf5=true
