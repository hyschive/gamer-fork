## Scripts for retrieving nuclear EoS tables for CCSN simulations

## Tables for the ORIG solver
## --> Available at https://stellarcollapse.org/equationofstate.html
# LS220
SRC=https://stellarcollapse.org/EOS/
FILE=LS220_234r_136t_50y_analmu_20091212_SVNr26.h5.bz2
curl $SRC/$FILE -o $FILE
bzip2 -d $FILE

# SFHo
SRC=https://stellarcollapse.org/~evanoc/
FILE=Hempel_SFHoEOS_rho222_temp180_ye60_version_1.1_20120817.h5.bz2
curl $SRC/$FILE -o $FILE
bzip2 -d $FILE


## Tables for the DIRECT solver
## --> Derived from the nuclear EoS table available at https://stellarcollapse.org/equationofstate.html
##     using the NuclearEoS package from https://github.com/hfhsieh/NuclearEoS
## --> The tables are stored on Google Drive, and gdown is utilized for downloading them
# LS220
# --> Auxiliary Table:
#     --> Resolution: (n_rho, n_eps/n_pres/n_entr, n_ye) = (234, 136, 50)
#         Energy    : 10^17 - 10^22   [erg/g]
#         Pressure  : 10^20 - 10^35.5 [dynes/cm^2]
#         Entropy   : 0.001 - 100.0   [kB/baryon]
# --> Checksum:
#     --> MD5 : 6f3bcd766854aca99d31c8b7ae6da8d7
#         SHA1: 514a6b2b8cf6e10cd00bc2fe195c2d80f689dde9
FILE=LS220_234r_136t_50y_SVNr26_136e_136s_136p_v1.h5.gz
TOKEN=11mnCgMTY6KqoL3jXJjBQXCMf9UMkBJ6U
gdown $TOKEN
gzip -d $FILE

# SFHo
# --> Auxiliary Table:
#     --> Resolution: (n_rho, n_eps/n_pres/n_entr, n_ye) = (222, 180, 60)
#         Energy    : 10^17 - 10^22   [erg/g]
#         Pressure  : 10^20 - 10^35.5 [dynes/cm^2]
#         Entropy   : 0.001 - 100.0   [kB/baryon]
# --> Checksum:
#     --> MD5 : db85822f0df1fe73cade76fb1041b012
#         SHA1: 40ae7c8417cbaf905d158404fff8737c98f5f4e8
FILE=Hempel_SFHoEOS_222r_180t_60y_v1.1_180e_180s_180p_v1.h5.gz
TOKEN=1g1J9ebH15TeYNzmuwujnxZUpRN1AZCBF
gdown $TOKEN
gzip -d $FILE
