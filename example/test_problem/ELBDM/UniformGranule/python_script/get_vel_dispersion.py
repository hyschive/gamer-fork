import yt
import numpy as np
import argparse
import sys

#-------------------------------------------------------------------------------------------------------------------------
# load the command-line parameters
parser = argparse.ArgumentParser( description='Get average velocity dispersion of the entire box' )

parser.add_argument( '-s', action='store', required=True,  type=int, dest='idx_start',
                     help='first data index' )
parser.add_argument( '-e', action='store', required=True,  type=int, dest='idx_end',
                     help='last data index' )
parser.add_argument( '-d', action='store', required=False, type=int, dest='didx',
                     help='delta data index [%(default)d]', default=1 )

args=parser.parse_args()

idx_start   = args.idx_start
idx_end     = args.idx_end
didx        = args.didx

# print command-line parameters
print( '\nCommand-line arguments:' )
print( '-------------------------------------------------------------------' )
for t in range( len(sys.argv) ):
   print( str(sys.argv[t]))
print( '' )
print( '-------------------------------------------------------------------\n' )

for idx in range(idx_start, idx_end+1, didx):
   ds = yt.load("../Data_%06d"%idx)
   N = ds.parameters["NX0"]
   N_tot = N[0]*N[1]*N[2]
   UNIT_L = ds.parameters["Unit_L"]
   ma   = ds.parameters["ELBDM_Mass"]
   hbar = ds.parameters["ELBDM_PlanckConst"]
   fac = hbar/ma

   grad_Real = ds.add_gradient_fields(("Real"))
   grad_Imag = ds.add_gradient_fields(("Imag"))

   dd = ds.all_data()

   if ( N_tot != len(dd["Dens"])):
      print('file size not matched!')
      sys.exit(1)

   dens = np.array(dd["Dens"])
   real = np.array(dd["Real"])
   imag = np.array(dd["Imag"])
   avedens = np.mean(dens)
   print("average density = " + str(avedens))
   print("minimum density = " + str(min(dens)))

   grad_real = np.array([dd["Real_gradient_x"], dd["Real_gradient_y"], dd["Real_gradient_z"]])*UNIT_L
   grad_imag = np.array([dd["Imag_gradient_x"], dd["Imag_gradient_y"], dd["Imag_gradient_z"]])*UNIT_L

   vx_bk = fac*(imag*grad_real[0] - real*grad_imag[0])/dens
   vy_bk = fac*(imag*grad_real[1] - real*grad_imag[1])/dens
   vz_bk = fac*(imag*grad_real[2] - real*grad_imag[2])/dens

   vx_qp = fac*(imag*grad_imag[0] + real*grad_real[0])/dens
   vy_qp = fac*(imag*grad_imag[1] + real*grad_real[1])/dens
   vz_qp = fac*(imag*grad_imag[2] + real*grad_real[2])/dens

   sigma_x_square_bk = np.average(dens*vx_bk**2)/avedens - (np.average(dens*vx_bk))**2/avedens**2
   sigma_y_square_bk = np.average(dens*vy_bk**2)/avedens - (np.average(dens*vy_bk))**2/avedens**2
   sigma_z_square_bk = np.average(dens*vz_bk**2)/avedens - (np.average(dens*vz_bk))**2/avedens**2

   sigma_x_square_qp = np.average(dens*vx_qp**2)/avedens - (np.average(dens*vx_qp))**2/avedens**2
   sigma_y_square_qp = np.average(dens*vy_qp**2)/avedens - (np.average(dens*vy_qp))**2/avedens**2
   sigma_z_square_qp = np.average(dens*vz_qp**2)/avedens - (np.average(dens*vz_qp))**2/avedens**2


   sigma_bk = ((sigma_x_square_bk + sigma_y_square_bk +sigma_z_square_bk)/3.)**0.5
   sigma_qp = ((sigma_x_square_qp + sigma_y_square_qp +sigma_z_square_qp)/3.)**0.5

   sigma_total = (sigma_bk**2+sigma_qp**2)**0.5
   print('bulk velocity dispersion = %g'%(sigma_bk))
   print('quantum pressure velocity dispersion = %g'%(sigma_qp))
   print('total velocity dispersion = %g'%(sigma_total))

   d = 0.35*2*np.pi*hbar/(ma*sigma_total)
   print('estimated granule diameter = %g\n'%(d))
