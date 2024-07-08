import yt
import numpy as np
import argparse
import sys

#-------------------------------------------------------------------------------------------------------------------------
# load the command-line parameters
parser = argparse.ArgumentParser( description='Get disk properties' )

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

field     = 'Dens'
colormap  = 'algae'
#colormap  = 'magma'
dpi       = 300


for idx in range(idx_start, idx_end+1, didx):
   ds = yt.load("../Data_%06d"%idx)
   dd = ds.all_data()
   dens = np.array(dd["Dens"])
   avedens = np.mean(dens)

   plt = yt.SlicePlot( ds, 0, fields = field, center = 'c')
   plt.set_zlim( field, avedens*1.0e-4, avedens*1.0e+1, dynamic_range=None)
   plt.set_axes_unit( 'kpc' )
   plt.set_unit( field, 'Msun/kpc**3')
   plt.set_cmap( field, colormap )
   plt.save( mpl_kwargs={"dpi":dpi} )

   plt = yt.SlicePlot( ds, 1, fields = field, center = 'c')
   plt.set_zlim( field, avedens*1.0e-4, avedens*1.0e+1, dynamic_range=None)
   plt.set_axes_unit( 'kpc' )
   plt.set_unit( field, 'Msun/kpc**3')
   plt.set_cmap( field, colormap )
   plt.save( mpl_kwargs={"dpi":dpi} )

   plt = yt.SlicePlot( ds, 2, fields = field, center = 'c')
   plt.set_zlim( field, avedens*1.0e-4, avedens*1.0e+1, dynamic_range=None)
   plt.set_axes_unit( 'kpc' )
   plt.set_unit( field, 'Msun/kpc**3')
   plt.set_cmap( field, colormap )
   plt.save( mpl_kwargs={"dpi":dpi} )


