import argparse
import sys
import yt
import add_ELBDM_derived_fields

# load the command-line parameters
parser = argparse.ArgumentParser( description='Slices of ELBDM derived fields' )

parser.add_argument( '-p', action='store', required=False, type=str, dest='prefix',
                     help='path prefix [%(default)s]', default='../' )
parser.add_argument( '-s', action='store', required=True,  type=int, dest='idx_start',
                     help='first data index' )
parser.add_argument( '-e', action='store', required=True,  type=int, dest='idx_end',
                     help='last data index' )
parser.add_argument( '-d', action='store', required=False, type=int, dest='didx',
                     help='delta data index [%(default)d]', default=1 )

args=parser.parse_args()

# take note
print( '\nCommand-line arguments:' )
print( '-------------------------------------------------------------------' )
print( ' '.join(map(str, sys.argv)) )
print( '-------------------------------------------------------------------\n' )


idx_start = args.idx_start
idx_end   = args.idx_end
didx      = args.didx
prefix    = args.prefix

yt.enable_parallelism()

ts = yt.DatasetSeries( [ prefix+'/Data_%06d'%idx for idx in range(idx_start, idx_end+1, didx) ] )
for ds in ts.piter():

   ds.force_periodicity() # for the computation of gradient
   add_ELBDM_derived_fields.Add_ELBDM_derived_fields( ds )

   for field in ds.derived_field_list:

      if field[0] != 'gamer':
         continue

      try:
         sz = yt.SlicePlot( ds, 'z', field, center='c' )

         sz.set_axes_unit( 'code_length' )
         sz.annotate_timestamp( corner='upper_right' )
         sz.annotate_grids()
         sz.save()

      except Exception as e:
          print( e )
          pass

