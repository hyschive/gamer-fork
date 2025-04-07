import numpy as np
import matplotlib.pyplot as plt
import os.path


###################################################################################################################
# Note that the external potential table has to have the same radial bins as the density profile of ParCloud
ParCloud_DensProf_Filename = 'FDMHaloDensityProfile'
###################################################################################################################


###################################################################################################################
# code units (in cgs)
UNIT_L   = 4.4366320345075490e+24
UNIT_D   = 2.5758579724476994e-30
UNIT_T   = 4.4366320345075490e+17

NEWTON_G = 6.6738e-8 / ( 1.0/UNIT_D/(UNIT_T**2) )
###################################################################################################################


###################################################################################################################
def Potential_UniDenSph(r, M, R):
   return -NEWTON_G*M/r if r >= R else NEWTON_G*M/(2.0*R*R*R)*(r*r-3.0*R*R)
###################################################################################################################


###################################################################################################################
# parameters for the external potential
# the default is the potential of an uniform density sphere with 3x mass and 0.3x radius of the default CDM halo
ParCloud_ExtPot_UniDenSph_M = 3.0*3.618847000e-02  # in code_mass
ParCloud_ExtPot_UniDenSph_R = 0.3*3.234605475e-02  # in code_length
###################################################################################################################


###################################################################################################################
# output the information
print( 'Information'                                                                            )
print( 'ParCloud_DensProf_Filename  =   '+ParCloud_DensProf_Filename                            )
print( 'UNIT_L                      = {: >16.8e} cm'.format(      UNIT_L                      ) )
print( 'UNIT_D                      = {: >16.8e} g/cm**3'.format( UNIT_D                      ) )
print( 'UNIT_T                      = {: >16.8e} s'.format(       UNIT_T                      ) )
print( 'ParCloud_ExtPot_UniDenSph_M = {: >16.8e} UNIT_M'.format(  ParCloud_ExtPot_UniDenSph_M ) )
print( 'ParCloud_ExtPot_UniDenSph_R = {: >16.8e} UNIT_L'.format(  ParCloud_ExtPot_UniDenSph_R ) )
###################################################################################################################


###################################################################################################################
# create the external potential table
ParCloud_DensProf_radius, _     = np.loadtxt( ParCloud_DensProf_Filename, skiprows=1, unpack=True )
ParCloud_ExtPot_table_radius    = np.append( ParCloud_DensProf_radius, 2.0*ParCloud_DensProf_radius[-1] ) # probably a bug, we need to add one extra row, which will not be used
ParCloud_ExtPot_table_potential = np.array([Potential_UniDenSph( radius, ParCloud_ExtPot_UniDenSph_M, ParCloud_ExtPot_UniDenSph_R ) for radius in ParCloud_ExtPot_table_radius ])
###################################################################################################################


###################################################################################################################
# save to file
filename = 'ParCloud_ExtPotTable'

if os.path.exists( filename ):
   print( '\nWARNING: file "%s" already exists and will be overwritten !!'%filename )

np.savetxt( filename,
            np.column_stack( (ParCloud_ExtPot_table_radius, ParCloud_ExtPot_table_potential) ),
            fmt='%23.8e',
            header='%21s %23s'%( 'r', 'potential' ) )
###################################################################################################################


###################################################################################################################
# plot to images
fig = plt.figure()
ax  = fig.add_subplot( 111 )

# plot some important values for reference
ax.axvline( ParCloud_ExtPot_UniDenSph_R, linestyle='--', color='grey', label=r'$R$' )

# plot the external potential
ax.plot( ParCloud_ExtPot_table_radius, ParCloud_ExtPot_table_potential, '-', color='r', label=r'$\Phi(r)$' )

# annotate the information
ax.annotate( r'UniDenSph $M$ = %.8e'%ParCloud_ExtPot_UniDenSph_M+'\n'+
             r'UniDenSph $R$ = %.8e'%ParCloud_ExtPot_UniDenSph_R,
             xy=( 0.02, 0.80 ), xycoords='axes fraction' )

# setting for the figure
ax.set_xscale( 'log' )
ax.set_xlim( 0.5*np.min(ParCloud_ExtPot_table_radius), 2.0*np.max(ParCloud_ExtPot_table_radius) )

# set the labels
ax.set_xlabel( r'$r$'+' (code_length)'     )
ax.set_ylabel( r'$\Phi$'+' (code_length$^2$/code_time$^2$)' )
fig.suptitle(   'External Potential for ParCloud'  )
ax.legend( loc='lower right' )

# save the figure
fig.subplots_adjust( top=0.93, bottom=0.1, left=0.15, right=0.97 )

filename_fig = 'fig_ParCloud_ExtPotTable.png'

if os.path.exists( filename_fig ):
   print( '\nWARNING: file "%s" already exists and will be overwritten !!'%filename_fig )

fig.savefig( filename_fig )
plt.close()
###################################################################################################################
