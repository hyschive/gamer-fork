#include "NuclearEoS.h"
#include "GAMER.h"

#if ( MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )


#ifdef SUPPORT_HDF5

#  define H5_USE_16_API    1
#  include "hdf5.h"

#  ifdef FLOAT8
#     define H5T_GAMER_REAL H5T_NATIVE_DOUBLE
#  else
#     define H5T_GAMER_REAL H5T_NATIVE_FLOAT
#  endif

#else

#  error : ERROR : must enable SUPPORT_HDF5 for EOS_NUCLEAR !!

#endif // #ifdef SUPPORT_HDF5 ... else ...



extern int    g_nrho;
extern int    g_nye;
extern int    g_nrho_mode;
extern int    g_nmode;
extern int    g_nye_mode;
extern double g_energy_shift;

extern real  *g_alltables;
extern real  *g_alltables_mode;
extern real  *g_logrho;
extern real  *g_yes;
extern real  *g_logrho_mode;
extern real  *g_entr_mode;
extern real  *g_logprss_mode;
extern real  *g_yes_mode;

#if ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )
extern int    g_ntemp;
extern real  *g_logtemp;
extern real  *g_logeps_mode;
#else
extern int    g_neps;
extern real  *g_logeps;
extern real  *g_logtemp_mode;
#endif



// catch HDF5 errors
#define HDF5_ERROR( fn_call )                                           \
{                                                                       \
   const int _error_code = fn_call;                                     \
   if ( _error_code < 0 )                                               \
   {                                                                    \
      Aux_Error( ERROR_INFO, "HDF5 call '%s' returned error code %d",   \
                 #fn_call, _error_code );                               \
   }                                                                    \
}



//-------------------------------------------------------------------------------------
// Function    :  nuc_eos_C_ReadTable
// Description :  Load the EoS table from the disk
//
// Note        :  1. Invoked by EoS_Init_Nuclear()
//
// Parameter   :  nuceos_table_name : Filename
//
// Return      :  EoS tables
//-------------------------------------------------------------------------------------
void nuc_eos_C_ReadTable( char *nuceos_table_name )
{

   if ( MPI_Rank == 0 )    Aux_Message( stdout, "%s ...\n", __FUNCTION__ );


   if ( MPI_Rank == 0 )
      Aux_Message( stdout, "   Reading nuclear EoS table: %s\n", nuceos_table_name );

// check file existence
   if ( !Aux_CheckFileExist(nuceos_table_name) )
      Aux_Error( ERROR_INFO, "file \"%s\" does not exist !!\n", nuceos_table_name );


// use these two macros to easily read in a lot of variables in the same way
// --> the first reads in one variable of a given type completely
#  define READ_EOS_HDF5( NAME, VAR, TYPE, MEM )                                        \
   {                                                                                   \
      hid_t dataset;                                                                   \
      HDF5_ERROR(  dataset = H5Dopen( file, NAME, H5P_DEFAULT )  );                    \
      HDF5_ERROR(  H5Dread( dataset, TYPE, MEM, H5S_ALL, H5P_DEFAULT, VAR )  );        \
      HDF5_ERROR(  H5Dclose( dataset )  );                                             \
   }

#  define READ_EOSTABLE_HDF5( NAME, OFF )                                              \
   {                                                                                   \
      hsize_t offset[2] = { OFF, 0 };                                                  \
      H5Sselect_hyperslab( mem3, H5S_SELECT_SET, offset, NULL, var3, NULL );           \
      READ_EOS_HDF5( NAME, g_alltables, H5T_GAMER_REAL, mem3 );                        \
   }

#  define READ_EOSTABLE_MODE_HDF5( NAME, OFF )                                         \
   {                                                                                   \
      hsize_t offset[2] = { OFF, 0 };                                                  \
      H5Sselect_hyperslab( mem3_mode, H5S_SELECT_SET, offset, NULL, var3_mode, NULL ); \
      READ_EOS_HDF5( NAME, g_alltables_mode, H5T_GAMER_REAL, mem3_mode );              \
   }



// open file
   hid_t file;
   HDF5_ERROR(  file = H5Fopen( nuceos_table_name, H5F_ACC_RDONLY, H5P_DEFAULT )  );


// read size of tables
   READ_EOS_HDF5( "pointsrho",      &g_nrho,      H5T_NATIVE_INT, H5S_ALL );
#  if ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )
   READ_EOS_HDF5( "pointstemp",     &g_ntemp,     H5T_NATIVE_INT, H5S_ALL );
#  else
   READ_EOS_HDF5( "pointsenergy",   &g_neps,      H5T_NATIVE_INT, H5S_ALL );
#  endif
   READ_EOS_HDF5( "pointsye",       &g_nye,       H5T_NATIVE_INT, H5S_ALL );
#  if ( NUC_EOS_SOLVER != NUC_EOS_SOLVER_ORIG )
   READ_EOS_HDF5( "pointsrho_mode", &g_nrho_mode, H5T_NATIVE_INT, H5S_ALL );
   READ_EOS_HDF5( "points_mode",    &g_nmode,     H5T_NATIVE_INT, H5S_ALL );
   READ_EOS_HDF5( "pointsye_mode",  &g_nye_mode,  H5T_NATIVE_INT, H5S_ALL );
#  endif


#  if ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )
   int n_def_mode = g_ntemp;
#  else
   int n_def_mode = g_neps;
#  endif


// allocate memory for tables
   if (  ! ( g_alltables      = (real*)malloc(g_nrho*n_def_mode*g_nye*NUC_TABLE_NVAR*sizeof(real)) )  )
      Aux_Error( ERROR_INFO, "cannot allocate memory for EOS table !!\n" );

   if (  ! ( g_logrho         = (real*)malloc(g_nrho                                *sizeof(real)) )  )
      Aux_Error( ERROR_INFO, "cannot allocate memory for EOS table !!\n" );

   if (  ! ( g_yes            = (real*)malloc(g_nye                                 *sizeof(real)) )  )
      Aux_Error( ERROR_INFO, "cannot allocate memory for EOS table !!\n" );

#  if ( NUC_EOS_SOLVER != NUC_EOS_SOLVER_ORIG )
   if (  ! ( g_alltables_mode = (real*)malloc(g_nrho_mode*g_nmode*g_nye_mode*3      *sizeof(real)) )  )
      Aux_Error( ERROR_INFO, "cannot allocate memory for EOS table !!\n" );

   if (  ! ( g_logrho_mode    = (real*)malloc(g_nrho_mode                           *sizeof(real)) )  )
      Aux_Error( ERROR_INFO, "cannot allocate memory for EOS table !!\n" );

   if (  ! ( g_entr_mode      = (real*)malloc(g_nmode                               *sizeof(real)) )  )
      Aux_Error( ERROR_INFO, "cannot allocate memory for EOS table !!\n" );

   if (  ! ( g_logprss_mode   = (real*)malloc(g_nmode                               *sizeof(real)) )  )
      Aux_Error( ERROR_INFO, "cannot allocate memory for EOS table !!\n" );

   if (  ! ( g_yes_mode       = (real*)malloc(g_nye_mode                            *sizeof(real)) )  )
      Aux_Error( ERROR_INFO, "cannot allocate memory for EOS table !!\n" );
#  endif

#  if ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )

   if (  ! ( g_logtemp        = (real*)malloc(n_def_mode                            *sizeof(real)) )  )
      Aux_Error( ERROR_INFO, "cannot allocate memory for EOS table !!\n" );

#  if ( NUC_EOS_SOLVER != NUC_EOS_SOLVER_ORIG )
   if (  ! ( g_logeps_mode    = (real*)malloc(g_nmode                               *sizeof(real)) )  )
      Aux_Error( ERROR_INFO, "cannot allocate memory for EOS table !!\n" );
#  endif

#  else

   if (  ! ( g_logeps         = (real*)malloc(n_def_mode                            *sizeof(real)) )  )
      Aux_Error( ERROR_INFO, "cannot allocate memory for EOS table !!\n" );

   if (  ! ( g_logtemp_mode   = (real*)malloc(g_nmode                               *sizeof(real)) )  )
      Aux_Error( ERROR_INFO, "cannot allocate memory for EOS table !!\n" );

#  endif // if ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP ) ... else ...

// prepare HDF5 to read hyperslabs into g_alltables[]

   hsize_t table_dims[2]      = { NUC_TABLE_NVAR, g_nrho*n_def_mode*g_nye };
   hsize_t var3[2]            = { 1, g_nrho*n_def_mode*g_nye };
   hid_t   mem3               = H5Screate_simple( 2, table_dims, NULL );

   hsize_t table_dims_mode[2] = { 3, g_nrho_mode*g_nmode*g_nye_mode };
   hsize_t var3_mode[2]       = { 1, g_nrho_mode*g_nmode*g_nye_mode };
   hid_t   mem3_mode          = H5Screate_simple( 2, table_dims_mode, NULL );


// read g_alltables[]
   READ_EOSTABLE_HDF5( "logpress",  NUC_VAR_IDX_PRES  );
#  if ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )
   READ_EOSTABLE_HDF5( "logenergy", NUC_VAR_IDX_EORT  );
#  else
   READ_EOSTABLE_HDF5( "logtemp",   NUC_VAR_IDX_EORT  );
#  endif
   READ_EOSTABLE_HDF5( "entropy",   NUC_VAR_IDX_ENTR  );
   READ_EOSTABLE_HDF5( "munu",      NUC_VAR_IDX_MUNU  );
   READ_EOSTABLE_HDF5( "cs2",       NUC_VAR_IDX_CSQR  );

// chemical potentials
   READ_EOSTABLE_HDF5( "muhat",     NUC_VAR_IDX_MUHAT );
   READ_EOSTABLE_HDF5( "mu_e",      NUC_VAR_IDX_MUE   );
   READ_EOSTABLE_HDF5( "mu_p",      NUC_VAR_IDX_MUP   );
   READ_EOSTABLE_HDF5( "mu_n",      NUC_VAR_IDX_MUN   );

// compositions
   READ_EOSTABLE_HDF5( "Xa",        NUC_VAR_IDX_XA    );
   READ_EOSTABLE_HDF5( "Xh",        NUC_VAR_IDX_XH    );
   READ_EOSTABLE_HDF5( "Xn",        NUC_VAR_IDX_XN    );
   READ_EOSTABLE_HDF5( "Xp",        NUC_VAR_IDX_XP    );

// average nucleus
   READ_EOSTABLE_HDF5( "Abar",      NUC_VAR_IDX_ABAR  );
   READ_EOSTABLE_HDF5( "Zbar",      NUC_VAR_IDX_ZBAR  );

// Gamma
   READ_EOSTABLE_HDF5( "gamma",     NUC_VAR_IDX_GAMMA );

// energy for temp, entr modes
#  if ( NUC_EOS_SOLVER != NUC_EOS_SOLVER_ORIG )
#  if ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )
   READ_EOSTABLE_MODE_HDF5( "logtemp_ener",   NUC_VAR_IDX_EORT );
   READ_EOSTABLE_MODE_HDF5( "logtemp_entr",   NUC_VAR_IDX_ENTR );
   READ_EOSTABLE_MODE_HDF5( "logtemp_prss",   NUC_VAR_IDX_PRES );
#  else
   READ_EOSTABLE_MODE_HDF5( "logenergy_temp", NUC_VAR_IDX_EORT );
   READ_EOSTABLE_MODE_HDF5( "logenergy_entr", NUC_VAR_IDX_ENTR );
   READ_EOSTABLE_MODE_HDF5( "logenergy_prss", NUC_VAR_IDX_PRES );
#  endif
#  endif

// read additional tables and variables
   READ_EOS_HDF5( "logrho",         g_logrho,        H5T_GAMER_REAL,    H5S_ALL );
   READ_EOS_HDF5( "ye",             g_yes,           H5T_GAMER_REAL,    H5S_ALL );
   READ_EOS_HDF5( "energy_shift",  &g_energy_shift,  H5T_NATIVE_DOUBLE, H5S_ALL );
#  if ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )
   READ_EOS_HDF5( "logtemp",        g_logtemp,       H5T_GAMER_REAL,    H5S_ALL );
#  if ( NUC_EOS_SOLVER != NUC_EOS_SOLVER_ORIG )
   READ_EOS_HDF5( "logenergy_mode", g_logeps_mode,   H5T_GAMER_REAL,    H5S_ALL );
#  endif
#  else
   READ_EOS_HDF5( "logenergy",      g_logeps,        H5T_GAMER_REAL,    H5S_ALL );
   READ_EOS_HDF5( "logtemp_mode",   g_logtemp_mode,  H5T_GAMER_REAL,    H5S_ALL );
#  endif // if ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )
#  if ( NUC_EOS_SOLVER != NUC_EOS_SOLVER_ORIG )
   READ_EOS_HDF5( "logrho_mode",    g_logrho_mode,   H5T_GAMER_REAL,    H5S_ALL );
   READ_EOS_HDF5( "entropy_mode",   g_entr_mode,     H5T_GAMER_REAL,    H5S_ALL );
   READ_EOS_HDF5( "logpress_mode",  g_logprss_mode,  H5T_GAMER_REAL,    H5S_ALL );
   READ_EOS_HDF5( "ye_mode",        g_yes_mode,      H5T_GAMER_REAL,    H5S_ALL );
#  endif


   HDF5_ERROR(  H5Sclose( mem3      )  );
   HDF5_ERROR(  H5Sclose( mem3_mode )  );
   HDF5_ERROR(  H5Fclose( file      )  );


// set the EoS table pointers
   h_EoS_Table[NUC_TAB_ALL      ] = g_alltables;
   h_EoS_Table[NUC_TAB_RHO      ] = g_logrho;
   h_EoS_Table[NUC_TAB_YE       ] = g_yes;
#  if ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )
   h_EoS_Table[NUC_TAB_TORE     ] = g_logtemp;
#  if ( NUC_EOS_SOLVER != NUC_EOS_SOLVER_ORIG )
   h_EoS_Table[NUC_TAB_EORT_MODE] = g_logeps_mode;
#  endif
#  else
   h_EoS_Table[NUC_TAB_TORE     ] = g_logeps;
   h_EoS_Table[NUC_TAB_EORT_MODE] = g_logtemp_mode;
#  endif // if ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )
#  if ( NUC_EOS_SOLVER != NUC_EOS_SOLVER_ORIG )
   h_EoS_Table[NUC_TAB_ALL_MODE ] = g_alltables_mode;
   h_EoS_Table[NUC_TAB_RHO_MODE ] = g_logrho_mode;
   h_EoS_Table[NUC_TAB_ENTR_MODE] = g_entr_mode;
   h_EoS_Table[NUC_TAB_PRES_MODE] = g_logprss_mode;
   h_EoS_Table[NUC_TAB_YE_MODE  ] = g_yes_mode;
#  endif


   if ( MPI_Rank == 0 )    Aux_Message( stdout, "%s ... done\n", __FUNCTION__ );

} // FUNCTION : nuc_eos_C_ReadTable



#endif // #if ( MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )
