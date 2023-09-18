#include "NuclearEoS.h"

#if ( MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )



#ifdef __CUDACC__

#include "cubinterp_some.cu"
#include "linterp_some.cu"
#include "findtoreps.cu"
#include "findtoreps_direct.cu"
#include "findtemp_NR_bisection.cu"

#else

void nuc_eos_C_cubinterp_some( const real x, const real y, const real z,
                               const int *TargetIdx, real *output_vars, const real *alltables,
                               const int nx, const int ny, const int nz, const int nvars,
                               const real *xt, const real *yt, const real *zt );

void nuc_eos_C_linterp_some( const real x, const real y, const real z,
                             const int *TargetIdx, real *output_vars, const real *alltables,
                             const int nx, const int ny, const int nz, const int nvars,
                             const real *xt, const real *yt, const real *zt );

void findtoreps( const real x, const real y, const real z,
                 real *found_lt, const real *table_Aux,
                 const int nx, const int ny, const int nz, const int ntemp,
                 const real *xt, const real *yt, const real *zt, const real *logtoreps,
                 const int IntScheme_Aux, int *keyerr );

void findtoreps_direct( const real x, const real y, const real z,
                        real *found_ltoreps, const real *table_main, const real *table_aux,
                        const int nx, const int ny, const int nz, const int nvar,
                        const real *xt, const real *yt, const real *zt, const real *vart,
                        const int keymode, int *keyerr );

void findtemp_NR_bisection( const real lr, const real lt_IG, const real ye, const real varin, real *ltout,
                            const int nrho, const int ntemp, const int nye, const real *alltables,
                            const real *logrho, const real *logtemp, const real *yes,
                            const int keymode, int *keyerrt, const real prec );

#endif // #ifdef __CUDACC__ ... else ...




//-----------------------------------------------------------------------------------------------
// Function    :  nuc_eos_C_short
// Description :  Function to find thermodynamic variables by searching
//                a pre-calculated nuclear equation of state table
//
// Note        :  1. It will strictly return values in cgs or MeV
//                2. Four modes are supported
//                   --> Energy      mode (0)
//                       Temperature mode (1)
//                       Entropy     mode (2)
//                       Pressure    mode (3)
//                3. Out[] must have the size of at least NTarget+1:
//                   --> Out[NTarget] stores the internal energy or temperature either
//                       from the input value or the value found in the auxiliary nuclear EoS table
//
// Parameter   :  Out            : Output array
//                In             : Input array
//                                 --> In[0] = mass density    ( rho) in g/cm^3
//                                     In[1] = internal energy ( eps) in cm^2/s^2   (keymode = 0/NUC_MODE_ENGY)
//                                           = temperature     (temp) in MeV        (keymode = 1/NUC_MODE_TEMP)
//                                           = entropy         (entr) in kB/baryon  (keymode = 2/NUC_MODE_ENTR)
//                                           = pressure        (pres) in dyne/cm^2  (keymode = 3/NUC_MODE_PRES)
//                                     In[2] = Ye              (  Ye) dimensionless
//                NTarget        : Number of thermodynamic variables retrieved from the nuclear EoS table
//                TargetIdx      : Indices of thermodynamic variables to be returned
//                energy_shift   : Energy shift
//                Temp_InitGuess : Initial guess of temperature (for temperature-based table)
//                nrho           : Size of density         array in the Nuclear EoS table
//                ntoreps        : Size of internal energy array in the Nuclear EoS table (     energy-based)
//                                         temperature                                    (temperature-based)
//                nye            : Size of Ye              array in the Nuclear EoS table
//                nrho_Aux       : Size of density                     array in the auxiliary table
//                nmode_Aux      : Size of internal energy/temperature array in the auxiliary table
//                                         entropy
//                                         pressure
//                nye_Aux        : Size of Ye                          array in the auxiliary table
//                alltables      : Nuclear EoS table
//                alltables_Aux  : Auxiliary arrays for finding internal energy/temperature in different modes
//                logrho         : density                     index array in the Nuclear EoS table (log    scale)
//                logtoreps      : internal energy/temperature index array in the Nuclear EoS table (log    scale)
//                yes            : Ye                          index array in the Nuclear EoS table (linear scale)
//                logrho_Aux     : density                     index array in the auxiliary table   (log    scale)
//                mode_Aux       : internal energy/temperature index array in the auxiliary table   (log    scale)
//                                 entropy                                                          (linear scale)
//                                 pressure                                                         (log    scale)
//                yes_Aux        : Ye                          index array in the auxiliary table   (linear scale)
//                IntScheme_Aux  : Interpolation scheme for the auxiliary table
//                IntScheme_Main : Interpolation scheme for the Nuclear EoS table
//                keymode        : Which mode we will use
//                                 --> 0 : Energy mode
//                                     1 : Temperature mode
//                                     2 : Entropy mode
//                                     3 : Pressure mode
//                keyerr         : Output error
//                                 --> 100 : rho  too high
//                                     101 : rho  too low
//                                     102 : rho  NaN
//                                     110 : eps  too high
//                                     111 : eps  too low
//                                     112 : eps  NaN
//                                     120 : temp too high
//                                     121 : temp too low
//                                     122 : temp NaN
//                                     130 : entr too high
//                                     131 : entr too low
//                                     132 : entr NaN
//                                     140 : pres too high
//                                     141 : pres too low
//                                     142 : pres NaN
//                                     150 : Ye   too high
//                                     151 : Ye   too low
//                                     152 : Ye   NaN
//                                     660 : fail in finding temperature         in the direct method
//                                     665 : fail in finding internal energy or temperature
//                                     668 : fail in bracketing the target value in the bisection method
//                                     669 : fail in finding temperature         in the bisection method
//                rfeps          : Tolerance for Newton-Raphson and bisection methods
//
// Return      :  Out[]
//-----------------------------------------------------------------------------------------------
GPU_DEVICE
void nuc_eos_C_short( real *Out, const real *In,
                      const int NTarget, const int *TargetIdx,
                      const real energy_shift, real Temp_InitGuess,
                      const int nrho, const int ntoreps, const int nye,
                      const int nrho_Aux, const int nmode_Aux, const int nye_Aux,
                      const real *alltables, const real *alltables_Aux,
                      const real *logrho, const real *logtoreps, const real *yes,
                      const real *logrho_Aux, const real *mode_Aux, const real *yes_Aux,
                      const int IntScheme_Aux, const int IntScheme_Main,
                      const int keymode, int *keyerr, const real rfeps )
{

   const real  lr       = LOG10( In[0] );
   const real  xye      = In[2];
         real  ltoreps  = NULL_REAL;
         real  var_mode = NULL_REAL;
         int   var_idx;
              *keyerr   = 0;

#  if ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )
   real lt_IG = 1.0;

   if ( Temp_InitGuess != NULL_REAL )
   {
      lt_IG = LOG10( Temp_InitGuess );
      lt_IG = MAX(  MIN( lt_IG, logtoreps[ntoreps-1] ), logtoreps[0]  );
   }
#  endif


// check whether the input density and Ye are within the table
   if ( lr >  logrho[nrho-1] )  {  *keyerr = 100;  return;  }
   if ( lr <  logrho[     0] )  {  *keyerr = 101;  return;  }
   if ( lr != lr             )  {  *keyerr = 102;  return;  }

   if ( xye >  yes  [nye -1] )  {  *keyerr = 150;  return;  }
   if ( xye <  yes  [     0] )  {  *keyerr = 151;  return;  }
   if ( xye != xye           )  {  *keyerr = 152;  return;  }


   switch ( keymode )
   {
      case NUC_MODE_ENGY :
      {
         const real leps = LOG10( In[1] + energy_shift );

#        if ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )
         const int   npt_chk   = nmode_Aux;
         const real *table_chk = mode_Aux;
                     var_mode  = leps;
                     var_idx   = NUC_VAR_IDX_EORT;
#        else
         const int   npt_chk   = ntoreps;
         const real *table_chk = logtoreps;
                     ltoreps   = leps;
#        endif

#        if ( NUC_EOS_SOLVER != NUC_EOS_SOLVER_ORIG )
#        if ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )
         if ( leps >  table_chk[npt_chk-1]  )  {  *keyerr = 110;           }
         if ( leps <  table_chk[        0]  )  {  *keyerr = 111;           }
#        else
         if ( leps >  table_chk[npt_chk-1]  )  {  *keyerr = 110;  return;  }
         if ( leps <  table_chk[        0]  )  {  *keyerr = 111;  return;  }
#        endif // if ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )
#        endif // if ( NUC_EOS_SOLVER != NUC_EOS_SOLVER_ORIG )
         if ( leps != leps                  )  {  *keyerr = 112;  return;  }
      }
      break;


      case NUC_MODE_TEMP :
      {
         const real lt = LOG10( In[1] );

#        if ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )
         const int   npt_chk   = ntoreps;
         const real *table_chk = logtoreps;
                     ltoreps   = lt;
#        else
         const int   npt_chk   = nmode_Aux;
         const real *table_chk = mode_Aux;
                     var_mode  = lt;
                     var_idx   = NUC_VAR_IDX_EORT;
#        endif

         if ( lt   >  table_chk[npt_chk-1]  )  {  *keyerr = 120;  return;  }
         if ( lt   <  table_chk[        0]  )  {  *keyerr = 121;  return;  }
         if ( lt   != lt                    )  {  *keyerr = 122;  return;  }
      }
      break;


      case NUC_MODE_ENTR :
      {
         const real entr     = In[1];
                    var_mode = entr;
                    var_idx  = NUC_VAR_IDX_ENTR;

#        if ( NUC_EOS_SOLVER != NUC_EOS_SOLVER_ORIG )
         if ( entr >  mode_Aux[nmode_Aux-1] )  {  *keyerr = 130;           }
         if ( entr <  mode_Aux[          0] )  {  *keyerr = 131;           }
#        endif
         if ( entr != entr                  )  {  *keyerr = 132;  return;  }
      }
      break;


      case NUC_MODE_PRES :
      {
         const real lprs     = LOG10( In[1] );
                    var_mode = lprs;
                    var_idx  = NUC_VAR_IDX_PRES;

#        if ( NUC_EOS_SOLVER != NUC_EOS_SOLVER_ORIG )
         if ( lprs >  mode_Aux[nmode_Aux-1] )  {  *keyerr = 140;           }
         if ( lprs <  mode_Aux[          0] )  {  *keyerr = 141;           }
#        endif
         if ( lprs != lprs                  )  {  *keyerr = 142;  return;  }
      }
      break;
   } // switch ( keymode )


// find corresponding temperature or internal energy
   if ( ltoreps == NULL_REAL )
   {
//    (a) Table lookup or direct method
#     if ( NUC_EOS_SOLVER != NUC_EOS_SOLVER_ORIG )
      const real *table_Aux = alltables_Aux + var_idx*nrho_Aux*nmode_Aux*nye_Aux;

      if ( *keyerr == 0 )
      {
#        if   ( NUC_EOS_SOLVER == NUC_EOS_SOLVER_LUT    )
         findtoreps( lr, var_mode, xye, &ltoreps, table_Aux, nrho_Aux, nmode_Aux, nye_Aux, ntoreps,
                     logrho_Aux, mode_Aux, yes_Aux, logtoreps, IntScheme_Aux, keyerr );

#        elif ( NUC_EOS_SOLVER == NUC_EOS_SOLVER_DIRECT )
         findtoreps_direct( lr, var_mode, xye, &ltoreps, alltables, table_Aux, nrho, ntoreps, nye, nmode_Aux,
                            logrho, logtoreps, yes, mode_Aux, keymode, keyerr );
#        endif
      }
#     endif // if ( NUC_EOS_SOLVER != NUC_EOS_SOLVER_ORIG )


//    (b) Newton-Raphson and bisection methods (for temperature-based table only)
#     if ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )
#     if ( NUC_EOS_SOLVER != NUC_EOS_SOLVER_ORIG )
      if ( *keyerr != 0 )
#     endif
         findtemp_NR_bisection( lr, lt_IG, xye, var_mode, &ltoreps, nrho, ntoreps, nye, alltables,
                                logrho, logtoreps, yes, keymode, keyerr, rfeps );
#     endif // if ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )
   }

   if ( *keyerr != 0 ) return;


// find other thermodynamic variables
   if ( NTarget > 0 )
   {
      if ( IntScheme_Main == NUC_INT_LINEAR )
      {
         nuc_eos_C_linterp_some( lr, ltoreps, xye, TargetIdx, Out, alltables,
                                 nrho, ntoreps, nye, NTarget, logrho, logtoreps, yes );
      }

      else
      {
         nuc_eos_C_cubinterp_some( lr, ltoreps, xye, TargetIdx, Out, alltables,
                                   nrho, ntoreps, nye, NTarget, logrho, logtoreps, yes );
      }


//    convert scale and correct energy shift
      for (int i=0; i<NTarget; i++)
      {
         if ( TargetIdx[i] == NUC_VAR_IDX_PRES )   Out[i] = POW( (real)10.0, Out[i] );

#        if ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )
         if ( TargetIdx[i] == NUC_VAR_IDX_EORT )   Out[i] = POW( (real)10.0, Out[i] ) - energy_shift;
#        else
         if ( TargetIdx[i] == NUC_VAR_IDX_EORT )   Out[i] = POW( (real)10.0, Out[i] );
#        endif
      }
   }


// store the temperature or internal energy, from either the input value or the auxiliary table,
// in Out[NTarget], with scale conversion and energy shift correction
#  if ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )
   Out[NTarget] = POW( (real)10.0, ltoreps );
#  else
   Out[NTarget] = POW( (real)10.0, ltoreps ) - energy_shift;
#  endif

} // FUNCTION : nuc_eos_C_short



#endif // #if ( MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )
