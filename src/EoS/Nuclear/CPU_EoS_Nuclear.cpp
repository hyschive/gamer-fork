#include "NuclearEoS.h"
#ifdef __CUDACC__
#include "CUDA_CheckError.h"
#include "CUFLU_Shared_FluUtility.cu"
#endif

#if ( MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )



#ifdef __CUDACC__

#include "NuclearEoS.cu"
__device__ static real EoS_DensEint2Pres_Nuclear( const real Dens_Code, const real Eint_Code, const real Passive_Code[],
                                                  const double AuxArray_Flt[], const int AuxArray_Int[],
                                                  const real *const Table[EOS_NTABLE_MAX] );
__device__ static void EoS_General_Nuclear( const int Mode, real Out[], const real In_Flt[], const int In_Int[],
                                            const double AuxArray_Flt[], const int AuxArray_Int[],
                                            const real *const Table[EOS_NTABLE_MAX] );

#else

// global variables
int    g_nrho;
int    g_nye;
int    g_nrho_mode;
int    g_nmode;
int    g_nye_mode;
double g_energy_shift;

real  *g_alltables      = NULL;
real  *g_alltables_mode = NULL;
real  *g_logrho         = NULL;
real  *g_yes            = NULL;
real  *g_logrho_mode    = NULL;
real  *g_entr_mode      = NULL;
real  *g_logprss_mode   = NULL;
real  *g_yes_mode       = NULL;


#if ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )
int    g_ntemp;
real  *g_logtemp        = NULL;
real  *g_logeps_mode    = NULL;
#else
int    g_neps;
real  *g_logeps         = NULL;
real  *g_logtemp_mode   = NULL;
#endif


// prototypes
void nuc_eos_C_short( real *Out, const real *In,
                      const int NTarget, const int *TargetIdx,
                      const real energy_shift, real Temp_InitGuess,
                      const int nrho, const int ntoreps, const int nye,
                      const int nrho_Aux, const int nmode_Aux, const int nye_Aux,
                      const real *alltables, const real *alltables_Aux,
                      const real *logrho, const real *logtoreps, const real *yes,
                      const real *logrho_Aux, const real *mode_Aux, const real *yes_Aux,
                      const int IntScheme_Aux, const int IntScheme_Main,
                      const int keymode, int *keyerr, const real rfeps );
void nuc_eos_C_ReadTable( char *nuceos_table_name );
void CUAPI_PassNuclearEoSTable2GPU();

static real EoS_DensEint2Pres_Nuclear( const real Dens_Code, const real Eint_Code, const real Passive_Code[],
                                       const double AuxArray_Flt[], const int AuxArray_Int[],
                                       const real *const Table[EOS_NTABLE_MAX] );
static void EoS_General_Nuclear( const int Mode, real Out[], const real In_Flt[], const int In_Int[],
                                 const double AuxArray_Flt[], const int AuxArray_Int[],
                                 const real *const Table[EOS_NTABLE_MAX] );

#endif // #ifdef __CUDACC__ ... else ...




/********************************************************
1. Nuclear EoS (EOS_NUCLEAR)

2. This file is shared by both CPU and GPU

   GPU_EoS_Nuclear.cu -> CPU_EoS_Nuclear.cpp

3. Three steps are required to implement an EoS

   I.   Set EoS auxiliary arrays
   II.  Implement EoS conversion functions
   III. Set EoS initialization functions

4. All EoS conversion functions must be thread-safe and
   not use any global variable

5. When an EoS conversion function fails, it is recommended
   to return NAN in order to trigger auto-correction such as
   "OPT__1ST_FLUX_CORR" and "AUTO_REDUCE_DT"
********************************************************/



// =============================================
// I. Set EoS auxiliary arrays
// =============================================

//-------------------------------------------------------------------------------------------------------
// Function    :  EoS_SetAuxArray_Nuclear
// Description :  Set the auxiliary arrays AuxArray_Flt/Int[]
//
// Note        :  1. Invoked by EoS_Init_Nuclear()
//                2. AuxArray_Flt/Int[] have the size of EOS_NAUX_MAX defined in Macro.h (default = 20)
//                3. Add "#ifndef __CUDACC__" since this routine is only useful on CPU
//
// Parameter   :  AuxArray_Flt/Int : Floating-point/Integer arrays to be filled up
//
// Return      :  AuxArray_Flt/Int[]
//-------------------------------------------------------------------------------------------------------
#ifndef __CUDACC__
void EoS_SetAuxArray_Nuclear( double AuxArray_Flt[], int AuxArray_Int[] )
{

   AuxArray_Flt[NUC_AUX_ESHIFT    ] = g_energy_shift;
   AuxArray_Flt[NUC_AUX_DENS2CGS  ] = UNIT_D;
   AuxArray_Flt[NUC_AUX_PRES2CGS  ] = UNIT_P;
   AuxArray_Flt[NUC_AUX_VSQR2CGS  ] = SQR( UNIT_V );
   AuxArray_Flt[NUC_AUX_PRES2CODE ] = 1.0 / UNIT_P;
   AuxArray_Flt[NUC_AUX_VSQR2CODE ] = 1.0 / SQR(UNIT_V);
   AuxArray_Flt[NUC_AUX_KELVIN2MEV] = Const_kB_eV*1.0e-6;
   AuxArray_Flt[NUC_AUX_MEV2KELVIN] = 1.0 / AuxArray_Flt[NUC_AUX_KELVIN2MEV];

   AuxArray_Int[NUC_AUX_NRHO      ] = g_nrho;
#  if ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )
   AuxArray_Int[NUC_AUX_NTORE     ] = g_ntemp;
#  else
   AuxArray_Int[NUC_AUX_NTORE     ] = g_neps;
#  endif
   AuxArray_Int[NUC_AUX_NYE       ] = g_nye;
   AuxArray_Int[NUC_AUX_NRHO_MODE ] = g_nrho_mode;
   AuxArray_Int[NUC_AUX_NMODE     ] = g_nmode;
   AuxArray_Int[NUC_AUX_NYE_MODE  ] = g_nye_mode;
   AuxArray_Int[NUC_AUX_INT_AUX   ] = NUC_INT_SCHEME_AUX;
   AuxArray_Int[NUC_AUX_INT_MAIN  ] = NUC_INT_SCHEME_MAIN;

} // FUNCTION : EoS_SetAuxArray_Nuclear
#endif // #ifndef __CUDACC__



// =============================================
// II. Implement EoS conversion functions
//     (1) EoS_DensEint2Pres_*
//     (2) EoS_DensPres2Eint_*
//     (3) EoS_DensPres2CSqr_*
//     (4) EoS_DensEint2Temp_*
//     (5) EoS_DensTemp2Pres_*
//     (6) EoS_DensEint2Entr_*
//     (7) EoS_General_*
// =============================================

#ifdef GAMER_DEBUG
//-------------------------------------------------------------------------------------------------------
// Function    :  Nuc_Overflow
// Description :  Check whether the input floating-point value is finite
//
// Note        :  1. Definition of "finite" --> not NaN, Inf, -Inf
//                2. Overflow may occur during unit conversion for improper code units
//
// Parameter   :  x : Floating-point value to be checked
//
// Return      :  true  : infinite
//                false : finite
//-------------------------------------------------------------------------------------------------------
GPU_DEVICE static
bool Nuc_Overflow( const real x )
{

   if ( x != x  ||  x < -HUGE_NUMBER  ||  x > HUGE_NUMBER )    return true;
   else                                                        return false;

} // FUNCTION : Nuc_Overflow
#endif // #ifdef GAMER_DEBUG



//-------------------------------------------------------------------------------------------------------
// Function    :  EoS_DensEint2Pres_Nuclear
// Description :  Convert gas mass density and internal energy density to gas pressure
//
// Note        :  1. Internal energy density here is per unit volume instead of per unit mass
//                2. See EoS_SetAuxArray_Nuclear() for the values stored in AuxArray_Flt/Int[]
//
// Parameter   :  Dens_Code    : Gas mass density            (in code unit)
//                Eint_Code    : Gas internal energy density (in code unit)
//                Passive_Code : Passive scalars             (in code unit)
//                AuxArray_*   : Auxiliary arrays (see the Note above)
//                Table        : EoS tables
//
// Return      :  Gas pressure (in code unit)
//-------------------------------------------------------------------------------------------------------
GPU_DEVICE_NOINLINE
static real EoS_DensEint2Pres_Nuclear( const real Dens_Code, const real Eint_Code, const real Passive_Code[],
                                       const double AuxArray_Flt[], const int AuxArray_Int[],
                                       const real *const Table[EOS_NTABLE_MAX] )
{

// check
#  ifdef GAMER_DEBUG
#  if ( NCOMP_PASSIVE > 0 )
   if ( Passive_Code == NULL )   printf( "ERROR : Passive_Code == NULL in %s !!\n", __FUNCTION__ );
#  endif
   if ( AuxArray_Flt == NULL )   printf( "ERROR : AuxArray_Flt == NULL in %s !!\n", __FUNCTION__ );
   if ( AuxArray_Int == NULL )   printf( "ERROR : AuxArray_Int == NULL in %s !!\n", __FUNCTION__ );

   Hydro_CheckUnphysical( UNPHY_MODE_SING, &Dens_Code, "input density",                 ERROR_INFO, UNPHY_VERBOSE );
// still require Eint>0 for the nuclear EoS
   Hydro_CheckUnphysical( UNPHY_MODE_SING, &Eint_Code, "input internal energy density", ERROR_INFO, UNPHY_VERBOSE );
#  endif // GAMER_DEBUG


   const real EnergyShift = AuxArray_Flt[NUC_AUX_ESHIFT    ];
   const real Dens2CGS    = AuxArray_Flt[NUC_AUX_DENS2CGS  ];
   const real sEint2CGS   = AuxArray_Flt[NUC_AUX_VSQR2CGS  ];
   const real Pres2Code   = AuxArray_Flt[NUC_AUX_PRES2CODE ];
#  if ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )
   const real Kelvin2MeV  = AuxArray_Flt[NUC_AUX_KELVIN2MEV];
#  endif

   const int  NRho        = AuxArray_Int[NUC_AUX_NRHO      ];
   const int  NTorE       = AuxArray_Int[NUC_AUX_NTORE     ];
   const int  NYe         = AuxArray_Int[NUC_AUX_NYE       ];
   const int  NRho_Mode   = AuxArray_Int[NUC_AUX_NRHO_MODE ];
   const int  NMode       = AuxArray_Int[NUC_AUX_NMODE     ];
   const int  NYe_Mode    = AuxArray_Int[NUC_AUX_NYE_MODE  ];
   const int  Int_Aux     = AuxArray_Int[NUC_AUX_INT_AUX   ];
   const int  Int_Main    = AuxArray_Int[NUC_AUX_INT_MAIN  ];


   int  Mode      = NUC_MODE_ENGY;
   real Dens_CGS  = Dens_Code * Dens2CGS;
   real sEint_CGS = ( Eint_Code / Dens_Code ) * sEint2CGS - EnergyShift;
   real Ye        = Passive_Code[ YE - NCOMP_FLUID ] / Dens_Code;
   int  Err       = NULL_INT;

// set up the initial guess of temperature for temperature-based table
#  ifdef TEMP_IG
   real Temp_IG   = Passive_Code[ TEMP_IG - NCOMP_FLUID ];
        Temp_IG   = Temp_IG * Kelvin2MeV;
#  else
   real Temp_IG   = NULL_REAL;
#  endif

// check floating-point overflow and Ye
#  ifdef GAMER_DEBUG
   if ( Nuc_Overflow(Dens_CGS) )
      printf( "ERROR : EoS overflow (Dens_CGS %13.7e, Dens_Code %13.7e, Dens2CGS %13.7e) in %s() !!\n",
              Dens_CGS, Dens_Code, Dens2CGS, __FUNCTION__ );

   if ( Nuc_Overflow(sEint_CGS) )
      printf( "ERROR : EoS overflow (sEint_CGS %13.7e, Eint_Code %13.7e, Dens_Code %13.7e, sEint2CGS %13.7e) in %s() !!\n",
              sEint_CGS, Eint_Code, Dens_Code, sEint2CGS, __FUNCTION__ );

   if ( Ye < (real)Table[NUC_TAB_YE][0]  ||  Ye > (real)Table[NUC_TAB_YE][NYe-1] )
      printf( "ERROR : invalid Ye = %13.7e (min = %13.7e, max = %13.7e) in %s() !!\n",
              Ye, Table[NUC_TAB_YE][0], Table[NUC_TAB_YE][NYe-1], __FUNCTION__ );
#  endif // GAMER_DEBUG


   const int  NTarget = 1;
         int  TargetIdx[NTarget] = { NUC_VAR_IDX_PRES };
         real In[3], Out[NTarget+1];

   In[0] = Dens_CGS;
   In[1] = sEint_CGS;
   In[2] = Ye;

// invoke the nuclear EoS driver
   nuc_eos_C_short( Out, In, NTarget, TargetIdx,
                    EnergyShift, Temp_IG, NRho, NTorE, NYe, NRho_Mode, NMode, NYe_Mode,
                    Table[NUC_TAB_ALL], Table[NUC_TAB_ALL_MODE], Table[NUC_TAB_RHO], Table[NUC_TAB_TORE], Table[NUC_TAB_YE],
                    Table[NUC_TAB_RHO_MODE], Table[NUC_TAB_EORT_MODE], Table[NUC_TAB_YE_MODE],
                    Int_Aux, Int_Main, Mode, &Err, Tolerance );

// trigger a *hard failure* if the EoS driver fails
   if ( Err )   for (int i=0; i<NTarget+1; i++)   Out[i] = NAN;

   const real Pres_CGS  = Out[0];
   const real Pres_Code = Pres_CGS * Pres2Code;


// final check
#  ifdef GAMER_DEBUG
   if (  Hydro_CheckUnphysical( UNPHY_MODE_SING, &Pres_Code, "output pressure", ERROR_INFO, UNPHY_VERBOSE )  )
   {
      printf( "   Dens=%13.7e code units, Eint=%13.7e code units, Ye=%13.7e, Mode %d\n", Dens_Code, Eint_Code, Ye, Mode );
      printf( "   EoS error code: %d\n", Err );
   }
#  endif // GAMER_DEBUG


   return Pres_Code;

} // FUNCTION : EoS_DensEint2Pres_Nuclear



//-------------------------------------------------------------------------------------------------------
// Function    :  EoS_DensPres2Eint_Nuclear
// Description :  Convert gas mass density and pressure to gas internal energy density
//
// Note        :  1. See EoS_DensEint2Pres_Nuclear()
//
// Parameter   :  Dens_Code    : Gas mass density (in code unit)
//                Pres_Code    : Gas pressure     (in code unit)
//                Passive_Code : Passive scalars  (in code unit)
//                AuxArray_*   : Auxiliary arrays (see the Note above)
//                Table        : EoS tables
//
// Return      :  Gas internal energy density (in code unit)
//-------------------------------------------------------------------------------------------------------
GPU_DEVICE_NOINLINE
static real EoS_DensPres2Eint_Nuclear( const real Dens_Code, const real Pres_Code, const real Passive_Code[],
                                       const double AuxArray_Flt[], const int AuxArray_Int[],
                                       const real *const Table[EOS_NTABLE_MAX] )
{

// check
#  ifdef GAMER_DEBUG
#  if ( NCOMP_PASSIVE > 0 )
   if ( Passive_Code == NULL )   printf( "ERROR : Passive_Code == NULL in %s !!\n", __FUNCTION__ );
#  endif
   if ( AuxArray_Flt == NULL )   printf( "ERROR : AuxArray_Flt == NULL in %s !!\n", __FUNCTION__ );
   if ( AuxArray_Int == NULL )   printf( "ERROR : AuxArray_Int == NULL in %s !!\n", __FUNCTION__ );

   Hydro_CheckUnphysical( UNPHY_MODE_SING, &Dens_Code, "input density",  ERROR_INFO, UNPHY_VERBOSE );
   Hydro_CheckUnphysical( UNPHY_MODE_SING, &Pres_Code, "input pressure", ERROR_INFO, UNPHY_VERBOSE );
#  endif // GAMER_DEBUG


   const real EnergyShift = AuxArray_Flt[NUC_AUX_ESHIFT    ];
   const real Dens2CGS    = AuxArray_Flt[NUC_AUX_DENS2CGS  ];
   const real Pres2CGS    = AuxArray_Flt[NUC_AUX_PRES2CGS  ];
   const real sEint2Code  = AuxArray_Flt[NUC_AUX_VSQR2CODE ];
#  if ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )
   const real Kelvin2MeV  = AuxArray_Flt[NUC_AUX_KELVIN2MEV];
#  endif

   const int  NRho        = AuxArray_Int[NUC_AUX_NRHO      ];
   const int  NTorE       = AuxArray_Int[NUC_AUX_NTORE     ];
   const int  NYe         = AuxArray_Int[NUC_AUX_NYE       ];
   const int  NRho_Mode   = AuxArray_Int[NUC_AUX_NRHO_MODE ];
   const int  NMode       = AuxArray_Int[NUC_AUX_NMODE     ];
   const int  NYe_Mode    = AuxArray_Int[NUC_AUX_NYE_MODE  ];
   const int  Int_Aux     = AuxArray_Int[NUC_AUX_INT_AUX   ];
   const int  Int_Main    = AuxArray_Int[NUC_AUX_INT_MAIN  ];

   int  Mode     = NUC_MODE_PRES;
   real Dens_CGS = Dens_Code * Dens2CGS;
   real Pres_CGS = Pres_Code * Pres2CGS;
   real Ye       = Passive_Code[ YE - NCOMP_FLUID ] / Dens_Code;
   int  Err      = NULL_INT;

// set up the initial guess of temperature for temperature-based table
#  ifdef TEMP_IG
   real Temp_IG   = Passive_Code[ TEMP_IG - NCOMP_FLUID ];
        Temp_IG   = Temp_IG * Kelvin2MeV;
#  else
   real Temp_IG  = NULL_REAL;
#  endif

// check floating-point overflow and Ye
#  ifdef GAMER_DEBUG
   if ( Nuc_Overflow(Dens_CGS) )
      printf( "ERROR : EoS overflow (Dens_CGS %13.7e, Dens_Code %13.7e, Dens2CGS %13.7e) in %s() !!\n",
              Dens_CGS, Dens_Code, Dens2CGS, __FUNCTION__ );

   if ( Nuc_Overflow(Pres_CGS) )
      printf( "ERROR : EoS overflow (Pres_CGS %13.7e, Pres_Code %13.7e, Pres2CGS %13.7e) in %s() !!\n",
              Pres_CGS, Pres_Code, Pres2CGS, __FUNCTION__ );

   if ( Ye < (real)Table[NUC_TAB_YE][0]  ||  Ye > (real)Table[NUC_TAB_YE][NYe-1] )
      printf( "ERROR : invalid Ye = %13.7e (min = %13.7e, max = %13.7e) in %s() !!\n",
              Ye, Table[NUC_TAB_YE][0], Table[NUC_TAB_YE][NYe-1], __FUNCTION__ );
#  endif // GAMER_DEBUG


#  if   ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )
   const int  NTarget = 1;
         int  TargetIdx[NTarget] = { NUC_VAR_IDX_EORT };
#  else
   const int  NTarget = 0;
         int *TargetIdx = NULL;
#  endif
         real In[3], Out[NTarget+1];

   In[0] = Dens_CGS;
   In[1] = Pres_CGS;
   In[2] = Ye;

// invoke the nuclear EoS driver
   nuc_eos_C_short( Out, In, NTarget, TargetIdx,
                    EnergyShift, Temp_IG, NRho, NTorE, NYe, NRho_Mode, NMode, NYe_Mode,
                    Table[NUC_TAB_ALL], Table[NUC_TAB_ALL_MODE], Table[NUC_TAB_RHO], Table[NUC_TAB_TORE], Table[NUC_TAB_YE],
                    Table[NUC_TAB_RHO_MODE], Table[NUC_TAB_PRES_MODE], Table[NUC_TAB_YE_MODE],
                    Int_Aux, Int_Main, Mode, &Err, Tolerance );

// trigger a *hard failure* if the EoS driver fails
   if ( Err )   for (int i=0; i<NTarget+1; i++)   Out[i] = NAN;

   const real sEint_CGS = Out[0];
   const real Eint_Code = (  ( sEint_CGS + EnergyShift ) * sEint2Code  ) * Dens_Code;


// final check
#  ifdef GAMER_DEBUG
// still require Eint>0 for the nuclear EoS
   if (  Hydro_CheckUnphysical( UNPHY_MODE_SING, &Eint_Code, "output internal energy density", ERROR_INFO, UNPHY_VERBOSE )  )
   {
      printf( "   Dens=%13.7e code units, Pres=%13.7e code units, Ye=%13.7e, Mode %d\n", Dens_Code, Pres_Code, Ye, Mode );
      printf( "   EoS error code: %d\n", Err );
   }
#  endif // GAMER_DEBUG


   return Eint_Code;

} // FUNCTION : EoS_DensPres2Eint_Nuclear



//-------------------------------------------------------------------------------------------------------
// Function    :  EoS_DensPres2CSqr_Nuclear
// Description :  Convert gas mass density and pressure to sound speed squared
//
// Note        :  1. See EoS_DensEint2Pres_Nuclear()
//
// Parameter   :  Dens_Code    : Gas mass density (in code unit)
//                Pres_Code    : Gas pressure     (in code unit)
//                Passive_Code : Passive scalars  (in code unit)
//                AuxArray_*   : Auxiliary arrays (see the Note above)
//                Table        : EoS tables
//
// Return      :  Sound speed squared (in code unit)
//-------------------------------------------------------------------------------------------------------
GPU_DEVICE_NOINLINE
static real EoS_DensPres2CSqr_Nuclear( const real Dens_Code, const real Pres_Code, const real Passive_Code[],
                                       const double AuxArray_Flt[], const int AuxArray_Int[],
                                       const real *const Table[EOS_NTABLE_MAX] )
{

// check
#  ifdef GAMER_DEBUG
#  if ( NCOMP_PASSIVE > 0 )
   if ( Passive_Code == NULL )   printf( "ERROR : Passive_Code == NULL in %s !!\n", __FUNCTION__ );
#  endif
   if ( AuxArray_Flt == NULL )   printf( "ERROR : AuxArray_Flt == NULL in %s !!\n", __FUNCTION__ );
   if ( AuxArray_Int == NULL )   printf( "ERROR : AuxArray_Int == NULL in %s !!\n", __FUNCTION__ );

   Hydro_CheckUnphysical( UNPHY_MODE_SING, &Dens_Code, "input density",  ERROR_INFO, UNPHY_VERBOSE );
   Hydro_CheckUnphysical( UNPHY_MODE_SING, &Pres_Code, "input pressure", ERROR_INFO, UNPHY_VERBOSE );
#  endif // GAMER_DEBUG


   const real EnergyShift = AuxArray_Flt[NUC_AUX_ESHIFT    ];
   const real Dens2CGS    = AuxArray_Flt[NUC_AUX_DENS2CGS  ];
   const real Pres2CGS    = AuxArray_Flt[NUC_AUX_PRES2CGS  ];
   const real CsSqr2Code  = AuxArray_Flt[NUC_AUX_VSQR2CODE ];
#  if ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )
   const real Kelvin2MeV  = AuxArray_Flt[NUC_AUX_KELVIN2MEV];
#  endif

   const int  NRho        = AuxArray_Int[NUC_AUX_NRHO      ];
   const int  NTorE       = AuxArray_Int[NUC_AUX_NTORE     ];
   const int  NYe         = AuxArray_Int[NUC_AUX_NYE       ];
   const int  NRho_Mode   = AuxArray_Int[NUC_AUX_NRHO_MODE ];
   const int  NMode       = AuxArray_Int[NUC_AUX_NMODE     ];
   const int  NYe_Mode    = AuxArray_Int[NUC_AUX_NYE_MODE  ];
   const int  Int_Aux     = AuxArray_Int[NUC_AUX_INT_AUX   ];
   const int  Int_Main    = AuxArray_Int[NUC_AUX_INT_MAIN  ];


   int  Mode     = NUC_MODE_PRES;
   real Dens_CGS = Dens_Code * Dens2CGS;
   real Pres_CGS = Pres_Code * Pres2CGS;
   real Ye       = Passive_Code[ YE - NCOMP_FLUID ] / Dens_Code;
   int  Err      = NULL_INT;

// set up the initial guess of temperature for temperature-based table
#  ifdef TEMP_IG
   real Temp_IG   = Passive_Code[ TEMP_IG - NCOMP_FLUID ];
        Temp_IG   = Temp_IG * Kelvin2MeV;
#  else
   real Temp_IG  = NULL_REAL;
#  endif

// check floating-point overflow and Ye
#  ifdef GAMER_DEBUG
   if ( Nuc_Overflow(Dens_CGS) )
      printf( "ERROR : EoS overflow (Dens_CGS %13.7e, Dens_Code %13.7e, Dens2CGS %13.7e) in %s() !!\n",
              Dens_CGS, Dens_Code, Dens2CGS, __FUNCTION__ );

   if ( Nuc_Overflow(Pres_CGS) )
      printf( "ERROR : EoS overflow (Pres_CGS %13.7e, Pres_Code %13.7e, Pres2CGS %13.7e) in %s() !!\n",
              Pres_CGS, Pres_Code, Pres2CGS, __FUNCTION__ );

   if ( Ye < (real)Table[NUC_TAB_YE][0]  ||  Ye > (real)Table[NUC_TAB_YE][NYe-1] )
      printf( "ERROR : invalid Ye = %13.7e (min = %13.7e, max = %13.7e) in %s() !!\n",
              Ye, Table[NUC_TAB_YE][0], Table[NUC_TAB_YE][NYe-1], __FUNCTION__ );
#  endif // GAMER_DEBUG


   const int  NTarget = 1;
         int  TargetIdx[NTarget] = { NUC_VAR_IDX_CSQR };
         real In[3], Out[NTarget+1];

   In[0] = Dens_CGS;
   In[1] = Pres_CGS;
   In[2] = Ye;

// invoke the nuclear EoS driver
   nuc_eos_C_short( Out, In, NTarget, TargetIdx,
                    EnergyShift, Temp_IG, NRho, NTorE, NYe, NRho_Mode, NMode, NYe_Mode,
                    Table[NUC_TAB_ALL], Table[NUC_TAB_ALL_MODE], Table[NUC_TAB_RHO], Table[NUC_TAB_TORE], Table[NUC_TAB_YE],
                    Table[NUC_TAB_RHO_MODE], Table[NUC_TAB_PRES_MODE], Table[NUC_TAB_YE_MODE],
                    Int_Aux, Int_Main, Mode, &Err, Tolerance );

// trigger a *hard failure* if the EoS driver fails
   if ( Err )   for (int i=0; i<NTarget+1; i++)   Out[i] = NAN;

   const real Cs2_CGS = Out[0];
   const real Cs2_Code = Cs2_CGS * CsSqr2Code;


// final check
#  ifdef GAMER_DEBUG
   if (  Hydro_CheckUnphysical( UNPHY_MODE_SING, &Cs2_Code, "output sound speed squared", ERROR_INFO, UNPHY_VERBOSE )  )
   {
      printf( "   Dens=%13.7e code units, Pres=%13.7e code units, Ye=%13.7e, Mode %d\n", Dens_Code, Pres_Code, Ye, Mode );
      printf( "   EoS error code: %d\n", Err );
   }
#  endif // GAMER_DEBUG


   return Cs2_Code;

} // FUNCTION : EoS_DensPres2CSqr_Nuclear



//-------------------------------------------------------------------------------------------------------
// Function    :  EoS_DensEint2Temp_Nuclear
// Description :  Convert gas mass density and internal energy density to gas temperature
//
// Note        :  1. Internal energy density here is per unit volume instead of per unit mass
//                2. See EoS_SetAuxArray_Nuclear() for the values stored in AuxArray_Flt/Int[]
//                3. Temperature is in kelvin
//                4. Invoke EoS_General_Nuclear()
//
// Parameter   :  Dens_Code    : Gas mass density            (in code unit)
//                Eint_Code    : Gas internal energy density (in code unit)
//                Passive_Code : Passive scalars             (in code unit)
//                AuxArray_*   : Auxiliary arrays (see the Note above)
//                Table        : EoS tables
//
// Return      :  Gas temperature in kelvin
//-------------------------------------------------------------------------------------------------------
GPU_DEVICE_NOINLINE
static real EoS_DensEint2Temp_Nuclear( const real Dens_Code, const real Eint_Code, const real Passive_Code[],
                                       const double AuxArray_Flt[], const int AuxArray_Int[],
                                       const real *const Table[EOS_NTABLE_MAX] )
{

#  ifdef TEMP_IG
   const int  NTarget = 0;
         real In_Flt[4];
#  else
   const int  NTarget = 1;
         real In_Flt[3];
#  endif
         int  In_Int[NTarget+1];
         real Out[NTarget+1], Temp_Kelv;

   In_Flt[0] = Dens_Code;
   In_Flt[1] = Eint_Code;
   In_Flt[2] = Passive_Code[ YE - NCOMP_FLUID ] / Dens_Code;
#  ifdef TEMP_IG
   In_Flt[3] = Passive_Code[ TEMP_IG - NCOMP_FLUID ];
#  endif

   In_Int[0] = NTarget;
#  if ( NUC_TABLE_MODE == NUC_TABLE_MODE_ENGY )
   In_Int[1] = NUC_VAR_IDX_EORT;
#  endif

   EoS_General_Nuclear( NUC_MODE_ENGY, Out, In_Flt, In_Int, AuxArray_Flt, AuxArray_Int, Table );

   Temp_Kelv = Out[0];

   return Temp_Kelv;

} // FUNCTION : EoS_DensEint2Temp_Nuclear



//-------------------------------------------------------------------------------------------------------
// Function    :  EoS_DensTemp2Pres_Nuclear
// Description :  Convert gas mass density and temperature to gas pressure
//
// Note        :  1. See EoS_SetAuxArray_Nuclear() for the values stored in AuxArray_Flt/Int[]
//                2. Temperature is in kelvin
//                3. Invoke EoS_General_Nuclear()
//
// Parameter   :  Dens_Code    : Gas mass density (in code unit)
//                Temp_Kelv    : Gas temperature  (in kelvin   )
//                Passive_Code : Passive scalars  (in code unit)
//                AuxArray_*   : Auxiliary arrays (see the Note above)
//                Table        : EoS tables
//
// Return      :  Gas pressure (in code unit)
//-------------------------------------------------------------------------------------------------------
GPU_DEVICE_NOINLINE
static real EoS_DensTemp2Pres_Nuclear( const real Dens_Code, const real Temp_Kelv, const real Passive_Code[],
                                       const double AuxArray_Flt[], const int AuxArray_Int[],
                                       const real *const Table[EOS_NTABLE_MAX] )
{

   const int  NTarget = 1;
         int  In_Int[NTarget+1];
         real Out[NTarget+1], Pres_Code;
#  ifdef TEMP_IG
         real In_Flt[4];
#  else
         real In_Flt[3];
#  endif


   In_Flt[0] = Dens_Code;
   In_Flt[1] = Temp_Kelv;
   In_Flt[2] = Passive_Code[ YE - NCOMP_FLUID ] / Dens_Code;
#  ifdef TEMP_IG
   In_Flt[3] = Passive_Code[ TEMP_IG - NCOMP_FLUID ];
#  endif

   In_Int[0] = NTarget;
   In_Int[1] = NUC_VAR_IDX_PRES;

   EoS_General_Nuclear( NUC_MODE_TEMP, Out, In_Flt, In_Int, AuxArray_Flt, AuxArray_Int, Table );

   Pres_Code = Out[0];

   return Pres_Code;

} // FUNCTION : EoS_DensTemp2Pres_Nuclear



//-------------------------------------------------------------------------------------------------------
// Function    :  EoS_DensEint2Entr_Nuclear
// Description :  Convert gas mass density and internal energy density to gas entropy
//
// Note        :  1. See EoS_SetAuxArray_Nuclear() for the values stored in AuxArray_Flt/Int[]
//                2. Entropy is in kB per baryon
//                3. Invoke EoS_General_Nuclear()
//
// Parameter   :  Dens_Code    : Gas mass density            (in code unit)
//                Eint_Code    : Gas internal energy density (in code unit)
//                Passive_Code : Passive scalars             (in code unit)
//                AuxArray_*   : Auxiliary arrays (see the Note above)
//                Table        : EoS tables
//
// Return      :  Gas entropy (in kB per baryon)
//-------------------------------------------------------------------------------------------------------
GPU_DEVICE_NOINLINE
static real EoS_DensEint2Entr_Nuclear( const real Dens_Code, const real Eint_Code, const real Passive_Code[],
                                       const double AuxArray_Flt[], const int AuxArray_Int[],
                                       const real *const Table[EOS_NTABLE_MAX] )
{

   const int  NTarget = 1;
         int  In_Int[NTarget+1];

#  ifdef TEMP_IG
         real In_Flt[4];
#  else
         real In_Flt[3];
#  endif
         real Out[NTarget+1], Entr;

   In_Flt[0] = Dens_Code;
   In_Flt[1] = Eint_Code;
   In_Flt[2] = Passive_Code[ YE - NCOMP_FLUID ] / Dens_Code;
#  ifdef TEMP_IG
   In_Flt[3] = Passive_Code[ TEMP_IG - NCOMP_FLUID ];
#  endif

   In_Int[0] = NTarget;
   In_Int[1] = NUC_VAR_IDX_ENTR;

   EoS_General_Nuclear( NUC_MODE_ENGY, Out, In_Flt, In_Int, AuxArray_Flt, AuxArray_Int, Table );

   Entr = Out[0];

   return Entr;

} // FUNCTION : EoS_DensEint2Entr_Nuclear



//-------------------------------------------------------------------------------------------------------
// Function    :  EoS_General_Nuclear
// Description :  General EoS converter: In_*[] -> Out[]
//
// Note        :  1. See EoS_DensEint2Pres_Nuclear()
//                2. In_*[] and Out[] must NOT overlap
//                3. Support energy, temperature, entropy, and pressure modes:
//                   --> In_Flt[0] = mass density            in code unit
//                       In_Flt[1] = internal energy density in code unit (energy      mode)
//                                 = temperature             in kelvin    (temperature mode)
//                                 = entropy                 in kB/baryon (entropy     mode)
//                                 = pressure                in code unit (pressure    mode)
//                       In_Flt[2] = Ye                      dimensionless
//                       In_Flt[3] = initial guess for temperature in kelvin
//                4. The thermodynamic variables returned in Out[] are specified in In_Int[]:
//                   --> In_Int[ 0] = number of thermodynamic variables retrieved from the nuclear EoS table
//                       In_Int[>0] = indices of thermodynamic variables in the nuclear EoS table
//                                    (NUC_VAR_IDX_* defined in NuclearEoS.h)
//                5. The size of Out[] must at least be In_Int[0] + 1:
//                   --> Out[NTarget] stores the internal energy density or temperature either
//                       from the input value or the value found in the auxiliary nuclear EoS table
//                6. Unit conversion is applied to the returned thermodynamic variables:
//                   --> internal energy density (in code unit)
//                       temperature             (in kelvin   )
//                       pressure                (in code unit)
//                       sound speed squared     (in code unit)
//
// Parameter   :  Mode        : Which mode we will use
//                              --> Supported modes: NUC_MODE_ENGY (0)
//                                                   NUC_MODE_TEMP (1)
//                                                   NUC_MODE_ENTR (2)
//                                                   NUC_MODE_PRES (3)
//                Out         : Output array     (see the Note above)
//                In_*        : Input array      (see the Note above)
//                AuxArray_*  : Auxiliary arrays (see the Note above)
//                Table       : EoS tables
//
// Return      :  Out[]
//-------------------------------------------------------------------------------------------------------
GPU_DEVICE_NOINLINE
static void EoS_General_Nuclear( const int Mode, real Out[], const real In_Flt[], const int In_Int[],
                                 const double AuxArray_Flt[], const int AuxArray_Int[],
                                 const real *const Table[EOS_NTABLE_MAX] )
{

// general check
#  ifdef GAMER_DEBUG
   if ( Out          == NULL )   printf( "ERROR : Out == NULL in %s !!\n", __FUNCTION__ );
   if ( In_Flt       == NULL )   printf( "ERROR : In_Flt == NULL in %s !!\n", __FUNCTION__ );
   if ( In_Int       == NULL )   printf( "ERROR : In_Int == NULL in %s !!\n", __FUNCTION__ );
   if ( AuxArray_Flt == NULL )   printf( "ERROR : AuxArray_Flt == NULL in %s !!\n", __FUNCTION__ );
   if ( AuxArray_Int == NULL )   printf( "ERROR : AuxArray_Int == NULL in %s !!\n", __FUNCTION__ );
#  endif // GAMER_DEBUG


   const real EnergyShift = AuxArray_Flt[NUC_AUX_ESHIFT    ];
   const real Dens2CGS    = AuxArray_Flt[NUC_AUX_DENS2CGS  ];
   const real Pres2CGS    = AuxArray_Flt[NUC_AUX_PRES2CGS  ];
   const real sEint2CGS   = AuxArray_Flt[NUC_AUX_VSQR2CGS  ];
   const real Pres2Code   = AuxArray_Flt[NUC_AUX_PRES2CODE ];
   const real CsSqr2Code  = AuxArray_Flt[NUC_AUX_VSQR2CODE ];
   const real sEint2Code  = AuxArray_Flt[NUC_AUX_VSQR2CODE ];
   const real Kelvin2MeV  = AuxArray_Flt[NUC_AUX_KELVIN2MEV];
   const real MeV2Kelvin  = AuxArray_Flt[NUC_AUX_MEV2KELVIN];

   const int  NRho        = AuxArray_Int[NUC_AUX_NRHO      ];
   const int  NTorE       = AuxArray_Int[NUC_AUX_NTORE     ];
   const int  NYe         = AuxArray_Int[NUC_AUX_NYE       ];
   const int  NRho_Mode   = AuxArray_Int[NUC_AUX_NRHO_MODE ];
   const int  NMode       = AuxArray_Int[NUC_AUX_NMODE     ];
   const int  NYe_Mode    = AuxArray_Int[NUC_AUX_NYE_MODE  ];
   const int  Int_Aux     = AuxArray_Int[NUC_AUX_INT_AUX   ];
   const int  Int_Main    = AuxArray_Int[NUC_AUX_INT_MAIN  ];


   const real Dens_Code = In_Flt[0];
   const real Ye        = In_Flt[2];

// check the input density and Ye
#  ifdef GAMER_DEBUG
   Hydro_CheckUnphysical( UNPHY_MODE_SING, &Dens_Code, "input density", ERROR_INFO, UNPHY_VERBOSE );
   Hydro_CheckUnphysical( UNPHY_MODE_SING, &Ye,        "input Ye",      ERROR_INFO, UNPHY_VERBOSE );
#  endif // GAMER_DEBUG


   real Dens_CGS  = Dens_Code * Dens2CGS;

// check floating-point overflow and Ye
#  ifdef GAMER_DEBUG
   if ( Nuc_Overflow(Dens_CGS) )
      printf( "ERROR : EoS overflow (Dens_CGS %13.7e, Dens_Code %13.7e, Dens2CGS %13.7e, Mode %d) in %s() !!\n",
              Dens_CGS, Dens_Code, Dens2CGS, Mode, __FUNCTION__ );

   if ( Ye < (real)Table[NUC_TAB_YE][0]  ||  Ye > (real)Table[NUC_TAB_YE][NYe-1] )
      printf( "ERROR : invalid Ye = %13.7e (min = %13.7e, max = %13.7e, Mode %d) in %s() !!\n",
              Ye, Table[NUC_TAB_YE][0], Table[NUC_TAB_YE][NYe-1], Mode, __FUNCTION__ );
#  endif // GAMER_DEBUG


   const int  NTarget      = In_Int[0];
   const int *TargetIdx    = In_Int+1;
         int  TableIdx_Aux = NULL_INT;
         int  Err          = NULL_INT;
         real TmpIn[3];

   TmpIn[0] = Dens_CGS;
   TmpIn[2] = Ye;

// set up the initial guess of temperature for temperature-based table
#  ifdef TEMP_IG
   real Temp_IG  = In_Flt[3] * Kelvin2MeV;
#  else
   real Temp_IG  = NULL_REAL;
#  endif


   switch ( Mode )
   {
//    energy mode
      case NUC_MODE_ENGY :
      {
         const real Eint_Code = In_Flt[1];

//       check the input internal energy density
#        ifdef GAMER_DEBUG
         Hydro_CheckUnphysical( UNPHY_MODE_SING, &Eint_Code, "input internal energy density", ERROR_INFO, UNPHY_VERBOSE );
#        endif // GAMER_DEBUG


         real sEint_CGS = ( Eint_Code / Dens_Code ) * sEint2CGS - EnergyShift;

//       check floating-point overflow
#        ifdef GAMER_DEBUG
         if ( Nuc_Overflow(sEint_CGS) )
            printf( "ERROR : EoS overflow (sEint_CGS %13.7e, Eint_Code %13.7e, Dens_Code %13.7e, sEint2CGS %13.7e, Mode %d) in %s() !!\n",
                    sEint_CGS, Eint_Code, Dens_Code, sEint2CGS, Mode, __FUNCTION__ );
#        endif // GAMER_DEBUG


         TmpIn[1]     = sEint_CGS;
         TableIdx_Aux = NUC_TAB_EORT_MODE;
      } // case NUC_MODE_ENGY
      break;


//    temperature mode
      case NUC_MODE_TEMP :
      {
         const real Temp_Kelv = In_Flt[1];

//       check the input temperature
#        ifdef GAMER_DEBUG
         Hydro_CheckUnphysical( UNPHY_MODE_SING, &Temp_Kelv, "input temperature", ERROR_INFO, UNPHY_VERBOSE );
#        endif // GAMER_DEBUG


         real Temp_MeV = Temp_Kelv * Kelvin2MeV;

//       check floating-point overflow
#        ifdef GAMER_DEBUG
         if ( Nuc_Overflow(Temp_MeV) )
            printf( "ERROR : EoS overflow (Temp_MeV %13.7e, Temp_Kelv %13.7e, Kelvin2MeV %13.7e, Mode %d) in %s() !!\n",
                    Temp_MeV, Temp_Kelv, Kelvin2MeV, Mode, __FUNCTION__ );
#        endif // GAMER_DEBUG


         TmpIn[1]     = Temp_MeV;
         TableIdx_Aux = NUC_TAB_EORT_MODE;
      } // case NUC_MODE_TEMP
      break;


//    entropy mode
      case NUC_MODE_ENTR :
      {
         const real Entr = In_Flt[1];

//       check the input entropy
#        ifdef GAMER_DEBUG
         Hydro_CheckUnphysical( UNPHY_MODE_SING, &Entr, "input entropy", ERROR_INFO, UNPHY_VERBOSE );
#        endif // GAMER_DEBUG


         TmpIn[1]     = Entr;
         TableIdx_Aux = NUC_TAB_ENTR_MODE;
      } // case NUC_MODE_ENTR
      break;


//    pressure mode
      case NUC_MODE_PRES :
      {
         const real Pres_Code = In_Flt[1];

//       check the input pressure
#        ifdef GAMER_DEBUG
         Hydro_CheckUnphysical( UNPHY_MODE_SING, &Pres_Code, "input pressure", ERROR_INFO, UNPHY_VERBOSE );
#        endif // GAMER_DEBUG


         real Pres_CGS = Pres_Code * Pres2CGS;

//       check floating-point overflow
#        ifdef GAMER_DEBUG
         if ( Nuc_Overflow(Pres_CGS) )
            printf( "ERROR : EoS overflow (Pres_CGS %13.7e, Pres_Code %13.7e, Pres2CGS %13.7e, Mode %d) in %s() !!\n",
                    Pres_CGS, Pres_Code, Pres2CGS, Mode, __FUNCTION__ );
#        endif // GAMER_DEBUG


         TmpIn[1]     = Pres_CGS;
         TableIdx_Aux = NUC_TAB_PRES_MODE;
      } // case NUC_MODE_PRES
      break;


      default :
      {
#        ifdef GAMER_DEBUG
         printf( "ERROR : unsupported mode (%d) at file <%s>, line <%d>, function <%s>\n",
                 Mode, __FILE__, __LINE__, __FUNCTION__ );
#        endif

//       trigger a *hard failure* here
         Out[0] = NAN;

         return;
      } // default

   } // switch ( Mode )


// invoke the nuclear EoS driver
   nuc_eos_C_short( Out, TmpIn, NTarget, TargetIdx,
                    EnergyShift, Temp_IG, NRho, NTorE, NYe, NRho_Mode, NMode, NYe_Mode,
                    Table[NUC_TAB_ALL], Table[NUC_TAB_ALL_MODE], Table[NUC_TAB_RHO], Table[NUC_TAB_TORE], Table[NUC_TAB_YE],
                    Table[NUC_TAB_RHO_MODE], Table[TableIdx_Aux], Table[NUC_TAB_YE_MODE],
                    Int_Aux, Int_Main, Mode, &Err, Tolerance );

// trigger a *hard failure* if the EoS driver fails
   if ( Err )   for (int i=0; i<NTarget+1; i++)   Out[i] = NAN;


// convert to code units and final check for quantities from the nuclear EoS table
// apply to pressure, internal energy, temperature, and sound speed squared
   for (int i=0; i<NTarget; i++)
   {
      switch ( TargetIdx[i] )
      {
         case NUC_VAR_IDX_PRES :
         {
            Out[i] *= Pres2Code;

#           ifdef GAMER_DEBUG
            if (  Hydro_CheckUnphysical( UNPHY_MODE_SING, &Out[i], "output pressure", ERROR_INFO, UNPHY_VERBOSE )  )
            {
               printf( "   Dens=%13.7e code units, Var_mode=%13.7e code units, Ye=%13.7e, Mode %d\n", Dens_Code, In_Flt[1], Ye, Mode );
               printf( "   EoS error code: %d\n", Err );
            }
#           endif
         }
         break;


         case NUC_VAR_IDX_CSQR :
         {
            Out[i] *= CsSqr2Code;

#           ifdef GAMER_DEBUG
            if (  Hydro_CheckUnphysical( UNPHY_MODE_SING, &Out[i], "output sound speed squared", ERROR_INFO, UNPHY_VERBOSE )  )
            {
               printf( "   Dens=%13.7e code units, Var_mode=%13.7e code units, Ye=%13.7e, Mode %d\n", Dens_Code, In_Flt[1], Ye, Mode );
               printf( "   EoS error code: %d\n", Err );
            }
#           endif
         }
         break;


         case NUC_VAR_IDX_EORT :
         {
#           if ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )

            Out[i]  = (  ( Out[i] + EnergyShift ) * sEint2Code  ) * Dens_Code;

#           ifdef GAMER_DEBUG
            if (  Hydro_CheckUnphysical( UNPHY_MODE_SING, &Out[i], "output internal energy density", ERROR_INFO, UNPHY_VERBOSE )  )
            {
               printf( "   Dens=%13.7e code units, Var_mode=%13.7e code units, Ye=%13.7e, Mode %d\n", Dens_Code, In_Flt[1], Ye, Mode );
               printf( "   EoS error code: %d\n", Err );
            }
#           endif // GAMER_DEBUG

#           else

            Out[i] *= MeV2Kelvin;

#           ifdef GAMER_DEBUG
            if (  Hydro_CheckUnphysical( UNPHY_MODE_SING, &Out[i], "output temperature", ERROR_INFO, UNPHY_VERBOSE )  )
            {
               printf( "   Dens=%13.7e code units, Var_mode=%13.7e code units, Ye=%13.7e, Mode %d\n", Dens_Code, In_Flt[1], Ye, Mode );
               printf( "   EoS error code: %d\n", Err );
            }
#           endif // GAMER_DEBUG

#           endif // if ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP ) ... else ...
         }
         break;
      } // switch ( TargetIdx[i] )
   } // for (int i=0; i<NTarget; i++)


// convert to code units and final check for temperature/energy from the auxiliary nuclear EoS table
#  if ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )

   Out[NTarget] *= MeV2Kelvin;

#  ifdef GAMER_DEBUG
   if (  Hydro_CheckUnphysical( UNPHY_MODE_SING, &Out[NTarget], "output temperature", ERROR_INFO, UNPHY_VERBOSE )  )
   {
      printf( "   Dens=%13.7e code units, Var_mode=%13.7e code units, Ye=%13.7e, Mode %d\n", Dens_Code, In_Flt[1], Ye, Mode );
      printf( "   EoS error code: %d\n", Err );
   }
#  endif // GAMER_DEBUG

#  else

   Out[NTarget]  = (  ( Out[NTarget] + EnergyShift ) * sEint2Code  ) * Dens_Code;

#  ifdef GAMER_DEBUG
   if (  Hydro_CheckUnphysical( UNPHY_MODE_SING, &Out[NTarget], "output internal energy density", ERROR_INFO, UNPHY_VERBOSE )  )
   {
      printf( "   Dens=%13.7e code units, Var_mode=%13.7e code units, Ye=%13.7e, Mode %d\n", Dens_Code, In_Flt[1], Ye, Mode );
      printf( "   EoS error code: %d\n", Err );
   }
#  endif // GAMER_DEBUG

#  endif // if ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP ) ... else ...

} // FUNCTION : EoS_General_Nuclear



// =============================================
// III. Set EoS initialization functions
// =============================================

#ifdef __CUDACC__
#  define FUNC_SPACE __device__ static
#else
#  define FUNC_SPACE            static
#endif

FUNC_SPACE EoS_DE2P_t EoS_DensEint2Pres_Ptr = EoS_DensEint2Pres_Nuclear;
FUNC_SPACE EoS_DP2E_t EoS_DensPres2Eint_Ptr = EoS_DensPres2Eint_Nuclear;
FUNC_SPACE EoS_DP2C_t EoS_DensPres2CSqr_Ptr = EoS_DensPres2CSqr_Nuclear;
FUNC_SPACE EoS_DE2T_t EoS_DensEint2Temp_Ptr = EoS_DensEint2Temp_Nuclear;
FUNC_SPACE EoS_DT2P_t EoS_DensTemp2Pres_Ptr = EoS_DensTemp2Pres_Nuclear;
FUNC_SPACE EoS_DE2S_t EoS_DensEint2Entr_Ptr = EoS_DensEint2Entr_Nuclear;
FUNC_SPACE EoS_GENE_t EoS_General_Ptr       = EoS_General_Nuclear;

//-----------------------------------------------------------------------------------------
// Function    :  EoS_SetCPU/GPUFunc_Nuclear
// Description :  Return the function pointers of the CPU/GPU EoS routines
//
// Note        :  1. Invoked by EoS_Init_Nuclear()
//                2. Must obtain the CPU and GPU function pointers by **separate** routines
//                   since CPU and GPU functions are compiled completely separately in GAMER
//                   --> In other words, a unified routine like the following won't work
//
//                      EoS_SetFunc_Nuclear( CPU_FuncPtr, GPU_FuncPtr );
//
//                3. Call-by-reference
//
// Parameter   :  EoS_DensEint2Pres_CPU/GPUPtr : CPU/GPU function pointers to be set
//                EoS_DensPres2Eint_CPU/GPUPtr : ...
//                EoS_DensPres2CSqr_CPU/GPUPtr : ...
//                EoS_DensEint2Temp_CPU/GPUPtr : ...
//                EoS_DensTemp2Pres_CPU/GPUPtr : ...
//                EoS_DensEint2Entr_CPU/GPUPtr : ...
//                EoS_General_CPU/GPUPtr       : ...
//
// Return      :  EoS_DensEint2Pres_CPU/GPUPtr, EoS_DensPres2Eint_CPU/GPUPtr,
//                EoS_DensPres2CSqr_CPU/GPUPtr, EoS_DensEint2Temp_CPU/GPUPtr,
//                EoS_DensTemp2Pres_CPU/GPUPtr, EoS_DensEint2Entr_CPU/GPUPtr,
//                EoS_General_CPU/GPUPtr
//-----------------------------------------------------------------------------------------
#ifdef __CUDACC__
__host__
void EoS_SetGPUFunc_Nuclear( EoS_DE2P_t &EoS_DensEint2Pres_GPUPtr,
                             EoS_DP2E_t &EoS_DensPres2Eint_GPUPtr,
                             EoS_DP2C_t &EoS_DensPres2CSqr_GPUPtr,
                             EoS_DE2T_t &EoS_DensEint2Temp_GPUPtr,
                             EoS_DT2P_t &EoS_DensTemp2Pres_GPUPtr,
                             EoS_DE2S_t &EoS_DensEint2Entr_GPUPtr,
                             EoS_GENE_t &EoS_General_GPUPtr )
{
   CUDA_CHECK_ERROR(  cudaMemcpyFromSymbol( &EoS_DensEint2Pres_GPUPtr, EoS_DensEint2Pres_Ptr, sizeof(EoS_DE2P_t) )  );
   CUDA_CHECK_ERROR(  cudaMemcpyFromSymbol( &EoS_DensPres2Eint_GPUPtr, EoS_DensPres2Eint_Ptr, sizeof(EoS_DP2E_t) )  );
   CUDA_CHECK_ERROR(  cudaMemcpyFromSymbol( &EoS_DensPres2CSqr_GPUPtr, EoS_DensPres2CSqr_Ptr, sizeof(EoS_DP2C_t) )  );
   CUDA_CHECK_ERROR(  cudaMemcpyFromSymbol( &EoS_DensEint2Temp_GPUPtr, EoS_DensEint2Temp_Ptr, sizeof(EoS_DE2T_t) )  );
   CUDA_CHECK_ERROR(  cudaMemcpyFromSymbol( &EoS_DensTemp2Pres_GPUPtr, EoS_DensTemp2Pres_Ptr, sizeof(EoS_DT2P_t) )  );
   CUDA_CHECK_ERROR(  cudaMemcpyFromSymbol( &EoS_DensEint2Entr_GPUPtr, EoS_DensEint2Entr_Ptr, sizeof(EoS_DE2S_t) )  );
   CUDA_CHECK_ERROR(  cudaMemcpyFromSymbol( &EoS_General_GPUPtr,       EoS_General_Ptr,       sizeof(EoS_GENE_t) )  );
}

#else // #ifdef __CUDACC__

void EoS_SetCPUFunc_Nuclear( EoS_DE2P_t &EoS_DensEint2Pres_CPUPtr,
                             EoS_DP2E_t &EoS_DensPres2Eint_CPUPtr,
                             EoS_DP2C_t &EoS_DensPres2CSqr_CPUPtr,
                             EoS_DE2T_t &EoS_DensEint2Temp_CPUPtr,
                             EoS_DT2P_t &EoS_DensTemp2Pres_CPUPtr,
                             EoS_DE2S_t &EoS_DensEint2Entr_CPUPtr,
                             EoS_GENE_t &EoS_General_CPUPtr )
{
   EoS_DensEint2Pres_CPUPtr = EoS_DensEint2Pres_Ptr;
   EoS_DensPres2Eint_CPUPtr = EoS_DensPres2Eint_Ptr;
   EoS_DensPres2CSqr_CPUPtr = EoS_DensPres2CSqr_Ptr;
   EoS_DensEint2Temp_CPUPtr = EoS_DensEint2Temp_Ptr;
   EoS_DensTemp2Pres_CPUPtr = EoS_DensTemp2Pres_Ptr;
   EoS_DensEint2Entr_CPUPtr = EoS_DensEint2Entr_Ptr;
   EoS_General_CPUPtr       = EoS_General_Ptr;
}

#endif // #ifdef __CUDACC__ ... else ...



#ifndef __CUDACC__

// local function prototypes
void EoS_SetAuxArray_Nuclear( double [], int [] );
void EoS_SetCPUFunc_Nuclear( EoS_DE2P_t &, EoS_DP2E_t &, EoS_DP2C_t &, EoS_DE2T_t &, EoS_DT2P_t &, EoS_DE2S_t &, EoS_GENE_t & );
#ifdef GPU
void EoS_SetGPUFunc_Nuclear( EoS_DE2P_t &, EoS_DP2E_t &, EoS_DP2C_t &, EoS_DE2T_t &, EoS_DT2P_t &, EoS_DE2S_t &, EoS_GENE_t & );
#endif

//-----------------------------------------------------------------------------------------
// Function    :  EoS_Init_Nuclear
// Description :  Initialize EoS
//
// Note        :  1. Set auxiliary arrays by invoking EoS_SetAuxArray_*()
//                   --> It will be copied to GPU automatically in CUAPI_SetConstMemory()
//                2. Set the CPU/GPU EoS routines by invoking EoS_SetCPU/GPUFunc_*()
//                3. Invoked by EoS_Init()
//                   --> Enable it by linking to the function pointer "EoS_Init_Ptr"
//                4. Add "#ifndef __CUDACC__" since this routine is only useful on CPU
//
// Parameter   :  None
//
// Return      :  None
//-----------------------------------------------------------------------------------------
void EoS_Init_Nuclear()
{

// check
// default maximum table size must be large enough
   if ( EOS_NTABLE_MAX < NUC_TABLE_NPTR )
      Aux_Error( ERROR_INFO, "EOS_NTABLE_MAX (%d) < NUC_TABLE_NPTR (%d) for the nuclear EoS !!\n",
                 EOS_NTABLE_MAX, NUC_TABLE_NPTR );

// must enable units
   if ( ! OPT__UNIT )
      Aux_Error( ERROR_INFO, "must enable OPT__UNIT for EOS_NUCLEAR !!\n" );


   nuc_eos_C_ReadTable( NUC_TABLE );

   EoS_SetAuxArray_Nuclear( EoS_AuxArray_Flt, EoS_AuxArray_Int );
   EoS_SetCPUFunc_Nuclear( EoS_DensEint2Pres_CPUPtr, EoS_DensPres2Eint_CPUPtr,
                           EoS_DensPres2CSqr_CPUPtr, EoS_DensEint2Temp_CPUPtr,
                           EoS_DensTemp2Pres_CPUPtr, EoS_DensEint2Entr_CPUPtr,
                           EoS_General_CPUPtr );
#  ifdef GPU
   EoS_SetGPUFunc_Nuclear( EoS_DensEint2Pres_GPUPtr, EoS_DensPres2Eint_GPUPtr,
                           EoS_DensPres2CSqr_GPUPtr, EoS_DensEint2Temp_GPUPtr,
                           EoS_DensTemp2Pres_GPUPtr, EoS_DensEint2Entr_GPUPtr,
                           EoS_General_GPUPtr );
#  endif

#  ifdef GPU
   CUAPI_PassNuclearEoSTable2GPU();
#  endif

} // FUNCTION : EoS_Init_Nuclear

#endif // #ifndef __CUDACC__



#endif // #if ( MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )
