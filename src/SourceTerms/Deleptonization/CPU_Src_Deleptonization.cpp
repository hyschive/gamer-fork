#include "NuclearEoS.h"
#include "CUFLU.h"

#if ( MODEL == HYDRO )

#define SRC_AUX_DENS2CGS            0
#define SRC_AUX_DELEP_ENU           1
#define SRC_AUX_DELEP_RHO1          2
#define SRC_AUX_DELEP_RHO2          3
#define SRC_AUX_DELEP_YE1           4
#define SRC_AUX_DELEP_YE2           5
#define SRC_AUX_DELEP_YEC           6
#define SRC_AUX_KELVIN2MEV          7



// external functions and GPU-related set-up
#ifdef __CUDACC__

#include "Global.h"
#include "PhysicalConstant.h"
#include "CUDA_CheckError.h"
#include "CUFLU_Shared_FluUtility.cu"
#include "CUDA_ConstMemory.h"

#endif // #ifdef __CUDACC__


// local function prototypes
#ifndef __CUDACC__

void Src_SetAuxArray_Deleptonization( double [], int [] );
void Src_SetCPUFunc_Deleptonization( SrcFunc_t & );
#ifdef GPU
void Src_SetGPUFunc_Deleptonization( SrcFunc_t & );
#endif
void Src_SetConstMemory_Deleptonization( const double AuxArray_Flt[], const int AuxArray_Int[],
                                         double *&DevPtr_Flt, int *&DevPtr_Int );
void Src_End_Deleptonization();

#endif

GPU_DEVICE static
real YeOfRhoFunc( const real Dens_CGS, const real Delep_Rho1, const real Delep_Rho2,
                  const real Delep_Ye1, const real Delep_Ye2, const real Delep_Yec );



/********************************************************
1. Deleptonization source term
   --> Enabled by the runtime option "SRC_DELEPTONIZATION"

2. This file is shared by both CPU and GPU

   CUSRC_Src_Deleptonization.cu -> CPU_Src_Deleptonization.cpp

3. Four steps are required to implement a source term

   I.   Set auxiliary arrays
   II.  Implement the source-term function
   III. [Optional] Add the work to be done every time
        before calling the major source-term function
   IV.  Set initialization functions

4. The source-term function must be thread-safe and
   not use any global variable
********************************************************/



// =======================
// I. Set auxiliary arrays
// =======================

//-------------------------------------------------------------------------------------------------------
// Function    :  Src_SetAuxArray_Deleptonization
// Description :  Set the auxiliary arrays AuxArray_Flt/Int[]
//
// Note        :  1. Invoked by Src_Init_Deleptonization()
//                2. AuxArray_Flt/Int[] have the size of SRC_NAUX_DLEP defined in Macro.h (default = 8)
//                3. Add "#ifndef __CUDACC__" since this routine is only useful on CPU
//
// Parameter   :  AuxArray_Flt/Int : Floating-point/Integer arrays to be filled up
//
// Return      :  AuxArray_Flt/Int[]
//-------------------------------------------------------------------------------------------------------
#ifndef __CUDACC__
void Src_SetAuxArray_Deleptonization( double AuxArray_Flt[], int AuxArray_Int[] )
{

   AuxArray_Flt[SRC_AUX_DENS2CGS   ] = UNIT_D;
   AuxArray_Flt[SRC_AUX_DELEP_ENU  ] = SrcTerms.Dlep_Enu;
   AuxArray_Flt[SRC_AUX_DELEP_RHO1 ] = SrcTerms.Dlep_Rho1;
   AuxArray_Flt[SRC_AUX_DELEP_RHO2 ] = SrcTerms.Dlep_Rho2;
   AuxArray_Flt[SRC_AUX_DELEP_YE1  ] = SrcTerms.Dlep_Ye1;
   AuxArray_Flt[SRC_AUX_DELEP_YE2  ] = SrcTerms.Dlep_Ye2;
   AuxArray_Flt[SRC_AUX_DELEP_YEC  ] = SrcTerms.Dlep_Yec;
   AuxArray_Flt[SRC_AUX_KELVIN2MEV ] = Const_kB_eV*1.0e-6;

} // FUNCTION : Src_SetAuxArray_Deleptonization
#endif // #ifndef __CUDACC__



// ======================================
// II. Implement the source-term function
// ======================================

//-------------------------------------------------------------------------------------------------------
// Function    :  Src_Deleptonization
// Description :  Major source-term function
//
// Note        :  1. Invoked by CPU/GPU_SrcSolver_IterateAllCells()
//                2. See Src_SetAuxArray_Deleptonization() for the values stored in AuxArray_Flt/Int[]
//                3. Shared by both CPU and GPU
//                4. Ref: M. Liebendoerfer, 2005, ApJ, 603, 1042-1051 (arXiv: astro-ph/0504072)
//
// Parameter   :  fluid             : Fluid array storing both the input and updated values
//                                    --> Including both active and passive variables
//                B                 : Cell-centered magnetic field
//                SrcTerms          : Structure storing all source-term variables
//                dt                : Time interval to advance solution
//                dh                : Grid size
//                x/y/z             : Target physical coordinates
//                TimeNew           : Target physical time to reach
//                TimeOld           : Physical time before update
//                                    --> This function updates physical time from TimeOld to TimeNew
//                MinDens/Pres/Eint : Density, pressure, and internal energy floors
//                EoS               : EoS object
//                AuxArray_*        : Auxiliary arrays (see the Note above)
//
// Return      :  fluid[]
//-----------------------------------------------------------------------------------------
GPU_DEVICE_NOINLINE
static void Src_Deleptonization( real fluid[], const real B[],
                                 const SrcTerms_t *SrcTerms, const real dt, const real dh,
                                 const double x, const double y, const double z,
                                 const double TimeNew, const double TimeOld,
                                 const real MinDens, const real MinPres, const real MinEint,
                                 const EoS_t *EoS, const double AuxArray_Flt[], const int AuxArray_Int[] )
{

// check
#  ifdef GAMER_DEBUG
   if ( AuxArray_Flt == NULL )   printf( "ERROR : AuxArray_Flt == NULL in %s !!\n", __FUNCTION__ );
   if ( AuxArray_Int == NULL )   printf( "ERROR : AuxArray_Int == NULL in %s !!\n", __FUNCTION__ );
#  endif


   const real Dens2CGS          = AuxArray_Flt[SRC_AUX_DENS2CGS   ];
   const real Delep_Enu         = AuxArray_Flt[SRC_AUX_DELEP_ENU  ];
   const real Delep_Rho1        = AuxArray_Flt[SRC_AUX_DELEP_RHO1 ];
   const real Delep_Rho2        = AuxArray_Flt[SRC_AUX_DELEP_RHO2 ];
   const real Delep_Ye1         = AuxArray_Flt[SRC_AUX_DELEP_YE1  ];
   const real Delep_Ye2         = AuxArray_Flt[SRC_AUX_DELEP_YE2  ];
   const real Delep_Yec         = AuxArray_Flt[SRC_AUX_DELEP_YEC  ];
   const real Kelvin2MeV        = AuxArray_Flt[SRC_AUX_KELVIN2MEV ];

   const real Delep_minDens_CGS = (real)1.0e6; // [g/cm^3]


   const real MeV2Kelvin        = 1.0/Kelvin2MeV;

#  ifdef MHD
   const real Emag       = (real)0.5*(  SQR( B[MAGX] ) + SQR( B[MAGY] ) + SQR( B[MAGZ] )  );
#  else
   const real Emag       = NULL_REAL;
#  endif


// for entropy updates
   real Del_Ye   = NULL_REAL;
   real Del_Entr = NULL_REAL;

// output Ye
   real Yout;


// Deleptonization
   const real Dens_Code = fluid[DENS];
   const real Dens_CGS  = Dens_Code * Dens2CGS;
   const real Eint_Code = Hydro_Con2Eint( fluid[DENS], fluid[MOMX], fluid[MOMY], fluid[MOMZ], fluid[ENGY], true, MinEint, Emag );
         real Eint_Update;
         real Entr = NULL_REAL;
#  ifdef YE
         real Ye           = fluid[YE] / fluid[DENS];
#  else
         real Ye           = NULL_REAL;
#  endif
#  ifdef TEMP_IG
         real Temp_IG_Kelv = fluid[TEMP_IG];
#  else
         real Temp_IG_Kelv = NULL_REAL;
#  endif

   if ( Dens_CGS <= Delep_minDens_CGS )
   {
      Del_Ye = (real)0.0;
   }
   else
   {
      Yout   = YeOfRhoFunc( Dens_CGS, Delep_Rho1, Delep_Rho2,
                            Delep_Ye1, Delep_Ye2, Delep_Yec );
      Del_Ye = Yout - Ye;
      Del_Ye = MIN( (real)0.0, Del_Ye ); // Deleptonization cannot increase Ye
   }

   if ( Del_Ye < (real)0.0 )
   {
//    Nuclear EoS
#     if ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )
      const int  NTarget1 = 2;
#     else
      const int  NTarget1 = 3;
#     endif
            int  In_Int1[NTarget1+1];
#     ifdef TEMP_IG
            real In_Flt1[4];
                 In_Flt1[3] = Temp_IG_Kelv;
#     else
            real In_Flt1[3];
#     endif
            real Out1[NTarget1+1];

      In_Flt1[0] = Dens_Code;
      In_Flt1[1] = Eint_Code;
      In_Flt1[2] = Ye;

      In_Int1[0] = NTarget1;
      In_Int1[1] = NUC_VAR_IDX_ENTR;
      In_Int1[2] = NUC_VAR_IDX_MUNU;
#     if ( NUC_TABLE_MODE == NUC_TABLE_MODE_ENGY )
      In_Int1[3] = NUC_VAR_IDX_EORT;
#     endif


      EoS->General_FuncPtr( NUC_MODE_ENGY, Out1, In_Flt1, In_Int1, EoS->AuxArrayDevPtr_Flt, EoS->AuxArrayDevPtr_Int, EoS->Table );

                 Entr      = Out1[0];
            real mu_nu_MeV = Out1[1];
      const real Temp_MeV  = Out1[2] * Kelvin2MeV;

#     ifdef GAMER_DEBUG
      if ( mu_nu_MeV != mu_nu_MeV ) printf( "ERROR : couldn't get chemical potential munu (NaN) !!\n" );
#     endif // GAMER_DEBUG

      if (  ( mu_nu_MeV < Delep_Enu )  ||  ( Dens_CGS >= (real)2.0e12 )  )
      {
         Del_Entr = (real)0.0;
      }
      else
      {
         Del_Entr = - Del_Ye * ( mu_nu_MeV - Delep_Enu ) / Temp_MeV;
      }

//    update entropy and Ye
      Entr      = Entr + Del_Entr;
      Ye        = Ye + Del_Ye;
#     ifdef YE
      fluid[YE] = Dens_Code * Ye;
#     endif

//    input and output arrays for Nuclear EoS
#     if ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )
      const int  NTarget2 = 1;
#     else
      const int  NTarget2 = 0;
#     endif
            int  In_Int2[NTarget2+1];
#     ifdef TEMP_IG
            real In_Flt2[4];
                 In_Flt2[3] = Temp_MeV * MeV2Kelvin;
#     else
            real In_Flt2[3];
#     endif

            real Out2[NTarget2+1];

      In_Flt2[0]  = Dens_Code;
      In_Flt2[1]  = Entr;
      In_Flt2[2]  = Ye;

      In_Int2[0]  = NTarget2;
#     if ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )
      In_Int2[1]  = NUC_VAR_IDX_EORT;
#     endif

//    call Nuclear EoS with entropy mode
      EoS->General_FuncPtr( NUC_MODE_ENTR, Out2, In_Flt2, In_Int2, EoS->AuxArrayDevPtr_Flt, EoS->AuxArrayDevPtr_Int, EoS->Table );
      Eint_Update = Out2[0];
      fluid[ENGY] = Hydro_ConEint2Etot( fluid[DENS], fluid[MOMX], fluid[MOMY], fluid[MOMZ], Eint_Update, Emag );


      // final check
#     if GAMER_DEBUG
      if (  Hydro_CheckUnphysical( UNPHY_MODE_SING, &Eint_Update, "output internal energy density", ERROR_INFO, UNPHY_VERBOSE )  )
      {
         printf( "   Dens=%13.7e code units, Eint=%13.7e code units,  Ye=%13.7e\n",      Dens_Code, Eint_Code, Ye       );
         printf( "   Entr=%13.7e kb/baryon, Del_Ye=%13.7e, Del_Entr=%13.7e kb/baryon\n", Entr,      Del_Ye,    Del_Entr );
      }
#     endif // GAMER_DEBUG

   } // if ( Del_Ye < (real)0.0 )


// update temperature initial guess
#  ifdef TEMP_IG
   fluid[TEMP_IG] = Hydro_Con2Temp( fluid[DENS], fluid[MOMX], fluid[MOMY], fluid[MOMZ], fluid[ENGY], fluid+NCOMP_FLUID,
                                    false, 0.0, Emag, EoS->DensEint2Temp_FuncPtr,
                                    EoS->AuxArrayDevPtr_Flt, EoS->AuxArrayDevPtr_Int, EoS->Table );
#  endif

} // FUNCTION : Src_Deleptonization



// ==================================================
// III. [Optional] Add the work to be done every time
//      before calling the major source-term function
// ==================================================

//-------------------------------------------------------------------------------------------------------
// Function    :  Src_WorkBeforeMajorFunc_Deleptonization
// Description :  Specify work to be done every time before calling the major source-term function
//
// Note        :  1. Invoked by Src_WorkBeforeMajorFunc()
//                2. Add "#ifndef __CUDACC__" since this routine is only useful on CPU
//
// Parameter   :  lv               : Target refinement level
//                TimeNew          : Target physical time to reach
//                TimeOld          : Physical time before update
//                                   --> The major source-term function will update the system from TimeOld to TimeNew
//                dt               : Time interval to advance solution
//                                   --> Physical coordinates : TimeNew - TimeOld == dt
//                                       Comoving coordinates : TimeNew - TimeOld == delta(scale factor) != dt
//                AuxArray_Flt/Int : Auxiliary arrays
//                                   --> Can be used and/or modified here
//                                   --> Must call Src_SetConstMemory_Deleptonization() after modification
//
// Return      :  AuxArray_Flt/Int[]
//-------------------------------------------------------------------------------------------------------
#ifndef __CUDACC__
void Src_WorkBeforeMajorFunc_Deleptonization( const int lv, const double TimeNew, const double TimeOld, const double dt,
                                              double AuxArray_Flt[], int AuxArray_Int[] )
{

// not used by this source term

} // FUNCTION : Src_WorkBeforeMajorFunc_Deleptonization
#endif



// ================================
// IV. Set initialization functions
// ================================

#ifdef __CUDACC__
#  define FUNC_SPACE __device__ static
#else
#  define FUNC_SPACE            static
#endif

FUNC_SPACE SrcFunc_t SrcFunc_Ptr = Src_Deleptonization;

//-----------------------------------------------------------------------------------------
// Function    :  Src_SetCPU/GPUFunc_Deleptonization
// Description :  Return the function pointer of the CPU/GPU source-term function
//
// Note        :  1. Invoked by Src_Init_Deleptonization()
//                2. Call-by-reference
//
// Parameter   :  SrcFunc_CPU/GPUPtr : CPU/GPU function pointer to be set
//
// Return      :  SrcFunc_CPU/GPUPtr
//-----------------------------------------------------------------------------------------
#ifdef __CUDACC__
__host__
void Src_SetGPUFunc_Deleptonization( SrcFunc_t &SrcFunc_GPUPtr )
{
   CUDA_CHECK_ERROR(  cudaMemcpyFromSymbol( &SrcFunc_GPUPtr, SrcFunc_Ptr, sizeof(SrcFunc_t) )  );
}

#else

void Src_SetCPUFunc_Deleptonization( SrcFunc_t &SrcFunc_CPUPtr )
{
   SrcFunc_CPUPtr = SrcFunc_Ptr;
}

#endif // #ifdef __CUDACC__ ... else ...



#ifdef __CUDACC__
//-------------------------------------------------------------------------------------------------------
// Function    :  Src_SetConstMemory_Deleptonization
// Description :  Set the constant memory variables on GPU
//
// Note        :  1. Adopt the suggested approach for CUDA version >= 5.0
//                2. Invoked by Src_Init_Deleptonizatio() and, if necessary, Src_WorkBeforeMajorFunc_Deleptonizatio()
//                3. SRC_NAUX_DLEP is defined in Macro.h
//
// Parameter   :  AuxArray_Flt/Int : Auxiliary arrays to be copied to the constant memory
//                DevPtr_Flt/Int   : Pointers to store the addresses of constant memory arrays
//
// Return      :  c_Src_Dlep_AuxArray_Flt[], c_Src_Dlep_AuxArray_Int[], DevPtr_Flt, DevPtr_Int
//---------------------------------------------------------------------------------------------------
void Src_SetConstMemory_Deleptonization( const double AuxArray_Flt[], const int AuxArray_Int[],
                                         double *&DevPtr_Flt, int *&DevPtr_Int )
{

// copy data to constant memory
   CUDA_CHECK_ERROR(  cudaMemcpyToSymbol( c_Src_Dlep_AuxArray_Flt, AuxArray_Flt, SRC_NAUX_DLEP*sizeof(double) )  );
   CUDA_CHECK_ERROR(  cudaMemcpyToSymbol( c_Src_Dlep_AuxArray_Int, AuxArray_Int, SRC_NAUX_DLEP*sizeof(int   ) )  );

// obtain the constant-memory pointers
   CUDA_CHECK_ERROR(  cudaGetSymbolAddress( (void **)&DevPtr_Flt, c_Src_Dlep_AuxArray_Flt )  );
   CUDA_CHECK_ERROR(  cudaGetSymbolAddress( (void **)&DevPtr_Int, c_Src_Dlep_AuxArray_Int )  );

} // FUNCTION : Src_SetConstMemory_Deleptonization
#endif // #ifdef __CUDACC__



#ifndef __CUDACC__

//-----------------------------------------------------------------------------------------
// Function    :  Src_Init_Deleptonization
// Description :  Initialize the deleptonization source term
//
// Note        :  1. Set auxiliary arrays by invoking Src_SetAuxArray_*()
//                   --> Copy to the GPU constant memory and store the associated addresses
//                2. Set the source-term function by invoking Src_SetCPU/GPUFunc_*()
//                3. Invoked by Src_Init()
//                4. Add "#ifndef __CUDACC__" since this routine is only useful on CPU
//
// Parameter   :  None
//
// Return      :  None
//-----------------------------------------------------------------------------------------
void Src_Init_Deleptonization()
{

// set the auxiliary arrays
   Src_SetAuxArray_Deleptonization( Src_Dlep_AuxArray_Flt, Src_Dlep_AuxArray_Int );

// copy the auxiliary arrays to the GPU constant memory and store the associated addresses
#  ifdef GPU
   Src_SetConstMemory_Deleptonization( Src_Dlep_AuxArray_Flt, Src_Dlep_AuxArray_Int,
                                       SrcTerms.Dlep_AuxArrayDevPtr_Flt, SrcTerms.Dlep_AuxArrayDevPtr_Int );
#  else
   SrcTerms.Dlep_AuxArrayDevPtr_Flt = Src_Dlep_AuxArray_Flt;
   SrcTerms.Dlep_AuxArrayDevPtr_Int = Src_Dlep_AuxArray_Int;
#  endif

// set the major source-term function
   Src_SetCPUFunc_Deleptonization( SrcTerms.Dlep_CPUPtr );

#  ifdef GPU
   Src_SetGPUFunc_Deleptonization( SrcTerms.Dlep_GPUPtr );
   SrcTerms.Dlep_FuncPtr = SrcTerms.Dlep_GPUPtr;
#  else
   SrcTerms.Dlep_FuncPtr = SrcTerms.Dlep_CPUPtr;
#  endif

} // FUNCTION : Src_Init_Deleptonization



//-----------------------------------------------------------------------------------------
// Function    :  Src_End_Deleptonization
// Description :  Release the resources used by the deleptonization source term
//
// Note        :  1. Invoked by Src_End()
//                2. Add "#ifndef __CUDACC__" since this routine is only useful on CPU
//
// Parameter   :  None
//
// Return      :  None
//-----------------------------------------------------------------------------------------
void Src_End_Deleptonization()
{

// not used by this source term

} // FUNCTION : Src_End_Deleptonization

#endif // #ifndef __CUDACC__




//-----------------------------------------------------------------------------------------
// Function    :  YeOfRhoFunc
// Description :  Calculate electron fraction Ye from the given density and
//                deleptonization parameters
//
// Note        :  1. Invoked by Src_Deleptonization()
//                2. Ref: M. Liebendoerfer, 2005, ApJ, 603, 1042-1051 (arXiv: astro-ph/0504072)
//
// Parameter   :  Dens_CGS  : density in CGS from which Ye is caculated
//             :  Delep_Rho1: parameter for the parameterized deleptonization fitting formula
//             :  Delep_Rho2: parameter for the parameterized deleptonization fitting formula
//             :  Delep_Ye1 : parameter for the parameterized deleptonization fitting formula
//             :  Delep_Ye2 : parameter for the parameterized deleptonization fitting formula
//             :  Delep_Yec : parameter for the parameterized deleptonization fitting formula
//
// Return      :  YeOfRhoFunc
//-----------------------------------------------------------------------------------------
GPU_DEVICE static
real YeOfRhoFunc( const real Dens_CGS, const real Delep_Rho1, const real Delep_Rho2,
                  const real Delep_Ye1, const real Delep_Ye2, const real Delep_Yec )
{

   real XofRho, Ye;

   XofRho = LOG10(  SQR( Dens_CGS ) / ( Delep_Rho2 * Delep_Rho1 )  )
          / LOG10( Delep_Rho2 / Delep_Rho1 );
   XofRho = MAX( (real)-1.0, MIN( (real)1.0, XofRho ) );

   Ye = (real)0.5 * ( Delep_Ye2 + Delep_Ye1 ) + (real)0.5 * XofRho * ( Delep_Ye2 - Delep_Ye1 )
      + Delep_Yec * (  (real)1.0 - FABS( XofRho )
      + (real)4.0 * FABS( XofRho ) * ( FABS( XofRho ) - (real)0.5 ) * ( FABS( XofRho ) - (real)1.0 )  );


   return Ye;

} // FUNCTION : YeOfRhoFunc



#endif // #if ( MODEL == HYDRO )
