#include "CUFLU.h"
#include "NuclearEoS.h"

#if ( MODEL == HYDRO )

#define SRC_AUX_KELVIN2MEV    0     // AuxArray_Flt: convert kelvin to MeV
#define SRC_AUX_VSQR2CODE     1     // AuxArray_Flt: convert velocity^2 to code unit



// external functions and GPU-related set-up
#ifdef __CUDACC__

#include "CUDA_CheckError.h"
#include "CUFLU_Shared_FluUtility.cu"
#include "CUDA_ConstMemory.h"

#endif // #ifdef __CUDACC__


// local function prototypes
#ifndef __CUDACC__

void Src_SetAuxArray_Lightbulb( double [], int [] );
void Src_SetCPUFunc_Lightbulb( SrcFunc_t & );
#ifdef GPU
void Src_SetGPUFunc_Lightbulb( SrcFunc_t & );
#endif
void Src_SetConstMemory_Lightbulb( const double AuxArray_Flt[], const int AuxArray_Int[],
                                   double *&DevPtr_Flt, int *&DevPtr_Int );
void Src_PassData2GPU_Lightbulb();
void Src_End_Lightbulb();

#endif



/********************************************************
1. Lightbulb source term
   --> Enabled by the runtime option "SRC_LIGHTBULB"

2. This file is shared by both CPU and GPU

   CUSRC_Src_Lightbulb.cu -> CPU_Src_Lightbulb.cpp

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
// Function    :  Src_SetAuxArray_Lightbulb
// Description :  Set the auxiliary arrays AuxArray_Flt/Int[]
//
// Note        :  1. Invoked by Src_Init_Lightbulb()
//                2. AuxArray_Flt/Int[] have the size of SRC_NAUX_LIGHTBULB defined in Macro.h (default = 2)
//                3. Add "#ifndef __CUDACC__" since this routine is only useful on CPU
//
// Parameter   :  AuxArray_Flt/Int : Floating-point/Integer arrays to be filled up
//
// Return      :  AuxArray_Flt/Int[]
//-------------------------------------------------------------------------------------------------------
#ifndef __CUDACC__
void Src_SetAuxArray_Lightbulb( double AuxArray_Flt[], int AuxArray_Int[] )
{

   AuxArray_Flt[SRC_AUX_KELVIN2MEV] = Const_kB_eV*1.0e-6;
   AuxArray_Flt[SRC_AUX_VSQR2CODE ] = 1.0 / SQR( UNIT_V );

} // FUNCTION : Src_SetAuxArray_Lightbulb
#endif // #ifndef __CUDACC__



// ======================================
// II. Implement the source-term function
// ======================================

//-------------------------------------------------------------------------------------------------------
// Function    :  Src_Lightbulb
// Description :  Apply the lightbulb scheme for neutrino heating and cooling
//
// Note        :  1. Invoked by CPU/GPU_SrcSolver_IterateAllCells()
//                2. See Src_SetAuxArray_Lightbulb() for the values stored in AuxArray_Flt/Int[]
//                3. Shared by both CPU and GPU
//                4. Ref: Sean M. Couch, 2013, ApJ, 765, 29 (arXiv: 1206.4724), sec. 2
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
static void Src_Lightbulb( real fluid[], const real B[],
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


   const real Kelvin2MeV = AuxArray_Flt[SRC_AUX_KELVIN2MEV];
   const real sEint2Code = AuxArray_Flt[SRC_AUX_VSQR2CODE ];

   const double x0     = x - SrcTerms->BoxCenter[0];
   const double y0     = y - SrcTerms->BoxCenter[1];
   const double z0     = z - SrcTerms->BoxCenter[2];
   const double r2_CGS = SQR( SrcTerms->Unit_L ) * (  SQR( x0 ) + SQR( y0 ) + SQR( z0 )  );

#  ifdef MHD
   const real Emag = (real)0.5*(  SQR( B[MAGX] ) + SQR( B[MAGY] ) + SQR( B[MAGZ] )  );
#  else
   const real Emag = NULL_REAL;
#  endif

   const real Dens_Code = fluid[DENS];
   const real Eint_Code = Hydro_Con2Eint( fluid[DENS], fluid[MOMX], fluid[MOMY], fluid[MOMZ], fluid[ENGY],
                                          true, MinEint, Emag );
#  ifdef YE
   const real Ye        = fluid[YE] / fluid[DENS];
#  else
   const real Ye        = NULL_REAL;
#  endif
#  ifdef TEMP_IG
   const real Temp_IG   = fluid[TEMP_IG];
#  else
   const real Temp_IG   = NULL_REAL;
#  endif


// 1. call nuclear EoS driver to obtain Xn, Xp, and temperature, using energy mode
#  if ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )
   const int  NTarget = 2;
#  else
   const int  NTarget = 3;
#  endif
         int  In_Int[NTarget+1];
#  ifdef TEMP_IG
         real In_Flt[4];
              In_Flt[3] = Temp_IG;
#  else
         real In_Flt[3];
#  endif
         real Out[NTarget+1];

   In_Flt[0] = Dens_Code;
   In_Flt[1] = Eint_Code;
   In_Flt[2] = Ye;

   In_Int[0] = NTarget;
   In_Int[1] = NUC_VAR_IDX_XN;
   In_Int[2] = NUC_VAR_IDX_XP;
#  if ( NUC_TABLE_MODE == NUC_TABLE_MODE_ENGY )
   In_Int[3] = NUC_VAR_IDX_EORT;
#  endif


// note that EoS->General_FuncPtr binds to the GPU EoS driver when enabling GPU
// here we call the CPU EoS driver instead if the CPU source term solvers are invoked
#  ifdef __CUDACC__
   EoS->General_FuncPtr( NUC_MODE_ENGY, Out, In_Flt, In_Int, EoS->AuxArrayDevPtr_Flt, EoS->AuxArrayDevPtr_Int, EoS->Table );
#  else
   EoS_General_CPUPtr  ( NUC_MODE_ENGY, Out, In_Flt, In_Int, EoS_AuxArray_Flt,        EoS_AuxArray_Int,        h_EoS_Table );
#  endif

   const real Xn       = Out[0];               // neutron mass fraction
   const real Xp       = Out[1];               // proton  mass fraction
   const real Temp_MeV = Out[2] * Kelvin2MeV;  // temperature in MeV


// 2. calculate cooling and heating rate
   const real rate_heating = 1.544e20 * ( SrcTerms->Lightbulb_Lnue / 1.0e52 ) * ( 1.0e14 / r2_CGS )
                           * SQR( 0.25 * SrcTerms->Lightbulb_Tnue ) * SrcTerms->Unit_T * sEint2Code;
   const real rate_cooling = 1.399e20 * CUBE(  SQR( 0.5 * Temp_MeV )  ) * SrcTerms->Unit_T * sEint2Code;

// approximate the optical depth by density in unit of 10^11 g/cm^3
   const real tau       = Dens_Code * (real)( 1.0e-11 * SrcTerms->Unit_D );
   const real rate_Code = ( rate_heating - rate_cooling ) * ( Xn + Xp ) * EXP( -tau );


// 3. calculate the change in internal energy and update the input energy density and temperature for initial guess
   const real dEint_Code  = rate_Code * dt * Dens_Code;
   const real Eint_Update = Eint_Code + dEint_Code;

   fluid[ENGY] = Hydro_ConEint2Etot( fluid[DENS], fluid[MOMX], fluid[MOMY], fluid[MOMZ], Eint_Update, Emag );
#  ifdef DEDT_LB
   fluid[DEDT_LB] = FABS( rate_Code * Dens_Code );
#  endif
// calculate temperature for initial guess and store to fluid[TEMP_IG]
#  ifdef TEMP_IG
#  ifdef __CUDACC__
   fluid[TEMP_IG] = Hydro_Con2Temp( fluid[DENS], fluid[MOMX], fluid[MOMY], fluid[MOMZ], fluid[ENGY], fluid+NCOMP_FLUID,
                                    false, 0.0, Emag, EoS->DensEint2Temp_FuncPtr,
                                    EoS->AuxArrayDevPtr_Flt, EoS->AuxArrayDevPtr_Int, EoS->Table );
#  else
   fluid[TEMP_IG] = Hydro_Con2Temp( fluid[DENS], fluid[MOMX], fluid[MOMY], fluid[MOMZ], fluid[ENGY], fluid+NCOMP_FLUID,
                                    false, 0.0, Emag, EoS_DensEint2Temp_CPUPtr,
                                    EoS_AuxArray_Flt, EoS_AuxArray_Int, h_EoS_Table );
#  endif
#  endif


// final check
#  ifdef GAMER_DEBUG
   if (  Hydro_CheckUnphysical( UNPHY_MODE_SING, &Eint_Update, "output internal energy density", ERROR_INFO, UNPHY_VERBOSE )  )
   {
      printf( "   Dens=%13.7e code units, Eint=%13.7e code units, Ye=%13.7e\n", Dens_Code, Eint_Code, Ye );
      printf( "   Temp=%13.7e MeV, Xn=%13.7e, Xp=%13.7e\n", Temp_MeV, Xn, Xp );
      printf( "   heating=%13.7e, cooling=%13.7e, dt=%13.7e, dEint=%13.7e\n", rate_heating, rate_cooling, dt, dEint_Code );
   }
#  endif // GAMER_DEBUG

} // FUNCTION : Src_Lightbulb



// ==================================================
// III. [Optional] Add the work to be done every time
//      before calling the major source-term function
// ==================================================

//-------------------------------------------------------------------------------------------------------
// Function    :  Src_WorkBeforeMajorFunc_Lightbulb
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
//                                   --> Must call Src_SetConstMemory_Lightbulb() after modification
//
// Return      :  AuxArray_Flt/Int[]
//-------------------------------------------------------------------------------------------------------
#ifndef __CUDACC__
void Src_WorkBeforeMajorFunc_Lightbulb( const int lv, const double TimeNew, const double TimeOld, const double dt,
                                        double AuxArray_Flt[], int AuxArray_Int[] )
{

// not used by this source term

} // FUNCTION : Src_WorkBeforeMajorFunc_Lightbulb
#endif



#ifdef __CUDACC__
//-------------------------------------------------------------------------------------------------------
// Function    :  Src_PassData2GPU_Lightbulb
// Description :  Transfer data to GPU
//
// Note        :  1. Invoked by Src_WorkBeforeMajorFunc_Lightbulb()
//                2. Use synchronous transfer
//
// Parameter   :  None
//
// Return      :  None
//-------------------------------------------------------------------------------------------------------
void Src_PassData2GPU_Lightbulb()
{

// not used by this source term

} // FUNCTION : Src_PassData2GPU_Lightbulb
#endif // #ifdef __CUDACC__



// ================================
// IV. Set initialization functions
// ================================

#ifdef __CUDACC__
#  define FUNC_SPACE __device__ static
#else
#  define FUNC_SPACE            static
#endif

FUNC_SPACE SrcFunc_t SrcFunc_Ptr = Src_Lightbulb;

//-----------------------------------------------------------------------------------------
// Function    :  Src_SetCPU/GPUFunc_Lightbulb
// Description :  Return the function pointer of the CPU/GPU source-term function
//
// Note        :  1. Invoked by Src_Init_Lightbulb()
//                2. Call-by-reference
//
// Parameter   :  SrcFunc_CPU/GPUPtr : CPU/GPU function pointer to be set
//
// Return      :  SrcFunc_CPU/GPUPtr
//-----------------------------------------------------------------------------------------
#ifdef __CUDACC__
__host__
void Src_SetGPUFunc_Lightbulb( SrcFunc_t &SrcFunc_GPUPtr )
{
   CUDA_CHECK_ERROR(  cudaMemcpyFromSymbol( &SrcFunc_GPUPtr, SrcFunc_Ptr, sizeof(SrcFunc_t) )  );
}

#else

void Src_SetCPUFunc_Lightbulb( SrcFunc_t &SrcFunc_CPUPtr )
{
   SrcFunc_CPUPtr = SrcFunc_Ptr;
}

#endif // #ifdef __CUDACC__ ... else ...



#ifdef __CUDACC__
//-------------------------------------------------------------------------------------------------------
// Function    :  Src_SetConstMemory_Lightbulb
// Description :  Set the constant memory variables on GPU
//
// Note        :  1. Adopt the suggested approach for CUDA version >= 5.0
//                2. Invoked by Src_Init_Lightbulb() and, if necessary, Src_SetFunc_Lightbulb()
//                3. SRC_NAUX_LIGHTBULB is defined in Macro.h
//
// Parameter   :  AuxArray_Flt/Int : Auxiliary arrays to be copied to the constant memory
//                DevPtr_Flt/Int   : Pointers to store the addresses of constant memory arrays
//
// Return      :  c_Src_Lightbulb_AuxArray_Flt[], c_Src_Lightbulb_AuxArray_Int[], DevPtr_Flt, DevPtr_Int
//---------------------------------------------------------------------------------------------------
void Src_SetConstMemory_Lightbulb( const double AuxArray_Flt[], const int AuxArray_Int[],
                                   double *&DevPtr_Flt, int *&DevPtr_Int )
{

// copy data to constant memory
   CUDA_CHECK_ERROR(  cudaMemcpyToSymbol( c_Src_Lightbulb_AuxArray_Flt, AuxArray_Flt, SRC_NAUX_LIGHTBULB*sizeof(double) )  );
   CUDA_CHECK_ERROR(  cudaMemcpyToSymbol( c_Src_Lightbulb_AuxArray_Int, AuxArray_Int, SRC_NAUX_LIGHTBULB*sizeof(int   ) )  );

// obtain the constant-memory pointers
   CUDA_CHECK_ERROR(  cudaGetSymbolAddress( (void **)&DevPtr_Flt, c_Src_Lightbulb_AuxArray_Flt )  );
   CUDA_CHECK_ERROR(  cudaGetSymbolAddress( (void **)&DevPtr_Int, c_Src_Lightbulb_AuxArray_Int )  );

} // FUNCTION : Src_SetConstMemory_Lightbulb
#endif // #ifdef __CUDACC__



#ifndef __CUDACC__

//-----------------------------------------------------------------------------------------
// Function    :  Src_Init_Lightbulb
// Description :  Initialize the lightbulb source term
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
void Src_Init_Lightbulb()
{

// set the auxiliary arrays
   Src_SetAuxArray_Lightbulb( Src_Lightbulb_AuxArray_Flt, Src_Lightbulb_AuxArray_Int );

// copy the auxiliary arrays to the GPU constant memory and store the associated addresses
#  ifdef GPU
   Src_SetConstMemory_Lightbulb( Src_Lightbulb_AuxArray_Flt, Src_Lightbulb_AuxArray_Int,
                                 SrcTerms.Lightbulb_AuxArrayDevPtr_Flt, SrcTerms.Lightbulb_AuxArrayDevPtr_Int );
#  else
   SrcTerms.Lightbulb_AuxArrayDevPtr_Flt = Src_Lightbulb_AuxArray_Flt;
   SrcTerms.Lightbulb_AuxArrayDevPtr_Int = Src_Lightbulb_AuxArray_Int;
#  endif

// set the major source-term function
   Src_SetCPUFunc_Lightbulb( SrcTerms.Lightbulb_CPUPtr );

#  ifdef GPU
   Src_SetGPUFunc_Lightbulb( SrcTerms.Lightbulb_GPUPtr );
   SrcTerms.Lightbulb_FuncPtr = SrcTerms.Lightbulb_GPUPtr;
#  else
   SrcTerms.Lightbulb_FuncPtr = SrcTerms.Lightbulb_CPUPtr;
#  endif

} // FUNCTION : Src_Init_Lightbulb



//-----------------------------------------------------------------------------------------
// Function    :  Src_End_Lightbulb
// Description :  Release the resources used by the lightbulb source term
//
// Note        :  1. Invoked by Src_End()
//                2. Add "#ifndef __CUDACC__" since this routine is only useful on CPU
//
// Parameter   :  None
//
// Return      :  None
//-----------------------------------------------------------------------------------------
void Src_End_Lightbulb()
{

// not used by this source term

} // FUNCTION : Src_End_Lightbulb

#endif // #ifndef __CUDACC__



#endif // #if ( MODEL == HYDRO )
