#include "EoS.h"
#include "CUFLU.h"
#include "NuclearEoS.h"

#define SRC_AUX_ESHIFT              0
#define SRC_AUX_DENS2CGS            1
#define SRC_AUX_VSQR2CGS            2
#define SRC_AUX_KELVIN2MEV          3 
#define SRC_AUX_MEV2KELVIN          4
#define SRC_AUX_GAMMA               5
#define SRC_AUX_LB_LNU              6
#define SRC_AUX_LB_TNU              7
#define SRC_AUX_LB_HEATFACTOR       8


// external functions and GPU-related set-up
#ifdef __CUDACC__

#include "CUAPI.h"
#include "CUFLU_Shared_FluUtility.cu"
#include "CUDA_ConstMemory.h"

//extern real (*d_SrcDlepProf_Data)[SRC_DLEP_PROF_NBINMAX];
//extern real  *d_SrcDlepProf_Radius;

#endif // #ifdef __CUDACC__


// local function prototypes
#ifndef __CUDACC__

void Src_SetAuxArray_LightBulb( double [], int [] );
void Src_SetFunc_LightBulb( SrcFunc_t & );
void Src_SetConstMemory_LightBulb( const double AuxArray_Flt[], const int AuxArray_Int[],
                                   double *&DevPtr_Flt, int *&DevPtr_Int );
void Src_PassData2GPU_LightBulb();

#endif



/********************************************************
1. LightBulb source term
   --> Enabled by the runtime option "SRC_LIGHTBULB"

2. This file is shared by both CPU and GPU

   CUSRC_Src_LightBulb.cu -> CPU_Src_LightBulb.cpp

3. Four steps are required to implement a source term

   I.   Set auxiliary arrays
   II.  Implement the source-term function
   III. [Optional] Add the work to be done every time
        before calling the major source-term function
   IV.  Set initialization functions

4. The source-term function must be thread-safe and
   not use any global variable
********************************************************/
// declare as static so that other functions cannot invoke it directly and must use the function pointer
static void Src_LightBulb( real fluid[], const double x, const double y, const double z, const double Time,
                      const int lv, double AuxArray[], const double dt );

// this function pointer may be overwritten by various test problem initializers
void (*Src_LightBulb_Ptr)( real fluid[], const double x, const double y, const double z, const double Time,
                      const int lv, double AuxArray[], const double dt ) = Src_LightBulb;



//-------------------------------------------------------------------------------------------------------
// Function    :  Src_SetAuxArray_LightBulb
// Description :  Set the auxiliary arrays AuxArray_Flt/Int[]
//
// Note        :  1. Invoked by Src_Init_LightBulb()
//                2. AuxArray_Flt/Int[] have the size of SRC_NAUX_DLEP defined in Macro.h (default = 5)
//                3. Add "#ifndef __CUDACC__" since this routine is only useful on CPU
//
// Parameter   :  AuxArray_Flt/Int : Floating-point/Integer arrays to be filled up
//
// Return      :  AuxArray_Flt/Int[]
//-------------------------------------------------------------------------------------------------------
#ifndef __CUDACC__
void Src_SetAuxArray_LightBulb( double AuxArray_Flt[], int AuxArray_Int[] )
{

   AuxArray_Flt[SRC_AUX_ESHIFT            ] = EoS_AuxArray_Flt[NUC_AUX_ESHIFT];
   AuxArray_Flt[SRC_AUX_DENS2CGS          ] = UNIT_D;
   AuxArray_Flt[SRC_AUX_VSQR2CGS          ] = SQR( UNIT_V );
   AuxArray_Flt[SRC_AUX_KELVIN2MEV        ] = Const_kB_eV*1.0e-6;
   AuxArray_Flt[SRC_AUX_MEV2KELVIN        ] = 1.0  / AuxArray_Flt[NUC_AUX_KELVIN2MEV];
   AuxArray_Flt[SRC_AUX_GAMMA             ] = GAMMA;
   AuxArray_Flt[SRC_AUX_LB_LNU            ] = LB_LNU;
   AuxArray_Flt[SRC_AUX_LB_TNU            ] = LB_TNU;
   AuxArray_Flt[SRC_AUX_LB_HEATFACTOR     ] = LB_HEATFACTOR;

   AuxArray_Int[NUC_AUX_NRHO      ] = g_nrho;
   AuxArray_Int[NUC_AUX_NTEMP     ] = g_ntemp;
   AuxArray_Int[NUC_AUX_NYE       ] = g_nye;
   AuxArray_Int[NUC_AUX_NMODE     ] = g_nmode;

} // FUNCTION : Src_SetAuxArray_LightBulb
#endif // #ifndef __CUDACC__



// ======================================
// II. Implement the source-term function
// ======================================

//-------------------------------------------------------------------------------------------------------
// Function    :  Src_LightBulb
// Description :  User-defined source terms
//
// Note        :  1. Invoked by Src_AdvanceDt() using the function pointer "Src_LigB_Ptr"
//                   --> The function pointer may be reset by various test problem initializers, in which case
//                       this funtion will become useless
//                2. Enabled by the runtime option "SRC_USER"
//
// Parameter   :  fluid    : Fluid array storing both the input and updated values
//                           --> Array size = NCOMP_TOTAL (so it includes both active and passive variables)
//                x/y/z    : Target physical coordinates
//                Time     : Target physical time
//                lv       : Target refinement level
//                AuxArray : Auxiliary array
//                dt       : Time interval to advance solution
//
// Return      :  fluid[]
//-------------------------------------------------------------------------------------------------------
GPU_DEVICE_NOINLINE
void Src_LightBulb( real fluid[], const real B[],
                    const SrcTerms_t *SrcTerms, const real dt, const real dh,
                    const double x, const double y, const double z,
                    const double TimeNew, const double TimeOld,
                    const real MinDens, const real MinPres, const real MinEint,
                    const EoS_t *EoS, const double AuxArray_Flt[], const int AuxArray_Int[] )
//real fluid[], const double x, const double y, const double z, const double Time,
 //              const int lv, double AuxArray[], const double dt )
{

// check
#  ifdef GAMER_DEBUG
   if ( AuxArray_Flt == NULL )   printf( "ERROR : AuxArray_Flt == NULL in %s !!\n", __FUNCTION__ );
   if ( AuxArray_Int == NULL )   printf( "ERROR : AuxArray_Int == NULL in %s !!\n", __FUNCTION__ );
#  endif

// example
   /*
   const double CoolRate = 1.23; // set arbitrarily here
   double Ek, Eint;              // kinetic and internal energies

   Ek    = (real)0.5*( SQR(fluid[MOMX]) + SQR(fluid[MOMY]) + SQR(fluid[MOMZ]) ) / fluid[DENS];
   Eint  = fluid[ENGY] - Ek;
   Eint -= CoolRate*dt;
   fluid[ENGY] = Ek + Eint;
   */

   const real EnergyShift = AuxArray_Flt[SRC_AUX_ESHIFT    ];
   const real Dens2CGS    = AuxArray_Flt[SRC_AUX_DENS2CGS  ];
   const real sEint2CGS   = AuxArray_Flt[SRC_AUX_VSQR2CGS  ];
   const real MeV2Kelvin  = AuxArray_Flt[SRC_AUX_MEV2KELVIN];

   const int  NRho        = AuxArray_Int[NUC_AUX_NRHO ];
   const int  NTemp       = AuxArray_Int[NUC_AUX_NTEMP];
   const int  NYe         = AuxArray_Int[NUC_AUX_NYE  ];
   const int  NMode       = AuxArray_Int[NUC_AUX_NMODE];

   //const double mev_to_kelvin = 1.1604447522806e10; 
   
   //double xdens,dens, ener, entr, ye;
   //double xtmp, xenr, xprs, xent, xcs2, xdedt, xdpderho, xdpdrhoe, xmunu;
   real radius, xc, yc, zc;

#ifdef FLOAT8
   const real Tolerance = 1e-10;
#else
   const real Tolerance = 1e-6;
#endif

   real xXp, xXn;
   double dEneut, T6;

   const real  BoxCenter[3] = { 0.5*amr->BoxSize[0], 0.5*amr->BoxSize[1], 0.5*amr->BoxSize[2] }; 

   real logd, logt;
   real res[19];

   const real Gamma_m1 = GAMMA - 1.0;

   if (!EOS_POSTBOUNCE) 
   {
        return;
   }

   dEneut = 0.0;

   // code units
   real Dens_Code = fluid[DENS];
   real Dens_CGS  = Dens_Code * Dens2CGS;
   real Ye        = fluid[YE - NCOMP_FLUID] / Dens_Code;
   real Eint_Code = fluid[ENGY];
   Eint_Code      = Eint_Code - 0.5*( SQR(fluid[MOMX]) + SQR(fluid[MOMY]) + SQR(fluid[MOMZ]) )/Dens_Code;
   real sEint_CGS = ( Eint_Code * sEint2CGS / Dens_Code ) - EnergyShift; // specific internal energy

   real Temp_MeV = 10.0; // trial value [MeV]

   real Useless  = NULL_REAL;
   int Err = NULL_INT;
   nuc_eos_C_short( Dens_CGS, &Temp_MeV, Ye, &sEint_CGS, &Useless, &Useless, &Useless, &Useless,
                    EnergyShift, NRho, NTemp, NYe, NMode,
                    EoS.Table[NUC_TAB_ALL], EoS.Table[NUC_TAB_ALL_MODE], EoS.Table[NUC_TAB_RHO], EoS.Table[NUC_TAB_TEMP],
                    EoS.Table[NUC_TAB_YE],  EoS.Table[NUC_TAB_ENGY_MODE], EoS.Table[NUC_TAB_ENTR_MODE], EoS.Table[NUC_TAB_PRES_MODE],
                    0, &Err, Tolerance );

//       trigger a *hard failure* if the EoS driver fails
         if ( Err )
         {
            sEint_CGS = NAN;
            Entr      = NAN;
            Pres_CGS  = NAN;
         }


   real logd = MIN(MAX(Dens_CGS, EoS.Table[NUC_TAB_RHO] [0]), EoS.Table[NUC_TAB_RHO ][NRho] );
   real logt = MIN(MAX(Temp_MeV, EoS.Table[NUC_TAB_TEMP][0]), EoS.Table[NUC_TAB_TEMP][NTEMP]);

   logd = LOG10(logd);
   logt = LOG10(logt);


   real res[16];
   // find xp xn
   nuc_eos_C_linterp_some( logd, logt, Ye, res, EoS.Table[NUC_TAB_ALL], NRho, NTemp, NYe, 
                           16, EoS.Table[NUC_TAB_RHO], EoS.Table[NUC_TAB_TEMP], EoS.Table[NUC_TAB_YE] );
   xXn = res[11];
   xXp = res[12];

   //printf("debug: xXp %13.7e  xXn %13.7e \n", xXp, xXn);

   xc = x - BoxCenter[0]; // [code unit]
   yc = y - BoxCenter[1];
   zc = z - BoxCenter[2];

   radius = SQRT(xc*xc + yc*yc + zc*zc);
   radius = radius * UNIT_L; // [cm]

   // calculate heating
   dEneut = 1.544e20 * (LB_LNU/1.e52) * SQR(1.e7 / radius) * SQR(LB_TNU / 4.);

   // now subtract cooling 
   T6 = (0.5*Temp_MeV)*(0.5*Temp_MeV)*(0.5*Temp_MeV)*(0.5*Temp_MeV)*(0.5*Temp_MeV)*(0.5*Temp_MeV);
   dEneut = dEneut - 1.399e20 * T6;

   dEneut = dEneut * EXP(-Dens_CGS*1.e-11);
   dEneut = dEneut * (xXp + xXn);  // [cgs]

   sEint_CGS = sEint_CGS + dEneut * (UNIT_T * dt);

   fluid[ENGY] = (Eint_Code/sEint2CGS)*(sEint_CGS + EnergyShift) + 0.5*( SQR(fluid[MOMX]) + SQR(fluid[MOMY]) + SQR(fluid[MOMZ]) ) / fluid[DENS]; 

   nuc_eos_C_short( Dens_CGS, &Temp_MeV, Ye, &sEint_CGS, &Entr_CGS, &Useless, &Useless, &Useless,
                    EnergyShift, NRho, NTemp, NYe, NMode,
                    EoS.Table[NUC_TAB_ALL], EoS.Table[NUC_TAB_ALL_MODE], EoS.Table[NUC_TAB_RHO], EoS.Table[NUC_TAB_TEMP],
                    EoS.Table[NUC_TAB_YE],  EoS.Table[NUC_TAB_ENGY_MODE], EoS.Table[NUC_TAB_ENTR_MODE], EoS.Table[NUC_TAB_PRES_MODE],
                    0, &Err, Tolerance );

   // update entropy using the new energy
   fluid[ENTR] = Dens_Code * Entr_CGS; 
   fluid[YE]   = Dens_Code * Ye;  // lb doesn't change ye

//   // if using Dual energy
//#  ifdef DUAL_ENERGY
//#  if (DUAL_ENERGY == DE_ENPY)
//   nuc_eos_C_short(xdens,&Temp_MeV,ye,&xenr, &xprs, &xent, &xcs2, &xdedt, &xdpderho,
//                     &xdpdrhoe, &xmunu, 0, &keyerr, rfeps); // energy mode
//   xprs = xprs * UNIT_P;
//   fluid[ENPY] = Hydro_DensPres2Entropy( dens, xprs, Gamma_m1 );
//#  endif
//#  endif

} // FUNCTION : Src_Lightbulb


// ================================
// IV. Set initialization functions
// ================================

#ifdef __CUDACC__
#  define FUNC_SPACE __device__ static
#else
#  define FUNC_SPACE            static
#endif

FUNC_SPACE SrcFunc_t SrcFunc_Ptr = Src_LightBulb;

//-----------------------------------------------------------------------------------------
// Function    :  Src_SetFunc_LightBulb
// Description :  Return the function pointer of the CPU/GPU source-term function
//
// Note        :  1. Invoked by Src_Init_LightBulb()
//                2. Call-by-reference
//                3. Use either CPU or GPU but not both of them
//
// Parameter   :  SrcFunc_CPU/GPUPtr : CPU/GPU function pointer to be set
//
// Return      :  SrcFunc_CPU/GPUPtr
//-----------------------------------------------------------------------------------------
#ifdef __CUDACC__
__host__
void Src_SetFunc_LightBulb( SrcFunc_t &SrcFunc_GPUPtr )
{
   CUDA_CHECK_ERROR(  cudaMemcpyFromSymbol( &SrcFunc_GPUPtr, SrcFunc_Ptr, sizeof(SrcFunc_t) )  );
}

#elif ( !defined GPU )

void Src_SetFunc_LightBulb( SrcFunc_t &SrcFunc_CPUPtr )
{
   SrcFunc_CPUPtr = SrcFunc_Ptr;
}

#endif // #ifdef __CUDACC__ ... elif ...



#ifdef __CUDACC__
//-------------------------------------------------------------------------------------------------------
// Function    :  Src_SetConstMemory_LightBulb
// Description :  Set the constant memory variables on GPU
//
// Note        :  1. Adopt the suggested approach for CUDA version >= 5.0
//                2. Invoked by Src_Init_LightBulb() and, if necessary, Src_WorkBeforeMajorFunc_LightBulb()
//                3. SRC_NAUX_USER is defined in Macro.h
//
// Parameter   :  AuxArray_Flt/Int : Auxiliary arrays to be copied to the constant memory
//                DevPtr_Flt/Int   : Pointers to store the addresses of constant memory arrays
//
// Return      :  c_Src_LigB_AuxArray_Flt[], c_Src_LigB_AuxArray_Int[], DevPtr_Flt, DevPtr_Int
//---------------------------------------------------------------------------------------------------
void Src_SetConstMemory_LightBulb( const double AuxArray_Flt[], const int AuxArray_Int[],
                                       double *&DevPtr_Flt, int *&DevPtr_Int )
{

// copy data to constant memory
   CUDA_CHECK_ERROR(  cudaMemcpyToSymbol( c_Src_LigB_AuxArray_Flt, AuxArray_Flt, SRC_NAUX_LIGB*sizeof(double) )  );
   CUDA_CHECK_ERROR(  cudaMemcpyToSymbol( c_Src_LigB_AuxArray_Int, AuxArray_Int, SRC_NAUX_LIGB*sizeof(int   ) )  );

// obtain the constant-memory pointers
   CUDA_CHECK_ERROR(  cudaGetSymbolAddress( (void **)&DevPtr_Flt, c_Src_LigB_AuxArray_Flt) );
   CUDA_CHECK_ERROR(  cudaGetSymbolAddress( (void **)&DevPtr_Int, c_Src_LigB_AuxArray_Int) );

} // FUNCTION : Src_SetConstMemory_LightBulb
#endif // #ifdef __CUDACC__



#ifndef __CUDACC__

// function pointer
extern void (*Src_WorkBeforeMajorFunc_LightBulb_Ptr)( const int lv, const double TimeNew, const double TimeOld, const double dt,
                                                      double AuxArray_Flt[], int AuxArray_Int[] );
extern void (*Src_End_LightBulb_Ptr)();

//-----------------------------------------------------------------------------------------
// Function    :  Src_Init_LightBulb
// Description :  Initialize a user-specified source term
//
// Note        :  1. Set auxiliary arrays by invoking Src_SetAuxArray_*()
//                   --> Copy to the GPU constant memory and store the associated addresses
//                2. Set the source-term function by invoking Src_SetFunc_*()
//                   --> Unlike other modules (e.g., EoS), here we use either CPU or GPU but not
//                       both of them
//                3. Set the function pointers "Src_WorkBeforeMajorFunc_LigB_Ptr" and "Src_End_LigB_Ptr"
//                4. Invoked by Src_Init()
//                   --> Enable it by linking to the function pointer "Src_Init_LigB_Ptr"
//                5. Add "#ifndef __CUDACC__" since this routine is only useful on CPU
//
// Parameter   :  None
//
// Return      :  None
//-----------------------------------------------------------------------------------------
void Src_Init_LightBulb()
{

// set the auxiliary arrays
   Src_SetAuxArray_LightBulb( Src_LigB_AuxArray_Flt, Src_LigB_AuxArray_Int );

// copy the auxiliary arrays to the GPU constant memory and store the associated addresses
#  ifdef GPU
   Src_SetConstMemory_LightBulb( Src_LigB_AuxArray_Flt, Src_LigB_AuxArray_Int,
                                     SrcTerms.LigB_AuxArrayDevPtr_Flt, SrcTerms.LigB_AuxArrayDevPtr_Int );
#  else
   SrcTerms.LigB_AuxArrayDevPtr_Flt = Src_LigB_AuxArray_Flt;
   SrcTerms.LigB_AuxArrayDevPtr_Int = Src_LigB_AuxArray_Int;
#  endif

// set the major source-term function
   Src_SetFunc_LightBulb( SrcTerms.LigB_FuncPtr );

// set the auxiliary functions
   Src_WorkBeforeMajorFunc_LigB_Ptr = Src_WorkBeforeMajorFunc_LightBulb;
   Src_End_LigB_Ptr                 = Src_End_LightBulb;

} // FUNCTION : Src_Init_LightBulb



//-----------------------------------------------------------------------------------------
// Function    :  Src_End_LightBulb
// Description :  Free the resources used by a user-specified source term
//
// Note        :  1. Invoked by Src_End()
//                   --> Enable it by linking to the function pointer "Src_End_LigB_Ptr"
//                2. Add "#ifndef __CUDACC__" since this routine is only useful on CPU
//
// Parameter   :  None
//
// Return      :  None
//-----------------------------------------------------------------------------------------
void Src_End_LightBulb()
{


} // FUNCTION : Src_End_LightBulb

#endif // #ifndef __CUDACC__
