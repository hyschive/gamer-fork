#include "CUFLU.h"
#include "Global.h"
#include "NuclearEoS.h"
#include "Src_Leakage.h"

#if ( MODEL == HYDRO )



// external functions and GPU-related set-up
#ifdef __CUDACC__

#include "CUDA_CheckError.h"
#include "CUFLU_Shared_FluUtility.cu"
#include "CUDA_ConstMemory.h"
#include "CUSRC_Src_Leakage_ComputeLeak.cu"

extern real *d_SrcLeakage_Radius;
extern real *d_SrcLeakage_tau;
extern real *d_SrcLeakage_chi;
extern real *d_SrcLeakage_HeatFlux;
extern real *d_SrcLeakage_HeatERms;
extern real *d_SrcLeakage_HeatEAve;

#endif // #ifdef __CUDACC__


// external variables, local and external function prototypes
#ifndef __CUDACC__

// construct the leakage profiles sequentially to minimize memory usage
// --> avoid bad_alloc errors when invoking Aux_AllocateArray3D() in Aux_ComputeRay()
//#define LEAKAGE_SEQUENTIAL_MAPPING

static long Leakage_Step = -1;

#if ( EOS == EOS_NUCLEAR )
extern int g_nye;
#endif

void   Src_SetAuxArray_Leakage( double [], int [] );
void   Src_SetCPUFunc_Leakage( SrcFunc_t & );
#ifdef GPU
void   Src_SetGPUFunc_Leakage( SrcFunc_t & );
#endif
void   Src_SetConstMemory_Leakage( const double AuxArray_Flt[], const int AuxArray_Int[],
                                   double *&DevPtr_Flt, int *&DevPtr_Int );
void   Src_PassData2GPU_Leakage();
double Src_Leakage_ConstructSeries( const int NBin, const double xmin, const double xmax, const double dx );

extern void Src_Leakage_ComputeTau( Profile_t *Ray[], double *Edge,
                                    const int NRadius, const int NTheta, const int NPhi,
                                    real *tau_Ruff, real *chi_Ross, real *Heat_Flux,
                                    real *Heat_ERms, real *Heat_Eave );
extern void Src_Leakage_ComputeLeak( const real Dens_Code, const real Temp_Kelv, const real Ye, const real chi[], const real tau[],
                                     const real Heat_Flux[], const real *Heat_ERms, const real *Heat_EAve,
                                     real *dEdt, real *dYedt, real *Lum, real *Heat, real *NetHeat,
                                     const bool NuHeat, const real NuHeat_Fac, const real UNIT_D, const EoS_t *EoS );

#endif // #ifndef __CUDACC__


GPU_DEVICE static
int  Src_Leakage_BinarySearch( const real Array[], int Min, int Max, const real Key );

GPU_DEVICE static
real Src_Leakage_LinearInterp( const real *array, const real *xs, const real x_in );

GPU_DEVICE static
real Src_Leakage_BilinearInterp( const real *array, const real *xs, const real *ys,
                                 const real x_in, const real y_in );

GPU_DEVICE static
real Src_Leakage_TrilinearInterp( const real *array, const real *xs, const real *ys, const real *zs,
                                  const real x_in, const real y_in, const real z_in );



/********************************************************
1. Leakage source term
   --> Enabled by the runtime option "SRC_LEAKAGE"

2. This file is shared by both CPU and GPU

   CUSRC_Src_Leakage.cu -> CPU_Src_Leakage.cpp

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
// Function    :  Src_SetAuxArray_Leakage
// Description :  Set the auxiliary arrays AuxArray_Flt/Int[]
//
// Note        :  1. Invoked by Src_Init_Leakage()
//                2. AuxArray_Flt/Int[] have the size of SRC_NAUX_LEAKAGE defined in Macro.h (default = 11)
//                3. Add "#ifndef __CUDACC__" since this routine is only useful on CPU
//
// Parameter   :  AuxArray_Flt/Int : Floating-point/Integer arrays to be filled up
//
// Return      :  AuxArray_Flt/Int[]
//-------------------------------------------------------------------------------------------------------
#ifndef __CUDACC__
void Src_SetAuxArray_Leakage( double AuxArray_Flt[], int AuxArray_Int[] )
{

   const int    RayConfig = (  (SrcTerms.Leakage_NTheta == 1) << 1  ) + (SrcTerms.Leakage_NPhi == 1);
   const int    NRay      = SrcTerms.Leakage_NTheta * SrcTerms.Leakage_NPhi;
   const double Ye_Frac   = 0.01;

   AuxArray_Flt[SRC_AUX_PNS_X     ] = amr->BoxCenter[0];
   AuxArray_Flt[SRC_AUX_PNS_Y     ] = amr->BoxCenter[1];
   AuxArray_Flt[SRC_AUX_PNS_Z     ] = amr->BoxCenter[2];
   AuxArray_Flt[SRC_AUX_MAXRADIUS ] = SrcTerms.Leakage_RadiusMax;
   AuxArray_Flt[SRC_AUX_RADMIN_LOG] = int( SrcTerms.Leakage_RadiusMin_Log / SrcTerms.Leakage_BinSize_Radius )
                                    * SrcTerms.Leakage_BinSize_Radius;
   AuxArray_Flt[SRC_AUX_DRAD      ] = SrcTerms.Leakage_BinSize_Radius;
   AuxArray_Flt[SRC_AUX_DTHETA    ] = M_PI / SrcTerms.Leakage_NTheta;
   AuxArray_Flt[SRC_AUX_DPHI      ] = 2.0 * M_PI / SrcTerms.Leakage_NPhi;
#  if ( EOS == EOS_NUCLEAR )
   AuxArray_Flt[SRC_AUX_YEMIN     ] = ( 1.0 + Ye_Frac ) * h_EoS_Table[NUC_TAB_YE][0];
   AuxArray_Flt[SRC_AUX_YEMAX     ] = ( 1.0 - Ye_Frac ) * h_EoS_Table[NUC_TAB_YE][g_nye-1];
#  endif
   AuxArray_Flt[SRC_AUX_VSQR2CODE ] = 1.0 / SQR( UNIT_V );

   AuxArray_Int[SRC_AUX_RAYCONF   ] = RayConfig;
   AuxArray_Int[SRC_AUX_NRAD_LIN  ] = int( SrcTerms.Leakage_RadiusMin_Log / SrcTerms.Leakage_BinSize_Radius );
   AuxArray_Int[SRC_AUX_STRIDE    ] = NRay * NType_Neutrino;
   AuxArray_Int[SRC_AUX_MODE      ] = LEAK_MODE_EVOLVE;

} // FUNCTION : Src_SetAuxArray_Leakage
#endif // #ifndef __CUDACC__



// ======================================
// II. Implement the source-term function
// ======================================

//-------------------------------------------------------------------------------------------------------
// Function    :  Src_Leakage
// Description :  Apply the leakage scheme for neutrino heating and cooling, and the change in electron fraction
//
// Note        :  1. Invoked by CPU/GPU_SrcSolver_IterateAllCells()
//                2. See Src_SetAuxArray_Leakage() for the values stored in AuxArray_Flt/Int[]
//                3. Shared by both CPU and GPU
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
static void Src_Leakage( real fluid[], const real B[],
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


   const real PNS_x      = AuxArray_Flt[SRC_AUX_PNS_X     ];
   const real PNS_y      = AuxArray_Flt[SRC_AUX_PNS_Y     ];
   const real PNS_z      = AuxArray_Flt[SRC_AUX_PNS_Z     ];
   const real MaxRadius  = AuxArray_Flt[SRC_AUX_MAXRADIUS ];
   const real RadMin_Log = AuxArray_Flt[SRC_AUX_RADMIN_LOG];
   const real dRad       = AuxArray_Flt[SRC_AUX_DRAD      ];
   const real dTht       = AuxArray_Flt[SRC_AUX_DTHETA    ];
   const real dPhi       = AuxArray_Flt[SRC_AUX_DPHI      ];
   const real YeMin      = AuxArray_Flt[SRC_AUX_YEMIN     ];
   const real YeMax      = AuxArray_Flt[SRC_AUX_YEMAX     ];
   const real sEint2Code = AuxArray_Flt[SRC_AUX_VSQR2CODE ];

   const int  RayConfig  = AuxArray_Int[SRC_AUX_RAYCONF   ];
   const int  NRad_Lin   = AuxArray_Int[SRC_AUX_NRAD_LIN  ];
   const int  Stride     = AuxArray_Int[SRC_AUX_STRIDE    ];
   const int  Mode       = AuxArray_Int[SRC_AUX_MODE      ];


   const int  NRadius = SrcTerms->Leakage_NRadius;
   const int  NTheta  = SrcTerms->Leakage_NTheta;
   const int  NPhi    = SrcTerms->Leakage_NPhi;
#  ifdef __CUDACC__
   const real *Radius = SrcTerms->Leakage_Radius_DevPtr;
#  else
   const real *Radius = h_SrcLeakage_Radius;
#  endif

   const double x0  = x - PNS_x;
   const double y0  = y - PNS_y;
   const double z0  = z - PNS_z;
   const double Rad = sqrt( SQR(x0) + SQR(y0) + SQR(z0) );
         double Tht, Phi;
         real   xs[2], ys[2], zs[2];
         int    Idx_Rad, Idx_Tht, Idx_Phi;
         int    NPhi_half = NPhi>>1;


// do nothing if the cell is beyond the sampled rays
   if ( Rad > MaxRadius )
   {
      if ( Mode == LEAK_MODE_RECORD )
      {
         fluid[DENS] = (real)0.0;
         fluid[MOMX] = (real)0.0;
         fluid[MOMY] = (real)0.0;
         fluid[MOMZ] = (real)0.0;
         fluid[ENGY] = (real)0.0;
#        ifdef YE
         fluid[YE  ] = (real)0.0;
#        endif
      }

#     ifdef DYEDT_NU
      fluid[DEDT_NU ] = (real)0.0;
      fluid[DYEDT_NU] = (real)0.0;
#     endif

      return;
   }


// (1) set up the data index and coordinate
// (1-1) spherical radius:
//       --> use data at boundary for cells outside of the radius range
//       --> for Rad <= RadMin_Log, add MIN( ..., NRadius-2 ) to deal with the special case RadMin_Log = MaxRadius
   Idx_Rad = ( Rad <= RadMin_Log )
           ? MIN(  MAX( int( Rad / dRad - (real)0.5 ), 0 ), NRadius-2  )
           : Src_Leakage_BinarySearch( Radius, NRad_Lin-1, NRadius-2, Rad );

   xs[0] = Radius[Idx_Rad  ];
   xs[1] = Radius[Idx_Rad+1];

// (1-2) polar angle, theta:
//       --> for cells close to the pole (Tht < 0.5 * dTht or Tht > PI - 0.5 * dTht)
//           (a) NPhi > 1, use data at Phi + PI for interpolation
//               --> Idx_Tht = -1                       if Tht <      0.5 * dTht
//                             NTheta - 1               if Tht > PI - 0.5 * dTht
//                             int( Tht / dTht - 0.5 )  otherwise
//           (b) NPhi = 1, use data at boundary
//               --> Idx_Tht = 0                        if Tht <      0.5 * dTht
//                             NTheta - 2               if Tht > PI - 0.5 * dTht
//                             int( Tht / dTht - 0.5 )  otherwise
   if ( NTheta > 1 )
   {
//    set theta = 0.5 * PI arbitrarily if the cell is at the reference center
      Tht     = ( Rad == 0.0 )
              ? 0.5 * M_PI
              : acos( z0 / Rad );
      Idx_Tht = ( NPhi > 1 )
              ? int( Tht / dTht + 0.5 ) - 1
              : MIN(  MAX( int( Tht / dTht - 0.5 ), 0 ), NTheta-2  );

      ys[0] = ( (real)0.5 + (real)Idx_Tht ) * dTht;
      ys[1] = ( (real)1.5 + (real)Idx_Tht ) * dTht;
   }

// (1-3) azimuthal angle, phi:
//       --> use periodic boundary condition
//           --> Idx_Phi = NPhi - 1                 if Phi < 0.5 * dPhi or Phi > 2 * PI - 0.5 * dPhi
//                         int( Phi / dPhi - 0.5 )  otherwise
   if ( NPhi > 1 )
   {
      Phi     = (  ( x0 == 0.0 )  &&  ( y0 == 0.0 )  )
              ? 0.0
              : (  ( y0 >= 0.0 ) ? atan2( y0, x0 ) : atan2( y0, x0 ) + 2.0 * M_PI  );
      Idx_Phi = int( Phi / dPhi + 0.5 ) - 1;

//    deal with the case Phi < 0.5 * dPhi
      if ( Idx_Phi == -1 )  {  Idx_Phi += NPhi;  Phi += 2.0 * M_PI;  }

      zs[0] = ( (real)0.5 + (real)Idx_Phi ) * dPhi;
      zs[1] = ( (real)1.5 + (real)Idx_Phi ) * dPhi;
   }


// (2-1) prepare the leakage data using linear interpolation
#  ifdef __CUDACC__
   real *Ray_tau  = SrcTerms->Leakage_tau_DevPtr;
   real *Ray_chi  = SrcTerms->Leakage_chi_DevPtr;
   real *Ray_Flux = SrcTerms->Leakage_HeatFlux_DevPtr;
   real *Ray_ERms = SrcTerms->Leakage_HeatERms_DevPtr;
   real *Ray_EAve = SrcTerms->Leakage_HeatEAve_DevPtr;
#  else
   real *Ray_tau  = h_SrcLeakage_tau;
   real *Ray_chi  = h_SrcLeakage_chi;
   real *Ray_Flux = h_SrcLeakage_HeatFlux;
   real *Ray_ERms = h_SrcLeakage_HeatERms;
   real *Ray_EAve = h_SrcLeakage_HeatEAve;
#  endif

   real tau[NType_Neutrino], chi[NType_Neutrino], Heat_Flux[NType_Neutrino];
   real Heat_EAve[NType_Neutrino], Heat_ERms[NType_Neutrino];

   switch ( RayConfig )
   {
      case 0: // NTheta != 1, NPhi != 1
      {
         real tau_tmp[8], chi_tmp[8], Flux_tmp[8], ERms_tmp[4], EAve_tmp[4];

         for (int n=0; n<NType_Neutrino; n++)
         {
            int Idx_Out_2D = 0;
            int Idx_Out_3D = 0;

            for (int k=Idx_Phi; k<=Idx_Phi+1; k++              )
            for (int j=Idx_Tht; j<=Idx_Tht+1; j++, Idx_Out_2D++)
            {
               const int jj   = ( j >= 0 )
                              ? ( j == NTheta ? NTheta-1 : j )
                              : 0;
               const int kk   = ( j == -1  ||  j == NTheta )
                              ? k + NPhi_half
                              : k;

               const int IRay      = jj + NTheta * ( kk % NPhi );
               const int Idx_In_2D = n + IRay * NType_Neutrino;

               for (int i=Idx_Rad; i<=Idx_Rad+1; i++, Idx_Out_3D++)
               {
                  const int Idx_In_3D = Idx_In_2D + i * Stride;

                  tau_tmp [Idx_Out_3D] = Ray_tau [Idx_In_3D];
                  chi_tmp [Idx_Out_3D] = Ray_chi [Idx_In_3D];
                  Flux_tmp[Idx_Out_3D] = Ray_Flux[Idx_In_3D];
               }

               ERms_tmp[Idx_Out_2D] = Ray_ERms[Idx_In_2D];
               EAve_tmp[Idx_Out_2D] = Ray_EAve[Idx_In_2D];
            }

            tau      [n] = Src_Leakage_TrilinearInterp( tau_tmp , xs, ys, zs, Rad, Tht, Phi );
            chi      [n] = Src_Leakage_TrilinearInterp( chi_tmp , xs, ys, zs, Rad, Tht, Phi );
            Heat_Flux[n] = Src_Leakage_TrilinearInterp( Flux_tmp, xs, ys, zs, Rad, Tht, Phi );
            Heat_ERms[n] = Src_Leakage_BilinearInterp ( ERms_tmp,     ys, zs,      Tht, Phi );
            Heat_EAve[n] = Src_Leakage_BilinearInterp ( EAve_tmp,     ys, zs,      Tht, Phi );
         } // for (int n=0; n<NType_Neutrino; n++)
      }
      break; // case 0


      case 1: // NTheta != 1, NPhi == 1
      {
         real tau_tmp[4], chi_tmp[4], Flux_tmp[4], ERms_tmp[2], EAve_tmp[2];

         for (int n=0; n<NType_Neutrino; n++)
         {
            int Idx_Out_2D = 0;
            int Idx_Out_3D = 0;

            for (int j=Idx_Tht; j<=Idx_Tht+1; j++, Idx_Out_2D++)
            {
               const int IRay      = j;
               const int Idx_In_2D = n + IRay * NType_Neutrino;

               for (int i=Idx_Rad; i<=Idx_Rad+1; i++, Idx_Out_3D++)
               {
                  const int Idx_In_3D = Idx_In_2D + i * Stride;

                  tau_tmp [Idx_Out_3D] = Ray_tau [Idx_In_3D];
                  chi_tmp [Idx_Out_3D] = Ray_chi [Idx_In_3D];
                  Flux_tmp[Idx_Out_3D] = Ray_Flux[Idx_In_3D];
               }

               ERms_tmp[Idx_Out_2D] = Ray_ERms[Idx_In_2D];
               EAve_tmp[Idx_Out_2D] = Ray_EAve[Idx_In_2D];
            }

            tau      [n] = Src_Leakage_BilinearInterp( tau_tmp , xs, ys, Rad, Tht );
            chi      [n] = Src_Leakage_BilinearInterp( chi_tmp , xs, ys, Rad, Tht );
            Heat_Flux[n] = Src_Leakage_BilinearInterp( Flux_tmp, xs, ys, Rad, Tht );
            Heat_ERms[n] = Src_Leakage_LinearInterp  ( ERms_tmp,     ys,      Tht );
            Heat_EAve[n] = Src_Leakage_LinearInterp  ( EAve_tmp,     ys,      Tht );
         } // for (int n=0; n<NType_Neutrino; n++)
      }
      break; // case 1


      case 2: // NTheta == 1, NPhi != 1
      {
         real tau_tmp[4], chi_tmp[4], Flux_tmp[4], ERms_tmp[2], EAve_tmp[2];

         for (int n=0; n<NType_Neutrino; n++)
         {
            int Idx_Out_2D = 0;
            int Idx_Out_3D = 0;

            for (int j=Idx_Phi; j<=Idx_Phi+1; j++, Idx_Out_2D++)
            {
               const int IRay      = j % NPhi;
               const int Idx_In_2D = n + IRay * NType_Neutrino;

               for (int i=Idx_Rad; i<=Idx_Rad+1; i++, Idx_Out_3D++)
               {
                  const int Idx_In_3D = Idx_In_2D + i * Stride;

                  tau_tmp [Idx_Out_3D] = Ray_tau [Idx_In_3D];
                  chi_tmp [Idx_Out_3D] = Ray_chi [Idx_In_3D];
                  Flux_tmp[Idx_Out_3D] = Ray_Flux[Idx_In_3D];
               }

               ERms_tmp[Idx_Out_2D] = Ray_ERms[Idx_In_2D];
               EAve_tmp[Idx_Out_2D] = Ray_EAve[Idx_In_2D];
            }

            tau      [n] = Src_Leakage_BilinearInterp( tau_tmp , xs, zs, Rad, Phi );
            chi      [n] = Src_Leakage_BilinearInterp( chi_tmp , xs, zs, Rad, Phi );
            Heat_Flux[n] = Src_Leakage_BilinearInterp( Flux_tmp, xs, zs, Rad, Phi );
            Heat_ERms[n] = Src_Leakage_LinearInterp  ( ERms_tmp,     zs,      Phi );
            Heat_EAve[n] = Src_Leakage_LinearInterp  ( EAve_tmp,     zs,      Phi );
         } // for (int n=0; n<NType_Neutrino; n++)
      }
      break; // case 2


      case 3: // NTheta == 1, NPhi == 1
      {
         real tau_tmp[2], chi_tmp[2], Flux_tmp[2];

         for (int n=0; n<NType_Neutrino; n++)
         {
            const int IRay       = 0;
            const int Idx_In_2D  = n + IRay * NType_Neutrino;
                  int Idx_Out_3D = 0;

            for (int i=Idx_Rad; i<=Idx_Rad+1; i++, Idx_Out_3D++)
            {
               const int Idx_In_3D = Idx_In_2D + i * Stride;

               tau_tmp [Idx_Out_3D] = Ray_tau [Idx_In_3D];
               chi_tmp [Idx_Out_3D] = Ray_chi [Idx_In_3D];
               Flux_tmp[Idx_Out_3D] = Ray_Flux[Idx_In_3D];
            }

            Heat_ERms[n] = Ray_ERms[Idx_In_2D];
            Heat_EAve[n] = Ray_EAve[Idx_In_2D];

            tau      [n] = Src_Leakage_LinearInterp( tau_tmp , xs, Rad );
            chi      [n] = Src_Leakage_LinearInterp( chi_tmp , xs, Rad );
            Heat_Flux[n] = Src_Leakage_LinearInterp( Flux_tmp, xs, Rad );
         } // for (int n=0; n<NType_Neutrino; n++)
      }
      break; // case 3
   } // switch ( RayConfig )


// (2-2) prepare fluid data for the leakage scheme
#  ifdef MHD
   const real Emag = (real)0.5*(  SQR( B[MAGX] ) + SQR( B[MAGY] ) + SQR( B[MAGZ] )  );
#  else
   const real Emag = NULL_REAL;
#  endif

   const real Dens_Code = fluid[DENS];

#  ifdef __CUDACC__
   const real Eint_Code = Hydro_Con2Eint( fluid[DENS], fluid[MOMX], fluid[MOMY], fluid[MOMZ], fluid[ENGY],
                                          true, MinEint, Emag, EoS->GuessHTilde_FuncPtr,
                                          EoS->HTilde2Temp_FuncPtr, EoS->AuxArrayDevPtr_Flt,
                                          EoS->AuxArrayDevPtr_Int, EoS->Table );
#  else
   const real Eint_Code = Hydro_Con2Eint( fluid[DENS], fluid[MOMX], fluid[MOMY], fluid[MOMZ], fluid[ENGY],
                                          true, MinEint, Emag, EoS_GuessHTilde_CPUPtr, EoS_HTilde2Temp_CPUPtr,
                                          EoS_AuxArray_Flt, EoS_AuxArray_Int, h_EoS_Table );
#  endif

#  ifdef YE
   const real Ye = fluid[YE] / fluid[DENS];
#  else
   const real Ye = NULL_REAL;
#  endif

#  ifdef TEMP_IG
   const real Temp_IG_Kelv = fluid[TEMP_IG];
#  else
   const real Temp_IG_Kelv = NULL_REAL;
#  endif

#  if ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )
   const int  NTarget = 0;
#  else
   const int  NTarget = 1;
#  endif
         int  In_Int[NTarget+1];
         real In_Flt[4], Out[NTarget+1];

   In_Flt[0] = Dens_Code;
   In_Flt[1] = Eint_Code;
   In_Flt[2] = Ye;
   In_Flt[3] = Temp_IG_Kelv;

   In_Int[0] = NTarget;
#  if ( NUC_TABLE_MODE == NUC_TABLE_MODE_ENGY )
   In_Int[1] = NUC_VAR_IDX_EORT;
#  endif

#  ifdef __CUDACC__
   EoS->General_FuncPtr( NUC_MODE_ENGY, Out, In_Flt, In_Int, EoS->AuxArrayDevPtr_Flt, EoS->AuxArrayDevPtr_Int, EoS->Table );
#  else
   EoS_General_CPUPtr  ( NUC_MODE_ENGY, Out, In_Flt, In_Int, EoS_AuxArray_Flt,        EoS_AuxArray_Int,        h_EoS_Table );
#  endif

   const real Temp_Kelv = Out[0];


// (3) compute the local heating rates for internal energy and electron fraction
   const bool NuHeat     = SrcTerms->Leakage_NuHeat;
   const real NuHeat_Fac = SrcTerms->Leakage_NuHeat_Fac;
   const real Unit_D     = SrcTerms->Unit_D;
   const real Unit_T     = SrcTerms->Unit_T;
         real Lum[NType_Neutrino], Heat[NType_Neutrino], NetHeat[NType_Neutrino];
         real dEdt_CGS, dYedt;

   Src_Leakage_ComputeLeak( Dens_Code, Temp_Kelv, Ye, chi, tau, Heat_Flux, Heat_ERms, Heat_EAve,
                            &dEdt_CGS, &dYedt, Lum, Heat, NetHeat, NuHeat, NuHeat_Fac, Unit_D, EoS );

// convert the heating rates to code unit
// --> the returned dEdt_CGS is in erg/g/s and dYedt is in 1/s
   const real dEdt_Code  = dEdt_CGS * sEint2Code * Unit_T * Dens_Code;
   const real dYedt_Code = dYedt * Unit_T;


// (4) for recording mode, store the dEdt, luminosity, heating rate, and net heating rate, then return
   if ( Mode == LEAK_MODE_RECORD )
   {
      fluid[DENS    ] = dEdt_CGS * Dens_Code * Unit_D;
      fluid[MOMX    ] = Lum [0];
      fluid[MOMY    ] = Lum [1];
      fluid[MOMZ    ] = Lum [2];
      fluid[ENGY    ] = Heat[0];
#     ifdef YE
      fluid[YE      ] = Heat[1];
#     endif
#     ifdef DYEDT_NU
      fluid[DEDT_NU ] = NetHeat[0];
      fluid[DYEDT_NU] = NetHeat[1];
#     endif

      return;
   }


// (5-1) store the heating rate before the corrections in (5-2)
//       --> for LEAK_MODE_CORRSIGN mode, store the actual sign instead of the absolute value
#  ifdef DYEDT_NU
   fluid[DEDT_NU ] = ( Mode == LEAK_MODE_CORRSIGN ) ? dEdt_Code  : FABS( dEdt_Code  );
   fluid[DYEDT_NU] = ( Mode == LEAK_MODE_CORRSIGN ) ? dYedt_Code : FABS( dYedt_Code );

   if ( Mode == LEAK_MODE_CORRSIGN )   return;
#  endif

// (5-2) check whether the updated Ye is at least 1% away from the table boundary
//       --> skip this cell if it is not
   const real dYe = dYedt_Code * dt;

   if (  ( Ye + dYe < YeMin )  ||  ( Ye + dYe > YeMax )  )
   {
#     ifdef GAMER_DEBUG
      printf( "Invalid heating rate of Ye: Ye=%13.7e, dYedt_Code=%13.7e, dt_Code=%13.7e, YeMin=%13.7e, YeMax=%13.7e\n",
              Ye, dYedt_Code, dt, YeMin, YeMax );
#     endif

      return;
   }

// (5-3) verify whether the updated internal energy density and Ye are valid for the nuclear EoS table
   const real Eint_Update = Eint_Code + dEdt_Code  * dt;
   const real Ye_Update   = Ye        + dYedt_Code * dt;

   In_Flt[1] = Eint_Update;
   In_Flt[2] = Ye_Update;

#  ifdef __CUDACC__
   EoS->General_FuncPtr( NUC_MODE_ENGY, Out, In_Flt, In_Int, EoS->AuxArrayDevPtr_Flt, EoS->AuxArrayDevPtr_Int, EoS->Table );
#  else
   EoS_General_CPUPtr  ( NUC_MODE_ENGY, Out, In_Flt, In_Int, EoS_AuxArray_Flt,        EoS_AuxArray_Int,        h_EoS_Table );
#  endif

   const real Temp_Chk = Out[0];

// (5-4) update the internal energy density and Ye if the EoS solver succeeds (Temp_Chk != NAN)
   if ( Temp_Chk == Temp_Chk )
   {
      fluid[ENGY] = Hydro_ConEint2Etot( fluid[DENS], fluid[MOMX], fluid[MOMY], fluid[MOMZ], Eint_Update, Emag );
#     ifdef YE
      fluid[YE  ] = Ye_Update * fluid[DENS];
#     endif
   }


// (6) final check
#  ifdef GAMER_DEBUG
   if (  Hydro_IsUnphysical( UNPHY_MODE_SING, &Eint_Update, "output internal energy density", (real)0.0,
                             __FLT_MAX__, NULL_REAL, NULL, NULL, NULL, NULL, NULL, NULL,
                             ERROR_INFO, UNPHY_VERBOSE )  )
   {
      printf( "   Dens=%13.7e code units, Eint=%13.7e code units, Ye=%13.7e\n", Dens_Code, Eint_Code, Ye );
      printf( "   Radius=%13.7e cm, Temp=%13.7e Kelvin, dEdt=%13.7e, dYedt=%13.7e\n", Rad * SrcTerms->Unit_L, Temp_Kelv, dEdt_Code, dYedt );

      for (int n=0; n<NType_Neutrino; n++)
      printf( "   n=%d: tau=%13.7e, chi=%13.7e, Heat_Flux=%13.7e, Heat_ERms=%13.7e, Heat_EAve=%13.7e\n",
                  n, tau[n], chi[n], Heat_Flux[n] / Const_hc_MeVcm_CUBE, Heat_ERms[n], Heat_EAve[n] );
   }
#  endif // GAMER_DEBUG

} // FUNCTION : Src_Leakage



// ==================================================
// III. [Optional] Add the work to be done every time
//      before calling the major source-term function
// ==================================================

//-------------------------------------------------------------------------------------------------------
// Function    :  Src_WorkBeforeMajorFunc_Leakage
// Description :  Prepare the leakage profile before calling the major source-term function
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
//                                   --> Must call Src_SetConstMemory_Leakage() after modification
//
// Return      :  AuxArray_Flt/Int[]
//-------------------------------------------------------------------------------------------------------
#ifndef __CUDACC__
void Src_WorkBeforeMajorFunc_Leakage( const int lv, const double TimeNew, const double TimeOld, const double dt,
                                      double AuxArray_Flt[], int AuxArray_Int[] )
{

   const int NRad   = SrcTerms.Leakage_NRadius;
   const int NTheta = SrcTerms.Leakage_NTheta;
   const int NPhi   = SrcTerms.Leakage_NPhi;
   const int NRay   = NTheta * NPhi;


// (0) do nothing if the leakage profile for current step is already constructed
   if ( Step == Leakage_Step )   return;


// (1) set the leakage center to the GREP center, if available
   double Leakage_Center[3];

#  ifdef GRAVITY
   for (int i=0; i<3; i++)   Leakage_Center[i] = GREP_Center[i];
#  else
   for (int i=0; i<3; i++)   Leakage_Center[i] = amr->BoxCenter[i];
#  endif


// (2) set up the coordinate of leakage rays
//     --> in linear scale for                 r <= RadiusMin_Log
//            log              RadiusMin_Log < r <  MaxRadius
   const double BinSize_Linear = AuxArray_Flt[SRC_AUX_DRAD      ];
   const double MaxRadius      = AuxArray_Flt[SRC_AUX_MAXRADIUS ];
   const int    NRad_Linear    = AuxArray_Int[SRC_AUX_NRAD_LIN  ];
   const double RadiusMin_Log  = AuxArray_Flt[SRC_AUX_RADMIN_LOG];

   const double Factor      = Src_Leakage_ConstructSeries( NRad - NRad_Linear, RadiusMin_Log, MaxRadius, BinSize_Linear );
         double BinSize_Log = BinSize_Linear;
         double Edge[NRad+1];

   Edge[0] = 0.0;

   for (int i=1; i<=NRad_Linear; i++)
      Edge[i] = Edge[i-1] + BinSize_Linear;

   for (int i=NRad_Linear+1; i<=NRad; i++)
   {
      BinSize_Log *= Factor;
      Edge[i]      = Edge[i-1] + BinSize_Log;
   }

// compute and store radius in the host array with typecasting
   for (int i=0; i<NRad; i++)   h_SrcLeakage_Radius[i] = 0.5 * ( Edge[i] + Edge[i+1] );


// (3) construct the leakage ray
//     --> prepare data at TimeOld because data on higher level at TimeNew are not available
#  ifndef _YE
   const long _YE = NULL_INT;
#  endif
   const int  NProf  = 3;
   const int  NData  = NRad * NTheta * NPhi;
         long TVar[] = { _DENS, _NONE, _YE };

   Profile_t  Ray_Dens_Code, Ray_Temp_Kelv, Ray_Ye;
   Profile_t *Leakage_Ray[] = { &Ray_Dens_Code, &Ray_Temp_Kelv, &Ray_Ye };

   switch ( SrcTerms.Leakage_Opt_Temp )
   {
      case LEAK_TEMP_INDCELL:   TVar[1] = _TEMP;   break;
      case LEAK_TEMP_BINDATA:   TVar[1] = _EINT;   break;
      default:
         Aux_Error( ERROR_INFO, "unsupported temperature computation scheme %s = %d !!\n",
                    "SRC_LEAKAGE_OPT_TEMP", SrcTerms.Leakage_Opt_Temp );
   } // switch ( SrcTerms.Leakage_Opt_Temp )


#  ifdef LEAKAGE_SEQUENTIAL_MAPPING
   for (int i=0; i<NProf; i++)
      Aux_ComputeRay( &Leakage_Ray[i], Leakage_Center, Edge, NRad_Linear, NRad, NTheta, NPhi,
                      BinSize_Linear, RadiusMin_Log, MaxRadius, &TVar[i], 1, TimeOld );
#  else
   Aux_ComputeRay( Leakage_Ray, Leakage_Center, Edge, NRad_Linear, NRad, NTheta, NPhi,
                   BinSize_Linear, RadiusMin_Log, MaxRadius, TVar, NProf, TimeOld );
#  endif


// retrieve the temperature profile from the density, internal energy density, and Ye
   if ( SrcTerms.Leakage_Opt_Temp == LEAK_TEMP_BINDATA )
   {
#     ifdef YE
      real Passive[NCOMP_PASSIVE] = { 0.0 };

      for (int i=0; i<NData; i++)
      {
         if ( Ray_Temp_Kelv.NCell[i] == 0L )   continue;

         Passive[ YE      - NCOMP_FLUID ] = Ray_Ye.Data[i];

#        ifdef TEMP_IG
//       set the initial guess of temperature to 1 MeV
         Passive[ TEMP_IG - NCOMP_FLUID ] = 1.0e6 / Const_kB_eV;
#        endif

         const real Temp = EoS_DensEint2Temp_CPUPtr( Ray_Dens_Code.Data[i], Ray_Temp_Kelv.Data[i],
                                                     Passive, EoS_AuxArray_Flt, EoS_AuxArray_Int, h_EoS_Table );

         if ( Temp != Temp )
            Aux_Error( ERROR_INFO, "Invalid temperature found in %s: %.7e %.7e %.7e !!\n",
                       __FUNCTION__, Ray_Dens_Code.Data[i], Ray_Temp_Kelv.Data[i], Ray_Ye.Data[i] );

         Ray_Temp_Kelv.Data[i] = Temp;
      }
#     endif // ifdef YE
   } // if ( SrcTerms.Leakage_Opt_Temp == LEAK_TEMP_BINDATA )


// convert electron density to electron fraction
   for (int i=0; i<NData; i++)
      if ( Ray_Ye.NCell[i] != 0L )   Ray_Ye.Data[i] /= Ray_Dens_Code.Data[i];


// (4) compute tau_Ruff, chi_Ross, HeatE_Flux, Heat_ERms, and Heat_EAve
   Src_Leakage_ComputeTau( Leakage_Ray, Edge, NRad, NTheta, NPhi,
                           h_SrcLeakage_tau, h_SrcLeakage_chi, h_SrcLeakage_HeatFlux,
                           h_SrcLeakage_HeatERms, h_SrcLeakage_HeatEAve );


// (5) update AuxArray_Flt and AuxArray_Int
   AuxArray_Flt[SRC_AUX_PNS_X] = Leakage_Center[0];
   AuxArray_Flt[SRC_AUX_PNS_Y] = Leakage_Center[1];
   AuxArray_Flt[SRC_AUX_PNS_Z] = Leakage_Center[2];

#  ifdef GPU
   Src_SetConstMemory_Leakage( AuxArray_Flt, AuxArray_Int,
                               SrcTerms.Leakage_AuxArrayDevPtr_Flt, SrcTerms.Leakage_AuxArrayDevPtr_Int );
#  endif


// (6) pass the leakage data to GPU
#  ifdef GPU
   Src_PassData2GPU_Leakage();
#  endif


// (7) free Profile_t objects
   for (int p=0; p<NProf; p++)   Leakage_Ray[p]->FreeMemory();


// (8) update the time stamp
   Leakage_Step = Step;

} // FUNCTION : Src_WorkBeforeMajorFunc_Leakage
#endif // #ifndef __CUDACC__



#ifdef __CUDACC__
//-------------------------------------------------------------------------------------------------------
// Function    :  Src_PassData2GPU_Leakage
// Description :  Transfer data to GPU
//
// Note        :  1. Invoked by Src_WorkBeforeMajorFunc_Leakage()
//                2. Use synchronous transfer
//
// Parameter   :  None
//
// Return      :  None
//-------------------------------------------------------------------------------------------------------
void Src_PassData2GPU_Leakage()
{

   const int  NRadius = SrcTerms.Leakage_NRadius;
   const int  NTheta  = SrcTerms.Leakage_NTheta;
   const int  NPhi    = SrcTerms.Leakage_NPhi;

   const long Size_Radius   = sizeof(real)*NRadius;
   const long Size_tau      = sizeof(real)*NRadius*NTheta*NPhi*NType_Neutrino;
   const long Size_chi      = sizeof(real)*NRadius*NTheta*NPhi*NType_Neutrino;
   const long Size_HeatFlux = sizeof(real)*NRadius*NTheta*NPhi*NType_Neutrino;
   const long Size_HeatERms = sizeof(real)*        NTheta*NPhi*NType_Neutrino;
   const long Size_HeatEAve = sizeof(real)*        NTheta*NPhi*NType_Neutrino;

// use synchronous transfer
   CUDA_CHECK_ERROR(  cudaMemcpy( d_SrcLeakage_Radius,   h_SrcLeakage_Radius,   Size_Radius,   cudaMemcpyHostToDevice )  );
   CUDA_CHECK_ERROR(  cudaMemcpy( d_SrcLeakage_tau,      h_SrcLeakage_tau,      Size_tau,      cudaMemcpyHostToDevice )  );
   CUDA_CHECK_ERROR(  cudaMemcpy( d_SrcLeakage_chi,      h_SrcLeakage_chi,      Size_chi,      cudaMemcpyHostToDevice )  );
   CUDA_CHECK_ERROR(  cudaMemcpy( d_SrcLeakage_HeatFlux, h_SrcLeakage_HeatFlux, Size_HeatFlux, cudaMemcpyHostToDevice )  );
   CUDA_CHECK_ERROR(  cudaMemcpy( d_SrcLeakage_HeatERms, h_SrcLeakage_HeatERms, Size_HeatERms, cudaMemcpyHostToDevice )  );
   CUDA_CHECK_ERROR(  cudaMemcpy( d_SrcLeakage_HeatEAve, h_SrcLeakage_HeatEAve, Size_HeatEAve, cudaMemcpyHostToDevice )  );

} // FUNCTION : Src_PassData2GPU_Leakage
#endif // #ifdef __CUDACC__



// ================================
// IV. Set initialization functions
// ================================

#ifdef __CUDACC__
#  define FUNC_SPACE __device__ static
#else
#  define FUNC_SPACE            static
#endif

FUNC_SPACE SrcFunc_t SrcFunc_Ptr = Src_Leakage;

//-----------------------------------------------------------------------------------------
// Function    :  Src_SetCPU/GPUFunc_Leakage
// Description :  Return the function pointer of the CPU/GPU source-term function
//
// Note        :  1. Invoked by Src_Init_Leakage()
//                2. Call-by-reference
//
// Parameter   :  SrcFunc_CPU/GPUPtr : CPU/GPU function pointer to be set
//
// Return      :  SrcFunc_CPU/GPUPtr
//-----------------------------------------------------------------------------------------
#ifdef __CUDACC__
__host__
void Src_SetGPUFunc_Leakage( SrcFunc_t &SrcFunc_GPUPtr )
{
   CUDA_CHECK_ERROR(  cudaMemcpyFromSymbol( &SrcFunc_GPUPtr, SrcFunc_Ptr, sizeof(SrcFunc_t) )  );
}

#else

void Src_SetCPUFunc_Leakage( SrcFunc_t &SrcFunc_CPUPtr )
{
   SrcFunc_CPUPtr = SrcFunc_Ptr;
}

#endif // #ifdef __CUDACC__ ... else ...



#ifdef __CUDACC__
//-------------------------------------------------------------------------------------------------------
// Function    :  Src_SetConstMemory_Leakage
// Description :  Set the constant memory variables on GPU
//
// Note        :  1. Adopt the suggested approach for CUDA version >= 5.0
//                2. Invoked by Src_Init_Leakage() and, if necessary, Src_WorkBeforeMajorFunc_Leakage()
//                3. SRC_NAUX_LEAKAGE is defined in Macro.h
//
// Parameter   :  AuxArray_Flt/Int : Auxiliary arrays to be copied to the constant memory
//                DevPtr_Flt/Int   : Pointers to store the addresses of constant memory arrays
//
// Return      :  c_Src_Leakage_AuxArray_Flt[], c_Src_Leakage_AuxArray_Int[], DevPtr_Flt, DevPtr_Int
//---------------------------------------------------------------------------------------------------
void Src_SetConstMemory_Leakage( const double AuxArray_Flt[], const int AuxArray_Int[],
                                 double *&DevPtr_Flt, int *&DevPtr_Int )
{

// copy data to constant memory
   CUDA_CHECK_ERROR(  cudaMemcpyToSymbol( c_Src_Leakage_AuxArray_Flt, AuxArray_Flt, SRC_NAUX_LEAKAGE*sizeof(double) )  );
   CUDA_CHECK_ERROR(  cudaMemcpyToSymbol( c_Src_Leakage_AuxArray_Int, AuxArray_Int, SRC_NAUX_LEAKAGE*sizeof(int   ) )  );

// obtain the constant-memory pointers
   CUDA_CHECK_ERROR(  cudaGetSymbolAddress( (void **)&DevPtr_Flt, c_Src_Leakage_AuxArray_Flt )  );
   CUDA_CHECK_ERROR(  cudaGetSymbolAddress( (void **)&DevPtr_Int, c_Src_Leakage_AuxArray_Int )  );

} // FUNCTION : Src_SetConstMemory_Leakage
#endif // #ifdef __CUDACC__



#ifndef __CUDACC__

//-----------------------------------------------------------------------------------------
// Function    :  Src_Init_Leakage
// Description :  Initialize the leakage source term
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
void Src_Init_Leakage()
{

// set the auxiliary arrays
   Src_SetAuxArray_Leakage( Src_Leakage_AuxArray_Flt, Src_Leakage_AuxArray_Int );

// copy the auxiliary arrays to the GPU constant memory and store the associated addresses
#  ifdef GPU
   Src_SetConstMemory_Leakage( Src_Leakage_AuxArray_Flt, Src_Leakage_AuxArray_Int,
                               SrcTerms.Leakage_AuxArrayDevPtr_Flt, SrcTerms.Leakage_AuxArrayDevPtr_Int );
#  else
   SrcTerms.Leakage_AuxArrayDevPtr_Flt = Src_Leakage_AuxArray_Flt;
   SrcTerms.Leakage_AuxArrayDevPtr_Int = Src_Leakage_AuxArray_Int;
#  endif

// set the major source-term function
   Src_SetCPUFunc_Leakage( SrcTerms.Leakage_CPUPtr );

#  ifdef GPU
   Src_SetGPUFunc_Leakage( SrcTerms.Leakage_GPUPtr );
   SrcTerms.Leakage_FuncPtr = SrcTerms.Leakage_GPUPtr;
#  else
   SrcTerms.Leakage_FuncPtr = SrcTerms.Leakage_CPUPtr;
#  endif

} // FUNCTION : Src_Init_Leakage



//-----------------------------------------------------------------------------------------
// Function    :  Src_End_Leakage
// Description :  Release the resources used by the leakage source term
//
// Note        :  1. Invoked by Src_End()
//                2. Add "#ifndef __CUDACC__" since this routine is only useful on CPU
//
// Parameter   :  None
//
// Return      :  None
//-----------------------------------------------------------------------------------------
void Src_End_Leakage()
{

// not used by this source term

} // FUNCTION : Src_End_Leakage

#endif // #ifndef __CUDACC__



//-------------------------------------------------------------------------------------------------------
// Function    :  Src_Leakage_ConstructSeries
// Description :  Construct a series with NBin+1 points in which the spacing increases logarithmically
//                --> dx * Factor, dx * Factor^2, ..., dx * Factor^NBin
//
// Note        :  1. Invoked by Src_WorkBeforeMajorFunc_Leakage()
//                2. Add "#ifndef __CUDACC__" since this routine is only useful on CPU
//
// Parameter   :  NBin : Number of bin
//                xmin : The minimum value of constructed series
//                xmax : The maximum value of constructed series
//                dx   : Initial value of spacing
//
// Return      :  Factor
//-------------------------------------------------------------------------------------------------------
#ifndef __CUDACC__
double Src_Leakage_ConstructSeries( const int NBin, const double xmin, const double xmax, const double dx )
{

   const double Tolerance   = 1.0e-10;
   const double Target      = ( xmax - xmin ) / dx;
   const double NBin_double = double(NBin);
         int    NIter       = 100;
         double Factor, Correction, Sum, Derivative;


// estimate the factor
   Factor = exp( log(Target) / NBin_double );

// solve ( xmax - xmin ) / dx = Factor + Factor^2 + ... + Factor^NBin
   while ( NIter-- )
   {
//    analytical formula of Sum        = Factor + Factor^2 + ... + Factor^NBin
//                          Derivative = d(Sum) / d(Factor)
      Sum        = ( pow( Factor, NBin_double + 1.0 ) - Factor ) / ( Factor - 1.0 );
      Derivative = ( NBin_double + 1.0 ) * Sum / Factor + ( NBin_double - Sum ) / ( Factor - 1.0 );
      Correction = ( Sum - Target ) / Derivative;

      if (  fabs( Correction / Factor ) < Tolerance  )   break;

      Factor -= Correction;
   }


   return Factor;

} // FUNCTION : Src_Leakage_ConstructSeries
#endif



//-------------------------------------------------------------------------------------------------------
// Function    :  Src_Leakage_BinarySearch
// Description :  Use binary search to find the proper array index "Idx" in the input "Array" satisfying
//
//                   Array[Idx] <= Key < Array[Idx+1]
//
// Note        :  1. A variant function of Mis_BinarySearch_Real(), which is shared by both CPU and GPU
//                2. Return "Min" instead if "Key < Array[Min]"
//                          "Max" instead if "Key > Array[Max]"
//
// Parameter   :  Array : Sorted look-up array (in ascending numerical order)
//                Min   : Minimum array index for searching
//                Max   : Maximum array index for searching
//                Key   : Target value to search for
//
// Return      :  Idx : if target is found
//                Min : if Key <  Array[Min]
//                Max : if Key >= Array[Max]
//-------------------------------------------------------------------------------------------------------
GPU_DEVICE static
int Src_Leakage_BinarySearch( const real Array[], int Min, int Max, const real Key )
{

// check whether the input key lies outside the target range
   if ( Key <  Array[Min] )   return Min;
   if ( Key >= Array[Max] )   return Max;

// binary search
   int Idx = -2;

   while (  ( Idx=(Min+Max)/2 ) != Min  )
   {
      if   ( Array[Idx] > Key )  Max = Idx;
      else                       Min = Idx;
   }


   return Idx;

} // FUNCTION : Src_Leakage_BinarySearch



//-------------------------------------------------------------------------------------
// Function    :  Src_Leakage_LinearInterp
// Description :  Linear interpolation
//
// Note        :  1. Return array[0] instead if x_in < xs[0]
//                          array[1] instead if x_in > xs[1]
//-------------------------------------------------------------------------------------
GPU_DEVICE static
real Src_Leakage_LinearInterp( const real *array, const real *xs, const real x_in )
{

   if ( x_in < xs[0] )   return array[0];
   if ( x_in > xs[1] )   return array[1];

   const real frac_x = ( x_in - xs[0] ) / ( xs[1] - xs[0] );


   return array[0] + ( array[1] - array[0] ) * frac_x;

} // FUNCTION : Src_Leakage_LinearInterp



//-------------------------------------------------------------------------------------
// Function    :  Src_Leakage_BilinearInterp
// Description :  Bilinear interpolation
//-------------------------------------------------------------------------------------
GPU_DEVICE static
real Src_Leakage_BilinearInterp( const real *array, const real *xs, const real *ys,
                                 const real x_in, const real y_in )
{

   real v[2] = {  Src_Leakage_LinearInterp( array,   xs, x_in ),
                  Src_Leakage_LinearInterp( array+2, xs, x_in )  };


   return Src_Leakage_LinearInterp( v, ys, y_in );

} // FUNCTION : Src_Leakage_BilinearInterp



//-------------------------------------------------------------------------------------
// Function    :  Src_Leakage_TrilinearInterp
// Description :  Trilinear interpolation
//-------------------------------------------------------------------------------------
GPU_DEVICE static
real Src_Leakage_TrilinearInterp( const real *array, const real *xs, const real *ys, const real *zs,
                                  const real x_in, const real y_in, const real z_in )
{

   real v[2] = {  Src_Leakage_BilinearInterp( array,   xs, ys, x_in, y_in ),
                  Src_Leakage_BilinearInterp( array+4, xs, ys, x_in, y_in )  };


   return Src_Leakage_LinearInterp( v, zs, z_in );

} // FUNCTION : Src_Leakage_TrilinearInterp



#endif // #if ( MODEL == HYDRO )
