#include "GAMER.h"


static void Poi_Prepare_GREP( const double Time, const int lv );
static void GREP_Compute_Profile( const int lv, const int Sg, const PatchType_t PatchType );
static void GREP_Combine_Profile( Profile_t *Prof[][2], const int lv, const int Sg, const double PrepTime,
                                  const bool RemoveEmpty );
static void GREP_Check_Profile( const int lv, Profile_t *Prof[], const int NProf );

extern void SetExtPotAuxArray_GREP( double AuxArray_Flt[], int AuxArray_Int[], const double Time );
extern void SetTempIntPara( const int lv, const int Sg_Current, const double PrepTime, const double Time0, const double Time1,
                            bool &IntTime, int &Sg, int &Sg_IntT, real &Weighting, real &Weighting_IntT );

#ifdef GPU
extern void ExtPot_PassData2GPU_GREP( const real *h_Table );
#endif


Profile_t *GREP_DensAve [NLEVEL+1][2];
Profile_t *GREP_EngyAve [NLEVEL+1][2];
Profile_t *GREP_VrAve   [NLEVEL+1][2];
Profile_t *GREP_PresAve [NLEVEL+1][2];
Profile_t *GREP_EffPot  [NLEVEL  ][2];

int    GREP_LvUpdate;
int    GREP_Sg     [NLEVEL];
double GREP_SgTime [NLEVEL][2];

extern bool CCSN_Is_PostBounce;
extern real *h_ExtPotGREP;



//-------------------------------------------------------------------------------------------------------
// Function    :  Poi_UserWorkBeforePoisson_GREP
// Description :  Compute the effective GR potential, transfer data to GPU device,
//                and update CPU/GPU data pointer before invoking the Poisson solver
//
// Note        :  1. Invoked by Gra_AdvanceDt() using the function pointer "Poi_UserWorkBeforePoisson_Ptr"
//
// Parameter   :  Time : Target physical time
//                lv   : Target refinement level
//
// Return      :  None
//-------------------------------------------------------------------------------------------------------
void Poi_UserWorkBeforePoisson_GREP( const double Time, const int lv )
{

// ignore level containing no patches
   if ( NPatchTotal[lv] == 0 )   return;

// update the GREP center at each global step
   if ( lv == 0 )
   {
      Extrema_t Extrema;

      switch ( GREP_CENTER_METHOD )
      {
         case GREP_CENTER_NONE:
         break;

         case GREP_CENTER_BOX: // box center
            for (int i=0; i<3; i++)   GREP_Center[i] = amr->BoxCenter[i];
         break;

         case GREP_CENTER_DENS: // density maximum
         {
            if ( ! CCSN_Is_PostBounce  &&  TESTPROB_ID == TESTPROB_HYDRO_CCSN )
            {
//             fix the center to the box center during the collapse stage in CCSN simulations
               for (int i=0; i<3; i++)   GREP_Center[i] = amr->BoxCenter[i];
            }

            else
            {
               Extrema.Field     = _DENS;
               Extrema.Radius    = HUGE_NUMBER;
               Extrema.Center[0] = amr->BoxCenter[0];
               Extrema.Center[1] = amr->BoxCenter[1];
               Extrema.Center[2] = amr->BoxCenter[2];

               Aux_FindExtrema( &Extrema, EXTREMA_MAX, 0, TOP_LEVEL, PATCH_LEAF );

               for (int i=0; i<3; i++)   GREP_Center[i] = Extrema.Coord[i];
            } // if ( ! CCSN_Is_PostBounce  &&  TESTPROB_ID == TESTPROB_HYDRO_CCSN ) ... else ...
         }
         break;

         case GREP_CENTER_POT: // potential minimum
         {
            Extrema.Field     = _POTE;
            Extrema.Radius    = HUGE_NUMBER;
            Extrema.Center[0] = amr->BoxCenter[0];
            Extrema.Center[1] = amr->BoxCenter[1];
            Extrema.Center[2] = amr->BoxCenter[2];

            Aux_FindExtrema( &Extrema, EXTREMA_MIN, 0, TOP_LEVEL, PATCH_LEAF );

            for (int i=0; i<3; i++)   GREP_Center[i] = Extrema.Coord[i];
         }
         break;

         case GREP_CENTER_COM: // center of mass
            Aux_Error( ERROR_INFO, "GREP_CENTER_COM has not been implemented yet!!\n" );
         break;

         default:
            Aux_Error( ERROR_INFO, "unsupported %s = %d !!\n", "GREP_CENTER_METHOD", GREP_CENTER_METHOD );
      } // switch ( GREP_CENTER_METHOD )


//    shift the center to the box center if it coincides with one of the innermost cells
      if (  GREP_CENTER_METHOD == GREP_CENTER_DENS  ||
            GREP_CENTER_METHOD == GREP_CENTER_POT     )
      {
         const double Extrema_dh = amr->dh[ Extrema.Level ];

         if (  fabs( GREP_Center[0] - amr->BoxCenter[0] ) < Extrema_dh  &&
               fabs( GREP_Center[1] - amr->BoxCenter[1] ) < Extrema_dh  &&
               fabs( GREP_Center[2] - amr->BoxCenter[2] ) < Extrema_dh    )
            for (int i=0; i<3; i++)   GREP_Center[i] = amr->BoxCenter[i];
      }
   } // if ( lv == 0 )


// compute effective GR potential
   Poi_Prepare_GREP( Time, lv );

// update the auxiliary arrays for GREP
   SetExtPotAuxArray_GREP( ExtPot_AuxArray_Flt, ExtPot_AuxArray_Int, Time );


// store the profiles in the host arrays
// --> note the typecasting from double to real
   const int Lv      = GREP_LvUpdate;
   const int FaLv    = ( Lv > 0 ) ? Lv - 1 : Lv;
   const int Sg_Lv   = GREP_Sg[Lv];
   const int Sg_FaLv = GREP_Sg[FaLv];

   Profile_t *Phi_Lv_New   = GREP_EffPot[ Lv   ][     Sg_Lv   ];
   Profile_t *Phi_FaLv_New = GREP_EffPot[ FaLv ][     Sg_FaLv ];
   Profile_t *Phi_FaLv_Old = GREP_EffPot[ FaLv ][ 1 - Sg_FaLv ];

   for (int b=0; b<Phi_Lv_New->NBin; b++) {
//    check whether the number of bins exceeds EXT_POT_GREP_NAUX_MAX
//    Phi_FaLv_New and Phi_FaLv_Old have been checked earlier and are skipped here
      if ( Phi_Lv_New->NBin > EXT_POT_GREP_NAUX_MAX )
         Aux_Error( ERROR_INFO, "number of bins (%d) > EXT_POT_GREP_NAUX_MAX (%d) for GREP on level = %d and SaveSg = %d !!\n",
                    Phi_Lv_New->NBin, EXT_POT_GREP_NAUX_MAX, Lv, Sg_Lv );

      h_ExtPotGREP[b                          ] = (real) Phi_Lv_New   ->Data  [b];
      h_ExtPotGREP[b +   EXT_POT_GREP_NAUX_MAX] = (real) Phi_Lv_New   ->Radius[b];
   }

   for (int b=0; b<Phi_FaLv_New->NBin; b++) {
      h_ExtPotGREP[b + 2*EXT_POT_GREP_NAUX_MAX] = (real) Phi_FaLv_New->Data  [b];
      h_ExtPotGREP[b + 3*EXT_POT_GREP_NAUX_MAX] = (real) Phi_FaLv_New->Radius[b];
   }

   for (int b=0; b<Phi_FaLv_Old->NBin; b++) {
      h_ExtPotGREP[b + 4*EXT_POT_GREP_NAUX_MAX] = (real) Phi_FaLv_Old->Data  [b];
      h_ExtPotGREP[b + 5*EXT_POT_GREP_NAUX_MAX] = (real) Phi_FaLv_Old->Radius[b];
   }


// assign the value of h_ExtPotGenePtr
   for (int i=0; i<6; i++)   h_ExtPotGenePtr[i] = (real**) (h_ExtPotGREP + i*EXT_POT_GREP_NAUX_MAX);


#  ifdef GPU
// update the GPU auxiliary arrays
   CUAPI_SetConstMemory_ExtAccPot();

// transfer GREP profiles to GPU
   ExtPot_PassData2GPU_GREP( h_ExtPotGREP );
#  endif

} // FUNCTION : Poi_UserWorkBeforePoisson_GREP



//-------------------------------------------------------------------------------------------------------
// Function    :  Mis_UserWorkBeforeNextLevel_GREP
// Description :  Update the spherically averaged profiles before entering the next AMR level in EvolveLevel()
//
// Note        :  1. Invoked by EvolveLevel() using the function pointer "Mis_UserWorkBeforeNextLevel_Ptr"
//                2. Update the profiles to account for the Poisson and Gravity solvers, and source terms
//
// Parameter   :  lv      : Target refinement level
//                TimeNew : Target physical time to reach
//                TimeOld : Physical time before update
//                dt      : Time interval to advance solution (can be different from TimeNew-TimeOld in COMOVING)
//-------------------------------------------------------------------------------------------------------
void Mis_UserWorkBeforeNextLevel_GREP( const int lv, const double TimeNew, const double TimeOld, const double dt )
{

   if ( lv == TOP_LEVEL )   return;

   if (  ( NPatchTotal[lv+1] == 0 )                      &&
         ( AdvanceCounter[lv] + 1 ) % REGRID_COUNT != 0     )   return;


   const int Sg = GREP_Sg[lv];

   GREP_Compute_Profile( lv, Sg, PATCH_LEAF );

} // FUNCTION : Mis_UserWorkBeforeNextLevel_GREP



//-------------------------------------------------------------------------------------------------------
// Function    :  Mis_UserWorkBeforeNextSubstep_GREP
// Description :  Update the spherically averaged profiles before proceeding to the next sub-step in EvolveLevel()
//                --> After fix-up and grid refinement on lv
//
// Note        :  1. Invoked by EvolveLevel() using the function pointer "Mis_UserWorkBeforeNextSubstep_Ptr"
//                2. Update the profiles to account for the flux correction and grid allocation/deallocation
//
// Parameter   :  lv      : Target refinement level
//                TimeNew : Target physical time to reach
//                TimeOld : Physical time before update
//                dt      : Time interval to advance solution (can be different from TimeNew-TimeOld in COMOVING)
//-------------------------------------------------------------------------------------------------------
void Mis_UserWorkBeforeNextSubstep_GREP( const int lv, const double TimeNew, const double TimeOld, const double dt )
{

   if ( lv == TOP_LEVEL )   return;

   if (  ( !GREP_OPT_FIXUP  &&  AdvanceCounter[lv] % REGRID_COUNT != 0 )  ||
         ( NPatchTotal[lv+1] == 0 )                                          )   return;


   const int Sg = GREP_Sg[lv];

   GREP_Compute_Profile( lv, Sg, PATCH_LEAF );

} // FUNCTION : Mis_UserWorkBeforeNextSubstep_GREP



//-------------------------------------------------------------------------------------------------------
// Function    :  Poi_Prepare_GREP
// Description :  Update the spherically averaged profiles before Poisson and Gravity solvers,
//                and compute the effective GR potential.
//
// Note        :  1. Invoked by Poi_UserWorkBeforePoisson_GREP()
//                2. The contributions from     leaf patches on level=lv are stored in GREP_*Ave[    lv]
//                                          non-leaf patches                           GREP_*Ave[NLEVEL]
//                3. The effective GR potential is stored at GREP_EffPot[lv]
//
// Parameter   :  Time : Target physical time
//                lv   : Target refinement level
//-------------------------------------------------------------------------------------------------------
void Poi_Prepare_GREP( const double Time, const int lv )
{

// compare the input time with the stored time to determine the appropriate sandglass
   int Sg;

   if      (  Mis_CompareRealValue( Time, GREP_SgTime[lv][0], NULL, false )  )   Sg = 0;
   else if (  Mis_CompareRealValue( Time, GREP_SgTime[lv][1], NULL, false )  )   Sg = 1;
   else                                                                          Sg = 1 - GREP_Sg[lv];


// update the spherically averaged profiles contributed from non-leaf and leaf patches
   GREP_Compute_Profile( lv, Sg, PATCH_LEAF    );
   GREP_Compute_Profile( lv, Sg, PATCH_NONLEAF );


// combine the spherically averaged profiles
   const bool RemoveEmpty_Yes = true;

   GREP_Combine_Profile( GREP_DensAve, lv, Sg, Time, RemoveEmpty_Yes );
   GREP_Combine_Profile( GREP_VrAve,   lv, Sg, Time, RemoveEmpty_Yes );
   GREP_Combine_Profile( GREP_PresAve, lv, Sg, Time, RemoveEmpty_Yes );
   GREP_Combine_Profile( GREP_EngyAve, lv, Sg, Time, RemoveEmpty_Yes );


// compute the pressure if GREP_OPT_PRES == GREP_PRES_BINDATA
   Profile_t *Dens_Tot = GREP_DensAve[NLEVEL][Sg];
   Profile_t *Vr_Tot   = GREP_VrAve  [NLEVEL][Sg];
   Profile_t *Pres_Tot = GREP_PresAve[NLEVEL][Sg];
   Profile_t *Engy_Tot = GREP_EngyAve[NLEVEL][Sg];

//###REVISE: support EOS != EOS_NUCLEAR, especially for EoS that does not need any passive scalar in EoS_DensEint2Pres_CPUPtr()
   if ( GREP_OPT_PRES == GREP_PRES_BINDATA )
   {
#     ifdef YE
      real Passive[NCOMP_PASSIVE] = { 0.0 };

      for (int b=0; b<Dens_Tot->NBin; b++)
      {
         if ( Dens_Tot->NCell[b] == 0 )   continue;

//       the Ye profile has been stored in the pressure profile temporarily
         Passive[ YE - NCOMP_FLUID ] = Pres_Tot->Data[b];

#        ifdef TEMP_IG
//       set the initial guess of temperature to 1 MeV
//###REVISE: support Temp_IG from Aux_ComputeProfile()
         Passive[ TEMP_IG - NCOMP_FLUID ] = 1.0e6 / Const_kB_eV;
#        endif

         Pres_Tot->Data[b] = EoS_DensEint2Pres_CPUPtr( Dens_Tot->Data[b], Engy_Tot->Data[b],
                                                       Passive, EoS_AuxArray_Flt, EoS_AuxArray_Int, h_EoS_Table );
      }
#     endif // ifdef YE
   } // if ( GREP_OPT_PRES == GREP_PRES_BINDATA )


// check the profiles before computing the effective GR potential
   Profile_t *GREP_Check_List[] = { Dens_Tot, Engy_Tot, Vr_Tot, Pres_Tot };

   GREP_Check_Profile( lv, GREP_Check_List, 4 );


// compute the effective GR potential
   CPU_ComputeGREP( lv, Time, Dens_Tot, Engy_Tot, Vr_Tot, Pres_Tot, GREP_EffPot[lv][Sg] );


// update the level, sandglass, and time
   GREP_LvUpdate       = lv;
   GREP_Sg    [lv]     = Sg;
   GREP_SgTime[lv][Sg] = Time;

} // FUNCTION : Poi_Prepare_GREP



//-------------------------------------------------------------------------------------------------------
// Function    :  GREP_Compute_Profile
// Description :  Interface for computing the spherically averaged profiles of density, radial velocity,
//                internal energy density, and pressure/Ye
//
// Note        :  1. Invoked by Mis_UserWorkBeforeNextLevel_GREP(), Mis_UserWorkBeforeNextSubstep_GREP(),
//                   and Poi_Prepare_GREP()
//                2. Keep empty bins to maintain consistent leaf and non-leaf profiles when merging them
//                3. The pressure profile is computed after combination if GREP_OPT_PRES is set to GREP_PRES_BINDATA
//
// Parameter   :  lv        : Target refinement level
//                Sg        : Sandglass indicating which Profile_t object the data are stored
//                PatchType : Types of patches to be considered
//                            --> Supported types: PATCH_LEAF, PATCH_NONLEAF
//-------------------------------------------------------------------------------------------------------
void GREP_Compute_Profile( const int lv, const int Sg, const PatchType_t PatchType )
{

// check
#  ifdef GAMER_DEBUG
   if ( PatchType != PATCH_LEAF  &&  PatchType != PATCH_NONLEAF )
      Aux_Error( ERROR_INFO, "incorrect PatchType (%d) !!\n", PatchType );
#  endif


   const bool   RemoveEmpty_No = false;
   const double PrepTime_No    = -1.0;
   const int    NVar           = 4;
   const int    Lv_Stored      = ( PatchType == PATCH_LEAF ) ? lv : NLEVEL;

   Profile_t *Dens = GREP_DensAve[Lv_Stored][Sg];
   Profile_t *Vr   = GREP_VrAve  [Lv_Stored][Sg];
   Profile_t *Engy = GREP_EngyAve[Lv_Stored][Sg];
   Profile_t *Pres = GREP_PresAve[Lv_Stored][Sg];

   Profile_t *Prof_List[] = {  Dens,    Vr,  Engy,  Pres };
   long       TVar     [] = { _DENS, _VELR, _EINT, _NONE };

   switch ( GREP_OPT_PRES )
   {
      case GREP_PRES_INDIVCELL:  // compute the pressure profile directly
      {
         TVar[3] = _PRES;
      }
      break;


      case GREP_PRES_BINDATA:  // compute the Ye profile and store it in GREP_PresAve[Lv_Stored][Sg] temporarily
      {
#        ifdef YE
         TVar[3] = _YE;
#        endif
      }
      break;


      default:
         Aux_Error( ERROR_INFO, "unsupported pressure computation scheme %s = %d !!\n", "GREP_OPT_PRES", GREP_OPT_PRES );
   } // switch ( GREP_OPT_PRES )


   Aux_ComputeProfile( Prof_List, GREP_Center, GREP_MAXRADIUS, GREP_MINBINSIZE, GREP_LOGBIN,
                       GREP_LOGBINRATIO, RemoveEmpty_No, TVar, NVar, lv, lv, PatchType, PrepTime_No );

} // FUNCTION : GREP_Compute_Profile



//-------------------------------------------------------------------------------------------------------
// Function    :  GREP_Combine_Profile
// Description :  Combine the spherically averaged profiles from leaf patches at each level
//                and from non-leaf patches at current level, then remove any empty bins
//
// Note        :  1. The total averaged profile is stored at Prof[NLEVEL][Sg]
//
// Parameter   :  Prof        : Profile_t object array to be combined
//                lv          : Target refinement level
//                Sg          : Sandglass indicating which Profile_t object the data are stored
//                PrepTime    : Target physical time to combine the spherically averaged profiles
//                RemoveEmpty : true  --> remove empty bins from the data
//                              false --> these empty bins will still be in the profile arrays with
//                                        Data[empty_bin]=Weight[empty_bin]=NCell[empty_bin]=0
//-------------------------------------------------------------------------------------------------------
void GREP_Combine_Profile( Profile_t *Prof[][2], const int lv, const int Sg, const double PrepTime,
                           const bool RemoveEmpty )
{

   Profile_t *Prof_NonLeaf = Prof[NLEVEL][Sg];
   Profile_t *Prof_Leaf;


// multiply the stored data by weight to reduce round-off errors
   for (int b=0; b<Prof_NonLeaf->NBin; b++)
   {
      if ( Prof_NonLeaf->NCell[b] != 0L )   Prof_NonLeaf->Data[b] *= Prof_NonLeaf->Weight[b];
   }


// combine the contributions from the leaf and non-leaf patches on level = lv
   Prof_Leaf = Prof[lv][Sg];

   for (int b=0; b<Prof_Leaf->NBin; b++)
   {
      if ( Prof_Leaf->NCell[b] == 0L )  continue;

      Prof_NonLeaf->Data  [b] += Prof_Leaf->Data  [b] * Prof_Leaf->Weight[b];
      Prof_NonLeaf->Weight[b] += Prof_Leaf->Weight[b];
      Prof_NonLeaf->NCell [b] += Prof_Leaf->NCell [b];
   }


// combine the contributions from the leaf patches on level < lv with temporal interpolation
   for (int level=0; level<lv; level++)
   {
      bool FluIntTime;
      int  FluSg, FluSg_IntT;
      int  Sg_Lv = GREP_Sg[level];
      real FluWeighting, FluWeighting_IntT;

      SetTempIntPara( level, Sg_Lv, PrepTime, GREP_SgTime[level][Sg_Lv], GREP_SgTime[level][1 - Sg_Lv],
                      FluIntTime, FluSg, FluSg_IntT, FluWeighting, FluWeighting_IntT );

                 Prof_Leaf      = Prof[level][FluSg];
      Profile_t *Prof_Leaf_IntT = ( FluIntTime ) ? Prof[level][FluSg_IntT] : NULL;

      for (int b=0; b<Prof_Leaf->NBin; b++)
      {
         if ( Prof_Leaf->NCell[b] == 0L )  continue;

         Prof_NonLeaf->Data  [b] += ( FluIntTime )
                                  ?   FluWeighting      * Prof_Leaf     ->Weight[b] * Prof_Leaf     ->Data[b]
                                    + FluWeighting_IntT * Prof_Leaf_IntT->Weight[b] * Prof_Leaf_IntT->Data[b]
                                  :                       Prof_Leaf     ->Weight[b] * Prof_Leaf     ->Data[b];

         Prof_NonLeaf->Weight[b] += ( FluIntTime )
                                  ?   FluWeighting      * Prof_Leaf     ->Weight[b]
                                    + FluWeighting_IntT * Prof_Leaf_IntT->Weight[b]
                                  :                       Prof_Leaf     ->Weight[b];

         Prof_NonLeaf->NCell [b] += Prof_Leaf->NCell [b];
      }
   } // for (int level=0; level<=lv; level++)


// divide the combined data by weight
   for (int b=0; b<Prof_NonLeaf->NBin; b++)
   {
      if ( Prof_NonLeaf->NCell[b] != 0L )   Prof_NonLeaf->Data[b] /= Prof_NonLeaf->Weight[b];
   }



// remove the empty bins in the combined profile stored in 'Prof_NonLeaf'
   if ( RemoveEmpty )
   {
      for (int b=0; b<Prof_NonLeaf->NBin; b++)
      {
         if ( Prof_NonLeaf->NCell[b] != 0L )   continue;

//       for cases of consecutive empty bins
         int b_up;
         for (b_up=b+1; b_up<Prof_NonLeaf->NBin; b_up++)
            if ( Prof_NonLeaf->NCell[b_up] != 0L )   break;

         const int stride = b_up - b;

         for (int b_up=b+stride; b_up<Prof_NonLeaf->NBin; b_up++)
         {
            const int b_up_ms = b_up - stride;

            Prof_NonLeaf->Radius[b_up_ms] = Prof_NonLeaf->Radius[b_up];
            Prof_NonLeaf->Data  [b_up_ms] = Prof_NonLeaf->Data  [b_up];
            Prof_NonLeaf->Weight[b_up_ms] = Prof_NonLeaf->Weight[b_up];
            Prof_NonLeaf->NCell [b_up_ms] = Prof_NonLeaf->NCell [b_up];
         }

//       reset the total number of bins
         Prof_NonLeaf->NBin -= stride;
      } // for (int b=0; b<Prof_NonLeaf->NBin; b++)

//    update the maximum radius since the last bin may have not been removed
      const int LastBin = Prof_NonLeaf->NBin-1;

      Prof_NonLeaf->MaxRadius = ( Prof_NonLeaf->LogBin )
                              ? Prof_NonLeaf->Radius[LastBin] * sqrt( Prof_NonLeaf->LogBinRatio )
                              : Prof_NonLeaf->Radius[LastBin] + 0.5*GREP_MINBINSIZE;
   } // if ( RemoveEmpty )

} // FUNCTION : GREP_Combine_Profile



//-------------------------------------------------------------------------------------------------------
// Function    :  GREP_Check_Profile
// Description :  Check if there are any unphysical bins in the spherically averaged profiles
//
// Note        :  1. Terminate the program if any invalid fluid variables are present
//
// Parameter   :  Prof  : Profile_t object array to be verified
//                NProf : Number of input profiles
//-------------------------------------------------------------------------------------------------------
void GREP_Check_Profile( const int lv, Profile_t *Prof[], const int NProf )
{

   const int NBin = Prof[0]->NBin;

   if ( MPI_Rank == 0 )
   {
      for (int p=0; p<NProf; p++)
      for (int b=0; b<NBin;  b++)
      {
         if (  ! Aux_IsFinite( Prof[p]->Data[b] )  )
         {
//          troubleshooting information
            char FileName[MAX_STRING];

            sprintf( FileName, "GREP_Lv%02d_InvalidProfile", lv );
            FILE *File = fopen( FileName, "w" );

//          metadata
            fprintf( File, "# GREP_CENTER_METHOD : %d\n",                   GREP_CENTER_METHOD );
            fprintf( File, "# Center             : %13.7e %13.7e %13.7e\n", Prof[0]->Center[0], Prof[0]->Center[1], Prof[0]->Center[2] );
            fprintf( File, "# Maximum Radius     : %13.7e\n",               Prof[0]->MaxRadius );
            fprintf( File, "# LogBin             : %d\n",                   Prof[0]->LogBin );
            fprintf( File, "# LogBinRatio        : %13.7e\n",               Prof[0]->LogBinRatio );
            fprintf( File, "# NBin               : %d\n",                   NBin );
            fprintf( File, "# -----------------------------------------------\n" );
            fprintf( File, "%5s %9s %22s %22s %22s %22s %22s\n",
                           "# Bin", "NCell", "Radius", "Density", "Energy", "Vr", "Pressure" );

//          data
            for (int i=0; i<NBin; i++)
            fprintf( File, "%5d %9ld %22.15e %22.15e %22.15e %22.15e %22.15e\n",
                           i, Prof[0]->NCell[i], Prof[0]->Radius[i],
                           Prof[0]->Data[i], Prof[1]->Data[i], Prof[2]->Data[i], Prof[3]->Data[i] );

            fclose( File );

            Aux_Error( ERROR_INFO, "invalid fluid variables (%14.7e) at GREP profile (%d), bin (%d) !!\n",
                       Prof[p]->Data[b], p, b );
         } // if (  ! Aux_IsFinite( Prof[p]->Data[b] )  )
      } // p,b
   } // if ( MPI_Rank == 0 )

} // FUNCTION : GREP_Check_Profile
