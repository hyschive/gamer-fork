#include "GAMER.h"


static void Poi_Prepare_GREP( const double Time, const int lv );
static void Combine_GREP_Profile( Profile_t *Prof[][2], const int lv, const int Sg, const double PrepTime,
                                  const bool RemoveEmpty );

extern void SetExtPotAuxArray_GREP( double AuxArray_Flt[], int AuxArray_Int[], const double Time );
extern void SetTempIntPara( const int lv, const int Sg_Current, const double PrepTime, const double Time0, const double Time1,
                            bool &IntTime, int &Sg, int &Sg_IntT, real &Weighting, real &Weighting_IntT );

#ifdef GPU
extern void ExtPot_PassData2GPU_GREP( const real *h_Table );
#endif


Profile_t *DensAve [NLEVEL+1][2];
Profile_t *EngyAve [NLEVEL+1][2];
Profile_t *VrAve   [NLEVEL+1][2];
Profile_t *PresAve [NLEVEL+1][2];
Profile_t *Phi_eff [NLEVEL  ][2];

int    GREP_LvUpdate;
double GREP_Prof_MaxRadius;
double GREP_Prof_MinBinSize;

int    GREPSg     [NLEVEL];
double GREPSgTime [NLEVEL][2];
double GREP_Prof_Center   [3];

extern real *h_ExtPotGREP;



//-------------------------------------------------------------------------------------------------------
// Function    :  Poi_UserWorkBeforePoisson_GREP
// Description :  Compute the GREP, transfer data to GPU device, and update CPU/GPU data pointer
//                before invoking the Poisson solver
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

// compute effective GR potential
   Poi_Prepare_GREP( Time, lv );

// update the auxiliary arrays for GREP
   SetExtPotAuxArray_GREP( ExtPot_AuxArray_Flt, ExtPot_AuxArray_Int, Time );


// store the profiles in the host arrays
// --> note the typecasting from double to real
   const int Lv      = GREP_LvUpdate;
   const int FaLv    = ( Lv > 0 ) ? Lv - 1 : Lv;
   const int Sg_Lv   = GREPSg[Lv];
   const int Sg_FaLv = GREPSg[FaLv];

   Profile_t *Phi_Lv_New   = Phi_eff[ Lv   ][     Sg_Lv   ];
   Profile_t *Phi_FaLv_New = Phi_eff[ FaLv ][     Sg_FaLv ];
   Profile_t *Phi_FaLv_Old = Phi_eff[ FaLv ][ 1 - Sg_FaLv ];

   for (int b=0; b<Phi_Lv_New->NBin; b++) {
//    check whether the number of bins exceeds EXT_POT_GREP_NAUX_MAX
//    Phi_FaLv_New and Phi_FaLv_Old have been checked earlier and are skipped here
      if ( Phi_Lv_New->NBin > EXT_POT_GREP_NAUX_MAX )
         Aux_Error( ERROR_INFO, "Number of bins = %d > EXT_POT_GREP_NAUX_MAX = %d for GREP at lv = %d and SaveSg = %d !!\n",
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
// Description :  Update the spherical-averaged profiles before entering the next AMR level in EvolveLevel()
//
// Note        :  1. Invoked by EvolveLevel() using the function pointer "Mis_UserWorkBeforeNextLevel_Ptr"
//                2. Update the radial velocity, internal energy, and pressure profiles
//                   to account for the Poisson + gravity solvers and source terms
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


   int        Sg           = GREPSg[lv];
   long       TVar      [] = {         _VELR,           _PRES,           _EINT };
   Profile_t *Prof_Leaf [] = { VrAve[lv][Sg], PresAve[lv][Sg], EngyAve[lv][Sg] };

   Aux_ComputeProfile( Prof_Leaf, GREP_Prof_Center, GREP_Prof_MaxRadius, GREP_Prof_MinBinSize,
                       GREP_LOGBIN, GREP_LOGBINRATIO, false, TVar, 3, lv, lv, PATCH_LEAF, -1.0 );

} // FUNCTION : Mis_UserWorkBeforeNextLevel_GREP



//-------------------------------------------------------------------------------------------------------
// Function    :  Mis_UserWorkBeforeNextSubstep_GREP
// Description :  Update the spherical-averaged profiles before proceeding to the next sub-step in EvolveLevel()
//                --> After fix-up and grid refinement on lv
//
// Note        :  1. Invoked by EvolveLevel() using the function pointer "Mis_UserWorkBeforeNextSubstep_Ptr"
//                2. Update the density, radial velocity, internal energy, and pressure profiles
//                   to account for the flux correction and grid allocation/deallocation
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


   int        Sg           = GREPSg[lv];
   long       TVar      [] = {           _DENS,         _VELR,           _PRES,           _EINT };
   Profile_t *Prof_Leaf [] = { DensAve[lv][Sg], VrAve[lv][Sg], PresAve[lv][Sg], EngyAve[lv][Sg] };

   Aux_ComputeProfile( Prof_Leaf, GREP_Prof_Center, GREP_Prof_MaxRadius, GREP_Prof_MinBinSize,
                       GREP_LOGBIN, GREP_LOGBINRATIO, false, TVar, 4, lv, lv, PATCH_LEAF, -1.0 );

} // FUNCTION : Mis_UserWorkBeforeNextSubstep_GREP



//-------------------------------------------------------------------------------------------------------
// Function    :  Poi_Prepare_GREP
// Description :  Update the spherical-averaged profiles before Poisson and Gravity solvers,
//                and compute the GR effective potential.
//
// Note        :  1. Invoked by Poi_UserWorkBeforePoisson_GREP()
//                2. The contribution from     leaf patches on level = lv (<= lv) is stored QUANT[    lv]
//                                         non-leaf patches on level = lv         is stored QUANT[NLEVEL]
//                3. The GR effective potential is stored at Phi_eff[lv]
//
// Parameter   :  Time : Target physical time
//                lv   : Target refinement level
//-------------------------------------------------------------------------------------------------------
void Poi_Prepare_GREP( const double Time, const int lv )
{

// compare the input Time with stored time to choose the suitable SaveSg
   int Sg;

   if      (  Mis_CompareRealValue( Time, GREPSgTime[lv][0], NULL, false )  )   Sg = 0;
   else if (  Mis_CompareRealValue( Time, GREPSgTime[lv][1], NULL, false )  )   Sg = 1;
   else                                                                         Sg = 1 - GREPSg[lv];


// update and combine the spherical-averaged profiles
   long       TVar         [] = {               _DENS,             _VELR,               _PRES,               _EINT };
   Profile_t *Prof_Leaf    [] = { DensAve[    lv][Sg], VrAve[    lv][Sg], PresAve[    lv][Sg], EngyAve[    lv][Sg] };
   Profile_t *Prof_NonLeaf [] = { DensAve[NLEVEL][Sg], VrAve[NLEVEL][Sg], PresAve[NLEVEL][Sg], EngyAve[NLEVEL][Sg] };


   if ( false ) {}
   /*
   if ( GREP_ALGO_SWITCH )
   {
//    contributions from the leaf patches on level <= lv and the non-leaf patches on level = lv
      Aux_ComputeProfile( Prof_NonLeaf, GREP_Prof_Center, GREP_Prof_MaxRadius, GREP_Prof_MinBinSize,
                          GREP_LOGBIN, GREP_LOGBINRATIO, true,  TVar, 4,  0, lv, PATCH_LEAF_PLUS_MAXNONLEAF, Time );
   }
   */

   else
   {
//    retain the empty bins to avoid inconsistent leaf- and non-leaft profiles during combining the profiles

//    contributions from the leaf patches on level = lv
      Aux_ComputeProfile( Prof_Leaf,    GREP_Prof_Center, GREP_Prof_MaxRadius, GREP_Prof_MinBinSize,
                          GREP_LOGBIN, GREP_LOGBINRATIO, false, TVar, 4, lv, lv, PATCH_LEAF,    -1.0 );

//    contributions from the non-leaf patches on level = lv
      Aux_ComputeProfile( Prof_NonLeaf, GREP_Prof_Center, GREP_Prof_MaxRadius, GREP_Prof_MinBinSize,
                          GREP_LOGBIN, GREP_LOGBINRATIO, false, TVar, 4, lv, lv, PATCH_NONLEAF, -1.0 );

//    combine the profiles on each level
      Combine_GREP_Profile( DensAve, lv, Sg, Time, true );
      Combine_GREP_Profile( EngyAve, lv, Sg, Time, true );
      Combine_GREP_Profile( VrAve,   lv, Sg, Time, true );
      Combine_GREP_Profile( PresAve, lv, Sg, Time, true );
   }


// record the level, Sg, and SgTime
   GREP_LvUpdate      = lv;
   GREPSg    [lv]     = Sg;
   GREPSgTime[lv][Sg] = Time;


// compute the effective GR potential
   CPU_ComputeGREP( DensAve[NLEVEL][Sg], EngyAve[NLEVEL][Sg], VrAve[NLEVEL][Sg], PresAve[NLEVEL][Sg],
                    Phi_eff[lv]    [Sg] );

} // FUNCTION : Poi_Prepare_GREP



//-------------------------------------------------------------------------------------------------------
// Function    :  Combine_GREP_Profile
// Description :  Combine the stored spherical-averaged profiles on each level
//                and remove the empty bins in the combined profile
//
// Note        :  1. The total averaged profile is stored at QUANT[NLEVEL]
//
// Parameter   :  Prof        : Profile_t object array to be combined
//                lv          : Target refinement level
//                Sg          : Sandglass indicating which Profile_t object the data are stored
//                PrepTime    : Target physical time to combine the spherical-averaged profiles
//                RemoveEmpty : true  --> remove empty bins from the data
//                              false --> these empty bins will still be in the profile arrays with
//                                        Data[empty_bin]=Weight[empty_bin]=NCell[empty_bin]=0
//-------------------------------------------------------------------------------------------------------
void Combine_GREP_Profile( Profile_t *Prof[][2], const int lv, const int Sg, const double PrepTime,
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
      int  Sg_Lv = GREPSg[level];
      real FluWeighting, FluWeighting_IntT;

      SetTempIntPara( level, Sg_Lv, PrepTime, GREPSgTime[level][Sg_Lv], GREPSgTime[level][1 - Sg_Lv],
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
                              : Prof_NonLeaf->Radius[LastBin] + 0.5*GREP_Prof_MinBinSize;
   } // if ( RemoveEmpty )

} // FUNCTION : Combine_GREP_Profile
