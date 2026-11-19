#include "GAMER.h"

static const double TWOPI = 2.0 * M_PI;


static void Aux_GetMinMax_Vertex_Radius( const double x, const double y, const double z, const double HalfWidth,
                                         double *RadMin, double *RadMax );

static void Aux_GetMinMax_Vertex_Angle( const double x, const double y, const double z, const double HalfWidth,
                                        double *ThtMin, double *ThtMax, double *PhiMin, double *PhiMax );

extern void SetTempIntPara( const int lv, const int Sg0, const double PrepTime, const double Time0, const double Time1,
                            bool &IntTime, int &Sg, int &Sg_IntT, real &Weighting, real &Weighting_IntT );




//-------------------------------------------------------------------------------------------------------
// Function    :  Aux_ComputeRay
// Description :  Maps the target field(s) from cells to rays
//
// Note        :  1. Results will be stored in the input "Prof" object
//                   --> with index "Idx_Radius + NRadius * ( Idx_Theta + NTheta * Idx_Phi )"
//                2. Support hybrid OpenMP/MPI parallelization
//                   --> All ranks will share the same ray data after invoking this function
//                3. Weighting of each cell:
//                   --> Cell mass  : gas velocity, gravitational potential
//                       Cell volume: other fields
//                4. Support computing multiple fields
//                   --> The order of fields to be returned follows TVarBitIdx[]
//                5. This routine is thread-unsafe when the temporal interpolation set by PrepTime and OPT__INT_TIME
//                   are inconsistent with each other
//                   --> But it shouldn't be a big issue since this routine itself has been parallelized with OpenMP
//
// Parameter   :  Ray            : Profile_t object array to store the results
//                Center         : Target center coordinates
//                Edge           : Edge of bins in the radial direction
//                NRadius_Linear : Number of linear bins in the radial    direction
//                NRadius        : Number of        bins in the radial    direction
//                NTheta         : Number of        bins in the polar     direction
//                NPhi           : Number of        bins in the azimuthal direction
//                BinSize_Linear : Bin size of linear bins in the radial direction
//                MaxRad_Linear  : Maximum radius of linear bins
//                MaxRad         : Maximum radius in each ray profile
//                TVarBitIdx     : Bitwise indices of target variables for computing the rays
//                                 --> Supported indices (defined in Macro.h):
//                                        HYDRO : _DENS, _MOMX, _MOMY, _MOMZ, _ENGY, _VELX, _VELY, _VELZ, _VELR,
//                                                _PRES, _TEMP, _ENTR, _EINT
//                                                [, _DUAL, _CRAY, _POTE, __MAGX_CC, _MAGY_CC, _MAGZ_CC, _MAGE_CC]
//                                        ELBDM : _DENS, _REAL, _IMAG [, _POTE]
//                                 --> All fields supported by Prepare_PatchData() are also supported here
//                NProf          : Number of target fields in Ray
//                PrepTimeIn     : Target physical time to prepare data
//                                 --> If PrepTimeIn<0, turn off temporal interpolation and always use the most recent data
//
// Example     :  const double Center[3]      = { amr->BoxCenter[0], amr->BoxCenter[1], amr->BoxCenter[2] };
//                const int    NRadius_Linear = 256;
//                const int    NRadius        = 256;
//                const int    NTheta         = 1;
//                const int    NPhi           = 1;
//                const double BinSize_Linear = amr->dh[MAX_LEVEL];
//                const double MaxRad_Linear  = 0.5*amr->BoxSize[0];
//                const double MaxRad         = 0.5*amr->BoxSize[0];
//                const long   TVarBitIdx[]   = { _DENS, _TEMP };
//                const int    NProf          = 2;
//                const double PrepTimeIn     = -1.0;
//                      double Edge[NRadius+1];
//
//                Profile_t Ray_Dens, Ray_Temp;
//                Profile_t *Ray[NProf] = { &Ray_Dens, &Ray_Temp };
//
//                for (int i=0; i<NRadius+1; i++)   Edge[i] = double(i) * BinSize_Linear;
//
//                Aux_ComputeRay( Ray, Center, Edge, NRadius_Linear, NRadius, NTheta, NPhi
//                                BinSize_Linear, MaxRad_Linear, MaxRad, TVarBitIdx, NProf, PrepTimeIn );
//
// Return      :  Ray
//-------------------------------------------------------------------------------------------------------
void Aux_ComputeRay( Profile_t *Ray[], const double Center[], const double Edge[],
                     const int NRadius_Linear, const int NRadius, const int NTheta, const int NPhi,
                     const double BinSize_Linear, const double MaxRad_Linear, const double MaxRad,
                     const long TVarBitIdx[], const int NProf, const double PrepTimeIn )
{

// check
#  ifdef GAMER_DEBUG
   if ( NRadius < NRadius_Linear )
      Aux_Error( ERROR_INFO, "NRadius (%d) < NRadius_Linear (%d) !!\n", NRadius, NRadius_Linear );

   if ( MaxRad < MaxRad_Linear )
      Aux_Error( ERROR_INFO, "MaxRad (%14.7e) < MaxRad_Linear (%14.7e) !!\n", MaxRad, MaxRad_Linear );
#  endif


// list all supported fields
// --> all fields supported by Prepare_PatchData() should be supported here
   long SupportedFields = ( _TOTAL | _DERIVED );
#  ifdef GRAVITY
   SupportedFields |= _POTE;
#  endif
#  ifdef PARTICLE
   SupportedFields |= _PAR_DENS;
   SupportedFields |= _TOTAL_DENS;
#  endif

   for (int p=0; p<NProf; p++) {
      if ( TVarBitIdx[p] & ~SupportedFields )
         Aux_Error( ERROR_INFO, "unsupported field (TVarBitIdx[%d] = %ld) !!\n", p, TVarBitIdx[p] );
   }


// record whether particle density is requested
#  ifdef PARTICLE
   bool NeedPar = false;
   for (int p=0; p<NProf; p++) {
      if ( TVarBitIdx[p] == _PAR_DENS  ||  TVarBitIdx[p] == _TOTAL_DENS ) {
         NeedPar = true;
         break;
      }
   }
#  endif


// initialize the profile objects
   const int NBin = NRadius * NTheta * NPhi;

   for (int p=0; p<NProf; p++)
   {
      Ray[p]->NBin = NBin;
      Ray[p]->AllocateMemory();
   }


// allocate memory for the per-thread arrays
#  ifdef OPENMP
   const int NT = OMP_NTHREAD;
#  else
   const int NT = 1;
#  endif

   double ***OMP_Data=NULL, ***OMP_Weight=NULL;
   long   ***OMP_NCell=NULL;

   Aux_AllocateArray3D( OMP_Data,   NProf, NT, NBin );
   Aux_AllocateArray3D( OMP_Weight, NProf, NT, NBin );
   Aux_AllocateArray3D( OMP_NCell,  NProf, NT, NBin );

// initialize profile arrays
// --> use memset() instead to improve performance
//     OMP_Data and OMP_Weight are initialized during their first assignment
   for (int p=0; p<NProf; p++)
   for (int t=0; t<NT;    t++)
   {
      memset( OMP_NCell[p][t], 0, sizeof(long)*NBin );
   }

   real (*Patch_Data)       [8][PS1][PS1][PS1] = new real [NT][8][PS1][PS1][PS1];  // field data            of each cell
   int  (*Patch_BinMin_Rad )[8][PS1][PS1][PS1] = new int  [NT][8][PS1][PS1][PS1];  // minimum radial    bin of each cell
   int  (*Patch_BinMax_Rad )[8][PS1][PS1][PS1] = new int  [NT][8][PS1][PS1][PS1];  // maximum radial    bin of each cell
   int  (*Patch_BinMin_Tht )[8][PS1][PS1][PS1] = new int  [NT][8][PS1][PS1][PS1];  // minimum polar     bin of each cell
   int  (*Patch_BinMax_Tht )[8][PS1][PS1][PS1] = new int  [NT][8][PS1][PS1][PS1];  // maximum polar     bin of each cell
   int  (*Patch_BinMin_Phi )[8][PS1][PS1][PS1] = new int  [NT][8][PS1][PS1][PS1];  // minimum azimuthal bin of each cell
   int  (*Patch_BinMax_Phi )[8][PS1][PS1][PS1] = new int  [NT][8][PS1][PS1][PS1];  // maximum azimuthal bin of each cell


// set global constants
   const int    CellSkip       = NULL_INT;
   const int    WeightByVolume = 1;
   const int    WeightByMass   = 2;
   const double _dr            = 1.0 / BinSize_Linear;
   const double _dtheta        = (double) NTheta / M_PI;
   const double _dphi          = (double) NPhi   / TWOPI;
   const double HalfBox [3]    = { 0.5*amr->BoxSize[0], 0.5*amr->BoxSize[1], 0.5*amr->BoxSize[2] };
   const bool   Periodic[3]    = { OPT__BC_FLU[0] == BC_FLU_PERIODIC,
                                   OPT__BC_FLU[2] == BC_FLU_PERIODIC,
                                   OPT__BC_FLU[4] == BC_FLU_PERIODIC };


// temporarily overwrite OPT__INT_TIME
// --> necessary because SetTempIntPara() called by Prepare_PatchData() relies on OPT__INT_TIME
// --> must restore it before exiting this routine
// --> note that modifying OPT__INT_TIME renders this routine thread-unsafe
//###REVISE: make temporal interpolation a function parameter in Prepare_PatchData() to solve this thread-safety issue
   const bool IntTimeBackup = OPT__INT_TIME;
   OPT__INT_TIME = ( PrepTimeIn >= 0.0 ) ? true : false;


// loop over all levels
   for (int lv=0; lv<=MAX_LEVEL; lv++)
   {
      if ( NPatchTotal[lv] == 0 )   continue;

      const double dh      = amr->dh[lv];
      const double dh_half = 0.5*amr->dh[lv];
      const double dv      = CUBE( dh );


//    determine the temporal interpolation parameters
//    --> mainly for computing cell mass for weighting; Prepare_PatchData() needs PrepTime
      const int    FluSg0   = amr->FluSg[lv];
      const double PrepTime = ( PrepTimeIn >= 0.0 ) ? PrepTimeIn : amr->FluSgTime[lv][FluSg0];

      bool FluIntTime;
      int  FluSg, FluSg_IntT;
      real FluWeighting, FluWeighting_IntT;

      SetTempIntPara( lv, FluSg0, PrepTime, amr->FluSgTime[lv][FluSg0], amr->FluSgTime[lv][1-FluSg0],
                      FluIntTime, FluSg, FluSg_IntT, FluWeighting, FluWeighting_IntT );


//    initialize the particle density array (rho_ext) and collect particles to the target level
#     ifdef PARTICLE
      const bool TimingSendPar_No = false;
      const bool JustCountNPar_No = false;
#     ifdef LOAD_BALANCE
      const bool PredictPos       = amr->Par->PredictPos;
      const bool SibBufPatch      = true;
      const bool FaSibBufPatch    = true;
#     else
      const bool PredictPos       = false;
      const bool SibBufPatch      = NULL_BOOL;
      const bool FaSibBufPatch    = NULL_BOOL;
#     endif

      if ( NeedPar )
      {
//       these two routines should NOT be put inside an OpenMP parallel region
         Par_CollectParticle2OneLevel( lv, _PAR_MASS|_PAR_POSX|_PAR_POSY|_PAR_POSZ, _PAR_TYPE, PredictPos,
                                       PrepTime, SibBufPatch, FaSibBufPatch, JustCountNPar_No, TimingSendPar_No );

         Prepare_PatchData_InitParticleDensityArray( lv, PrepTime );
      } // if ( NeedPar )
#     endif // #ifdef PARTICLE


//    different OpenMP threads and MPI processes first compute profiles independently
//    --> their data will be combined later
#     pragma omp parallel
      {
#        ifdef OPENMP
         const int TID = omp_get_thread_num();
#        else
         const int TID = 0;
#        endif

//       use the "static" schedule for reproducibility
#        pragma omp for schedule( static )
         for (int PID0=0; PID0<amr->NPatchComma[lv][1]; PID0+=8)
         {
//          skip non-leaf patches
            bool SkipPatch[8], SkipPatchGroup=true;

            for (int LocalID=0; LocalID<8; LocalID++)
            {
               const int PID = PID0 + LocalID;
               SkipPatch[LocalID] = false;

               if ( amr->patch[0][lv][PID]->son != -1  &&  lv != MAX_LEVEL )   SkipPatch[LocalID] = true;
               if ( ! SkipPatch[LocalID]                                   )   SkipPatchGroup     = false;
            } // for (int LocalID=0; LocalID<8; LocalID++)

            if ( SkipPatchGroup )   continue;


//          store the bin indices associated with each cell
//          --> do it before looping over all target fields to avoid redundant calculations
            for (int LocalID=0; LocalID<8; LocalID++)
            {
               if ( SkipPatch[LocalID] )  continue;

               const int    PID = PID0 + LocalID;
               const double x0  = amr->patch[0][lv][PID]->EdgeL[0] + dh_half - Center[0];
               const double y0  = amr->patch[0][lv][PID]->EdgeL[1] + dh_half - Center[1];
               const double z0  = amr->patch[0][lv][PID]->EdgeL[2] + dh_half - Center[2];

               for (int k=0; k<PS1; k++)  {  double dz = z0 + k*dh;
                                             if ( Periodic[2] ) {
                                                if      ( dz > +HalfBox[2] )  {  dz -= amr->BoxSize[2];  }
                                                else if ( dz < -HalfBox[2] )  {  dz += amr->BoxSize[2];  }
                                             }
               for (int j=0; j<PS1; j++)  {  double dy = y0 + j*dh;
                                             if ( Periodic[1] ) {
                                                if      ( dy > +HalfBox[1] )  {  dy -= amr->BoxSize[1];  }
                                                else if ( dy < -HalfBox[1] )  {  dy += amr->BoxSize[1];  }
                                             }
               for (int i=0; i<PS1; i++)  {  double dx = x0 + i*dh;
                                             if ( Periodic[0] ) {
                                                if      ( dx > +HalfBox[0] )  {  dx -= amr->BoxSize[0];  }
                                                else if ( dx < -HalfBox[0] )  {  dx += amr->BoxSize[0];  }
                                             }
//                find the minimum/maximum radius among the eight cell vertices
                  double MinRad_Cell, MaxRad_Cell;

                  Aux_GetMinMax_Vertex_Radius( dx, dy, dz, dh_half, &MinRad_Cell, &MaxRad_Cell );


                  if ( MinRad_Cell >= MaxRad )
                  {
                     Patch_BinMin_Rad[TID][LocalID][k][j][i] = CellSkip;
                     Patch_BinMax_Rad[TID][LocalID][k][j][i] = CellSkip;
                  }

                  else
                  {
//                   find the minimum/maximum azimuthal angle and polar angle among the eight cell vertices
                     double MinTht_Cell, MaxTht_Cell, MinPhi_Cell, MaxPhi_Cell;

                     Aux_GetMinMax_Vertex_Angle( dx, dy, dz, dh_half, &MinTht_Cell, &MaxTht_Cell, &MinPhi_Cell, &MaxPhi_Cell );


//                   find the index range of bins overlapping with this cell
//                   --> use Cell_Skip for handling the vertex lies on the bin edge
//                   case (a): the center coordinate is inside the cell
//                             --> contribute to all rays
                     if (  ( fabs(dx) < dh_half )  &&
                           ( fabs(dy) < dh_half )  &&
                           ( fabs(dz) < dh_half )      )
                     {
                        Patch_BinMin_Rad[TID][LocalID][k][j][i] = 0;
                        Patch_BinMax_Rad[TID][LocalID][k][j][i] = ( MaxRad_Cell <= MaxRad_Linear )
                                                                ? int( MaxRad_Cell * _dr )
                                                                : Mis_BinarySearch_Real( Edge, NRadius_Linear, NRadius-1, MaxRad_Cell );
                        Patch_BinMin_Tht[TID][LocalID][k][j][i] = 0;
                        Patch_BinMax_Tht[TID][LocalID][k][j][i] = NTheta - 1;
                        Patch_BinMin_Phi[TID][LocalID][k][j][i] = 0;
                        Patch_BinMax_Phi[TID][LocalID][k][j][i] = NPhi - 1;
                     }

                     else
                     {
                        Patch_BinMin_Rad[TID][LocalID][k][j][i] = ( MinRad_Cell <= MaxRad_Linear )
                                                                ? int( MinRad_Cell * _dr )
                                                                : Mis_BinarySearch_Real( Edge, NRadius_Linear, NRadius-1, MinRad_Cell );
                        Patch_BinMax_Rad[TID][LocalID][k][j][i] = ( MaxRad_Cell <= MaxRad_Linear )
                                                                ? int( MaxRad_Cell * _dr )
                                                                : Mis_BinarySearch_Real( Edge, NRadius_Linear, NRadius-1, MaxRad_Cell );

//                      case (b): the cell intersects the reference z-axis
//                                --> contributes to rays at any azimuthal angle
                        if (  ( fabs(dx) < dh_half )  &&  ( fabs(dy) < dh_half )  )
                        {
                           Patch_BinMin_Tht[TID][LocalID][k][j][i] = ( dz > 0.0 ) ? 0                            : int( MinTht_Cell * _dtheta );
                           Patch_BinMax_Tht[TID][LocalID][k][j][i] = ( dz > 0.0 ) ? int( MaxTht_Cell * _dtheta ) : NTheta - 1;
                           Patch_BinMin_Phi[TID][LocalID][k][j][i] = 0;
                           Patch_BinMax_Phi[TID][LocalID][k][j][i] = NPhi - 1;
                        }

                        else
                        {
                           Patch_BinMin_Tht[TID][LocalID][k][j][i] = int( MinTht_Cell * _dtheta );
                           Patch_BinMax_Tht[TID][LocalID][k][j][i] = int( MaxTht_Cell * _dtheta );

//                         case (c): the cell intersects the positive xz-plane (azimuthal discontinuity)
//                                   --> handle the index range of the azimuthal angle carefully
                           if (  ( dx > dh_half )  &&  ( fabs(dy) < dh_half )  )
                           {
                              MinPhi_Cell = atan2( dy - dh_half, dx - dh_half ) + TWOPI;
                              MaxPhi_Cell = atan2( dy + dh_half, dx - dh_half );

//                            add NPhi to enable looping later
                              Patch_BinMin_Phi[TID][LocalID][k][j][i] = int( MinPhi_Cell * _dphi );
                              Patch_BinMax_Phi[TID][LocalID][k][j][i] = ( NPhi > 1 )
                                                                      ? int( MaxPhi_Cell * _dphi ) + NPhi
                                                                      : int( MaxPhi_Cell * _dphi );
                           }

//                         case (d): general case
                           else
                           {
                              Patch_BinMin_Phi[TID][LocalID][k][j][i] = int( MinPhi_Cell * _dphi );
                              Patch_BinMax_Phi[TID][LocalID][k][j][i] = int( MaxPhi_Cell * _dphi );
                           }
                        }
                     }

//                   handle the maximum bin indices carefully to prevent redundant mapping
                     double BinMax_Tht = MaxTht_Cell * _dtheta;
                     double BinMax_Phi = MaxPhi_Cell * _dphi;

                     if (  Mis_CompareRealValue( floor(BinMax_Tht), BinMax_Tht, NULL, false )  )
                        Patch_BinMax_Tht[TID][LocalID][k][j][i] -= 1;

                     if (  Mis_CompareRealValue( floor(BinMax_Phi), BinMax_Phi, NULL, false )  )
                        Patch_BinMax_Phi[TID][LocalID][k][j][i] -= 1;
                  } // if ( MaxRad_Cell > MaxRad ) ... else ...
               }}} // i,j,k
            } // for (int LocalID=0; LocalID<8; LocalID++)


//          compute one field at a time
            for (int p=0; p<NProf; p++)
            {
//             collect the data of the target field
               switch ( TVarBitIdx[p] )
               {
//                _VELR is currently not supported by Prepare_PatchData()
#                 ifdef _VELR
                  case _VELR:
                     for (int LocalID=0; LocalID<8; LocalID++)
                     {
                        if ( SkipPatch[LocalID] )  continue;

                        const int PID = PID0 + LocalID;
                        const real (*FluidPtr     )[PS1][PS1][PS1] =                  amr->patch[ FluSg      ][lv][PID]->fluid;
                        const real (*FluidPtr_IntT)[PS1][PS1][PS1] = ( FluIntTime ) ? amr->patch[ FluSg_IntT ][lv][PID]->fluid : NULL;

                        const double x0 = amr->patch[0][lv][PID]->EdgeL[0] + dh_half - Center[0];
                        const double y0 = amr->patch[0][lv][PID]->EdgeL[1] + dh_half - Center[1];
                        const double z0 = amr->patch[0][lv][PID]->EdgeL[2] + dh_half - Center[2];

                        for (int k=0; k<PS1; k++)  {  double dz = z0 + k*dh;
                                                      if ( Periodic[2] ) {
                                                         if      ( dz > +HalfBox[2] )  {  dz -= amr->BoxSize[2];  }
                                                         else if ( dz < -HalfBox[2] )  {  dz += amr->BoxSize[2];  }
                                                      }
                        for (int j=0; j<PS1; j++)  {  double dy = y0 + j*dh;
                                                      if ( Periodic[1] ) {
                                                         if      ( dy > +HalfBox[1] )  {  dy -= amr->BoxSize[1];  }
                                                         else if ( dy < -HalfBox[1] )  {  dy += amr->BoxSize[1];  }
                                                      }
                        for (int i=0; i<PS1; i++)  {  double dx = x0 + i*dh;
                                                      if ( Periodic[0] ) {
                                                         if      ( dx > +HalfBox[0] )  {  dx -= amr->BoxSize[0];  }
                                                         else if ( dx < -HalfBox[0] )  {  dx += amr->BoxSize[0];  }
                                                      }

                           if ( Patch_BinMin_Rad[TID][LocalID][k][j][i] == CellSkip )   continue;

                           const double r        = sqrt( SQR(dx) + SQR(dy) + SQR(dz) );
                           const real _Dens      =                  (real)1.0 / FluidPtr     [DENS][k][j][i];
                           const real _Dens_IntT = ( FluIntTime ) ? (real)1.0 / FluidPtr_IntT[DENS][k][j][i] : NULL_REAL;

                           real VelR;
                           if ( r == 0.0 ) {
                              VelR = (real)0.0;    // take care of the corner case where the profile center coincides with a cell center
                           }

                           else {
                              VelR = ( FluIntTime )
                                   ? ( FluWeighting     *( FluidPtr     [MOMX][k][j][i]*dx +
                                                           FluidPtr     [MOMY][k][j][i]*dy +
                                                           FluidPtr     [MOMZ][k][j][i]*dz )*_Dens
                                     + FluWeighting_IntT*( FluidPtr_IntT[MOMX][k][j][i]*dx +
                                                           FluidPtr_IntT[MOMY][k][j][i]*dy +
                                                           FluidPtr_IntT[MOMZ][k][j][i]*dz )*_Dens_IntT ) / r
                                   :                     ( FluidPtr     [MOMX][k][j][i]*dx +
                                                           FluidPtr     [MOMY][k][j][i]*dy +
                                                           FluidPtr     [MOMZ][k][j][i]*dz )*_Dens / r;
                           }

                           Patch_Data[TID][LocalID][k][j][i] = VelR;
                        }}} // i,j,k
                     } // for (int LocalID=0; LocalID<8; LocalID++)
                  break; // _VELR
#                 endif // #ifdef _VELR

                  default:
                     const int  NGhost             = 0;
                     const int  NPG                = 1;
                     const bool IntPhase_No        = false;
                     const real MinDens_No         = -1.0;
                     const real MinPres_No         = -1.0;
                     const real MinTemp_No         = -1.0;
                     const real MinEntr_No         = -1.0;
                     const bool DE_Consistency_Yes = true;

                     Prepare_PatchData( lv, PrepTime, &Patch_Data[TID][0][0][0][0], NULL, NGhost, NPG, &PID0,
                                        TVarBitIdx[p], _NONE, INT_NONE, INT_NONE, UNIT_PATCH, NSIDE_00, IntPhase_No,
                                        OPT__BC_FLU, BC_POT_NONE, MinDens_No, MinPres_No, MinTemp_No, MinEntr_No,
                                        DE_Consistency_Yes );
                  break; // default
               } // switch ( TVarBitIdx[p] )


//             set the weight field
//###REVISE: allow users to choose the weight field
               int WeightField=-1;

               switch ( TVarBitIdx[p] )
               {
#                 ifdef _VELX
                  case _VELX : WeightField = WeightByMass;     break;
#                 endif
#                 ifdef _VELY
                  case _VELY : WeightField = WeightByMass;     break;
#                 endif
#                 ifdef _VELZ
                  case _VELZ : WeightField = WeightByMass;     break;
#                 endif
#                 ifdef _VELR
                  case _VELR : WeightField = WeightByMass;     break;
#                 endif
#                 ifdef _POTE
                  case _POTE : WeightField = WeightByMass;     break;
#                 endif
                  default    : WeightField = WeightByVolume;   break;
               } // switch ( TVarBitIdx[p] )


//             compute the ray profile
               for (int LocalID=0; LocalID<8; LocalID++)
               {
                  if ( SkipPatch[LocalID] )  continue;

                  const int PID = PID0 + LocalID;
                  const real (*DensPtr     )[PS1][PS1] =                  amr->patch[ FluSg      ][lv][PID]->fluid[DENS];
                  const real (*DensPtr_IntT)[PS1][PS1] = ( FluIntTime ) ? amr->patch[ FluSg_IntT ][lv][PID]->fluid[DENS] : NULL;

                  for (int k=0; k<PS1; k++)
                  for (int j=0; j<PS1; j++)
                  for (int i=0; i<PS1; i++)
                  {
                     if ( Patch_BinMin_Rad[TID][LocalID][k][j][i] == CellSkip )   continue;

//                   compute the weight
                     real Weight;
                     switch ( WeightField )
                     {
                        case WeightByMass   :   Weight = ( FluIntTime )
                                                       ? ( FluWeighting     *DensPtr     [k][j][i]
                                                         + FluWeighting_IntT*DensPtr_IntT[k][j][i] )*dv
                                                       :                     DensPtr     [k][j][i]  *dv;
                        break;

                        case WeightByVolume :   Weight = dv;
                        break;

                        default:
                           Aux_Error( ERROR_INFO, "unsupported weight field (%d) !!\n", WeightField );
                           exit( 1 );
                     }

//                   update the ray profile
                     for (int kk=Patch_BinMin_Phi[TID][LocalID][k][j][i]; kk<=Patch_BinMax_Phi[TID][LocalID][k][j][i]; kk++)  {  const int iv1    = NTheta  * ( kk % NPhi );
                     for (int jj=Patch_BinMin_Tht[TID][LocalID][k][j][i]; jj<=Patch_BinMax_Tht[TID][LocalID][k][j][i]; jj++)  {  const int iv2    = NRadius * ( jj + iv1  );
                     for (int ii=Patch_BinMin_Rad[TID][LocalID][k][j][i]; ii<=Patch_BinMax_Rad[TID][LocalID][k][j][i]; ii++)  {  const int BinIdx = ii + iv2;

                        if ( OMP_NCell[p][TID][BinIdx] == 0L )
                        {
                           OMP_Data  [p][TID][BinIdx] = Patch_Data[TID][LocalID][k][j][i]*Weight;
                           OMP_Weight[p][TID][BinIdx] = Weight;
                        }

                        else
                        {
                           OMP_Data  [p][TID][BinIdx] += Patch_Data[TID][LocalID][k][j][i]*Weight;
                           OMP_Weight[p][TID][BinIdx] += Weight;
                        }

                        OMP_NCell[p][TID][BinIdx] ++;

                     }}} // ii,jj,kk
                  } // i,j,k
               } // for (int LocalID=0; LocalID<8; LocalID++)
            } // for (int p=0; p<NProf; p++)
         } // for (int PID0=0; PID0<amr->NPatchComma[lv][1]; PID0+=8)
      } // OpenMP parallel region


//    free particle resources
//    --> these two routines should NOT be put inside an OpenMP parallel region
#     ifdef PARTICLE
      if ( NeedPar )
      {
         Par_CollectParticle2OneLevel_FreeMemory( lv, SibBufPatch, FaSibBufPatch );

         Prepare_PatchData_FreeParticleDensityArray( lv );
      }
#     endif
   } // for (int lv=MinLv; lv<=MaxLv; lv++)


#  pragma omp parallel
   {
//    initialize Data and Weight of empty bins in the first OpenMP thread to zero for later summation
#     pragma omp for schedule( static ) collapse( 2 )
      for (int p=0; p<NProf; p++)  {
      for (int b=0; b<NBin;  b++)  {
         if ( OMP_NCell[p][0][b] == 0L )
         {
            OMP_Data  [p][0][b] = 0.0;
            OMP_Weight[p][0][b] = 0.0;
         }
      }}


//    sum over all OpenMP threads
      for (int p=0; p<NProf; p++)  {
      for (int t=1; t<NT;    t++)  {
#     pragma omp for schedule( static )
      for (int b=0; b<NBin;  b++)  {
         if ( OMP_NCell[p][t][b] != 0L )
         {
            OMP_Data  [p][0][b] += OMP_Data  [p][t][b];
            OMP_Weight[p][0][b] += OMP_Weight[p][t][b];
            OMP_NCell [p][0][b] += OMP_NCell [p][t][b];
         }
      }}}
   } // OpenMP parallel region


// collect data from all ranks (in-place reduction)
#  ifndef SERIAL
   for (int p=0; p<NProf; p++)
   {
      MPI_Allreduce( MPI_IN_PLACE, OMP_Data  [p][0], NBin, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
      MPI_Allreduce( MPI_IN_PLACE, OMP_Weight[p][0], NBin, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
      MPI_Allreduce( MPI_IN_PLACE, OMP_NCell [p][0], NBin, MPI_LONG,   MPI_SUM, MPI_COMM_WORLD );
   }
#  endif


// compute profile in each rank
#  pragma omp parallel
   {
      for (int p=0; p<NProf; p++)
#     pragma omp for schedule( static )
      for (int b=0; b<NBin;  b++)
      {
         Ray[p]->NCell[b] = OMP_NCell[p][0][b];
         Ray[p]->Data [b] = ( Ray[p]->NCell[b] > 0L ) ? OMP_Data[p][0][b] / OMP_Weight[p][0][b] : 0.0;
      }
   } // OpenMP parallel region


// free per-thread arrays
   Aux_DeallocateArray3D( OMP_Data   );
   Aux_DeallocateArray3D( OMP_Weight );
   Aux_DeallocateArray3D( OMP_NCell  );

   delete [] Patch_Data;
   delete [] Patch_BinMin_Rad;
   delete [] Patch_BinMax_Rad;
   delete [] Patch_BinMin_Tht;
   delete [] Patch_BinMax_Tht;
   delete [] Patch_BinMin_Phi;
   delete [] Patch_BinMax_Phi;


// restore the original temporal interpolation setup
   OPT__INT_TIME = IntTimeBackup;

} // FUNCTION : Aux_ComputeRay



//-------------------------------------------------------------------------------------------------------
// Function    :  Aux_GetMinMax_Vertex_Radius
// Description :  Find the minimum and maximum radius among the cell vertices
//
// Parameter   :  x/y/z     : Physical coordinates of cell center
//                HalfWidth : Half cell width
//                RadMin    : Variable to store the output minimum radius
//                RadMax    : Variable to store the output maximum radius
//
// Return      :  RadMin, RadMax
//-------------------------------------------------------------------------------------------------------
void Aux_GetMinMax_Vertex_Radius( const double x, const double y, const double z, const double HalfWidth,
                                  double *RadMin, double *RadMax )
{

   *RadMin =  HUGE_NUMBER;
   *RadMax = -HUGE_NUMBER;

   for (int k=-1; k<=1; k+=2)  {  const double zz = z + k * HalfWidth;  const double zz2 = SQR(zz);
   for (int j=-1; j<=1; j+=2)  {  const double yy = y + j * HalfWidth;  const double yy2 = SQR(yy);
   for (int i=-1; i<=1; i+=2)  {  const double xx = x + i * HalfWidth;  const double xx2 = SQR(xx);

      const double Rad = sqrt( xx2 + yy2 + zz2 );

      *RadMin = fmin( *RadMin, Rad );
      *RadMax = fmax( *RadMax, Rad );

   }}}

} // FUNCTION : Aux_GetMinMax_Vertex_Radius



//-------------------------------------------------------------------------------------------------------
// Function    :  Aux_GetMinMax_Vertex_Angle
// Description :  Find the minimum and maximum azimuthal angle and polar angle among the cell vertices
//
// Parameter   :  x/y/z     : Physical coordinates of cell center
//                HalfWidth : Half cell width
//                ThtMin    : Variable to store the output minimum polar     angle
//                ThtMax    : Variable to store the output maximum polar     angle
//                PhiMin    : Variable to store the output minimum azimuthal angle
//                PhiMax    : Variable to store the output maximum azimuthal angle
//
// Return      :  ThtMin, ThtMax, PhiMin, PhiMax
//-------------------------------------------------------------------------------------------------------
void Aux_GetMinMax_Vertex_Angle( const double x, const double y, const double z, const double HalfWidth,
                                 double *ThtMin, double *ThtMax, double *PhiMin, double *PhiMax )
{

   *ThtMin =  HUGE_NUMBER;
   *ThtMax = -HUGE_NUMBER;
   *PhiMin =  HUGE_NUMBER;
   *PhiMax = -HUGE_NUMBER;

   for (int k=-1; k<=1; k+=2)  {  const double zz = z + k * HalfWidth;  const double zz2 = SQR(zz);
   for (int j=-1; j<=1; j+=2)  {  const double yy = y + j * HalfWidth;  const double yy2 = SQR(yy);
   for (int i=-1; i<=1; i+=2)  {  const double xx = x + i * HalfWidth;  const double xx2 = SQR(xx);

      const double Rad = sqrt( xx2 + yy2 + zz2 );

//    exclude the corner case where the vertex lies on the reference z-axis
      if ( xx != 0.0  ||  yy != 0.0 )
      {
         double Phi = ( yy >= 0.0 )
                    ? atan2( yy, xx )
                    : atan2( yy, xx ) + TWOPI;

         if (  ( Phi == 0.0 )  &&  ( y < 0.0 )  )   Phi += TWOPI;

         *PhiMin = fmin( *PhiMin, Phi );
         *PhiMax = fmax( *PhiMax, Phi );
      }


//    exclude the corner case where the vertex coincides with the reference center
      if ( Rad != 0.0 )
      {
         const double Tht = acos( zz / Rad );

         *ThtMin = fmin( *ThtMin, Tht );
         *ThtMax = fmax( *ThtMax, Tht );
      }
   }}} // i,j,k

} // FUNCTION : Aux_GetMinMax_Vertex_Angle
