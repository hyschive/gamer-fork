#include "GAMER.h"

static void Aux_GetMinMax_Vertex_Radius( const double x, const double y, const double z, const double HalfWidth,
                                         double *rad_min, double *rad_max );

static void Aux_GetMinMax_Vertex_Angle( const double x, const double y, const double z, const double HalfWidth,
                                        double *theta_min, double *theta_max, double *phi_min, double *phi_max );

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
//                3. Use cell volume as the weighting of each cell
//                   --> Use mass as the weighting for temperature
//                4. Support computing multiple fields
//                   --> The order of fields to be returned follows TVarBitIdx[]
//
// Parameter   :  Ray            : Profile_t object array to store the results
//                Center         : Target center coordinates
//                                 --> Must be the box center or the cell center
//                Edge           : Edge of bins in the radial direction
//                NRadius_Linear : Number of linear bins in the radial    direction
//                NRadius        : Number of        bins in the radial    direction
//                NTheta         : Number of        bins in the polar     direction
//                NPhi           : Number of        bins in the azimuthal direction
//                BinSize_Linear : Bin size of linear bins in the radial direction
//                MaxRad_Linear  : Maximum radius of linear bins
//                MaxRad         : Maximum radius for computing the rays
//                TVarBitIdx     : Bitwise indices of target variables for computing the rays
//                                 --> Supported indices (defined in Macro.h):
//                                        HYDRO : _DENS, _MOMX, _MOMY, _MOMZ, _ENGY, _TEMP
//                                        ELBDM : _DENS, _REAL, _IMAG
//                                 --> For a passive scalar with an integer field index FieldIdx returned by AddField(),
//                                     one can convert it to a bitwise field index by BIDX(FieldIdx)
//                NProf          : Number of target fields in Ray
//                PrepTime       : Target physical time to prepare data
//                                 --> If PrepTime<0, turn off temporal interpolation and always use the most recent data
//
// Example     :  const double      Center[3]      = { amr->BoxCenter[0], amr->BoxCenter[1], amr->BoxCenter[2] };
//                const int         NRadius_Linear = 128;
//                const int         NRadius        = 256;
//                const int         NTheta         = 1;
//                const int         NPhi           = 1;
//                const double      BinSize_Linear = amr->dh[MAX_LEVEL];
//                const double      MaxRad_Linear  = 0.25*amr->BoxSize[0];
//                const double      MaxRad         = 0.5 *amr->BoxSize[0];
//                const long        TVarBitIdx[]   = { _DENS, _TEMP };
//                const int         NProf          = 2;
//                const double      PrepTime       = -1.0;
//
//                Profile_t Ray_Dens, Ray_Temp;
//                Profile_t *Ray[NProf] = { &Ray_Dens, &Ray_Temp };
//
//                Aux_ComputeRay( Ray, Center, Edge, NRadius_Linear, NRadius, NTheta, NPhi
//                                BinSize_Linear, MaxRad_Linear, MaxRad, TVarBitIdx, NProf, PrepTime );
//
// Return      :  Ray
//-------------------------------------------------------------------------------------------------------
void Aux_ComputeRay( Profile_t *Ray[], const double Center[], const double Edge[],
                     const int NRadius_Linear, const int NRadius, const int NTheta, const int NPhi,
                     const double BinSize_Linear, const double MaxRad_Linear, const double MaxRad,
                     const long TVarBitIdx[], const int NProf, const double PrepTime )
{

// check
#  ifdef GAMER_DEBUG
   if ( NRadius < NRadius_Linear )
      Aux_Error( ERROR_INFO, "NRadius (%14.7e) < NRadius_Linear (%14.7e) !!\n", NRadius, NRadius_Linear );

   if ( MaxRad < MaxRad_Linear )
      Aux_Error( ERROR_INFO, "MaxRad (%14.7e) < MaxRad_Linear (%14.7e) !!\n", MaxRad, MaxRad_Linear );
#  endif


// precompute the integer indices of intrinsic fluid fields for better performance
   const int IdxUndef = -1;
   int TFluIntIdx[NProf];

   for (int p=0; p<NProf; p++)
   {
      TFluIntIdx[p] = IdxUndef;

      for (int v=0; v<NCOMP_TOTAL; v++)
         if ( TVarBitIdx[p] & (1L<<v) )   TFluIntIdx[p] = v;
   }


// allocate all member arrays of Profile
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


// collect profile data in this rank
   const double _dr         = 1.0 / BinSize_Linear;
   const double _dtheta     = (double) NTheta / M_PI;
   const double _dphi       = (double) NPhi / ( 2.0 * M_PI );
   const double HalfBox[3]  = { 0.5*amr->BoxSize[0], 0.5*amr->BoxSize[1], 0.5*amr->BoxSize[2] };
   const bool   Periodic[3] = { OPT__BC_FLU[0] == BC_FLU_PERIODIC,
                                OPT__BC_FLU[2] == BC_FLU_PERIODIC,
                                OPT__BC_FLU[4] == BC_FLU_PERIODIC };

#  pragma omp parallel
   {
#     ifdef OPENMP
      const int TID = omp_get_thread_num();
#     else
      const int TID = 0;
#     endif

//    initialize arrays
      for (int p=0; p<NProf; p++)
      for (int b=0; b<NBin;  b++)
      {
         OMP_Data  [p][TID][b] = 0.0;
         OMP_Weight[p][TID][b] = 0.0;
         OMP_NCell [p][TID][b] = 0;
      }

//    allocate passive scalar arrays
#     if ( MODEL == HYDRO )
      real *Passive      = new real [NCOMP_PASSIVE];
      real *Passive_IntT = new real [NCOMP_PASSIVE];
#     endif

//    loop over all levels
      for (int lv=0; lv<=MAX_LEVEL; lv++)
      {
         if ( NPatchTotal[lv] == 0 )   continue;

         const double dh      = amr->dh[lv];
         const double dv      = CUBE( dh );
         const double dh_half = 0.5*dh;


//       determine temporal interpolation parameters
         bool FluIntTime = false;
         int  FluSg      = amr->FluSg[lv];
         int  FluSg_IntT;
         real FluWeighting, FluWeighting_IntT;

#        ifdef MHD
         bool MagIntTime = false;
         int  MagSg      = amr->MagSg[lv];
         int  MagSg_IntT;
         real MagWeighting, MagWeighting_IntT;
#        endif

         if ( PrepTime >= 0.0 )
         {
//          fluid
            const int FluSg0 = amr->FluSg[lv];
            SetTempIntPara( lv, FluSg0, PrepTime, amr->FluSgTime[lv][FluSg0], amr->FluSgTime[lv][1-FluSg0],
                            FluIntTime, FluSg, FluSg_IntT, FluWeighting, FluWeighting_IntT );

//          magnetic field
#           ifdef MHD
            const int MagSg0 = amr->MagSg[lv];
            SetTempIntPara( lv, MagSg0, PrepTime, amr->MagSgTime[lv][MagSg0], amr->MagSgTime[lv][1-MagSg0],
                            MagIntTime, MagSg, MagSg_IntT, MagWeighting, MagWeighting_IntT );
#           endif
         }


//       use the "static" schedule for reproducibility
#        pragma omp for schedule( static )
         for (int PID=0; PID<amr->NPatchComma[lv][1]; PID++)
         {
//          skip non-leaf patches
            if ( amr->patch[0][lv][PID]->son != -1 )   continue;

            const real (*FluidPtr)[PS1][PS1][PS1] = amr->patch[ FluSg ][lv][PID]->fluid;

//          pointer for temporal interpolation
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

//             find the min/max radius, azimuthal angle, and polar angle among the cell vertices
               double rad_min, rad_max, theta_min, theta_max, phi_min, phi_max;

               Aux_GetMinMax_Vertex_Radius( dx, dy, dz, dh_half, &rad_min, &rad_max );

               if ( rad_min < MaxRad )
               {
                  Aux_GetMinMax_Vertex_Angle( dx, dy, dz, dh_half, &theta_min, &theta_max, &phi_min, &phi_max );

//                find the index range of bins that overlay with this cell
                  int idx_rad_min, idx_rad_max, idx_theta_min, idx_theta_max, idx_phi_min, idx_phi_max;

//                deal with the special case that the center locates at this cell
//                --> contribute to all rays
                  if (  Mis_CompareRealValue( rad_min, rad_max, NULL, false )  )
                  {
                     idx_rad_min   = 0;
                     idx_rad_max   = ( rad_max <= MaxRad_Linear )
                                   ? int( rad_max * _dr )
                                   : Mis_BinarySearch_Real( Edge, NRadius_Linear, NRadius-1, rad_max );
                     idx_theta_min = 0;
                     idx_theta_max = NTheta-1;
                     idx_phi_min   = 0;
                     idx_phi_max   = NPhi-1;
                  }

                  else
                  {
                     idx_rad_min   = ( rad_min <= MaxRad_Linear )
                                   ? int( rad_min * _dr )
                                   : Mis_BinarySearch_Real( Edge, NRadius_Linear, NRadius-1, rad_min );
                     idx_rad_max   = ( rad_max <= MaxRad_Linear )
                                   ? int( rad_max * _dr )
                                   : Mis_BinarySearch_Real( Edge, NRadius_Linear, NRadius-1, rad_max );

//                   deal with the special case that the cell overlaps with the z-axis (dx = dy = 0)
//                   --> contribute to rays close to the z-axis with arbitrary azimuthal angle
                     if (  ( fabs(dx) < dh_half )  &&  ( fabs(dy) < dh_half )  )
                     {
                        idx_theta_min = ( dz > 0.0 ) ? 0                          : int( theta_max * _dtheta );
                        idx_theta_max = ( dz > 0.0 ) ? int( theta_min * _dtheta ) : NTheta-1;
                        idx_phi_min   = 0;
                        idx_phi_max   = NPhi-1;
                     }

                     else
                     {
                        idx_theta_min = int( theta_min * _dtheta );
                        idx_theta_max = MIN( int( theta_max * _dtheta ), NTheta-1 );

//                      deal with the special case that the cell overlay with the positive xz plane (dx > 0, dy = 0)
                        if (  ( dx > 0 )  &&  ( fabs(dy) < dh_half )  )
                        {
//                         get the correct azimuthal angle
                           phi_min = atan2( dy - dh_half, dx - dh_half ) + 2.0 * M_PI;
                           phi_max = atan2( dy + dh_half, dx - dh_half );

                           idx_phi_min = int( phi_min * _dphi );
                           idx_phi_max = ( NPhi > 1 )
                                       ? int( phi_max * _dphi ) + NPhi
                                       : int( phi_max * _dphi );
                        }

                        else
                        {
                           idx_phi_min = int( phi_min * _dphi );
                           idx_phi_max = int( phi_max * _dphi );
                        }
                     }
                  } // if (  Mis_CompareRealValue( rad_min, rad_max, NULL, false )  ) ... else ...


//                prepare passive scalars (for better sustainability, always do it even when unnecessary)
#                 if ( MODEL == HYDRO )
                  for (int v_out=0; v_out<NCOMP_PASSIVE; v_out++)
                  {
                     const int v_in = v_out + NCOMP_FLUID;

                     Passive     [v_out] = FluidPtr     [v_in][k][j][i];
                     if ( FluIntTime )
                     Passive_IntT[v_out] = FluidPtr_IntT[v_in][k][j][i];
                  }
#                 endif

                  for (int p=0; p<NProf; p++)
                  {
                     real Data   = NULL_REAL;
                     real Weight = NULL_REAL;

//                   intrinsic fluid fields
                     if ( TFluIntIdx[p] != IdxUndef )
                     {
                        Weight = dv;
                        Data   = ( FluIntTime )
                               ? ( FluWeighting     *FluidPtr     [ TFluIntIdx[p] ][k][j][i]
                                 + FluWeighting_IntT*FluidPtr_IntT[ TFluIntIdx[p] ][k][j][i] )*Weight
                               :                     FluidPtr     [ TFluIntIdx[p] ][k][j][i]  *Weight;
                     }

//                   other fields
                     else
                     {
                        switch ( TVarBitIdx[p] )
                        {
//                         derived fields
#                          if ( MODEL == HYDRO )
                           case _TEMP:
                           {
                              const bool CheckMinTemp_No = false;
#                             ifdef MHD
                              const real Emag            = MHD_GetCellCenteredBEnergyInPatch( lv, PID, i, j, k, MagSg      );
                              const real Emag_IntT       = ( MagIntTime )
                                                         ? MHD_GetCellCenteredBEnergyInPatch( lv, PID, i, j, k, MagSg_IntT )
                                                         : NULL_REAL;
#                             else
                              const real Emag            = NULL_REAL;
                              const real Emag_IntT       = NULL_REAL;
#                             endif
                              const real Temp = ( FluIntTime )
                                              ?   FluWeighting     *Hydro_Con2Temp( FluidPtr     [DENS][k][j][i],
                                                                                    FluidPtr     [MOMX][k][j][i],
                                                                                    FluidPtr     [MOMY][k][j][i],
                                                                                    FluidPtr     [MOMZ][k][j][i],
                                                                                    FluidPtr     [ENGY][k][j][i],
                                                                                    Passive,
                                                                                    CheckMinTemp_No, NULL_REAL, Emag,
                                                                                    EoS_DensEint2Temp_CPUPtr, EoS_AuxArray_Flt,
                                                                                    EoS_AuxArray_Int, h_EoS_Table )
                                                + FluWeighting_IntT*Hydro_Con2Temp( FluidPtr_IntT[DENS][k][j][i],
                                                                                    FluidPtr_IntT[MOMX][k][j][i],
                                                                                    FluidPtr_IntT[MOMY][k][j][i],
                                                                                    FluidPtr_IntT[MOMZ][k][j][i],
                                                                                    FluidPtr_IntT[ENGY][k][j][i],
                                                                                    Passive_IntT,
                                                                                    CheckMinTemp_No, NULL_REAL, Emag_IntT,
                                                                                    EoS_DensEint2Temp_CPUPtr, EoS_AuxArray_Flt,
                                                                                    EoS_AuxArray_Int, h_EoS_Table )
                                              :                     Hydro_Con2Temp( FluidPtr     [DENS][k][j][i],
                                                                                    FluidPtr     [MOMX][k][j][i],
                                                                                    FluidPtr     [MOMY][k][j][i],
                                                                                    FluidPtr     [MOMZ][k][j][i],
                                                                                    FluidPtr     [ENGY][k][j][i],
                                                                                    Passive,
                                                                                    CheckMinTemp_No, NULL_REAL, Emag,
                                                                                    EoS_DensEint2Temp_CPUPtr, EoS_AuxArray_Flt,
                                                                                    EoS_AuxArray_Int, h_EoS_Table );

//                            use cell mass as the weighting of each cell
                              Weight = ( FluIntTime )
                                     ? ( FluWeighting     *FluidPtr     [DENS][k][j][i]
                                       + FluWeighting_IntT*FluidPtr_IntT[DENS][k][j][i] )*dv
                                     :                     FluidPtr     [DENS][k][j][i]  *dv;
                              Data   = Temp*Weight;
                           }
                           break;
#                          endif // if ( MODEL == HYDRO )

                           default:
                              Aux_Error( ERROR_INFO, "unsupported field (%ld) !!\n", TVarBitIdx[p] );
                              exit( 1 );
                        } // switch ( TVarBitIdx[p] )
                     } // if ( TFluIntIdx[p] != IdxUndef ) ... else ...


//                   add the data to all the rays that overlay with this cell
                     for (int kk=idx_phi_min;   kk<=idx_phi_max;   kk++)  {
                     for (int jj=idx_theta_min; jj<=idx_theta_max; jj++)  {
                     for (int ii=idx_rad_min;   ii<=idx_rad_max;   ii++)  {

                        const int idx = ii + NRadius * (  jj + NTheta * ( kk % NPhi )  );

                        OMP_Data  [p][TID][idx] += Data;
                        OMP_Weight[p][TID][idx] += Weight;
                        OMP_NCell [p][TID][idx] ++;

                     }}} // ii,jj,kk
                  } // for (int p=0; p<NProf; p++)
               } // if ( rad_min < MaxRad )
            }}} // i,j,k
         } // for (int PID=0; PID<amr->NPatchComma[lv][1]; PID++)
      } // for (int lv=0; lv<=MAX_LEVEL; lv++)

#     if ( MODEL == HYDRO )
      delete [] Passive;         Passive      = NULL;
      delete [] Passive_IntT;    Passive_IntT = NULL;
#     endif

   } // OpenMP parallel region


// sum over all OpenMP threads
   for (int p=0; p<NProf; p++)
   for (int t=1; t<NT;    t++)
   for (int b=0; b<NBin;  b++)
   {
      OMP_Data  [p][0][b] += OMP_Data  [p][t][b];
      OMP_Weight[p][0][b] += OMP_Weight[p][t][b];
      OMP_NCell [p][0][b] += OMP_NCell [p][t][b];
   }


// collect data from all ranks (in-place reduction)
#  ifndef SERIAL
   for (int p=0; p<NProf; p++)
   {
      if ( MPI_Rank == 0 )
      {
         MPI_Reduce( MPI_IN_PLACE,     OMP_Data  [p][0], NBin, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );
         MPI_Reduce( MPI_IN_PLACE,     OMP_Weight[p][0], NBin, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );
         MPI_Reduce( MPI_IN_PLACE,     OMP_NCell [p][0], NBin, MPI_LONG,   MPI_SUM, 0, MPI_COMM_WORLD );
      }

      else
      {
         MPI_Reduce( OMP_Data  [p][0], NULL,             NBin, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );
         MPI_Reduce( OMP_Weight[p][0], NULL,             NBin, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );
         MPI_Reduce( OMP_NCell [p][0], NULL,             NBin, MPI_LONG,   MPI_SUM, 0, MPI_COMM_WORLD );
      }
   }
#  endif


// compute profile by the root rank
   if ( MPI_Rank == 0 )
   {
      for (int p=0; p<NProf; p++)
      for (int b=0; b<NBin;  b++)
      {
         Ray[p]->NCell[b] = OMP_NCell[p][0][b];
         Ray[p]->Data [b] = ( Ray[p]->NCell[b] > 0L ) ? OMP_Data[p][0][b] / OMP_Weight[p][0][b] : 0.0;
      }
   }


// broadcast data to all ranks
#  ifndef SERIAL
   for (int p=0; p<NProf; p++)
   {
      MPI_Bcast( Ray[p]->Data,  Ray[p]->NBin, MPI_DOUBLE, 0, MPI_COMM_WORLD );
      MPI_Bcast( Ray[p]->NCell, Ray[p]->NBin, MPI_DOUBLE, 0, MPI_COMM_WORLD );
   }
#  endif


// free per-thread arrays
   Aux_DeallocateArray3D( OMP_Data   );
   Aux_DeallocateArray3D( OMP_Weight );
   Aux_DeallocateArray3D( OMP_NCell  );

} // FUNCTION : Aux_ComputeRay



//-------------------------------------------------------------------------------------------------------
// Function    :  Aux_GetMinMax_Vertex_Radius
// Description :  Find the minimum and maximum radius among the cell vertices
//
// Parameter   :  x/y/z     : Physical coordinates of cell center
//                HalfWidth : Half cell width
//                rad_min   : Variable to store the minimum radius
//                rad_max   : Variable to store the maximum radius
//
// Return      :  rad_min, rad_max
//-------------------------------------------------------------------------------------------------------
void Aux_GetMinMax_Vertex_Radius( const double x, const double y, const double z, const double HalfWidth,
                                  double *rad_min, double *rad_max )
{

   *rad_min   =  HUGE_NUMBER;
   *rad_max   = -HUGE_NUMBER;

   for (int k=-1; k<=1; k+=2)  {  const double zz = z + k * HalfWidth;  const double zz2 = SQR(zz);
   for (int j=-1; j<=1; j+=2)  {  const double yy = y + j * HalfWidth;  const double yy2 = SQR(yy);
   for (int i=-1; i<=1; i+=2)  {  const double xx = x + i * HalfWidth;  const double xx2 = SQR(xx);

      const double radius = sqrt( xx2 + yy2 + zz2 );

      *rad_min = fmin( *rad_min, radius );
      *rad_max = fmax( *rad_max, radius );

   }}}

} // FUNCTION : Aux_GetMinMax_Vertex_Radius



//-------------------------------------------------------------------------------------------------------
// Function    :  Aux_GetMinMax_Vertex_Angle
// Description :  Find the minimum and maximum azimuthal angle and polar angle among the cell vertices
//
// Parameter   :  x/y/z     : Physical coordinates of cell center
//                HalfWidth : Half cell width
//                theta_min : Variable to store the minimum polar     angle
//                theta_max : Variable to store the maximum polar     angle
//                phi_min   : Variable to store the minimum azimuthal angle
//                phi_max   : Variable to store the maximum azimuthal angle
//
// Return      :  theta_min, theta_max, phi_min, phi_max
//-------------------------------------------------------------------------------------------------------
void Aux_GetMinMax_Vertex_Angle( const double x, const double y, const double z, const double HalfWidth,
                                 double *theta_min, double *theta_max, double *phi_min, double *phi_max )
{

   *theta_min =  HUGE_NUMBER;
   *theta_max = -HUGE_NUMBER;
   *phi_min   =  HUGE_NUMBER;
   *phi_max   = -HUGE_NUMBER;

   for (int k=-1; k<=1; k+=2)  {  const double zz = z + k * HalfWidth;  const double zz2 = SQR(zz);
   for (int j=-1; j<=1; j+=2)  {  const double yy = y + j * HalfWidth;  const double yy2 = SQR(yy);
   for (int i=-1; i<=1; i+=2)  {  const double xx = x + i * HalfWidth;  const double xx2 = SQR(xx);

//    deal the special case that the vertex locates on the z-axis (xx = yy = 0.0)
      if ( xx != 0.0  ||  yy != 0.0 )
      {
         const double phi = ( yy >= 0.0 )
                          ? atan2( yy, xx )
                          : atan2( yy, xx ) + 2.0 * M_PI;

         *phi_min = fmin( *phi_min, phi );
         *phi_max = fmax( *phi_max, phi );
      }


      const double radius = sqrt( xx2 + yy2 + zz2 );

//    deal the special case that the vertex locates at the center (radius = 0.0)
      if ( radius != 0.0 )
      {
         const double theta = acos( zz / radius );

         *theta_min = fmin( *theta_min, theta );
         *theta_max = fmax( *theta_max, theta );
      }
   }}} // i,j,k

} // FUNCTION : Aux_GetMinMax_Vertex_Angle
