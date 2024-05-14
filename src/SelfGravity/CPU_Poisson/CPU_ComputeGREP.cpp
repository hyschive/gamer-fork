#include "GAMER.h"


#define LinearInterp( x, xa, xb, ya, yb )   (  (ya) + ((yb) - (ya)) * ((x) - (xa)) / ((xb) - (xa))  )




//-------------------------------------------------------------------------------------------------------
// Function    :  CPU_ComputeGREP
// Description :  Construct the effective GR potential
//
// Note        :  1. Enabled if macro GRAVITY and GREP are set
//                2. Invoked by Poi_UserWorkBeforePoisson_GREP()
//                3. The profile EffPot stores the value of -Phi_NW(r) + Phi_TOV(r) at the bin center
//                4. Ref: Marek et al., 2006, A&A, 445, 273 (arXiv: 0502161), sec. 2, case A
//
// Parameter   :  lv      : Current AMR level
//                Time    : Current physical time at the specified level
//                DensAve : Profile_t object storing the spherically averaged profile of density
//                EngyAve : Profile_t object storing the spherically averaged profile of internal energy density
//                VrAve   : Profile_t object storing the spherically averaged profile of radial velocity
//                PresAve : Profile_t object storing the spherically averaged profile of pressure
//                EffPot  : Profile_t object storing the effective GR potential
//
// Return      :  EffPot
//-------------------------------------------------------------------------------------------------------
void CPU_ComputeGREP( const int lv, const double Time, const Profile_t *DensAve, const Profile_t *EngyAve,
                      const Profile_t *VrAve, const Profile_t *PresAve, Profile_t *EffPot )
{

   const double c2          = SQR( Const_c / UNIT_V );
   const double _c2         = 1.0 / c2;
   const double FourPI      = 4.0 * M_PI;
   const double FourThirdPI = FourPI / 3.0;
   const double Tolerance   = 1.0e-10;

   int     NIter     = 0;
   int     NBin      = DensAve->NBin;
   double *Radius    = DensAve->Radius;
   double  MaxRadius = DensAve->MaxRadius;
   double *Dens      = DensAve->Data;
   double *Engy      = EngyAve->Data;
   double *Vr        = VrAve  ->Data;
   double *Pres      = PresAve->Data;

   double *Weight       = new double [NBin];  // volume of each bin
   double *Edge_L       = new double [NBin];  // left edge of each bin
   double *Vr_L         = new double [NBin];  // radial velocity at the left edge of each bin
   double *Mass_NW      = new double [NBin];  // enclosed Newtonian mass for \bar_Phi(r)     in Eq. (7)
   double *Mass_TOV     = new double [NBin];  // enclosed TOV mass       for \bar_Phi(r)_TOV in Eq. (7)
   double *Mass_TOV_USG = new double [NBin];  // enclosed TOV mass in the previous iteration
   double *Dens_TOV     = new double [NBin];  // empirical TOV density                          Eq. (4)
   double *Gamma_TOV    = new double [NBin];  // metric function                                Eq. (5)

// (1) preparation stage
   for (int i=0; i<NBin; i++)   Mass_TOV_USG[i] = 0.0;
   for (int i=0; i<NBin; i++)   Gamma_TOV   [i] = 1.0;

                                Edge_L[0] = 0.0;
   for (int i=1; i<NBin; i++)   Edge_L[i] = ( GREP_LOGBIN ) ? sqrt( Radius[i] * Radius[i-1] )
                                                            : 0.5*( Radius[i] + Radius[i-1] );

// find the maximum radius that the stored weights are reliable
   for (int d=0; d<3; d++)
   {
      MaxRadius = MIN(  MIN( amr->BoxSize[d] - DensAve->Center[d], DensAve->Center[d] ),
                        MaxRadius  );
   }

// update the weight
   for (int i=0; i<NBin-1; i++)
   {
      Weight[i] = ( Edge_L[i+1] > MaxRadius )
                ? FourThirdPI * ( CUBE(Edge_L[i+1]) - CUBE(Edge_L[i]) )
                : DensAve->Weight[i];
   }

   Weight[NBin-1] = FourThirdPI * ( CUBE(DensAve->MaxRadius) - CUBE(Edge_L[NBin-1]) );

// compute the mass of each bin
   for (int i=0; i<NBin; i++)   Mass_NW[i] = Weight[i] * Dens[i];

// interpolate the radial velocity at the bin edge
                                Vr_L[0] = 0.0;
   for (int i=1; i<NBin; i++)   Vr_L[i] = LinearInterp( Edge_L[i], Radius[i-1], Radius[i],
                                                        Mass_NW[i-1] * Vr[i-1], Mass_NW[i] * Vr[i] )
                                        / LinearInterp( Edge_L[i], Radius[i-1], Radius[i],
                                                        Mass_NW[i-1],           Mass_NW[i]         );

// compute the enclosed Newtonian mass, defined at the right edge of each bin
   for (int i=1; i<NBin; i++)   Mass_NW[i] += Mass_NW[i-1];


// (2) iteratively compute Mass_TOV and Gamma_TOV, defined at the right edge of each bin
//     --> ignore the last bin since it is not used later
   bool IsConverged = false;

   while ( ! IsConverged  &&  ( NIter++ < GREP_MAXITER ) )
   {
//    update Mass_TOV
//    --> include the last bin here for troubleshooting information
      for (int i=0; i<NBin;   i++)   Dens_TOV[i]  = Gamma_TOV[i] * ( Dens[i] + Engy[i] * _c2 );

      for (int i=0; i<NBin;   i++)   Mass_TOV[i]  = Weight[i] * Dens_TOV[i];
      for (int i=1; i<NBin;   i++)   Mass_TOV[i] += Mass_TOV[i-1];

//    update Gamma_TOV
      for (int i=0; i<NBin-1; i++)   Gamma_TOV[i] = 1.0 + (  SQR( Vr_L[i+1] )
                                                           - 2.0  * NEWTON_G * Mass_TOV[i] / Edge_L[i+1]  ) * _c2;

      for (int i=0; i<NBin-1; i++)   Gamma_TOV[i] = sqrt( MAX( TINY_NUMBER, Gamma_TOV[i] ) );

//    check whether the Mass_TOV is converged
      IsConverged = true;

      for (int i=0; i<NBin-1; i++)
      {
         double RelErr = fabs( Mass_TOV_USG[i] - Mass_TOV[i] ) / Mass_TOV[i];

         if ( RelErr > Tolerance )
         {
            IsConverged = false;
            break;
         }
      }

//    backup the old Mass_TOV if not converged yet
      if ( ! IsConverged )
         for (int i=0; i<NBin; i++)   Mass_TOV_USG[i] = Mass_TOV[i];
   } // while ( ! IsConverged  &&  ( NIter++ < GREP_MAXITER ) )


// troubleshooting information in case convergent solutions cannot be found.
   if ( ! IsConverged )
   {
      if ( MPI_Rank == 0 )
      {
         char FileName[MAX_STRING];

         sprintf( FileName, "GREP_Lv%02d_FailedTOVProfile", lv );
         FILE *File = fopen( FileName, "w" );

//       metadata
         fprintf( File, "# Step               : %ld\n",                  Step );
         fprintf( File, "# Time               : %.7e\n",                 Time );
         fprintf( File, "# GREP_CENTER_METHOD : %d\n",                   GREP_CENTER_METHOD );
         fprintf( File, "# Center             : %13.7e %13.7e %13.7e\n", DensAve->Center[0], DensAve->Center[1], DensAve->Center[2] );
         fprintf( File, "# Maximum Radius     : %13.7e\n",               DensAve->MaxRadius );
         fprintf( File, "# LogBin             : %d\n",                   DensAve->LogBin );
         fprintf( File, "# LogBinRatio        : %13.7e\n",               DensAve->LogBinRatio );
         fprintf( File, "# NBin               : %d\n",                   NBin );
         fprintf( File, "# Maximum Iteration  : %d\n",                   GREP_MAXITER );
         fprintf( File, "# Tolerance          : %13.7e\n",               Tolerance );
         fprintf( File, "------------------------------------------------------\n" );
         fprintf( File, "%5s %9s %22s %22s %22s %22s %22s %22s %22s %22s %22s %22s\n",
                        "# Bin", "NCell", "Radius", "Density", "Energy", "Vr", "Pressure",
                        "Mass_NW", "Mass_TOV", "Mass_TOV_Usg", "Gamma_TOV", "Rel_Err (Mass_TOV)");

//       data
         for (int i=0; i<NBin; i++)
         fprintf( File, "%5d %9ld %22.15e %22.15e %22.15e %22.15e %22.15e %22.15e %22.15e %22.15e %22.15e %22.15e\n",
                        i, DensAve->NCell[i], Radius[i], Dens[i], Engy[i], Vr[i], Pres[i],
                        Mass_NW[i], Mass_TOV[i], Mass_TOV_USG[i], Gamma_TOV[i],
                        fabs( Mass_TOV_USG[i] - Mass_TOV[i] ) / Mass_TOV[i] );

         fclose( File );

//       terminate the program
         Aux_Error( ERROR_INFO, "the solutions for Mass_TOV and Gamma_TOV are not converging !!\n" );
      }
   } // if ( ! IsConverged )


// (3) compute the effective GR potential, defined at the left edge of each bin
   EffPot->NBin        = DensAve->NBin;
   EffPot->LogBin      = DensAve->LogBin;
   EffPot->LogBinRatio = DensAve->LogBinRatio;
   EffPot->MaxRadius   = DensAve->MaxRadius;
   EffPot->AllocateMemory();

   for (int d=0; d<3;    d++)   EffPot->Center[d] = DensAve->Center[d];
   for (int b=0; b<NBin; b++)   EffPot->Radius[b] = Edge_L[b];

// set the outer boundary condition of the potential to -GM/r at the left edge of last bin
   EffPot->Data[NBin-1] = -NEWTON_G * ( Mass_TOV[NBin-2] - Mass_NW[NBin-2] ) / EffPot->Radius[NBin-1];

   for (int i=NBin-2; i>0; i--)
   {
      const double dr = EffPot->Radius[i] - EffPot->Radius[i+1];

//    compute the integrand using data at the bin center
//    --> approximate the Mass_NW, Mass_TOV, and Gamma_TOV at the bin center via linear interpolation
      const double ratio = ( Radius[i] - Edge_L[i] ) / ( Edge_L[i+1] - Edge_L[i] );

      double Mass_NW_C   = Mass_NW  [i-1] + ratio * ( Mass_NW  [i] - Mass_NW  [i-1] );
      double Mass_TOV_C  = Mass_TOV [i-1] + ratio * ( Mass_TOV [i] - Mass_TOV [i-1] );
      double Gamma_TOV_C = Gamma_TOV[i-1] + ratio * ( Gamma_TOV[i] - Gamma_TOV[i-1] );

      Mass_TOV_C = ( Mass_TOV_C + FourPI * CUBE( Radius[i] ) * Pres[i] * _c2 )
                 * ( 1.0 + ( Engy[i] + Pres[i] ) / ( Dens[i] * c2 ) ) / SQR( Gamma_TOV_C );


      EffPot->Data[i] = EffPot->Data[i+1] - dr * NEWTON_G * ( Mass_NW_C - Mass_TOV_C ) / SQR( Radius[i] );
   }

// set the potential for the innermost bin
   EffPot->Data[0] = EffPot->Data[1];


// troubleshooting information
#  ifdef GAMER_DEBUG
   if ( MPI_Rank == 0 )
   {
      char FileName[MAX_STRING];

      sprintf( FileName, "GREP_Lv%02d", lv );
      FILE *File = fopen( FileName, "w" );

//    metadata
      fprintf( File, "# Step                 : %ld\n",                  Step );
      fprintf( File, "# Time                 : %.7e\n",                 Time );
      fprintf( File, "# GREP_CENTER_METHOD   : %d\n",                   GREP_CENTER_METHOD );
      fprintf( File, "# Center               : %13.7e %13.7e %13.7e\n", EffPot->Center[0], EffPot->Center[1], EffPot->Center[2] );
      fprintf( File, "# Maximum Radius       : %13.7e\n",               EffPot->MaxRadius );
      fprintf( File, "# LogBin               : %d\n",                   EffPot->LogBin );
      fprintf( File, "# LogBinRatio          : %13.7e\n",               EffPot->LogBinRatio );
      fprintf( File, "# Number of Iterations : %d\n",                   NIter );
      fprintf( File, "# NBin                 : %d\n",                   NBin );
      fprintf( File, "# -------------------------------------------------\n" );
      fprintf( File, "%5s %9s %22s %22s %22s %22s %22s %22s %22s %22s %22s %23s\n",
                     "# Bin", "NCell", "Bin_Center", "Bin_Edge", "Density", "Energy", "Vr", "Pressure",
                     "Mass_NW", "Mass_TOV", "Gamma_TOV", "Eff_Pot");

//    data
//    --> Dens, Engy, Vr, Pres         are defined at the bin center
//        Mass_NW, Mass_TOV, Gamma_TOV are defined at the right edge
//        EffPot                       is  defined at the left  edge
      fprintf( File, "%5d %9ld %22.15e %22.15e %22.15e %22.15e %22.15e %22.15e %22.15e %22.15e %22.15e %23.15e\n",
                     0, DensAve->NCell[0], Radius[0], Edge_L[0], Dens[0], Engy[0], Vr[0], Pres[0],
                     NULL_REAL,    NULL_REAL,     NULL_REAL,      EffPot->Data[0] );

      for (int i=1; i<NBin; i++)
      fprintf( File, "%5d %9ld %22.15e %22.15e %22.15e %22.15e %22.15e %22.15e %22.15e %22.15e %22.15e %23.15e\n",
                     i, DensAve->NCell[i], Radius[i], Edge_L[i], Dens[i], Engy[i], Vr[i], Pres[i],
                     Mass_NW[i-1], Mass_TOV[i-1], Gamma_TOV[i-1], EffPot->Data[i] );

      fclose( File );
   }
#  endif


// free memory
   delete [] Weight;
   delete [] Edge_L;
   delete [] Vr_L;
   delete [] Mass_NW;
   delete [] Mass_TOV;
   delete [] Mass_TOV_USG;
   delete [] Dens_TOV;
   delete [] Gamma_TOV;

} // FUNCTION : CPU_ComputeGREP
