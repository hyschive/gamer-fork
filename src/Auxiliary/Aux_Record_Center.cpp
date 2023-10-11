#include "GAMER.h"
#include "TestProb.h"




//-------------------------------------------------------------------------------------------------------
// Function    :  Aux_Record_Center
// Description :  Record various center coordinates
//
// Note        :  1. Invoked by main()
//                2. Enabled by the runtime option "OPT__RECORD_CENTER"
//                3. This function will be called both during the program initialization and after each full update
//                4. It will record the position of maximum density, minimum potential, and center of mass
//                5. Output filename is fixed to "Record__Center"
//
// Parameter   :  None
//
// Return      :  None
//-------------------------------------------------------------------------------------------------------
void Aux_Record_Center()
{

   const char FileName[] = "Record__Center";
   static bool FirstTime = true;

// 1. Maximum fluid density in HYDRO/ELBDM
   Extrema_t Max_Dens;
   Max_Dens.Field     = _DENS;
   Max_Dens.Radius    = HUGE_NUMBER; // entire domain
   Max_Dens.Center[0] = amr->BoxCenter[0];
   Max_Dens.Center[1] = amr->BoxCenter[1];
   Max_Dens.Center[2] = amr->BoxCenter[2];

   Aux_FindExtrema_ParDens( &Max_Dens, EXTREMA_MAX, 0, TOP_LEVEL, PATCH_LEAF );


#  ifdef PARTICLE
// 2. Maximum particle density
   Extrema_t Max_ParDens;
   Max_ParDens.Field     = _PAR_DENS;
   Max_ParDens.Radius    = HUGE_NUMBER; // entire domain
   Max_ParDens.Center[0] = amr->BoxCenter[0];
   Max_ParDens.Center[1] = amr->BoxCenter[1];
   Max_ParDens.Center[2] = amr->BoxCenter[2];

   Aux_FindExtrema_ParDens( &Max_ParDens, EXTREMA_MAX, 0, TOP_LEVEL, PATCH_LEAF );


// 3. Maximun total density including fluid density and particle density
   Extrema_t Max_TotDens;
   Max_TotDens.Field     = _TOTAL_DENS;
   Max_TotDens.Radius    = HUGE_NUMBER; // entire domain
   Max_TotDens.Center[0] = amr->BoxCenter[0];
   Max_TotDens.Center[1] = amr->BoxCenter[1];
   Max_TotDens.Center[2] = amr->BoxCenter[2];

   Aux_FindExtrema_ParDens( &Max_TotDens, EXTREMA_MAX, 0, TOP_LEVEL, PATCH_LEAF );
#  endif


// 4. Minimun gravitational potential
   Extrema_t Min_Pote;
   Min_Pote.Field     = _POTE;
   Min_Pote.Radius    = HUGE_NUMBER; // entire domain
   Min_Pote.Center[0] = amr->BoxCenter[0];
   Min_Pote.Center[1] = amr->BoxCenter[1];
   Min_Pote.Center[2] = amr->BoxCenter[2];

   Aux_FindExtrema_ParDens( &Min_Pote, EXTREMA_MIN, 0, TOP_LEVEL, PATCH_LEAF );


// 5. Center of mass for the total density field
// set an initial guess by the peak density position or the user-specified center
   double CoM_ref[3];
   if ( COM_CEN_X < 0.0  ||  COM_CEN_Y < 0.0  ||  COM_CEN_Z < 0.0 )
   {
#     ifdef PARTICLE
      for (int d=0; d<3; d++) CoM_ref[d] = Max_TotDens.Coord[d];
#     else
      for (int d=0; d<3; d++) CoM_ref[d] = Max_Dens.Coord[d];
#     endif
   }
   else
   {
      CoM_ref[0] = COM_CEN_X;
      CoM_ref[1] = COM_CEN_Y;
      CoM_ref[2] = COM_CEN_Z;
   }

// find the center of mass
   double CoM_Coord[3];
   double FinaldR;
   int    FinalNIter;
   Aux_FindWeightedCenter( CoM_Coord, CoM_ref, COM_MAX_R, COM_MIN_RHO, _TOTAL_DENS, COM_TOL_ERR_R, COM_N_ITER_MAX, &FinaldR, &FinalNIter );


// Output the center to file
   if ( MPI_Rank == 0 )
   {
//    Output the header
      if ( FirstTime )
      {
         if ( Aux_CheckFileExist(FileName) )
            Aux_Message( stderr, "WARNING : file \"%s\" already exists !!\n", FileName );
         else
         {
            FILE *File = fopen( FileName, "w" );
            fprintf( File, "#%19s  %10s", "Time", "Step" );
            fprintf( File, "  %14s  %14s  %14s  %14s",
                           "MaxDens", "MaxDens_x", "MaxDens_y", "MaxDens_z" );
#           ifdef PARTICLE
            fprintf( File, "  %14s  %14s  %14s  %14s",
                           "MaxParDens", "MaxParDens_x", "MaxParDens_y", "MaxParDens_z" );
            fprintf( File, "  %14s  %14s  %14s  %14s",
                           "MaxTotalDens", "MaxTotalDens_x", "MaxTotalDens_y", "MaxTotalDens_z" );
#           endif
            fprintf( File, "  %14s  %14s  %14s  %14s",
                           "MinPote", "MinPote_x", "MinPote_y", "MinPote_z" );
            fprintf( File, "  %14s  %14s  %14s  %14s  %14s",
                           "Final_NIter", "Final_dR", "CoM_x", "CoM_y", "CoM_z" );
            fprintf( File, "\n" );
            fclose( File );
         }

         FirstTime = false;
      }

      FILE *File = fopen( FileName, "a" );
      fprintf( File, "%20.14e  %10ld", Time[0], Step );
      fprintf( File, "  %14.7e  %14.7e  %14.7e  %14.7e",
                     Max_Dens.Value, Max_Dens.Coord[0], Max_Dens.Coord[1], Max_Dens.Coord[2] );
#     ifdef PARTICLE
      fprintf( File, "  %14.7e  %14.7e  %14.7e  %14.7e",
                     Max_ParDens.Value, Max_ParDens.Coord[0], Max_ParDens.Coord[1], Max_ParDens.Coord[2] );
      fprintf( File, "  %14.7e  %14.7e  %14.7e  %14.7e",
                     Max_TotDens.Value, Max_TotDens.Coord[0], Max_TotDens.Coord[1], Max_TotDens.Coord[2] );
#     endif
      fprintf( File, "  %14.7e  %14.7e  %14.7e  %14.7e",
                     Min_Pote.Value, Min_Pote.Coord[0], Min_Pote.Coord[1], Min_Pote.Coord[2] );
      fprintf( File, "  %14d  %14.7e  %14.7e  %14.7e  %14.7e",
                     FinalNIter, FinaldR, CoM_Coord[0], CoM_Coord[1], CoM_Coord[2] );
      fprintf( File, "\n" );
      fclose( File );
   }

} // FUNCTION : Aux_Record_Center
