#include "GAMER.h"
#include "TestProb.h"
#include "Profile_with_Sigma.h"

static void Init_Load_StepTable( void );
static void AddNewField_ELBDM_UniformGranule( void );
static void Init_User_ELBDM_UniformGranule( void );
static void Do_CF( void );

void Aux_ComputeProfile_with_Sigma( Profile_with_Sigma_t *Prof[], const double Center[], const double r_max_input, const double dr_min,
                                    const bool LogBin, const double LogBinRatio, const bool RemoveEmpty, const long TVarBitIdx[],
                                    const int NProf, const int MinLv, const int MaxLv, const PatchType_t PatchType,
                                    const double PrepTimeIn );

void Aux_ComputeCorrelation( Profile_t *Correlation[], const Profile_with_Sigma_t *prof_init[], const double Center[],
                             const double r_max_input, const double dr_min, const bool LogBin, const double LogBinRatio,
                             const bool RemoveEmpty, const long TVarBitIdx[], const int NProf, const int MinLv, const int MaxLv,
                             const PatchType_t PatchType, const double PrepTime, const double dr_min_prof );


// problem-specific global variables
// =======================================================================================
static FieldIdx_t Idx_Dens0 = Idx_Undefined;  // field index for storing the **initial** density
static double   Center[3];                    // use CoM coordinate of the whole halo as center
static double   dr_min_prof;                  // bin size of correlation function statistics (minimum size if logarithic bin) (profile)
static double   LogBinRatio_prof;             // ratio of bin size growing rate for logarithmic bin (profile)
static double   RadiusMax_prof;               // maximum radius for correlation function statistics (profile)
static double   dr_min_corr;                  // bin size of correlation function statistics (minimum size if logarithic bin) (correlation)
static double   LogBinRatio_corr;             // ratio of bin size growing rate for logarithmic bin (correlation)
static double   RadiusMax_corr;               // maximum radius for correlation function statistics (correlation)
static bool     ComputeCorrelation;           // flag for compute correlation
static bool     ReComputeCorrelation;         // flag for recompute correlation for restart; use the simulation time of RESTART as initial time for computing time correlation; only available for RESTART
static bool     LogBin_prof;                  // logarithmic bin or not (profile)
static bool     RemoveEmpty_prof;             // remove 0 sample bins; false: Data[empty_bin]=Weight[empty_bin]=NCell[empty_bin]=0 (profile)
static bool     LogBin_corr;                  // logarithmic bin or not (correlation)
static bool     RemoveEmpty_corr;             // remove 0 sample bins; false: Data[empty_bin]=Weight[empty_bin]=NCell[empty_bin]=0 (correlation)
static int      MinLv;                        // do statistics from MinLv to MaxLv
static int      MaxLv;                        // do statistics from MinLv to MaxLv
static int      StepInitial;                  // inital step for recording correlation function
static int      StepInterval;                 // interval for recording correlation function
static int      StepEnd;                      // end step for recording correlation function
static char     FilePath_corr[MAX_STRING];    // output path for correlation function text files

static Profile_with_Sigma_t  Prof_Dens_initial;                      // pointer to save initial density profile
static Profile_with_Sigma_t *Prof[] = { &Prof_Dens_initial };
static Profile_t             Correlation_Dens;                       // pointer to save density correlation function
static Profile_t            *Correlation[] = { &Correlation_Dens };
// =======================================================================================

//-------------------------------------------------------------------------------------------------------
// Function    :  Validate
// Description :  Validate the compilation flags and runtime parameters for this test problem
//
// Note        :  None
//
// Parameter   :  None
//
// Return      :  None
//-------------------------------------------------------------------------------------------------------
void Validate()
{

   if ( MPI_Rank == 0 )    Aux_Message( stdout, "   Validating test problem %d ...\n", TESTPROB_ID );

// errors
#  if ( MODEL != ELBDM )
   Aux_Error( ERROR_INFO, "MODEL != ELBDM !!\n" );
#  endif

#  ifndef GRAVITY
   Aux_Error( ERROR_INFO, "GRAVITY must be enabled !!\n" );
#  endif

#  ifdef COMOVING
   Aux_Error( ERROR_INFO, "COMOVING must be disabled !!\n" );
#  endif

// only accept OPT__INIT == INIT_BY_RESTART or OPT__INIT == INIT_BY_FILE
   if ( OPT__INIT != INIT_BY_RESTART && OPT__INIT != INIT_BY_FILE )
      Aux_Error( ERROR_INFO, "enforced to accept only OPT__INIT == INIT_BY_RESTART or OPT__INIT == INIT_BY_FILE !!\n" );

   if ( MPI_Rank == 0 )    Aux_Message( stdout, "   Validating test problem %d ... done\n", TESTPROB_ID );

} // FUNCTION : Validate



// replace HYDRO by the target model (e.g., MHD/ELBDM) and also check other compilation flags if necessary (e.g., GRAVITY/PARTICLE)
#if ( MODEL == ELBDM  && defined GRAVITY )
//-------------------------------------------------------------------------------------------------------
// Function    :  SetParameter
// Description :  Load and set the problem-specific runtime parameters
//
// Note        :  1. Filename is set to "Input__TestProb" by default
//                2. Major tasks in this function:
//                   (1) load the problem-specific runtime parameters
//                   (2) set the problem-specific derived parameters
//                   (3) reset other general-purpose parameters if necessary
//                   (4) make a note of the problem-specific parameters
//                3. Must NOT call any EoS routine here since it hasn't been initialized at this point
//
// Parameter   :  None
//
// Return      :  None
//-------------------------------------------------------------------------------------------------------
void SetParameter()
{

   if ( MPI_Rank == 0 )    Aux_Message( stdout, "   Setting runtime parameters ...\n" );


// (1) load the problem-specific runtime parameters
   const char FileName[] = "Input__TestProb";
   ReadPara_t *ReadPara  = new ReadPara_t;

// (1-1) add parameters in the following format:
// --> note that VARIABLE, DEFAULT, MIN, and MAX must have the same data type
// --> some handy constants (e.g., Useless_bool, Eps_double, NoMin_int, ...) are defined in "include/ReadPara.h"
// ********************************************************************************************************************************
// ReadPara->Add( "KEY_IN_THE_FILE",          &VARIABLE,                DEFAULT,       MIN,              MAX               );
// ********************************************************************************************************************************
   ReadPara->Add( "ComputeCorrelation",       &ComputeCorrelation,      false,         Useless_bool,     Useless_bool      );
   ReadPara->Add( "dr_min_corr",              &dr_min_corr,             Eps_double,    Eps_double,       NoMax_double      );
   ReadPara->Add( "LogBinRatio_corr",         &LogBinRatio_corr,        1.0,           Eps_double,       NoMax_double      );
   ReadPara->Add( "RadiusMax_corr",           &RadiusMax_corr,          Eps_double,    Eps_double,       NoMax_double      );
   ReadPara->Add( "LogBin_corr",              &LogBin_corr,             false,         Useless_bool,     Useless_bool      );
   ReadPara->Add( "RemoveEmpty_corr",         &RemoveEmpty_corr,        false,         Useless_bool,     Useless_bool      );
   ReadPara->Add( "dr_min_prof",              &dr_min_prof,             Eps_double,    Eps_double,       NoMax_double      );
   ReadPara->Add( "MinLv",                    &MinLv,                   0,             0,                MAX_LEVEL         );
   ReadPara->Add( "MaxLv",                    &MaxLv,                   MAX_LEVEL,     0,                MAX_LEVEL         );
   ReadPara->Add( "StepInitial",              &StepInitial,             0,             0,                NoMax_int         );
   ReadPara->Add( "StepInterval",             &StepInterval,            1,             1,                NoMax_int         );
   ReadPara->Add( "StepEnd",                  &StepEnd,                 NoMax_int,     0,                NoMax_int         );
   ReadPara->Add( "FilePath_corr",            FilePath_corr,            Useless_str,   Useless_str,      Useless_str       );
   if ( OPT__INIT == INIT_BY_RESTART )
      ReadPara->Add( "ReComputeCorrelation", &ReComputeCorrelation,     false,         Useless_bool,     Useless_bool      );

   ReadPara->Read( FileName );

   delete ReadPara;

// (1-2) set the default values
   if (ComputeCorrelation)
   {
       if ( dr_min_corr <=Eps_double )          dr_min_corr = 1e-3*0.5*amr->BoxSize[0];
       if ( RadiusMax_corr<=Eps_double )        RadiusMax_corr = 0.5*amr->BoxSize[0];
       if ( LogBinRatio_corr<=1.0 )             LogBinRatio_corr = 2.0;

       if ( dr_min_prof <=Eps_double )          dr_min_prof = dr_min_corr;
       RadiusMax_prof                           = RadiusMax_corr * 1.05;   // assigned by Test Problem
       LogBinRatio_prof                         = 1.0;                     // hard-coded by Test Problem (no effect)
       LogBin_prof                              = false;                   // hard-coded by Test Problem
       RemoveEmpty_prof                         = false;                   // hard-coded by Test Problem

       if ( MinLv < 0 ) MinLv = 0;
       if ( MaxLv < MinLv ) MaxLv = MAX_LEVEL;
       if ( FilePath_corr == "\0" )  sprintf( FilePath_corr, "./" );
       else
       {
          FILE *file_checker = fopen(FilePath_corr, "r");
          if (!file_checker)
             Aux_Error( ERROR_INFO, "File path %s for saving correlation function text files does not exist!! Please create!!\n", FilePath_corr );
          else
             fclose(file_checker);
       }
   }

// (1-3) check the runtime parameters
   if ( OPT__INIT == INIT_BY_FUNCTION )
      Aux_Error( ERROR_INFO, "OPT__INIT = 1 is not supported for this test problem !!\n" );

// check whether fluid boundary condition in Input__Parameter is set properly
   for ( int direction = 0; direction < 6; direction++ )
   {
      if ( OPT__BC_FLU[direction] != BC_FLU_PERIODIC )
         Aux_Error( ERROR_INFO, "must set periodic BC for fluid --> reset OPT__BC_FLU[%d] to 1 !!\n", direction );
   }
   if ( OPT__BC_POT != BC_POT_PERIODIC )
      Aux_Error( ERROR_INFO, "must set periodic BC for gravity --> reset OPT__BC_POT to 1 !!\n" );


// (2) set the problem-specific derived parameters

// (3) reset other general-purpose parameters
//     --> a helper macro PRINT_WARNING is defined in TestProb.h
   const long   End_Step_Default = 5.0e4;
   const double End_T_Default    = 1.0;

   if ( END_STEP < 0 ) {
      END_STEP = End_Step_Default;
      PRINT_RESET_PARA( END_STEP, FORMAT_LONG, "" );
   }

   if ( END_T < 0.0 ) {
      END_T = End_T_Default;
      PRINT_RESET_PARA( END_T, FORMAT_REAL, "" );
   }


// (4) make a note
   if ( MPI_Rank == 0 )
   {
      Aux_Message( stdout, "=================================================================================\n" );
      Aux_Message( stdout, "  test problem ID                             = %d\n",         TESTPROB_ID           );
      Aux_Message( stdout, "  compute correlation                         = %d\n"    , ComputeCorrelation        );
      if (ComputeCorrelation)
      {
         Aux_Message( stdout, "  histogram bin size  (correlation)           = %13.7e\n", dr_min_corr            );
         Aux_Message( stdout, "  log bin ratio       (correlation)           = %13.7e\n", LogBinRatio_corr       );
         Aux_Message( stdout, "  radius maximum      (correlation)           = %13.7e\n", RadiusMax_corr         );
         Aux_Message( stdout, "  use logarithmic bin (correlation)           = %d\n"    , LogBin_corr            );
         Aux_Message( stdout, "  remove empty bin    (correlation)           = %d\n"    , RemoveEmpty_corr       );
         Aux_Message( stdout, "  histogram bin size  (profile)               = %13.7e\n", dr_min_prof            );
         Aux_Message( stdout, "  log bin ratio       (profile, no effect)    = %13.7e\n", LogBinRatio_prof       );
         Aux_Message( stdout, "  radius maximum      (profile, assigned)     = %13.7e\n", RadiusMax_prof         );
         Aux_Message( stdout, "  use logarithmic bin (profile, assigned)     = %d\n"    , LogBin_prof            );
         Aux_Message( stdout, "  remove empty bin    (profile, assigned)     = %d\n"    , RemoveEmpty_prof       );
         Aux_Message( stdout, "  minimum level                               = %d\n"    , MinLv                  );
         Aux_Message( stdout, "  maximum level                               = %d\n"    , MaxLv                  );
         Aux_Message( stdout, "  file path for correlation text file         = %s\n"    , FilePath_corr          );
         if ( OPT__INIT == INIT_BY_RESTART )
         Aux_Message( stdout, "  re-compute correlation using restart time as initial time = %d\n", ReComputeCorrelation );
      }
      Aux_Message( stdout, "=================================================================================\n" );
   }

   if ( MPI_Rank == 0 )    Aux_Message( stdout, "   Setting runtime parameters ... done\n" );

} // FUNCTION : SetParameter

//-------------------------------------------------------------------------------------------------------
// Function    :  AddNewField_ELBDM_UniformGranule
// Description :  Add the problem-specific fields
//
// Note        :  1. Ref: https://github.com/gamer-project/gamer/wiki/Adding-New-Simulations#v-add-problem-specific-grid-fields-and-particle-attributes
//                2. Invoke AddField() for each of the problem-specific field:
//                   --> Field label sent to AddField() will be used as the output name of the field
//                   --> Field index returned by AddField() can be used to access the field data
//                3. Pre-declared field indices are put in Field.h
//
// Parameter   :  None
//
// Return      :  None
//-------------------------------------------------------------------------------------------------------
static void AddNewField_ELBDM_UniformGranule(void)
{

#  if ( NCOMP_PASSIVE_USER > 0 )
   Idx_Dens0 = AddField( "Dens0", FIXUP_FLUX_NO, FIXUP_REST_NO, NORMALIZE_NO, INTERP_FRAC_NO );
#  endif

} // FUNCTION : AddNewField_ELBDM_Halo_Stability_Test

//-------------------------------------------------------------------------------------------------------
// Function    :  Init_User_ELBDM_UniformGranule
// Description :  Store the initial density
//
// Note        :  1. Invoked by Init_GAMER() using the function pointer "Init_User_Ptr",
//                   which must be set by a test problem initializer
//
// Parameter   :  None
//
// Return      :  None
//-------------------------------------------------------------------------------------------------------
static void Init_User_ELBDM_UniformGranule(void)
{

#  if ( NCOMP_PASSIVE_USER > 0 )
   for (int lv=0; lv<NLEVEL; lv++)
   for (int PID=0; PID<amr->NPatchComma[lv][1]; PID++)
   for (int k=0; k<PS1; k++)
   for (int j=0; j<PS1; j++)
   for (int i=0; i<PS1; i++)
   {
//    store the initial density in both Sg so that we don't have to worry about which Sg to be used
//    a. for restart and ReComputeCorrelation disabled, the initial density has already been loaded and we just need to copy the data to another Sg
      if ( ( OPT__INIT == INIT_BY_RESTART ) && ( !ReComputeCorrelation ) ) {
         const real Dens0 = amr->patch[ amr->FluSg[lv] ][lv][PID]->fluid[Idx_Dens0][k][j][i];

         amr->patch[ 1-amr->FluSg[lv] ][lv][PID]->fluid[Idx_Dens0][k][j][i] = Dens0;
      }

//    b. for starting a new simulation, we must copy the initial density to both Sg
      else {
         const real Dens0 = amr->patch[ amr->FluSg[lv] ][lv][PID]->fluid[DENS][k][j][i];

         amr->patch[   amr->FluSg[lv] ][lv][PID]->fluid[Idx_Dens0][k][j][i] = Dens0;
         amr->patch[ 1-amr->FluSg[lv] ][lv][PID]->fluid[Idx_Dens0][k][j][i] = Dens0;
      }
   }

   if (ComputeCorrelation)
   {
       const double InitialTime = Time[0];
       if ( MPI_Rank==0 ) Aux_Message( stdout, "StepInitial = %d ; StepInterval = %d ; StepEnd = %d\n", StepInitial, StepInterval, StepEnd);

       if ( MPI_Rank == 0 )  Aux_Message( stdout, "InitialTime = %13.6e \n", InitialTime );

//     compute the enter position for passive field
       if ( MPI_Rank == 0 )  Aux_Message( stdout, "Calculate halo center for passive field:\n");

       double FinaldR;
       int    FinalNIter;
       double CoM_ref[3];
       Extrema_t Max_Dens;

       Max_Dens.Field     = _DENS;
       Max_Dens.Radius    = __FLT_MAX__; // entire domain
       Max_Dens.Center[0] = amr->BoxCenter[0];
       Max_Dens.Center[1] = amr->BoxCenter[1];
       Max_Dens.Center[2] = amr->BoxCenter[2];

       Aux_FindExtrema( &Max_Dens, EXTREMA_MAX, 0, TOP_LEVEL, PATCH_LEAF );
       if ( COM_CEN_X < 0.0  ||  COM_CEN_Y < 0.0  ||  COM_CEN_Z < 0.0 )
       {
          for (int d=0; d<3; d++) CoM_ref[d] = Max_Dens.Coord[d];
       }
       else
       {
          CoM_ref[0] = COM_CEN_X;
          CoM_ref[1] = COM_CEN_Y;
          CoM_ref[2] = COM_CEN_Z;
       }
       Aux_FindWeightedAverageCenter( Center, CoM_ref, COM_MAX_R, COM_MIN_RHO, _TOTAL_DENS, COM_TOLERR_R, COM_MAX_ITER, &FinaldR, &FinalNIter );

       if ( MPI_Rank == 0 )  Aux_Message( stdout, "Center of passive field is ( %14.11e,%14.11e,%14.11e )\n", Center[0], Center[1], Center[2] );

//     commpute density profile for passive field
       if ( MPI_Rank == 0 )  Aux_Message( stdout, "Calculate density profile for passive field:\n");

       const long TVar[] = {BIDX(Idx_Dens0)};
       Aux_ComputeProfile_with_Sigma( Prof, Center, RadiusMax_prof, dr_min_prof, LogBin_prof, LogBinRatio_prof, RemoveEmpty_prof, TVar, 1, MinLv, MaxLv, PATCH_LEAF, InitialTime );

       if ( MPI_Rank == 0 )
       {
          char Filename[MAX_STRING];
          sprintf( Filename, "%s/initial_profile_with_Sigma.txt", FilePath_corr );
          FILE *output_initial_prof = fopen(Filename, "w");
          fprintf( output_initial_prof, "#%19s  %21s  %21s  %21s  %11s\n", "Radius", "Dens", "Dens_Sigma" , "Weighting", "Cell_Number");
          for (int b=0; b<Prof[0]->NBin; b++)
             fprintf( output_initial_prof, "%20.14e  %21.14e  %21.14e  %21.14e  %11ld\n",
                      Prof[0]->Radius[b], Prof[0]->Data[b], Prof[0]->Data_Sigma[b], Prof[0]->Weight[b], Prof[0]->NCell[b] );
          fclose(output_initial_prof);
       }
   }
#  endif

} // FUNCTION : Init_User_ELBDM_UniformGranule

#endif // #if ( MODEL == ELBDM )


static void Do_CF( void )
{
// Compute correlation if ComputeCorrelation flag is true
   if (ComputeCorrelation)
   {
      if ( (Step>=StepInitial) && (((Step-StepInitial)%StepInterval)==0) && (Step<=StepEnd) )
      {
         const long TVar[] = {_DENS};
         Aux_ComputeCorrelation( Correlation, (const Profile_with_Sigma_t**)Prof, Center, RadiusMax_corr, dr_min_corr, LogBin_corr, LogBinRatio_corr,
                                 RemoveEmpty_corr, TVar, 1, MinLv, MaxLv, PATCH_LEAF, Time[0], dr_min_prof);

         if ( MPI_Rank == 0 )
         {
            char Filename[MAX_STRING];
            sprintf( Filename, "%s/correlation_function_t=%.4e.txt", FilePath_corr, Time[0] );
            FILE *output_correlation = fopen(Filename, "w");
            fprintf( output_correlation, "#%19s  %21s  %21s  %11s\n", "Radius", "Correlation_Function", "Weighting", "Cell_Number");
            for (int b=0; b<Correlation[0]->NBin; b++)
                fprintf( output_correlation, "%20.14e  %21.14e  %21.14e  %11ld\n",
                         Correlation[0]->Radius[b], Correlation[0]->Data[b], Correlation[0]->Weight[b], Correlation[0]->NCell[b] );
            fclose(output_correlation);
         }
      }
   }  // end of if ComputeCorrelation
}


//-------------------------------------------------------------------------------------------------------
// Function    :  Init_TestProb_Template
// Description :  Test problem initializer
//
// Note        :  None
//
// Parameter   :  None
//
// Return      :  None
//-------------------------------------------------------------------------------------------------------
void Init_TestProb_ELBDM_UniformGranule()
{

   if ( MPI_Rank == 0 )    Aux_Message( stdout, "%s ...\n", __FUNCTION__ );


// validate the compilation flags and runtime parameters
   Validate();


// replace HYDRO by the target model (e.g., MHD/ELBDM) and also check other compilation flags if necessary (e.g., GRAVITY/PARTICLE)
#  if ( MODEL == ELBDM )
// set the problem-specific runtime parameters
   SetParameter();

   Init_Field_User_Ptr    = AddNewField_ELBDM_UniformGranule;
   Init_User_Ptr          = Init_User_ELBDM_UniformGranule;
   Aux_Record_User_Ptr    = Do_CF;
#  endif // #if ( MODEL == ELBDM )

   if ( MPI_Rank == 0 )    Aux_Message( stdout, "%s ... done\n", __FUNCTION__ );

} // FUNCTION : Init_TestProb_ELBDM_DiskHeating
