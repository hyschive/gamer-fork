#include "GAMER.h"
#include "TestProb.h"



// problem-specific global variables
// =======================================================================================
static double GAMMA_CR;
static double CR_Shocktube_Rho_R;
static double CR_Shocktube_Rho_L;
static double CR_Shocktube_Pres_R;
static double CR_Shocktube_Pres_L;
static double CR_Shocktube_PresCR_R;
static double CR_Shocktube_PresCR_L;
static int    CR_Shocktube_Dir;
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
#  if ( MODEL != HYDRO )
   Aux_Error( ERROR_INFO, "MODEL != HYDRO !!\n" );
#  endif

#  ifndef MHD
   Aux_Error( ERROR_INFO, "MHD must be enabled !!\n" );
#  endif

// @@@ EOS GAMMA_CR

// warnings
   if ( MPI_Rank == 0 )
   {
#     ifndef DUAL_ENERGY
         Aux_Message( stderr, "WARNING : it's recommended to enable DUAL_ENERGY for this test !!\n" );
#     endif
      // What is this for?@@@
      if ( FLAG_BUFFER_SIZE < 5 )
         Aux_Message( stderr, "WARNING : it's recommended to set FLAG_BUFFER_SIZE >= 5 for this test !!\n" );
   } // if ( MPI_Rank == 0 )



   if ( MPI_Rank == 0 )    Aux_Message( stdout, "   Validating test problem %d ... done\n", TESTPROB_ID );

} // FUNCTION : Validate



// replace HYDRO by the target model (e.g., MHD/ELBDM) and also check other compilation flags if necessary (e.g., GRAVITY/PARTICLE)
#if ( MODEL == HYDRO )
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
// ReadPara->Add( "KEY_IN_THE_FILE",       &VARIABLE,              DEFAULT,           MIN,              MAX               );
// ********************************************************************************************************************************
   ReadPara->Add( "GAMMA_CR",              &GAMMA_CR,                  0.0,           0.0,              NoMax_double);
   ReadPara->Add( "CR_Shocktube_Rho_R",    &CR_Shocktube_Rho_R,        0.0,           0.0,              NoMax_double);
   ReadPara->Add( "CR_Shocktube_Rho_L",    &CR_Shocktube_Rho_L,        0.0,           0.0,              NoMax_double);
   ReadPara->Add( "CR_Shocktube_Pres_R",   &CR_Shocktube_Pres_R,       0.0,           0.0,              NoMax_double);
   ReadPara->Add( "CR_Shocktube_Pres_L",   &CR_Shocktube_Pres_L,       0.0,           0.0,              NoMax_double);
   ReadPara->Add( "CR_Shocktube_PresCR_R", &CR_Shocktube_PresCR_R,     0.0,           0.0,              NoMax_double);
   ReadPara->Add( "CR_Shocktube_PresCR_L", &CR_Shocktube_PresCR_L,     0.0,           0.0,              NoMax_double);
   ReadPara->Add( "CR_Shocktube_Dir",      &CR_Shocktube_Dir,             0,             0,                 NoMax_int);

   ReadPara->Read( FileName );

   delete ReadPara;

// (1-2) set the default values

// (1-3) check the runtime parameters


// (2) set the problem-specific derived parameters
//   Linear_Wavelength = amr->BoxSize[0] / 2.0;  // 2 wavelength along the x-axix

// (3) reset other general-purpose parameters
//     --> a helper macro PRINT_WARNING is defined in TestProb.h
   const long   End_Step_Default = __INT_MAX__;
   const double End_T_Default    = __FLT_MAX__;

   if ( END_STEP < 0 ) {
      END_STEP = End_Step_Default;
      PRINT_WARNING( "END_STEP", END_STEP, FORMAT_LONG );
   }

   if ( END_T < 0.0 ) {
      END_T = End_T_Default;
      PRINT_WARNING( "END_T", END_T, FORMAT_REAL );
   }


// (4) make a note
   if ( MPI_Rank == 0 )
   {
      Aux_Message( stdout, "=============================================================================\n" );
      Aux_Message( stdout, "  test problem ID           = %d\n",         TESTPROB_ID );
      Aux_Message( stdout, "  GAMMA_CR                  = %14.7e\n",     GAMMA_CR );
      Aux_Message( stdout, "  CR_Shocktube_Rho_R        = %14.7e\n",     CR_Shocktube_Rho_R );
      Aux_Message( stdout, "  CR_Shocktube_RhoR_L       = %14.7e\n",     CR_Shocktube_Rho_L );
      Aux_Message( stdout, "  CR_Shocktube_Pres_R       = %14.7e\n",     CR_Shocktube_Pres_R );
      Aux_Message( stdout, "  CR_Shocktube_Pres_L       = %14.7e\n",     CR_Shocktube_Pres_L );
      Aux_Message( stdout, "  CR_Shocktube_PresCR_R     = %14.7e\n",     CR_Shocktube_PresCR_R );
      Aux_Message( stdout, "  CR_Shocktube_PresCR_L     = %14.7e\n",     CR_Shocktube_PresCR_L );
      Aux_Message( stdout, "  CR_Shocktube_Dir          = %d\n",         CR_Shocktube_Dir );
      Aux_Message( stdout, "=============================================================================\n" );
   }


   if ( MPI_Rank == 0 )    Aux_Message( stdout, "   Setting runtime parameters ... done\n" );

} // FUNCTION : SetParameter



//-------------------------------------------------------------------------------------------------------
// Function    :  SetGridIC
// Description :  Set the problem-specific initial condition on grids
//
// Note        :  1. This function may also be used to estimate the numerical errors when OPT__OUTPUT_USER is enabled
//                   --> In this case, it should provide the analytical solution at the given "Time"
//                2. This function will be invoked by multiple OpenMP threads when OPENMP is enabled
//                   (unless OPT__INIT_GRID_WITH_OMP is disabled)
//                   --> Please ensure that everything here is thread-safe
//                3. Even when DUAL_ENERGY is adopted for HYDRO, one does NOT need to set the dual-energy variable here
//                   --> It will be calculated automatically
//                4. For MHD, do NOT add magnetic energy (i.e., 0.5*B^2) to fluid[ENGY] here
//                   --> It will be added automatically later
//
// Parameter   :  fluid    : Fluid field to be initialized
//                x/y/z    : Physical coordinates
//                Time     : Physical time
//                lv       : Target refinement level
//                AuxArray : Auxiliary array
//
// Return      :  fluid
//-------------------------------------------------------------------------------------------------------
void SetGridIC( real fluid[], const double x, const double y, const double z, const double Time,
                const int lv, double AuxArray[] )
{

// HYDRO example
   double Dens, MomX, MomY, MomZ, Pres, Eint, Etot;
   double P_cr, CRay;
   double GAMMA_CR_m1_inv = 1.0 / (GAMMA_CR - 1.0);
   
   if ( CR_Shocktube_Dir == 0)
   {
      if (x < 0.5)
      {
      Dens = CR_Shocktube_Rho_L;
      MomX = 0.0;
      MomY = 0.0;
      MomZ = 0.0;
      Pres = CR_Shocktube_Pres_L;
      P_cr = CR_Shocktube_PresCR_L;
      Pres = Pres + P_cr;
      CRay = GAMMA_CR_m1_inv * P_cr;
      } else if (x == 0.5){
      Dens = 0.5*( CR_Shocktube_Rho_L + CR_Shocktube_Rho_R );
      MomX = 0.0;
      MomY = 0.0;
      MomZ = 0.0;
      Pres = 0.5*( CR_Shocktube_Pres_L + CR_Shocktube_Pres_R );
      P_cr = 0.5*( CR_Shocktube_PresCR_L + CR_Shocktube_PresCR_R );
      Pres = Pres + P_cr;
      CRay = GAMMA_CR_m1_inv * P_cr;
      }else if (x > 0.5)
      {
      Dens = CR_Shocktube_Rho_R;
      MomX = 0.0;
      MomY = 0.0;
      MomZ = 0.0;
      Pres = CR_Shocktube_Pres_R;
      P_cr = CR_Shocktube_PresCR_R;
      Pres = Pres + P_cr;
      CRay = GAMMA_CR_m1_inv * P_cr;
      }
   }
   else if ( CR_Shocktube_Dir == 1)
   {
      
      Aux_Error( ERROR_INFO, "CR_Shocktube_Dir = %d is NOT supported [0] !!\n", CR_Shocktube_Dir );
   }
   else
   {
      Aux_Error( ERROR_INFO, "CR_Shocktube_Dir = %d is NOT supported [0/1] !!\n", CR_Shocktube_Dir );
   }

// set the output array of passive scaler
#ifdef COSMIC_RAY
   fluid[CRAY] = CRay;
#endif
   /*
   FILE * pFile;
   pFile = fopen("./soundspeed.txt", "a");
   
   //real cs = EoS_DensPres2CSqr_CPUPtr(Dens, Pres, fluid+NCOMP_FLUID, EoS_AuxArray);

   fprintf(pFile, "%.3f %.3f %.3f %.8f %.8f %.8f %.8f\n", x, y, z, Dens, MomX, Pres, P_cr);

   fclose(pFile);
   */
   Eint = EoS_DensPres2Eint_CPUPtr( Dens, Pres, fluid+NCOMP_FLUID, EoS_AuxArray_Flt, EoS_AuxArray_Int, h_EoS_Table, NULL );
   Etot = Hydro_ConEint2Etot( Dens, MomX, MomY, MomZ, Eint, 0.0 );      // do NOT include magnetic energy here

// set the output array
   fluid[DENS] = Dens;
   fluid[MOMX] = MomX;
   fluid[MOMY] = MomY;
   fluid[MOMZ] = MomZ;
   fluid[ENGY] = Etot;

} // FUNCTION : SetGridIC



#ifdef MHD
//-------------------------------------------------------------------------------------------------------
// Function    :  SetBFieldIC
// Description :  Set the problem-specific initial condition of magnetic field
//
// Note        :  1. This function will be invoked by multiple OpenMP threads when OPENMP is enabled
//                   (unless OPT__INIT_GRID_WITH_OMP is disabled)
//                   --> Please ensure that everything here is thread-safe
//
// Parameter   :  magnetic : Array to store the output magnetic field
//                x/y/z    : Target physical coordinates
//                Time     : Target physical time
//                lv       : Target refinement level
//                AuxArray : Auxiliary array
//
// Return      :  magnetic
//-------------------------------------------------------------------------------------------------------
void SetBFieldIC( real magnetic[], const double x, const double y, const double z, const double Time,
                  const int lv, double AuxArray[] )
{

   magnetic[MAGX] = 0.0;
   magnetic[MAGY] = 0.0;
   magnetic[MAGZ] = 0.0;   

} // FUNCTION : SetBFieldIC
#endif // #ifdef MHD
#endif // #if ( MODEL == HYDRO )



//-------------------------------------------------------------------------------------------------------
// Function    :  Init_TestProb_Hydro_Cosmic_Ray_Shocktube
// Description :  Test problem initializer
//
// Note        :  None
//
// Parameter   :  None
//
// Return      :  None
//-------------------------------------------------------------------------------------------------------
void Init_TestProb_Hydro_Cosmic_Ray_Shocktube()
{

   if ( MPI_Rank == 0 )    Aux_Message( stdout, "%s ...\n", __FUNCTION__ );


// validate the compilation flags and runtime parameters
   Validate();


// replace HYDRO by the target model (e.g., MHD/ELBDM) and also check other compilation flags if necessary (e.g., GRAVITY/PARTICLE)
#  if ( MODEL == HYDRO )
// set the problem-specific runtime parameters
   SetParameter();


// procedure to enable a problem-specific function:
// 1. define a user-specified function (example functions are given below)
// 2. declare its function prototype on the top of this file
// 3. set the corresponding function pointer below to the new problem-specific function
// 4. enable the corresponding runtime option in "Input__Parameter"
//    --> for instance, enable OPT__OUTPUT_USER for Output_User_Ptr
   Init_Function_User_Ptr         = SetGridIC;

#  ifdef MHD
   Init_Function_BField_User_Ptr  = SetBFieldIC;
#  endif

// comment out Init_ByFile_User_Ptr to use the default
// Init_ByFile_User_Ptr           = NULL; // option: OPT__INIT=3;             example: Init/Init_ByFile.cpp -> Init_ByFile_Default()
   Init_Field_User_Ptr            = NULL; // set NCOMP_PASSIVE_USER;          example: TestProblem/Hydro/Plummer/Init_TestProb_Hydro_Plummer.cpp --> AddNewField()
   Flag_User_Ptr                  = NULL; // option: OPT__FLAG_USER;          example: Refine/Flag_User.cpp
   Mis_GetTimeStep_User_Ptr       = NULL; // option: OPT__DT_USER;            example: Miscellaneous/Mis_GetTimeStep_User.cpp
   BC_User_Ptr                    = NULL; // option: OPT__BC_FLU_*=4;         example: TestProblem/ELBDM/ExtPot/Init_TestProb_ELBDM_ExtPot.cpp --> BC()

#  ifdef MHD
   BC_BField_User_Ptr             = NULL; // option: OPT__BC_FLU_*=4;
#  endif

   Flu_ResetByUser_Func_Ptr       = NULL; // option: OPT__RESET_FLUID;        example: Fluid/Flu_ResetByUser.cpp
   Output_User_Ptr                = NULL; // option: OPT__OUTPUT_USER;        example: TestProblem/Hydro/AcousticWave/Init_TestProb_Hydro_AcousticWave.cpp --> OutputError()
   Aux_Record_User_Ptr            = NULL; // option: OPT__RECORD_USER;        example: Auxiliary/Aux_Record_User.cpp
   Init_User_Ptr                  = NULL; // option: none;                    example: none
   End_User_Ptr                   = NULL; // option: none;                    example: TestProblem/Hydro/ClusterMerger_vs_Flash/Init_TestProb_ClusterMerger_vs_Flash.cpp --> End_ClusterMerger()

#  if ( EOS == EOS_COSMIC_RAY )
   EoS_Init_Ptr                   = NULL; // option: EOS in the Makefile;     example: EoS/User_Template/CPU_EoS_User_Template.cpp
#  endif

#  endif // #if ( MODEL == HYDRO )

   if ( MPI_Rank == 0 )    Aux_Message( stdout, "%s ... done\n", __FUNCTION__ );

} // FUNCTION : Init_TestProb_Hydro_Cosmic_Ray
