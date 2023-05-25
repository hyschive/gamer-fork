#include "GAMER.h"
#include "TestProb.h"
#include <gsl/gsl_integration.h> 


static void Output_ExactCooling();
double Lambda(double Temp, double ZIRON);
double integrand(double Temp, void *params);

// problem-specific global variables
// =======================================================================================
static double EC_Temp;
static double EC_Dens;
       int    count = 0;
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

#  if ( MODEL != HYDRO )
   Aux_Error( ERROR_INFO, "MODEL != HYDRO !!\n" );
#  endif

// examples
/*
// errors
#  if ( MODEL != HYDRO )
   Aux_Error( ERROR_INFO, "MODEL != HYDRO !!\n" );
#  endif

#  ifndef GRAVITY
   Aux_Error( ERROR_INFO, "GRAVITY must be enabled !!\n" );
#  endif

#  ifdef PARTICLE
   Aux_Error( ERROR_INFO, "PARTICLE must be disabled !!\n" );
#  endif

#  ifdef GRAVITY
   if ( OPT__BC_FLU[0] == BC_FLU_PERIODIC  ||  OPT__BC_POT == BC_POT_PERIODIC )
      Aux_Error( ERROR_INFO, "do not use periodic BC for this test !!\n" );
#  endif


// warnings
   if ( MPI_Rank == 0 )
   {
#     ifndef DUAL_ENERGY
         Aux_Message( stderr, "WARNING : it's recommended to enable DUAL_ENERGY for this test !!\n" );
#     endif

      if ( FLAG_BUFFER_SIZE < 5 )
         Aux_Message( stderr, "WARNING : it's recommended to set FLAG_BUFFER_SIZE >= 5 for this test !!\n" );
   } // if ( MPI_Rank == 0 )
*/


   if ( MPI_Rank == 0 )    Aux_Message( stdout, "   Validating test problem %d ... done\n", TESTPROB_ID );

} // FUNCTION : Validate



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
//                3. Must call EoS_Init() before calling any other EoS routine
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
// ReadPara->Add( "KEY_IN_THE_FILE",   &VARIABLE,              DEFAULT,       MIN,              MAX               );
// ********************************************************************************************************************************
   ReadPara->Add( "EC_Temp",        &EC_Temp,                  1000000.0,     Eps_double,       NoMax_double      );
   ReadPara->Add( "EC_Dens",        &EC_Dens,                  1.0,           Eps_double,       NoMax_double      );

   ReadPara->Read( FileName );

   delete ReadPara;

// (1-2) set the default values

// (1-3) check the runtime parameters


// (2) set the problem-specific derived parameters


// (3) reset other general-purpose parameters
//     --> a helper macro PRINT_WARNING is defined in TestProb.h
   const long   End_Step_Default = __INT_MAX__;
   const double End_T_Default    = 100.0*Const_Myr/UNIT_T;

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
      Aux_Message( stdout, "  test problem ID           = %d\n",     TESTPROB_ID );
      Aux_Message( stdout, "  EC_Temp                   = %13.7e\n", EC_Temp     );
      Aux_Message( stdout, "  EC_Dens                   = %13.7e\n", EC_Dens     );
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

   double Dens, MomX, MomY, MomZ, Pres, Eint, Etot;
// Convert the input number density into mass density rho
//   double cl_dens = (EC_Dens*Const_mp*0.61709348966) / UNIT_D;
//   double cl_dens = (EC_Dens*1.660538921e-24*1.007947*0.61709348966) / UNIT_D;
   double cl_dens = (EC_Dens*MU_NORM*0.61709348966) / UNIT_D;
//   double cl_pres = EC_Dens*Const_kB*EC_Temp / UNIT_P; 
   double cl_pres = EoS_DensTemp2Pres_CPUPtr( cl_dens, EC_Temp, NULL, EoS_AuxArray_Flt, EoS_AuxArray_Int, h_EoS_Table );

   Dens = cl_dens;
   MomX = 0.0;
   MomY = 0.0;
   MomZ = 0.0;
   Pres = cl_pres;
   Eint = EoS_DensPres2Eint_CPUPtr( Dens, Pres, NULL, EoS_AuxArray_Flt,
                                    EoS_AuxArray_Int, h_EoS_Table );   // assuming EoS requires no passive scalars
   Etot = Hydro_ConEint2Etot( Dens, MomX, MomY, MomZ, Eint, 0.0 );     // do NOT include magnetic energy here

// set the output array
   fluid[DENS] = Dens;
   fluid[MOMX] = MomX;
   fluid[MOMY] = MomY;
   fluid[MOMZ] = MomZ;
   fluid[ENGY] = Etot;

   double Temp_tmp;
   Temp_tmp = (real) Hydro_Con2Temp( fluid[DENS], fluid[MOMX], fluid[MOMY], fluid[MOMZ], fluid[ENGY], fluid+NCOMP_FLUID, 
                                     true, MIN_TEMP, 0.0, EoS_DensEint2Temp_CPUPtr, 
                                     EoS_AuxArray_Flt, EoS_AuxArray_Int, h_EoS_Table );
   count += 1;
//   if ( count < 300 && count > 287 ){
//      printf("Debugging in Init!! fluid[DENS] = %14.20e, fluid[MOMX] = %14.8e, fluid[ENGY] = %14.8e, Temp = %14.20e\n", fluid[DENS], fluid[MOMX], fluid[ENGY], Temp_tmp);
//   }
} // FUNCTION : SetGridIC



//-------------------------------------------------------------------------------------------------------
// Function    :  OutputExactCooling
// Description :  Output the temperature relative error in the exact cooling problem
//
// Note        :  1. Enabled by the runtime option "OPT__OUTPUT_USER"
//                2. Construct the analytical solution corresponding to the cooling function
//
// Parameter   :  None
//
// Return      :  None
//-------------------------------------------------------------------------------------------------------
void Output_ExactCooling()
{
   const char FileName[] = "Output__Error";
   static bool FirstTime = true;

// header
   if ( FirstTime ) {
      if ( MPI_Rank == 0 ) {   
         if ( Aux_CheckFileExist(FileName) )
            Aux_Message( stderr, "WARNING : file \"%s\" already exists !!\n", FileName );
 
         FILE *File_User = fopen( FileName, "a" );
         fprintf( File_User, "#%13s%10s ",  "Time", "DumpID" );
         fprintf( File_User, "%14s %14s %14s %14s %14s", "Temp_nume", "Temp_anal", "Err", "Tcool_nume", "Tcool_anal");
         fprintf( File_User, "\n" );
         fclose( File_User );
      }      
      FirstTime = false;
   } // if ( FirstTime )


   const double cl_X         = 0.7;      // mass-fraction of hydrogen
   const double cl_Z         = 0.018;    // metallicity (in Zsun)
   const double cl_mol       = 1.0/(2*cl_X+0.75*(1-cl_X-cl_Z)+cl_Z*0.5);   // mean (total) molecular weights 
   const double cl_mole      = 2.0/(1+cl_X);   // mean electron molecular weights
   const double cl_moli      = 1.0/cl_X;   // mean proton molecular weights
   const double cl_moli_mole = cl_moli*cl_mole;  // Assume the molecular weights are constant, mu_e*mu_i = 1.464
   const double Temp_cut     = 1e5;
 
// Get the numerical result
   real fluid[NCOMP_TOTAL];
   double Temp_nume = 0.0;
   double Temp_nume_tmp = 0.0;
   double Tcool_nume = 0.0;
   int    count = 0;
   const int lv = 0;

   for (int k=1; k<PS1; k++){
   for (int j=1; j<PS1; j++){
   for (int i=1; i<PS1; i++){
      for (int v=0; v<NCOMP_TOTAL; v++)   fluid[v] = amr->patch[ amr->FluSg[lv] ][lv][0]->fluid[v][k][j][i];
      Temp_nume_tmp = (real) Hydro_Con2Temp( fluid[0], fluid[1], fluid[2], fluid[3], fluid[4], fluid+NCOMP_FLUID, 
                                             true, MIN_TEMP, 0.0, EoS_DensEint2Temp_CPUPtr, 
                                             EoS_AuxArray_Flt, EoS_AuxArray_Int, h_EoS_Table );
      Tcool_nume += 1.0/(GAMMA-1)*(Const_kB*cl_moli_mole*Temp_nume_tmp)/(fluid[0]*UNIT_D/MU_NORM*cl_mol*3.2217e-27*sqrt(Temp_nume_tmp))/Const_Myr;
      Temp_nume += Temp_nume_tmp;
      count += 1;
   }}}
   Temp_nume /= count;
   Tcool_nume /= count;

// Compute the analytical solution
   double gsl_result, gsl_error;
   double K = -(GAMMA-1)*EC_Dens*cl_mol*cl_mol/cl_moli_mole/Const_kB;
   gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000);
   gsl_function F;
   F.function = &integrand;
   gsl_integration_qags(&F, EC_Temp, Temp_nume, 0, 1e-10, 1000, w, &gsl_result, &gsl_error);
   double Time_gsl = gsl_result/K;

/*
// Single branch
   double Temp_anal, Tcool_anal;
   if (sqrt(EC_Temp) >= 3.2217e-27/2.0*(GAMMA-1)*EC_Dens*cl_mol*cl_mol/cl_mole/cl_moli/Const_kB*Time[0]*UNIT_T){
      Temp_anal = pow(sqrt(EC_Temp) - 3.2217e-27/2.0*(GAMMA-1)*EC_Dens*cl_mol*cl_mol/cl_mole/cl_moli/Const_kB*Time[0]*UNIT_T, 2.0);
      if (Temp_anal < MIN_TEMP)   Temp_anal = MIN_TEMP;
   }
   else   Temp_anal = MIN_TEMP;

   Tcool_anal = 1.0/(GAMMA-1)*EC_Dens*Const_kB*Temp_anal/((EC_Dens*cl_mol/cl_mole)*(EC_Dens*cl_mol/cl_moli)*3.2217e-27*sqrt(Temp_anal))/Const_Myr;
*/
/*
// 2 branch
   const int size_anal = 1001;       
   double time_anal[size_anal];
   double Temp_anal_arr[size_anal] = {0.0};
   double Tcool_anal_arr[size_anal] = {0.0};
   double time_cut, Temp_anal, Tcool_anal;

   for (int i=0; i<size_anal; i++)   time_anal[i] = i*0.1;

   for (int i=0; i<size_anal; i++) {
      if (sqrt(EC_Temp) >= 3.2217e-27/2.0*(GAMMA-1)*EC_Dens*cl_mol*cl_mol/cl_mole/cl_moli/Const_kB*time_anal[i]*Const_Myr){
         Temp_anal_arr[i] = pow(sqrt(EC_Temp) - 3.2217e-27/2.0*(GAMMA-1)*EC_Dens*cl_mol*cl_mol/cl_mole/cl_moli/Const_kB*time_anal[i]*Const_Myr, 2.0);
         if (Temp_anal_arr[i] < MIN_TEMP)   Temp_anal_arr[i] = MIN_TEMP;
      }
      else   Temp_anal_arr[i] = MIN_TEMP;
   }

   for (int i=0; i<size_anal-1; i++) {
      if ( Temp_anal_arr[i] >= Temp_cut && Temp_anal_arr[i+1] <= Temp_cut ){
         time_cut = time_anal[i] + (time_anal[i+1]-time_anal[i])*(Temp_cut-Temp_anal_arr[i])/(Temp_anal_arr[i+1]-Temp_anal_arr[i]);
      }
   }

//   for (int i=0; i<size_anal; i++) {
//      Tcool_anal[i] = 1.0/(GAMMA-1)*EC_Dens*Const_kB*Temp_anal[i]/((EC_Dens*cl_mol/cl_mole)*(EC_Dens*cl_mol/cl_moli)*3.2217e-27*sqrt(Temp_anal[i]))/Const_Myr;
//   }

   if ( Time[0]*UNIT_T <= time_cut*Const_Myr ){
      if (sqrt(EC_Temp) >= 3.2217e-27/2.0*(GAMMA-1)*EC_Dens*cl_mol*cl_mol/cl_mole/cl_moli/Const_kB*Time[0]*UNIT_T){
          Temp_anal = pow(sqrt(EC_Temp) - 3.2217e-27/2.0*(GAMMA-1)*EC_Dens*cl_mol*cl_mol/cl_mole/cl_moli/Const_kB*Time[0]*UNIT_T, 2.0);
      if (Temp_anal < MIN_TEMP)   Temp_anal = MIN_TEMP;
      }
      else   Temp_anal = MIN_TEMP;
   
      Tcool_anal = 1.0/(GAMMA-1)*EC_Dens*Const_kB*Temp_anal/((EC_Dens*cl_mol/cl_mole)*(EC_Dens*cl_mol/cl_moli)*3.2217e-27*sqrt(Temp_anal))/Const_Myr;
   }
   else {
      if (pow(Temp_cut, 0.6) >= 3.2217e-27*0.6*(GAMMA-1)*EC_Dens*cl_mol*cl_mol/cl_mole/cl_moli/Const_kB*(Time[0]*UNIT_T-time_cut*Const_Myr)){
          Temp_anal = pow(pow(Temp_cut, 0.6) - 3.2217e-27*0.6*(GAMMA-1)*EC_Dens*cl_mol*cl_mol/cl_mole/cl_moli/Const_kB*(Time[0]*UNIT_T-time_cut*Const_Myr), 1.0/0.6);
      if (Temp_anal < MIN_TEMP)   Temp_anal = MIN_TEMP;
      }   
      else   Temp_anal = MIN_TEMP;
   
      Tcool_anal = 1.0/(GAMMA-1)*EC_Dens*Const_kB*Temp_anal/((EC_Dens*cl_mol/cl_mole)*(EC_Dens*cl_mol/cl_moli)*3.2217e-27*pow(Temp_anal, 0.4))/Const_Myr;
   }
*/

// Record
   if ( MPI_Rank == 0 ) {
      FILE *File_User = fopen( FileName, "a" );
      fprintf( File_User, "%14.7e%10d ", Time[0]*UNIT_T/Const_Myr, DumpID );
//      fprintf( File_User, "%14.7e %14.7e %14.7e %14.7e %14.7e", Temp_nume, Temp_anal, (Temp_nume-Temp_anal)/Temp_anal, Tcool_nume, Tcool_anal );
      fprintf( File_User, "%14.7e %14.7e %14.7e %14.7e %14.7e", Temp_nume, Time_gsl/Const_Myr, (Time[0]*UNIT_T-Time_gsl)/Time_gsl, Tcool_nume, 0.0 );
      fprintf( File_User, "\n" );
      fclose( File_User );
   }


} // FUNCTION : Output_ExactCooling
#endif


double Lambda(double TEMP, double ZIRON){
   double TLOGC = 5.65;
   double QLOGC = -21.566;
   double QLOGINFTY = -23.1;
   double PPP = 0.8;
   double TLOGM = 5.1;
   double QLOGM = -20.85;
   double SIG = 0.65;
   double TLOG = log10(TEMP);

   double QLOG0, QLOG1, QLAMBDA0, QLAMBDA1, ARG, BUMP1RHS, BUMP2LHS, Lambdat;
   if (TLOG >= 6.1)   QLOG0 = -26.39 + 0.471*log10(TEMP + 3.1623e6);
   else if (TLOG >= 4.9){         
      ARG = pow(10.0, (-(TLOG-4.9)/0.5)) + 0.077302;
      QLOG0 = -22.16 + log10(ARG);
   }                              
   else if (TLOG >= 4.25){        
      BUMP1RHS = -21.98 - ((TLOG-4.25)/0.55);
      BUMP2LHS = -22.16 - pow((TLOG-4.9)/0.284, 2); 
      QLOG0 = fmax(BUMP1RHS, BUMP2LHS);
   }                              
   else   QLOG0 = -21.98 - pow((TLOG-4.25)/0.2, 2); 
                                  
   if (QLOG0 < -30.0)   QLOG0 = -30.0;
   QLAMBDA0 = pow(10.0, QLOG0); 

   if (TLOG >= 5.65) {
       QLOG1 = QLOGC - PPP * (TLOG - TLOGC);
       QLOG1 = fmax(QLOG1, QLOGINFTY);
   } else {
       QLOG1 = QLOGM - pow((TLOG - TLOGM) / SIG, 2.0);
   }

   if (QLOG1 < -30.0)   QLOG1 = -30.0;
   QLAMBDA1 = pow(10.0, QLOG1);

//   Lambdat = QLAMBDA0;
   Lambdat = QLAMBDA0 + ZIRON * QLAMBDA1;
//   Lambdat = 3.2217e-27 * sqrt(TEMP);

   return Lambdat;
}

double integrand(double Temp, void *params){
    double Lambda_T = Lambda(Temp, 0.018);
    return 1.0/Lambda_T;
}



//-------------------------------------------------------------------------------------------------------
// Function    :  Init_TestProb_Hydro_ExactCooling
// Description :  Test problem initializer
//
// Note        :  None
//
// Parameter   :  None
//
// Return      :  None
//-------------------------------------------------------------------------------------------------------
void Init_TestProb_Hydro_ExactCooling()
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
   Output_User_Ptr                = Output_ExactCooling;
//   End_User_Ptr                   = End_ClusterMerger;
//   Aux_Record_User_Ptr            = Aux_Record_ClusterMerger;

#  endif
/*
   Init_Function_User_Ptr            = SetGridIC;
#  ifdef MHD
   Init_Function_BField_User_Ptr     = SetBFieldIC;
#  endif
// comment out Init_ByFile_User_Ptr to use the default
// Init_ByFile_User_Ptr              = NULL; // option: OPT__INIT=3;             example: Init/Init_ByFile.cpp -> Init_ByFile_Default()
   Init_Field_User_Ptr               = NULL; // set NCOMP_PASSIVE_USER;          example: TestProblem/Hydro/Plummer/Init_TestProb_Hydro_Plummer.cpp --> AddNewField()
   Flag_Region_Ptr                   = NULL; // option: OPT__FLAG_REGION;        example: Refing/Flag_Region.cpp
   Flag_User_Ptr                     = NULL; // option: OPT__FLAG_USER;          example: Refine/Flag_User.cpp
   Mis_GetTimeStep_User_Ptr          = NULL; // option: OPT__DT_USER;            example: Miscellaneous/Mis_GetTimeStep_User.cpp
   Mis_UserWorkBeforeNextLevel_Ptr   = NULL; //                                  example: Miscellaneous/Mis_UserWorkBeforeNextLevel.cpp
   Mis_UserWorkBeforeNextSubstep_Ptr = NULL; //                                  example: Miscellaneous/Mis_UserWorkBeforeNextSubstep.cpp
   BC_User_Ptr                       = NULL; // option: OPT__BC_FLU_*=4;         example: TestProblem/ELBDM/ExtPot/Init_TestProb_ELBDM_ExtPot.cpp --> BC()
#  ifdef MHD
   BC_BField_User_Ptr                = NULL; // option: OPT__BC_FLU_*=4;
#  endif
   Flu_ResetByUser_Func_Ptr          = NULL; // option: OPT__RESET_FLUID;        example: Fluid/Flu_ResetByUser.cpp
   Init_DerivedField_User_Ptr        = NULL; // option: OPT__OUTPUT_USER_FIELD;  example: Fluid/Flu_DerivedField_User.cpp
   Output_User_Ptr                   = NULL; // option: OPT__OUTPUT_USER;        example: TestProblem/Hydro/AcousticWave/Init_TestProb_Hydro_AcousticWave.cpp --> OutputError()
   Output_UserWorkBeforeOutput_Ptr   = NULL; // option: none;                    example: Output/Output_UserWorkBeforeOutput.cpp
   Aux_Record_User_Ptr               = NULL; // option: OPT__RECORD_USER;        example: Auxiliary/Aux_Record_User.cpp
   Init_User_Ptr                     = NULL; // option: none;                    example: none
   End_User_Ptr                      = NULL; // option: none;                    example: TestProblem/Hydro/ClusterMerger_vs_Flash/Init_TestProb_ClusterMerger_vs_Flash.cpp --> End_ClusterMerger()
#  ifdef GRAVITY
   Init_ExtAcc_Ptr                   = NULL; // option: OPT__EXT_ACC;            example: SelfGravity/CPU_Gravity/CPU_ExtAcc_PointMass.cpp
   End_ExtAcc_Ptr                    = NULL;
   Init_ExtPot_Ptr                   = NULL; // option: OPT__EXT_POT;            example: SelfGravity/CPU_Poisson/CPU_ExtPot_PointMass.cpp
   End_ExtPot_Ptr                    = NULL;
   Poi_AddExtraMassForGravity_Ptr    = NULL; // option: OPT__GRAVITY_EXTRA_MASS; example: none
   Poi_UserWorkBeforePoisson_Ptr     = NULL; // option: none;                    example: SelfGravity/Poi_UserWorkBeforePoisson.cpp
#  endif
#  ifdef PARTICLE
   Par_Init_ByFunction_Ptr           = NULL; // option: PAR_INIT=1;              example: Particle/Par_Init_ByFunction.cpp
   Par_Init_Attribute_User_Ptr       = NULL; // set PAR_NATT_USER;               example: TestProblem/Hydro/AGORA_IsolatedGalaxy/Init_TestProb_Hydro_AGORA_IsolatedGalaxy.cpp --> AddNewParticleAttribute()
#  endif
#  if ( EOS == EOS_USER )
   EoS_Init_Ptr                      = NULL; // option: EOS in the Makefile;     example: EoS/User_Template/CPU_EoS_User_Template.cpp
   EoS_End_Ptr                       = NULL;
#  endif
#  endif // #if ( MODEL == HYDRO )
   Src_Init_User_Ptr                 = NULL; // option: SRC_USER;                example: SourceTerms/User_Template/CPU_Src_User_Template.cpp
*/

   if ( MPI_Rank == 0 )    Aux_Message( stdout, "%s ... done\n", __FUNCTION__ );

} // FUNCTION : Init_TestProb_Hydro_ExactCooling
