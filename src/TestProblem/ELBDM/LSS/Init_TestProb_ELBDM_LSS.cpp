#include "GAMER.h"



// problem-specific global variables
// =======================================================================================
static int   LSS_InitMode;         // initialization mode: 1=density-only, 2=real and imaginary parts or phase and density
static int   ZoomIn_MaxLvOutside;  // maximum refinement level outside of the zoom-in box
static int   ZoomIn_NRow;             // number of changes of the zoom-in box
static real *ZoomIn_a, *ZoomIn_LX, *ZoomIn_LY, *ZoomIn_LZ, 
            *ZoomIn_CenterX, *ZoomIn_CenterY, *ZoomIn_CenterZ;   // arrays of the table of scale factor, length, and center in xyz-axis

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
#  if ( MODEL != ELBDM)
   Aux_Error( ERROR_INFO, "MODEL != ELBDM !!\n" );
#  endif

#  ifndef GRAVITY
   Aux_Error( ERROR_INFO, "GRAVITY must be enabled !!\n" );
#  endif

#  ifndef COMOVING
   Aux_Error( ERROR_INFO, "COMOVING must be enabled !!\n" );
#  endif

#  ifdef GRAVITY
   if ( OPT__BC_FLU[0] != BC_FLU_PERIODIC  ||  OPT__BC_POT != BC_POT_PERIODIC )
      Aux_Error( ERROR_INFO, "must adopt periodic BC for this test !!\n" );
#  endif

   if ( OPT__INIT == INIT_BY_FUNCTION )
      Aux_Error( ERROR_INFO, "OPT__INIT=INIT_BY_FUNCTION (1) is not supported for this test !!\n" );


// warnings

   if ( MPI_Rank == 0 )    Aux_Message( stdout, "   Validating test problem %d ... done\n", TESTPROB_ID );

} // FUNCTION : Validate



#if ( MODEL == ELBDM )
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
   const char FileNameZoomIn[] = "Input__ZoominBox";
   ReadPara_t *ReadPara  = new ReadPara_t;

// (1-1) add parameters in the following format:
// --> note that VARIABLE, DEFAULT, MIN, and MAX must have the same data type
// --> some handy constants (e.g., Useless_bool, Eps_double, NoMin_int, ...) are defined in "include/ReadPara.h"
// ********************************************************************************************************************************
// ReadPara->Add( "KEY_IN_THE_FILE",   &VARIABLE,              DEFAULT,       MIN,              MAX               );
// ********************************************************************************************************************************
   ReadPara->Add( "LSS_InitMode",        &LSS_InitMode,          1,             1,                2                 );
   ReadPara->Add( "ZoomIn_MaxLvOutside", &ZoomIn_MaxLvOutside,   MAX_LEVEL,     0,                MAX_LEVEL         );
 
   ReadPara->Read( FileName );

   delete ReadPara;
     
   if ( OPT__FLAG_REGION )
   {
      if ( ELBDM_FIRST_WAVE_LEVEL > ZoomIn_MaxLvOutside )
         Aux_Error( ERROR_INFO, " it is required to set ELBDM_FIRST_WAVE_LEVEL <= ZoomIn_MaxLvOutside for zoom-in simulation !!\n" );

      if ( !Aux_CheckFileExist( FileNameZoomIn ) )
         Aux_Error( ERROR_INFO, "%s does not exist !!\n", FileNameZoomIn );

      char *input_line = NULL;
      size_t len = 0;
      int n;
      FILE *File_zoom;
      File_zoom = fopen( FileNameZoomIn, "r" );

//    get the number of rows in Input__ZoominBox
      ZoomIn_NRow = -1;
      while ( getline( &input_line, &len, File_zoom ) != -1 ) ++ZoomIn_NRow;
      fclose( File_zoom );

//    load Zoom-in Lagrangian Box Volumne at different redshifts
      ZoomIn_a        = new real [ZoomIn_NRow];
      ZoomIn_LX       = new real [ZoomIn_NRow];
      ZoomIn_LY       = new real [ZoomIn_NRow];
      ZoomIn_LZ       = new real [ZoomIn_NRow];
      ZoomIn_CenterX  = new real [ZoomIn_NRow];
      ZoomIn_CenterY  = new real [ZoomIn_NRow];
      ZoomIn_CenterZ  = new real [ZoomIn_NRow];

      File_zoom = fopen( FileNameZoomIn, "r" );
      getline( &input_line, &len, File_zoom ); //   skip the header

      for (int s=0; s<ZoomIn_NRow; s++)
      {
         n = getline( &input_line, &len, File_zoom );
         if ( n <= 1 )
            Aux_Error( ERROR_INFO, "incorrect reading at zoom-in table at line %d of the file <%s> !!\n", s+1, FileNameZoomIn );

         sscanf( input_line, "%f%f%f%f%f%f%f",
		 &ZoomIn_a[s],
		 &ZoomIn_LX[s], &ZoomIn_LY[s], &ZoomIn_LZ[s],
		 &ZoomIn_CenterX[s], &ZoomIn_CenterY[s], &ZoomIn_CenterZ[s] );
       
         if ( s < 1 ) continue;

         if ( ZoomIn_a[s-1] < ZoomIn_a[s] )
            Aux_Error( ERROR_INFO, "Current a=%f is greater than the previous a=%f. Scale factors are not listed in descending order in %s!!\n", ZoomIn_a[s], ZoomIn_a[s-1], FileNameZoomIn );
 
         if ( ZoomIn_CenterX[s-1] - ZoomIn_LX[s-1]/2 < ZoomIn_CenterX[s] - ZoomIn_LX[s]/2  ||
	      ZoomIn_CenterY[s-1] - ZoomIn_LY[s-1]/2 < ZoomIn_CenterY[s] - ZoomIn_LY[s]/2  ||
	      ZoomIn_CenterZ[s-1] - ZoomIn_LZ[s-1]/2 < ZoomIn_CenterZ[s] - ZoomIn_LZ[s]/2  ||
	      ZoomIn_CenterX[s-1] + ZoomIn_LX[s-1]/2 > ZoomIn_CenterX[s] + ZoomIn_LX[s]/2  ||
	      ZoomIn_CenterY[s-1] + ZoomIn_LY[s-1]/2 > ZoomIn_CenterY[s] + ZoomIn_LY[s]/2  ||
	      ZoomIn_CenterZ[s-1] + ZoomIn_LZ[s-1]/2 > ZoomIn_CenterZ[s] + ZoomIn_LZ[s]/2 )
            Aux_Error( ERROR_INFO, "Zoom-in box at a = %.2f with LX=%.2f LY=%.2f LZ=%.2f CenterX=%.2f CenterY=%.2f CenterZ=%.2f is outside of the Zoom-in box at earlier a = %.2f with LX=%.2f LY=%.2f LZ=%.2f CenterX=%.2f CenterY=%.2f CenterZ=%.2f !!\n", 
		       ZoomIn_a[s-1], ZoomIn_LX[s-1], ZoomIn_LY[s-1], ZoomIn_LZ[s-1], ZoomIn_CenterX[s-1], ZoomIn_CenterY[s-1], ZoomIn_CenterZ[s-1],
		       ZoomIn_a[s], ZoomIn_LX[s], ZoomIn_LY[s], ZoomIn_LZ[s], ZoomIn_CenterX[s], ZoomIn_CenterY[s], ZoomIn_CenterZ[s] );

      } // for (int s=0; s<ZoomIn_NRow; s++)

      fclose( File_zoom );
   } // if ( OPT__FLAG_REGION )


// (2) reset other general-purpose parameters
//     --> a helper macro PRINT_RESET_PARA is defined in Macro.h
   const long   End_Step_Default = __INT_MAX__;
   const double End_T_Default    = 1.0;

   if ( END_STEP < 0 ) {
      END_STEP = End_Step_Default;
      PRINT_RESET_PARA( END_STEP, FORMAT_LONG, "" );
   }

   if ( END_T < 0.0 ) {
      END_T = End_T_Default;
      PRINT_RESET_PARA( END_T, FORMAT_REAL, "" );
   }


// (3) make a note
   if ( MPI_Rank == 0 )
   {
      Aux_Message( stdout, "=============================================================================\n" );
      Aux_Message( stdout, "  test problem ID     = %d\n", TESTPROB_ID  );
      Aux_Message( stdout, "  initialization mode = %d\n", LSS_InitMode );
      Aux_Message( stdout, "  ZoomIn_MaxLvOutside = %d\n", ZoomIn_MaxLvOutside );
      Aux_Message( stdout, "  Table in %s\n", FileNameZoomIn );
      Aux_Message( stdout, "  #ScaleFactor    LX(UNIT_L)    LY(UNIT_L)    LZ(UNIT_L)   CenterX(UNIT_L)   CenterY(UNIT_L)   CenterZ(UNIT_L)\n" );
      
      for (int i=0; i<ZoomIn_NRow; i++)
         Aux_Message( stdout, "  %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
		      ZoomIn_a[i], ZoomIn_LX[i], ZoomIn_LY[i], ZoomIn_LZ[i], ZoomIn_CenterX[i], ZoomIn_CenterY[i], ZoomIn_CenterZ[i] );
      Aux_Message( stdout, "=============================================================================\n" );
   }


   if ( MPI_Rank == 0 )    Aux_Message( stdout, "   Setting runtime parameters ... done\n" );

} // FUNCTION : SetParameter



//-------------------------------------------------------------------------------------------------------
// Function    :  Flag_Region_LSS
// Description :  Template for checking if the element (i,j,k) of the input patch is within
//                the regions allowed to be refined
//
// Note        :  1. Invoked by Flag_Check() using the function pointer "Flag_Region_Ptr",
//                   which must be set by a test problem initializer
//                2. Enabled by the runtime option "OPT__FLAG_REGION"
//
// Parameter   :  i,j,k       : Indices of the target element in the patch ptr[0][lv][PID]
//                lv          : Refinement level of the target patch
//                PID         : ID of the target patch
//
// Return      :  "true/false"  if the input cell "is/is not" within the region allowed for refinement
//-------------------------------------------------------------------------------------------------------
bool Flag_Region_LSS( const int i, const int j, const int k, const int lv, const int PID )
{

   if ( lv < ZoomIn_MaxLvOutside )   return true; // restrict the maximum refinement level outside the zoom-in box

   const double dh     = amr->dh[lv];                                         // cell size
   const double Pos[3] = { amr->patch[0][lv][PID]->EdgeL[0] + (i+0.5)*dh,     // x,y,z position
                           amr->patch[0][lv][PID]->EdgeL[1] + (j+0.5)*dh,
                           amr->patch[0][lv][PID]->EdgeL[2] + (k+0.5)*dh  };

   bool Within = true;

// get the index of ZoomIn_* that can collect the zoom-in box's volume and center based on the current simulation time
   int zoom_idx; 
   for (int s=0; s<ZoomIn_NRow; s++)
   {
      if ( Time[0] <= ZoomIn_a[s] )   continue;
      zoom_idx = s;
      break;
   } // for (int s=0; s<ZoomIn_NRow; s++)

// periodic BC checks
   const double Pos_x = ( Pos[0] - ZoomIn_CenterX[zoom_idx] >  0.5*amr->BoxSize[0] ) ? Pos[0] - amr->BoxSize[0] :
                        ( Pos[0] - ZoomIn_CenterX[zoom_idx] < -0.5*amr->BoxSize[0] ) ? Pos[0] + amr->BoxSize[0] : Pos[0];
   const double Pos_y = ( Pos[1] - ZoomIn_CenterY[zoom_idx] >  0.5*amr->BoxSize[1] ) ? Pos[1] - amr->BoxSize[1] :
                        ( Pos[1] - ZoomIn_CenterY[zoom_idx] < -0.5*amr->BoxSize[1] ) ? Pos[1] + amr->BoxSize[1] : Pos[1];
   const double Pos_z = ( Pos[2] - ZoomIn_CenterZ[zoom_idx] >  0.5*amr->BoxSize[2] ) ? Pos[2] - amr->BoxSize[2] :
                        ( Pos[2] - ZoomIn_CenterZ[zoom_idx] < -0.5*amr->BoxSize[2] ) ? Pos[2] + amr->BoxSize[2] : Pos[2];
   const double dR[3] = { Pos_x  - ZoomIn_CenterX[zoom_idx],
                          Pos_y  - ZoomIn_CenterY[zoom_idx],
                          Pos_z  - ZoomIn_CenterZ[zoom_idx] };

   Within = ( abs(dR[0]) < ZoomIn_LX[zoom_idx]/2 )  &&
            ( abs(dR[1]) < ZoomIn_LY[zoom_idx]/2 )  &&
            ( abs(dR[2]) < ZoomIn_LZ[zoom_idx]/2 );
   return Within;

} // FUNCTION : Flag_Region_LSS



//-------------------------------------------------------------------------------------------------------
// Function    :  Init_ByFile_ELBDM_LSS
// Description :  Function to actually set the fluid field from the input uniform-mesh array
//
// Note        :  1. Invoked by Init_ByFile_AssignData() using the function pointer Init_ByFile_User_Ptr()
//                   --> The function pointer may be reset by various test problem initializers, in which case
//                       this funtion will become useless
//                2. One can use LSS_InitMode to support different data formats
//                3. For ELBDM_SCHEME == ELBDM_WAVE this function expects:
//                       LSS_InitMode == 1: Density
//                       LSS_InitMode == 2: Real and imaginary part
//                   For ELBDM_SCHEME == ELBDM_HYBRID this function expects:
//                       LSS_InitMode == 1: Density
//                       LSS_InitMode == 2: Density and phase
// Parameter   :  fluid_out : Fluid field to be set
//                fluid_in  : Fluid field loaded from the uniform-mesh array (UM_IC)
//                nvar_in   : Number of variables in fluid_in
//                x/y/z     : Target physical coordinates
//                Time      : Target physical time
//                lv        : Target AMR level
//                AuxArray  : Auxiliary array
//
// Return      :  fluid_out
//-------------------------------------------------------------------------------------------------------
void Init_ByFile_ELBDM_LSS( real fluid_out[], const real fluid_in[], const int nvar_in,
                            const double x, const double y, const double z, const double Time,
                            const int lv, double AuxArray[] )
{

   double Re, Im, De;

#  if ( ELBDM_SCHEME == ELBDM_HYBRID )
   double Ph;
#  endif

   switch ( LSS_InitMode )
   {
      case 1:
      {
         if ( nvar_in != 1 )  Aux_Error( ERROR_INFO, "nvar_in (%d) != 1 for LSS_InitMode 1 !!\n", nvar_in );

         const double AveDens     = 1.0;        // assuming background density = 1.0
         const double GrowingFrac = 3.0/5.0;    // growing-mode amplitude = total amplitude * 3/5

         Re = sqrt( (fluid_in[0]-AveDens )/GrowingFrac + AveDens );
         Im = 0.0;   // constant phase
         De = SQR( Re ) + SQR( Im );

#        if ( ELBDM_SCHEME == ELBDM_HYBRID )
         Ph = 0.0;
#        endif

         break;
      }

      case 2:
      {
         if ( nvar_in != 2 )  Aux_Error( ERROR_INFO, "nvar_in (%d) != 2 for LSS_InitMode 2 !!\n", nvar_in );

//       ELBDM_WAVE   expects real and imaginary parts
//       ELBDM_HYBRID expects density and phase
#        if   ( ELBDM_SCHEME == ELBDM_WAVE )
         Re = fluid_in[0];
         Im = fluid_in[1];
         De = SQR( Re ) + SQR( Im );
#        elif ( ELBDM_SCHEME == ELBDM_HYBRID )
         De = fluid_in[0];
         Ph = fluid_in[1];
         Re = sqrt( De )*cos( Ph );
         Im = sqrt( De )*sin( Ph );
#        else
#        error : ERROR : unsupported ELBDM_SCHEME !!
#        endif
         break;
      }

      default:
         Aux_Error( ERROR_INFO, "unsupported initialization mode (%s = %d) !!\n",
                    "LSS_InitMode", LSS_InitMode );
   } // switch ( LSS_InitMode )

#  if ( ELBDM_SCHEME == ELBDM_HYBRID )
   if ( amr->use_wave_flag[lv] ) {
#  endif
   fluid_out[DENS] = De;
   fluid_out[REAL] = Re;
   fluid_out[IMAG] = Im;
#  if ( ELBDM_SCHEME == ELBDM_HYBRID )
   } else {
   fluid_out[DENS] = De;
   fluid_out[PHAS] = Ph;
   fluid_out[STUB] = 0.0;
   }
#  endif

} // Init_ByFile_ELBDM_LSS
#endif // #if ( MODEL == ELBDM )



//-------------------------------------------------------------------------------------------------------
// Function    :  End_Region_LSS
// Description :  Free memory before terminating the program
//
// Note        :  1. Linked to the function pointer "End_User_Ptr" to replace "End_User()"
//
// Parameter   :  None
//-------------------------------------------------------------------------------------------------------
void End_Region_LSS()
{

   delete [] ZoomIn_a;
   delete [] ZoomIn_LX;
   delete [] ZoomIn_LY;
   delete [] ZoomIn_LZ;
   delete [] ZoomIn_CenterX;
   delete [] ZoomIn_CenterY;
   delete [] ZoomIn_CenterZ;

} // FUNCTION : End_Region_LSS



//-------------------------------------------------------------------------------------------------------
// Function    :  Init_TestProb_ELBDM_LSS
// Description :  Test problem initializer
//
// Note        :  None
//
// Parameter   :  None
//
// Return      :  None
//-------------------------------------------------------------------------------------------------------
void Init_TestProb_ELBDM_LSS()
{

   if ( MPI_Rank == 0 )    Aux_Message( stdout, "%s ...\n", __FUNCTION__ );

// validate the compilation flags and runtime parameters
   Validate();

#  if ( MODEL == ELBDM )
// set the problem-specific runtime parameters
   SetParameter();

   Init_ByFile_User_Ptr = Init_ByFile_ELBDM_LSS;
   Flag_Region_Ptr      = Flag_Region_LSS;
   End_User_Ptr         = End_Region_LSS;

#  endif // #if ( MODEL == ELBDM )

   if ( MPI_Rank == 0 )    Aux_Message( stdout, "%s ... done\n", __FUNCTION__ );

} // FUNCTION : Init_TestProb_ELBDM_LSS
