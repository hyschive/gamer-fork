#include "GAMER.h"
#include "TestProb.h"

#ifdef SUPPORT_GSL
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#endif // #ifdef SUPPORT_GSL


static void SetGridIC( real fluid[], const double x, const double y, const double z, const double Time,
                       const int lv, double AuxArray[] );

static void BC( real Array[], const int ArraySize[], real fluid[], const int NVar_Flu,
                const int GhostSize, const int idx[], const double pos[], const double Time,
                const int lv, const int TFluVarIdxList[], double AuxArray[] );

static void End_DiskHeating();
static void AddNewParticleAttribute();
static double Get_Dispersion( double r );
static double Halo_Density( double r );

void Init_ExtPot_Soliton();
// problem-specific global variables
// =======================================================================================
FieldIdx_t ParLabel_Idx = Idx_Undefined;
static RandomNumber_t *RNG = NULL;
       double  Cen[3];                    // center
static bool    AddFixedHalo;              // add a fixed halo, must enable OPT__FREEZE_FLUID
static bool    HaloUseTable;              // 0 = from analytical profile, 1 = from table
       double  m_22;                      // ELBDM particle mass, used for soliton profile of the fixed halo
       double  CoreRadius;                // soliton radius of the fixed halo (in kpc)
static double  Rho_0;                     // halo rho_0 (in 1.0e+10 Msun*kpc^-3)
static double  Rs;                        // halo Rs (in kpc)
static double  Alpha;                     // dimensionless, used for alpha-beta-gamma density profile of the fixed halo
static double  Beta;                      // dimensionless, used for alpha-beta-gamma density profile of the fixed halo
static double  Gamma;                     // dimensionless, used for alpha-beta-gamma density profile of the fixed halo
static char    DensTableFile[MAX_STRING]; // fixed halo density profile filename
static double *DensTable = NULL;          // density table
static int     DensTable_Nbin;            // number of bins of density table
static bool    AddParWhenRestart;         // add a new disk to an existing snapshot, must enable OPT__RESTART_RESET
static bool    AddParWhenRestartByFile;   // add a new disk via PAR_IC
static long    AddParWhenRestartNPar;     // particle number of the new disk
static int     NewDisk_RSeed;             // random seed for setting new disk particle position and velocity
static double  Disk_Mass;                 // total mass of the new disk
static double  Disk_R;                    // scale radius of the new disk
static char    DispTableFile[MAX_STRING]; // velocity dispersion filename
static double *DispTable = NULL;          // radius of velocity dispersion table
static int     DispTable_Nbin;            // number of bins of velocity dispersion table
// =======================================================================================
#ifdef PARTICLE
void Par_Init_ByFunction_DiskHeating( const long NPar_ThisRank, const long NPar_AllRank,
                                      real *ParMass, real *ParPosX, real *ParPosY, real *ParPosZ,
                                      real *ParVelX, real *ParVelY, real *ParVelZ, real *ParTime,
                                      real *ParType, real *AllAttribute[PAR_NATT_TOTAL]);
static void Init_NewDiskRestart();
static void Par_AfterAcceleration( const long NPar_ThisRank, const long NPar_AllRank,
                                   real *ParMass, real *ParPosX, real *ParPosY, real *ParPosZ,
                                   real *ParVelX, real *ParVelY, real *ParVelZ,
                                   real *ParAccX, real *ParAccY, real *ParAccZ,
                                   real *ParTime, real *ParType, real *AllAttribute[PAR_NATT_TOTAL] );

#endif



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

#  ifndef PARTICLE
   Aux_Error( ERROR_INFO, "PARTICLE must be enabled !!\n" );
#  endif

#  ifdef COMOVING
   Aux_Error( ERROR_INFO, "COMOVING must be disabled !!\n" );
#  endif


// warnings
   if ( MPI_Rank == 0 )
   {
      if ( FLAG_BUFFER_SIZE < 5 )
         Aux_Message( stderr, "WARNING : it's recommended to set FLAG_BUFFER_SIZE >= 5 for this test !!\n" );
   } // if ( MPI_Rank == 0 )



   if ( MPI_Rank == 0 )    Aux_Message( stdout, "   Validating test problem %d ... done\n", TESTPROB_ID );

} // FUNCTION : Validate



// replace HYDRO by the target model (e.g., MHD/ELBDM) and also check other compilation flags if necessary (e.g., GRAVITY/PARTICLE)
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
// ReadPara->Add( "KEY_IN_THE_FILE",         &VARIABLE,                DEFAULT,       MIN,              MAX               );
// ********************************************************************************************************************************
   ReadPara->Add( "CenX",                    &Cen[0],                  NoDef_double,  NoMin_double,     NoMax_double      );
   ReadPara->Add( "CenY",                    &Cen[1],                  NoDef_double,  NoMin_double,     NoMax_double      );
   ReadPara->Add( "CenZ",                    &Cen[2],                  NoDef_double,  NoMin_double,     NoMax_double      );
   ReadPara->Add( "AddFixedHalo",            &AddFixedHalo,            false,         Useless_bool,     Useless_bool      );
   ReadPara->Add( "HaloUseTable",            &HaloUseTable,            false,         Useless_bool,     Useless_bool      );
   ReadPara->Add( "m_22",                    &m_22,                    0.4,           Eps_double,       NoMax_double      );
   ReadPara->Add( "CoreRadius",              &CoreRadius,              1.0,           Eps_double,       NoMax_double      );
   ReadPara->Add( "Rho_0",                   &Rho_0,                   1.0,           Eps_double,       NoMax_double      );
   ReadPara->Add( "Rs",                      &Rs,                      1.0,           Eps_double,       NoMax_double      );
   ReadPara->Add( "Alpha",                   &Alpha,                   1.0,           Eps_double,       NoMax_double      );
   ReadPara->Add( "Beta",                    &Beta,                    1.0,           Eps_double,       NoMax_double      );
   ReadPara->Add( "Gamma",                   &Gamma,                   1.0,           Eps_double,       NoMax_double      );
   ReadPara->Add( "DensTableFile",           DensTableFile,            NoDef_str,     Useless_str,      Useless_str       );
   ReadPara->Add( "AddParWhenRestart",       &AddParWhenRestart,       false,         Useless_bool,     Useless_bool      );
   ReadPara->Add( "AddParWhenRestartByFile", &AddParWhenRestartByFile, true,          Useless_bool,     Useless_bool      );
   ReadPara->Add( "AddParWhenRestartNPar",   &AddParWhenRestartNPar,   (long)0,       (long)0,          NoMax_long        );
   ReadPara->Add( "NewDisk_RSeed",           &NewDisk_RSeed,           1002,          0,                NoMax_int         );
   ReadPara->Add( "Disk_Mass",               &Disk_Mass,               1.0,           Eps_double,       NoMax_double      );
   ReadPara->Add( "Disk_R",                  &Disk_R,                  1.0,           Eps_double,       NoMax_double      );
   ReadPara->Add( "DispTableFile",           DispTableFile,            NoDef_str,     Useless_str,      Useless_str       );


   ReadPara->Read( FileName );

   delete ReadPara;

// (1-2) set the default values

// (1-3) check the runtime parameters

// (2) set the problem-specific derived parameters
   // use density table as background fixed halo profile
   if ( HaloUseTable == 1 ) {

      // read the density table
      const bool RowMajor_No = false;     // load data into the column-major order
      const bool AllocMem_Yes = true;     // allocate memory for AGORA_VcProf
      const int  NCol        = 2;         // total number of columns to load
      const int  Col[NCol]   = {0, 1};    // target columns: (radius, density)

      DensTable_Nbin = Aux_LoadTable( DensTable, DensTableFile, NCol, Col, RowMajor_No, AllocMem_Yes );

      double *DensTable_r = DensTable + 0*DensTable_Nbin;
      double *DensTable_d = DensTable + 1*DensTable_Nbin;

      // convert to log-log scale
      for (int b=0; b<DensTable_Nbin; b++)
      {
         DensTable_r[b] = log(DensTable_r[b]);
         DensTable_d[b] = log(DensTable_d[b]*CUBE(UNIT_L)/UNIT_M);
      }
   }
// (3) reset other general-purpose parameters
//     --> a helper macro PRINT_WARNING is defined in TestProb.h
   const long   End_Step_Default = 2500;
   const double End_T_Default    = 2.5e-1;

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
      Aux_Message( stdout, "=============================================================================\n" );
      Aux_Message( stdout, "  test problem ID           = %d\n",         TESTPROB_ID             );
      Aux_Message( stdout, "  CenX                      = %13.7e\n",     Cen[0]                  );
      Aux_Message( stdout, "  CenY                      = %13.7e\n",     Cen[1]                  );
      Aux_Message( stdout, "  Cenz                      = %13.7e\n",     Cen[2]                  );
      Aux_Message( stdout, "  AddFixedHalo              = %d\n",         AddFixedHalo            );
      Aux_Message( stdout, "  HaloUseTable              = %d\n",         HaloUseTable            );
      Aux_Message( stdout, "  m_22                      = %13.7e\n",     m_22                    );
      Aux_Message( stdout, "  CoreRadius                = %13.7e\n",     CoreRadius              );
      Aux_Message( stdout, "  Rho_0                     = %13.7e\n",     Rho_0                   );
      Aux_Message( stdout, "  Rs                        = %13.7e\n",     Rs                      );
      Aux_Message( stdout, "  Alpha                     = %13.7e\n",     Alpha                   );
      Aux_Message( stdout, "  Beta                      = %13.7e\n",     Beta                    );
      Aux_Message( stdout, "  Gamma                     = %13.7e\n",     Gamma                   );
      Aux_Message( stdout, "  DensTableFile             = %s\n",         DensTableFile           );
      Aux_Message( stdout, "  AddParWhenRestart         = %d\n",         AddParWhenRestart       );
      Aux_Message( stdout, "  AddParWhenRestartByFile   = %d\n",         AddParWhenRestartByFile );
      Aux_Message( stdout, "  AddParWhenRestartNPar     = %d\n",         AddParWhenRestartNPar   );
      Aux_Message( stdout, "  NewDisk_RSeed             = %d\n",         NewDisk_RSeed           );
      Aux_Message( stdout, "  Disk_Mass                 = %13.7e\n",     Disk_Mass               );
      Aux_Message( stdout, "  Disk_R                    = %13.7e\n",     Disk_R                  );
      Aux_Message( stdout, "  DispTableFile             = %s\n",         DispTableFile           );
      Aux_Message( stdout, "=============================================================================\n" );
   }


   if ( MPI_Rank == 0 )    Aux_Message( stdout, "   Setting runtime parameters ... done\n" );

} // FUNCTION : SetParameter

////-------------------------------------------------------------------------------------------------------
//// Function    :  Halo_Density
//// Description :  alpha-beta-gamma density profile for fixed background halo with soliton as function of radius
//// Note        :  r should have unit in kpc
////                Returned density is in 1.0e10*Msun/kpc^3
////-------------------------------------------------------------------------------------------------------
double Halo_Density(double r) {

   double rho_halo = Rho_0/pow(r/Rs, Alpha)/pow(1 + pow(r/Rs,Beta),(Gamma-Alpha)/Beta);
   if ( fabs(rho_halo) <  __FLT_MIN__) rho_halo = 0;

   double rho_soliton =  0.0019 / m_22 / m_22 * pow(1.0 / CoreRadius/ pow(1 + 0.091 * pow(r / CoreRadius, 2), 2), 4);
   if ( fabs(rho_soliton) <  __FLT_MIN__) rho_soliton = 0;

   double rho_max =  0.0019 / m_22 / m_22 * pow(1.0 / CoreRadius, 4);
   if ((rho_halo + rho_soliton) > rho_max) return rho_max;
   else return(rho_halo + rho_soliton);
} // FUNCTION : Halo_Density

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
// set the output array
   double dens = 0.0;

   if ( AddFixedHalo && OPT__FREEZE_FLUID)  // add a fixed halo at the background
   {
      const double dx     = (x - Cen[0]);
      const double dy     = (y - Cen[1]);
      const double dz     = (z - Cen[2]);
      const double r      = SQRT( dx*dx + dy*dy + dz*dz );
      const double Unit_L_GALIC = Const_kpc; //  1.0 kpc
      const double Unit_M_GALIC = 1.0e10*Const_Msun;    //  1.0e10 solar masses
      const double Unit_D_GALIC = Unit_M_GALIC/CUBE(Unit_L_GALIC);
      const double r_in_kpc = r*UNIT_L/Unit_L_GALIC;

      const double *DensTable_r = DensTable + 0*DensTable_Nbin;
      const double *DensTable_d = DensTable + 1*DensTable_Nbin;

      if (HaloUseTable)
      {
         if (r_in_kpc < exp(DensTable_r[0])) dens = exp(DensTable_d[0]);
         else if (r_in_kpc > exp(DensTable_r[DensTable_Nbin-1])) dens = 0;
         else dens = exp( Mis_InterpolateFromTable( DensTable_Nbin, DensTable_r, DensTable_d, log(r_in_kpc) ) );
      }
      else dens = Halo_Density(r_in_kpc)*Unit_D_GALIC/UNIT_D;
   }

//ELBDM example
   fluid[DENS] = (real)dens;
   fluid[REAL] = sqrt( fluid[DENS] );
   fluid[IMAG] = 0.0;

} // FUNCTION : SetGridIC

void BC( real Array[], const int ArraySize[], real fluid[], const int NVar_Flu,
         const int GhostSize, const int idx[], const double pos[], const double Time,
         const int lv, const int TFluVarIdxList[], double AuxArray[] )
{

// simply call the IC function
   SetGridIC( fluid, pos[0], pos[1], pos[2], Time, lv, AuxArray );

} // FUNCTION : BC


void End_DiskHeating()
{
   delete [] DensTable;
   delete [] DispTable;
} // FUNCTION : End_DiskHeating


# ifdef PARTICLE

//-------------------------------------------------------------------------------------------------------
//// Function    :  AddNewParticleAttribute
//// Description :  Add the problem-specific particle attributes
////
//// Note        :  1. Ref: https://github.com/gamer-project/gamer/wiki/Adding-New-Simulations#v-add-problem-specific-grid-fields-and-particle-attributes
////                2. Invoke AddParticleField() for each of the problem-specific particle attribute:
////                   --> Attribute label sent to AddParticleField() will be used as the output name of the attribute
////                   --> Attribute index returned by AddParticleField() can be used to access the particle attribute data
////                3. Pre-declared attribute indices are put in Field.h
////
//// Parameter   :  None
////
//// Return      :  None
////-------------------------------------------------------------------------------------------------------
void AddNewParticleAttribute()
{

   if ( ParLabel_Idx == Idx_Undefined )  ParLabel_Idx = AddParticleAttribute( "ParLabel" );

} // FUNCTION : AddNewParticleAttribute
#ifdef GRAVITY

//-------------------------------------------------------------------------------------------------------
//// Function    :  Init_NewDiskRestart()
//// Description :  Add a new disk from an existing snapshot
//// Note        :  Must enable OPT__RESTART_RESET and AddParWhenRestart
////-------------------------------------------------------------------------------------------------------
void Init_NewDiskRestart()
{

   if ( amr->Par->Init != PAR_INIT_BY_RESTART  || !OPT__RESTART_RESET || !AddParWhenRestart )   return;

   const long   NNewPar        = ( MPI_Rank == 0 ) ? AddParWhenRestartNPar : 0;
   const long   NPar_AllRank   = NNewPar;
   real *NewParAtt[PAR_NATT_TOTAL];

   for (int v=0; v<PAR_NATT_TOTAL; v++)   NewParAtt[v] = new real [NNewPar];

// set particle attributes
   real *Time_AllRank   = NewParAtt[PAR_TIME];
   real *Mass_AllRank   = NewParAtt[PAR_MASS];
   real *Pos_AllRank[3] = { NewParAtt[PAR_POSX], NewParAtt[PAR_POSY], NewParAtt[PAR_POSZ] };
   real *Vel_AllRank[3] = { NewParAtt[PAR_VELX], NewParAtt[PAR_VELY], NewParAtt[PAR_VELZ] };
   real *Type_AllRank   = NewParAtt[PAR_TYPE];
#  if ( PAR_NATT_USER == 1 )
   const long ParLabelStart = amr->Par->NPar_Active_AllRank;
   real *Label_AllRank  = NewParAtt[ParLabel_Idx];
#  endif

   if ( AddParWhenRestartByFile ) // add new disk via PAR_IC
   {
//    load data
      if ( MPI_Rank == 0 ){
         const char FileName[]     = "PAR_IC";
         const int  NParAtt        = 8;             // mass, pos*3, vel*3, type

//       check
         if ( !Aux_CheckFileExist(FileName) )
            Aux_Error( ERROR_INFO, "file \"%s\" does not exist !!\n", FileName );

         FILE *FileTemp = fopen( FileName, "rb" );

         fseek( FileTemp, 0, SEEK_END );

         const long ExpectSize = long(NParAtt)*NPar_AllRank*sizeof(real);
         const long FileSize   = ftell( FileTemp );
         if ( FileSize != ExpectSize )
            Aux_Error( ERROR_INFO, "size of the file <%s> = %ld != expect = %ld !!\n",
                       FileName, FileSize, ExpectSize );
         fclose( FileTemp );

         Aux_Message( stdout, "   Loading data ... " );

         real *ParData_AllRank = new real [ NPar_AllRank*NParAtt ];

//       note that fread() may fail for large files if sizeof(size_t) == 4 instead of 8
         FILE *File = fopen( FileName, "rb" );

         for (int v=0; v<NParAtt; v++)
         {
            fseek( File, v*NPar_AllRank*sizeof(real), SEEK_SET );
            fread( ParData_AllRank+v*NPar_AllRank, sizeof(real), NPar_AllRank, File );
         }

         fclose( File );
         Aux_Message( stdout, "done\n" );
         Aux_Message( stdout, "   Storing data into particle repository ... " );

         real *ParData1 = new real [NParAtt];

         for ( long p = 0; p < NPar_AllRank; p++)
         {
//          collect data for the target particle
//          [att][id]
            for (int v=0; v<NParAtt; v++)
               ParData1[v] = ParData_AllRank[ v*NPar_AllRank + p ];

            Time_AllRank[p] = Time[0];
//          mass
            Mass_AllRank[p] = ParData1[0];
//          label
            Type_AllRank[p] = ParData1[7]; // 1=CDM halo, 2=disk
#           if ( PAR_NATT_USER == 1 )      // add particle label, not very reliable due to floating point error
            Label_AllRank[p] = ParLabelStart + p;
#           endif

//          position
            Pos_AllRank[0][p] = ParData1[1];
            Pos_AllRank[1][p] = ParData1[2];
            Pos_AllRank[2][p] = ParData1[3];
//          velocity
            Vel_AllRank[0][p] = ParData1[4];
            Vel_AllRank[1][p] = ParData1[5];
            Vel_AllRank[2][p] = ParData1[6];

         } // for ( long p = 0; p < NPar_AllRank; p++)
         Aux_Message( stdout, "done\n" );

         // free memory
         delete [] ParData_AllRank;
         delete [] ParData1;
      } // if ( MPI_Rank == 0 )
   }// if ( AddParWhenRestartByFile )

   else // add a thin disk
   {

#     ifndef SUPPORT_GSL
      Aux_Error( ERROR_INFO, "SUPPORT_GSL must be enabled when AddParWhenRestart=1 and AddParWhenRestartByFile=0 !!\n" );
#     endif

      // read velocity dispersion table
      const bool RowMajor_No = false;     // load data into the column-major order
      const bool AllocMem_Yes = true;     // allocate memory for AGORA_VcProf
      const int  NCol        = 2;         // total number of columns to load
      const int  Col[NCol]   = {0, 1};    // target columns: (radius, density)

      DispTable_Nbin = Aux_LoadTable( DispTable, DispTableFile, NCol, Col, RowMajor_No, AllocMem_Yes );

      double *DispTable_r = DispTable + 0*DispTable_Nbin;
      double *DispTable_d = DispTable + 1*DispTable_Nbin;

      // convert to code unit
      for (int b=0; b<DispTable_Nbin; b++)
      {
         DispTable_r[b] = DispTable_r[b]*Const_kpc/UNIT_L;
         DispTable_d[b] = DispTable_d[b]*1e5/UNIT_V;
      }

      if ( MPI_Rank == 0 )
      {
         Aux_Message( stdout, "%s ...\n", __FUNCTION__ );

         const double ParM = Disk_Mass / NPar_AllRank;
         double Ran, RanR, RanV, R, Rold, f, f_, phi, RanVec[3];
         Aux_Message(stdout, " Particle Mass = %13.7e\n", ParM) ;

//       initialize the RNG
         RNG = new RandomNumber_t( 1 );
         RNG->SetSeed( 0, NewDisk_RSeed );


         for ( long p = 0; p < NPar_AllRank; p++)
         {
            Time_AllRank[p] = Time[0];
//          mass
            Mass_AllRank[p] = ParM;
//          label
            Type_AllRank[p] = 3;      // use 3 to represent thin disk particles
#           if ( PAR_NATT_USER == 1 ) // add particle label, not very reliable due to floating point error
            Label_AllRank[p] = ParLabelStart + p;
#           endif

//          position: statisfying surface density Sigma=Disk_Mass/(2*pi*Disk_R**2)*exp(-R/Disk_R)
            Ran  = RNG->GetValue( 0, 0.0, 1.0);
            R = 1.0;
            do
            {
               f = (1 + R) * exp(-R) + Ran - 1;
               f_ = -R * exp(-R);
               Rold = R;
               R = R - f / f_;
            }
            while(fabs(R - Rold) / R > 1e-7);
            RanR = Disk_R*R;
            phi = 2*M_PI*RNG->GetValue( 0, 0.0, 1.0 );
            RanVec[0] = RanR*cos(phi);
            RanVec[1] = RanR*sin(phi);
            RanVec[2] = 0;
            for (int d = 0; d < 3; d++) Pos_AllRank[d][p] = RanVec[d] + Cen[d];

         } // for ( long p = 0; p < NPar_AllRank; p++)

      } //if ( MPI_Rank == 0 )
   } //if ( !AddParWhenRestartByFile )

// add particles here
   Par_AddParticleAfterInit( NNewPar, NewParAtt );
// free memory
   for (int v=0; v<PAR_NATT_TOTAL; v++)   delete [] NewParAtt[v];

#  if ( defined PARTICLE  &&  defined LOAD_BALANCE )
   const double Par_Weight    = amr->LB->Par_Weight;
#  else
   const double Par_Weight    = 0.0;
#  endif
#  ifdef LOAD_BALANCE
   const UseLBFunc_t UseLB    = USELB_YES;
#  else
   const UseLBFunc_t UseLB    = USELB_NO;
#  endif

   for (int lv=0; lv<MAX_LEVEL; lv++)
   {
      if ( MPI_Rank == 0 )    Aux_Message( stdout, "   Refining level %d ...\n", lv );

      Flag_Real( lv, UseLB );

      Refine( lv, UseLB );

#     ifdef LOAD_BALANCE
      LB_Init_LoadBalance( true, true, Par_Weight, true, false, lv+1 );
#     endif

      if ( MPI_Rank == 0 )    Aux_Message( stdout, "   Refining level %d ... done\n", lv );
    } // for (int lv=OPT__UM_IC_LEVEL; lv<MAX_LEVEL; lv++)

//  compute garvitaional potential field to get the accleration for thin disk particles
    if ( !AddParWhenRestartByFile )
    {
//    initialize the k-space Green's function for the isolated BC.
      if ( OPT__BC_POT == BC_POT_ISOLATED )  Init_GreenFuncK();

//    evaluate the initial average density if it is not set yet (may already be set in Init_ByRestart)
      if ( AveDensity_Init <= 0.0 )    Poi_GetAverageDensity();

//    evaluate the gravitational potential
      if ( MPI_Rank == 0 )    Aux_Message( stdout, "%s ...\n", "Calculating gravitational potential" );

      for (int lv=0; lv<NLEVEL; lv++)
      {
         if ( MPI_Rank == 0 )    Aux_Message( stdout, "   Lv %2d ... ", lv );

         Buf_GetBufferData( lv, amr->FluSg[lv], NULL_INT, NULL_INT, DATA_GENERAL, _DENS, _NONE, Rho_ParaBuf, USELB_YES );

         Gra_AdvanceDt( lv, Time[lv], NULL_REAL, NULL_REAL, NULL_INT, amr->PotSg[lv], true, false, false, false, false );

         if ( lv > 0 )

         Buf_GetBufferData( lv, NULL_INT, NULL_INT, amr->PotSg[lv], POT_FOR_POISSON, _POTE, _NONE, Pot_ParaBuf, USELB_YES );

         if ( MPI_Rank == 0 )    Aux_Message( stdout, "done\n" );
      } // for (int lv=0; lv<NLEVEL; lv++)

      if ( MPI_Rank == 0 )    Aux_Message( stdout, "%s ... done\n", "Calculating gravitational potential" );
//    initialize particle acceleration
      if ( MPI_Rank == 0 )    Aux_Message( stdout, "%s ...\n", "Calculating particle acceleration" );

      const bool StoreAcc_Yes    = true;
      const bool UseStoredAcc_No = false;

      for (int lv=0; lv<NLEVEL; lv++)
      Par_UpdateParticle( lv, amr->PotSgTime[lv][ amr->PotSg[lv] ], NULL_REAL, PAR_UPSTEP_ACC_ONLY, StoreAcc_Yes, UseStoredAcc_No );

      if ( MPI_Rank == 0 )    Aux_Message( stdout, "%s ... done\n", "Calculating particle acceleration" );

      // compute particle velocities using acceleration fields
      Par_AfterAcceleration( amr->Par->NPar_AcPlusInac, amr->Par->NPar_Active_AllRank,
                             amr->Par->Mass, amr->Par->PosX, amr->Par->PosY, amr->Par->PosZ,
                             amr->Par->VelX, amr->Par->VelY, amr->Par->VelZ,
                             amr->Par->AccX, amr->Par->AccY, amr->Par->AccZ,
                             amr->Par->Time, amr->Par->Type, amr->Par->Attribute );

      // free memory
      root_fftw::fft_free(GreenFuncK); GreenFuncK      = NULL;

   }// if ( !AddParWhenRestartByFile )

} // FUNCTION : Init_NewDiskRestart

////-------------------------------------------------------------------------------------------------------
//// Function    :  Par_AfterAcceleration()
//// Description :  Compute particle veolcities using acceleration fields
////-------------------------------------------------------------------------------------------------------
void Par_AfterAcceleration( const long NPar_ThisRank, const long NPar_AllRank, real *ParMass,
                            real *ParPosX, real *ParPosY, real *ParPosZ,
                            real *ParVelX, real *ParVelY, real *ParVelZ,
                            real *ParAccX, real *ParAccY, real *ParAccZ,
                            real *ParTime, real *ParType,  real *AllAttribute[PAR_NATT_TOTAL])
{
#  ifdef SUPPORT_GSL

   if ( MPI_Rank == 0 )    Aux_Message( stdout, "%s ...\n", __FUNCTION__ );

   real *ParPos[3] = { ParPosX, ParPosY, ParPosZ };
   real *ParVel[3] = { ParVelX, ParVelY, ParVelZ };
   real *ParAcc[3] = { ParAccX, ParAccY, ParAccZ };

   real ParRadius[2];
   real NormParRadius[2];
   double V_acc, RanV[3], sigma;
   double ParR;

   // initialize the RNG
   gsl_rng *random_generator;
   random_generator = gsl_rng_alloc(gsl_rng_ranlxd1);
   gsl_rng_set(random_generator, NewDisk_RSeed + MPI_Rank);

   for (long p=0; p<NPar_ThisRank; p++)
   {
      if ( ParMass[p] < 0.0 )  continue;
      if ( ParType[p] == 3 ){
         ParRadius[0] = ParPos[0][p] - Cen[0];
         ParRadius[1] = ParPos[1][p] - Cen[1];
         ParR = sqrt( SQR(ParRadius[0]) + SQR(ParRadius[1]) );
         NormParRadius[0] = ParRadius[0]/ ParR;
         NormParRadius[1] = ParRadius[1]/ ParR;

         // compute radial acceleration
         V_acc = sqrt(fabs(ParRadius[0]*ParAcc[0][p]+ParRadius[1]*ParAcc[1][p]));

         // add velocity dispersion
         sigma = Get_Dispersion(ParR);
         RanV[0] = gsl_ran_gaussian(random_generator, sigma);
         RanV[1] = gsl_ran_gaussian(random_generator, sigma);
         RanV[2] = gsl_ran_gaussian(random_generator, sigma);

         ParVel[0][p] = - V_acc*NormParRadius[1]+RanV[0];
         ParVel[1][p] =   V_acc*NormParRadius[0]+RanV[1];
         ParVel[2][p] =   RanV[2];
      }

   }

   if ( MPI_Rank == 0 )    Aux_Message( stdout, "%s ... done\n", __FUNCTION__ );
#  endif //ifdef SUPPORT_GSL

} // FUNCTION : Par_AfterAcceleration

////-------------------------------------------------------------------------------------------------------
//// Function    :  Get_Dispersion
//// Description :  Get thin disk velocity dispersion from table
////-------------------------------------------------------------------------------------------------------
double Get_Dispersion(double r){
   double disp = 0.0;
   const double *DispTable_r = DispTable + 0*DispTable_Nbin;
   const double *DispTable_d = DispTable + 1*DispTable_Nbin;

   if (r < DispTable_r[0]) disp = DispTable_d[0];
   else if (r > DispTable_r[DispTable_Nbin-1]) disp = DispTable_d[DispTable_Nbin-1];
   else disp = Mis_InterpolateFromTable( DispTable_Nbin, DispTable_r, DispTable_d, r );

   return disp;
} // FUNCTION : Get_Dispersion




#endif //ifdef GRAVITY
#endif //ifdef PARTICLE
#endif // #if ( MODEL == ELBDM )



//-------------------------------------------------------------------------------------------------------
// Function    :  Init_TestProb_ELBDM_DiskHeating
// Description :  Test problem initializer
//
// Note        :  None
//
// Parameter   :  None
//
// Return      :  None
//-------------------------------------------------------------------------------------------------------
void Init_TestProb_ELBDM_DiskHeating()
{

   if ( MPI_Rank == 0 )    Aux_Message( stdout, "%s ...\n", __FUNCTION__ );


// validate the compilation flags and runtime parameters
   Validate();


#  if ( MODEL == ELBDM )
// set the problem-specific runtime parameters
   SetParameter();


// set the function pointers of various problem-specific routines
   Init_Function_User_Ptr         = SetGridIC;
   BC_User_Ptr                    = BC;
   End_User_Ptr                   = End_DiskHeating;
#  ifdef GRAVITY
   Init_ExtPot_Ptr                = Init_ExtPot_Soliton;
#  endif
#  ifdef PARTICLE
   Par_Init_ByFunction_Ptr        = Par_Init_ByFunction_DiskHeating;
#  if ( PAR_NATT_USER == 1 )
   Par_Init_Attribute_User_Ptr    = AddNewParticleAttribute;
#  endif
   Init_User_Ptr                  = Init_NewDiskRestart;
#  endif
#  endif // #if ( MODEL == ELBDM )



   if ( MPI_Rank == 0 )    Aux_Message( stdout, "%s ... done\n", __FUNCTION__ );

} // FUNCTION : Init_TestProb_ELBDM_DiskHeating