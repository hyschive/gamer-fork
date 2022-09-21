#include "GAMER.h"
#include "NuclearEoS.h"


extern double CCSN_LB_TimeFac;
extern double CCSN_CC_CentralDensFac;
extern double CCSN_CC_Red_DT;
extern double CCSN_CentralDens;



//-------------------------------------------------------------------------------------------------------
// Function    :  Mis_GetTimeStep_Lightbulb
// Description :  Estimate the evolution time-step constrained by the lightbulb source term
//
// Note        :  1. This function should be applied to both physical and comoving coordinates and always
//                   return the evolution time-step (dt) actually used in various solvers
//                   --> Physical coordinates : dt = physical time interval
//                       Comoving coordinates : dt = delta(scale_factor) / ( Hubble_parameter*scale_factor^3 )
//                   --> We convert dt back to the physical time interval, which equals "delta(scale_factor)"
//                       in the comoving coordinates, in Mis_GetTimeStep()
//                2. Invoked by Mis_GetTimeStep() using the function pointer "Mis_GetTimeStep_User_Ptr",
//                   which must be set by a test problem initializer
//                3. Enabled by the runtime option "OPT__DT_USER"
//
// Parameter   :  lv       : Target refinement level
//                dTime_dt : dTime/dt (== 1.0 if COMOVING is off)
//
// Return      :  dt
//-------------------------------------------------------------------------------------------------------
double Mis_GetTimeStep_Lightbulb( const int lv, const double dTime_dt )
{

   if ( !SrcTerms.Lightbulb )   return HUGE_NUMBER;


// allocate memory for per-thread arrays
#  ifdef OPENMP
   const int NT = OMP_NTHREAD;   // number of OpenMP threads
#  else
   const int NT = 1;
#  endif

   double  dt_LB         = HUGE_NUMBER;
   double  dt_LB_Inv     = -__DBL_MAX__;
   double *OMP_dt_LB_Inv = new double [NT];


#  pragma omp parallel
   {
#     ifdef OPENMP
      const int TID = omp_get_thread_num();
#     else
      const int TID = 0;
#     endif

//    initialize arrays
      OMP_dt_LB_Inv[TID] = -__DBL_MAX__;

      const double dh = amr->dh[lv];

#     pragma omp for schedule( runtime )
      for (int PID=0; PID<amr->NPatchComma[lv][1]; PID++)
      {
         for (int k=0; k<PS1; k++)  {
         for (int j=0; j<PS1; j++)  {
         for (int i=0; i<PS1; i++)  {

            const real Dens = amr->patch[ amr->FluSg[lv] ][lv][PID]->fluid[DENS][k][j][i];
            const real Momx = amr->patch[ amr->FluSg[lv] ][lv][PID]->fluid[MOMX][k][j][i];
            const real Momy = amr->patch[ amr->FluSg[lv] ][lv][PID]->fluid[MOMY][k][j][i];
            const real Momz = amr->patch[ amr->FluSg[lv] ][lv][PID]->fluid[MOMZ][k][j][i];
            const real Engy = amr->patch[ amr->FluSg[lv] ][lv][PID]->fluid[ENGY][k][j][i];

#           ifdef MHD
            const real Emag = MHD_GetCellCenteredBEnergyInPatch( lv, PID, i, j, k, amr->MagSg[lv] );
#           else
            const real Emag = NULL_REAL;
#           endif

            const real Eint_Code = Hydro_Con2Eint( Dens, Momx, Momy, Momz, Engy, true, MIN_EINT, Emag );


#           ifdef DEDT_LB
            real dEint_Code     = amr->patch[     amr->FluSg[lv] ][lv][PID]->fluid[DEDT_LB][k][j][i];
            real dEint_Code_Old = amr->patch[ 1 - amr->FluSg[lv] ][lv][PID]->fluid[DEDT_LB][k][j][i];
#           else
            real dEint_Code     = DEDT_UNINITIALIZED;
            real dEint_Code_Old = DEDT_UNINITIALIZED;
#           endif


//          call Src_Lightbulb() to compute the neutrino heating/cooling rate if not initialized yet
//
//          check DEDT_LB at both the Sg = 0 and 1
//          since the sandglass Sg = amr->FluSg[lv] may not equal to that used during initialization
            if ( dEint_Code     == DEDT_UNINITIALIZED ||
                 dEint_Code_Old == DEDT_UNINITIALIZED    )
            {
               const double z = amr->patch[0][lv][PID]->EdgeL[2] + (k+0.5)*dh;
               const double y = amr->patch[0][lv][PID]->EdgeL[1] + (j+0.5)*dh;
               const double x = amr->patch[0][lv][PID]->EdgeL[0] + (i+0.5)*dh;


//             get the input arrays
               real fluid[FLU_NIN_S];

               for (int v=0; v<FLU_NIN_S; v++)  fluid[v] = amr->patch[ amr->FluSg[lv] ][lv][PID]->fluid[v][k][j][i];

#              ifdef MHD
               real B[NCOMP_MAG] = { Emag, 0.0, 0.0 };
#              else
               real *B = NULL;
#              endif

               SrcTerms.Lightbulb_CPUPtr( fluid, B, &SrcTerms, 0.0, NULL_REAL, x, y, z, NULL_REAL, NULL_REAL,
                                          MIN_DENS, MIN_PRES, MIN_EINT, NULL,
                                          Src_Lightbulb_AuxArray_Flt, Src_Lightbulb_AuxArray_Int );

#              ifdef DEDT_LB
               dEint_Code = fluid[DEDT_LB];
#              endif
            } // if ( dEint_Code == DEDT_UNINITIALIZED )


            const double dt_LB_Inv_ThisCell = FABS( dEint_Code / Eint_Code );

//          compare the inverse of ratio to avoid zero division, and store the maximum value
            OMP_dt_LB_Inv[TID] = FMAX( OMP_dt_LB_Inv[TID], dt_LB_Inv_ThisCell );

         }}} // i,j,k
      } // for (int PID=0; PID<amr->NPatchComma[lv][1]; PID++)
   } // OpenMP parallel region


// find the maximum over all OpenMP threads
   for (int TID=0; TID<NT; TID++)   dt_LB_Inv = FMAX( dt_LB_Inv, OMP_dt_LB_Inv[TID] );

// free per-thread arrays
   delete [] OMP_dt_LB_Inv;


// find the maximum over all MPI processes
#  ifndef SERIAL
   MPI_Allreduce( MPI_IN_PLACE, &dt_LB_Inv, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
#  endif


   dt_LB = CCSN_LB_TimeFac / dt_LB_Inv;

   return dt_LB;

} // FUNCTION : Mis_GetTimeStep_Lightbulb



//-------------------------------------------------------------------------------------------------------
// Function    :  Mis_GetTimeStep_CoreCollapse
// Description :  Set the time-step to CCSN_CC_Red_DT during the core collape once the central
//                density reaches CCSN_CC_CentralDensFac (g/cm^3) to get a more accurate bounce time
//
// Note        :  1. This function should be applied to both physical and comoving coordinates and always
//                   return the evolution time-step (dt) actually used in various solvers
//                   --> Physical coordinates : dt = physical time interval
//                       Comoving coordinates : dt = delta(scale_factor) / ( Hubble_parameter*scale_factor^3 )
//                   --> We convert dt back to the physical time interval, which equals "delta(scale_factor)"
//                       in the comoving coordinates, in Mis_GetTimeStep()
//                2. Invoked by Mis_GetTimeStep() using the function pointer "Mis_GetTimeStep_User_Ptr",
//                   which must be set by a test problem initializer
//                3. Enabled by the runtime option "OPT__DT_USER"
//
// Parameter   :  lv       : Target refinement level
//                dTime_dt : dTime/dt (== 1.0 if COMOVING is off)
//
// Return      :  dt
//-------------------------------------------------------------------------------------------------------
double Mis_GetTimeStep_CoreCollapse( const int lv, const double dTime_dt )
{

   double dt = HUGE_NUMBER;

   const double CentralDens = CCSN_CentralDens / UNIT_D;

   if ( CentralDens > CCSN_CC_CentralDensFac / UNIT_D ) dt = CCSN_CC_Red_DT / UNIT_T;

   return dt;

} // FUNCTION : Mis_GetTimeStep_CoreCollapse
