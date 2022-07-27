#include "GAMER.h"
#include "NuclearEoS.h"


extern double CCSN_LB_TimeFac;
extern double CCSN_CentralDens;



//-------------------------------------------------------------------------------------------------------
// Function    :  Mis_GetTimeStep_Lightbulb
// Description :  estimate the evolution time-step constrained by the lightbulb source term
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

   const double BoxCenter[3] = { amr->BoxCenter[0], amr->BoxCenter[1], amr->BoxCenter[2] };
   const double Kelvin2MeV   = Const_kB_eV*1.0e-6;
   const double sEint2Code   = 1.0 / SQR( UNIT_V );

// allocate memory for per-thread arrays
#  ifdef OPENMP
   const int NT = OMP_NTHREAD;   // number of OpenMP threads
#  else
   const int NT = 1;
#  endif

   double   dt_LB         = HUGE_NUMBER;
   double   dt_LB_Inv     = -__DBL_MAX__;
   double **OMP_dt_LB_Inv = NULL;
   Aux_AllocateArray2D( OMP_dt_LB_Inv, NT, 1 );


#  pragma omp parallel
   {
#     ifdef OPENMP
      const int TID = omp_get_thread_num();
#     else
      const int TID = 0;
#     endif

//    initialize arrays
      OMP_dt_LB_Inv[TID][0] = -__DBL_MAX__;

      const double dh = amr->dh[lv];

#     pragma omp for schedule( runtime )
      for (int PID=0; PID<amr->NPatchComma[lv][1]; PID++)
      {
         for (int k=0; k<PS1; k++)  {  const double z0 = amr->patch[0][lv][PID]->EdgeL[2] + (k+0.5)*dh - BoxCenter[2];
         for (int j=0; j<PS1; j++)  {  const double y0 = amr->patch[0][lv][PID]->EdgeL[1] + (j+0.5)*dh - BoxCenter[1];
         for (int i=0; i<PS1; i++)  {  const double x0 = amr->patch[0][lv][PID]->EdgeL[0] + (i+0.5)*dh - BoxCenter[0];

            const double r2_CGS = SQR( UNIT_L ) * (  SQR( x0 ) + SQR( y0 ) + SQR( z0 )  );

            const real Dens   = amr->patch[ amr->FluSg[lv] ][lv][PID]->fluid[DENS][k][j][i];
            const real Momx   = amr->patch[ amr->FluSg[lv] ][lv][PID]->fluid[MOMX][k][j][i];
            const real Momy   = amr->patch[ amr->FluSg[lv] ][lv][PID]->fluid[MOMY][k][j][i];
            const real Momz   = amr->patch[ amr->FluSg[lv] ][lv][PID]->fluid[MOMZ][k][j][i];
            const real Engy   = amr->patch[ amr->FluSg[lv] ][lv][PID]->fluid[ENGY][k][j][i];
#           ifdef YE
            const real YeDens = amr->patch[ amr->FluSg[lv] ][lv][PID]->fluid[YE  ][k][j][i];
#           else
            const real YeDens = NULL_REAL;
#           endif

#           ifdef MHD
            const real Emag = MHD_GetCellCenteredBEnergy( amr->patch[ amr->MagSg[lv] ][lv][PID]->magnetic[MAGX],
                                                          amr->patch[ amr->MagSg[lv] ][lv][PID]->magnetic[MAGY],
                                                          amr->patch[ amr->MagSg[lv] ][lv][PID]->magnetic[MAGZ],
                                                          PS1, PS1, PS1, i, j, k );
#           else
            const real Emag = NULL_REAL;
#           endif

            const real Dens_Code = Dens;
            const real Eint_Code = Hydro_Con2Eint( Dens, Momx, Momy, Momz, Engy, true, MIN_EINT, Emag );
            const real Ye        = YeDens / Dens;


//          compute the neutrino heating rate
#           if ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )
            const int  NTarget = 2;
#           else
            const int  NTarget = 3;
#           endif
                  int  In_Int[NTarget+1];
                  real In_Flt[3], Out[NTarget+1];

            In_Flt[0] = Dens_Code;
            In_Flt[1] = Eint_Code;
            In_Flt[2] = Ye;

            In_Int[0] = NTarget;
            In_Int[1] = NUC_VAR_IDX_XN;
            In_Int[2] = NUC_VAR_IDX_XP;
#           if ( NUC_TABLE_MODE == NUC_TABLE_MODE_ENGY )
            In_Int[3] = NUC_VAR_IDX_EORT;
#           endif

            EoS_General_CPUPtr( NUC_MODE_ENGY, Out, In_Flt, In_Int, EoS_AuxArray_Flt, EoS_AuxArray_Int, h_EoS_Table );

            const real Xn       = Out[0];               // neutron mass fraction
            const real Xp       = Out[1];               // proton  mass fraction
            const real Temp_MeV = Out[2] * Kelvin2MeV;  // temperature in MeV

            const double rate_heating = 1.544e20 * ( SrcTerms.Lightbulb_Lnue / 1.0e52 ) * ( 1.0e14 / r2_CGS )
                                      * SQR( 0.25 * SrcTerms.Lightbulb_Tnue );
            const double rate_cooling = 1.399e20 * CUBE(  SQR( 0.5 * Temp_MeV )  );

            const double tau      = 1.0e-11 * Dens_Code * UNIT_D;
            const double rate_CGS = ( rate_heating - rate_cooling ) * ( Xn + Xp ) * exp( -tau );

            const double dEint_Code  = rate_CGS * UNIT_T * sEint2Code * Dens_Code;  // dEint per UNIT_T
            const double _Eint_Ratio = fabs( dEint_Code / Eint_Code );

//          compare the inverse of ratio to avoid zero division, and store the maximum value
            if ( _Eint_Ratio > OMP_dt_LB_Inv[TID][0] )   OMP_dt_LB_Inv[TID][0] = _Eint_Ratio;

         }}} // i,j,k
      } // for (int PID=0; PID<amr->NPatchComma[lv][1]; PID++)
   } // OpenMP parallel region


// find the maximum over all OpenMP threads
   for (int TID=0; TID<NT; TID++)
   {
      if ( OMP_dt_LB_Inv[TID][0] > dt_LB_Inv )   dt_LB_Inv = OMP_dt_LB_Inv[TID][0];
   }

// free per-thread arrays
   Aux_DeallocateArray2D( OMP_dt_LB_Inv );


// find the maximum over all MPI processes
#  ifndef SERIAL
   MPI_Allreduce( MPI_IN_PLACE, &dt_LB_Inv, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
#  endif


   dt_LB = CCSN_LB_TimeFac / dt_LB_Inv;

   return dt_LB;

} // FUNCTION : Mis_GetTimeStep_Lightbulb



//-------------------------------------------------------------------------------------------------------
// Function    :  Mis_GetTimeStep_Deleptonization
// Description :  estimate the evolution time-step constrained by the lightbulb source term
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
double Mis_GetTimeStep_Deleptonization( const int lv, const double dTime_dt )
{

   real dt = HUGE_NUMBER;

   if (  ( lv == 0 )  &&  ( CCSN_CentralDens > 1.0e13 )  )
      dt = 0.1 * DT__MAX;

   return dt;

} // FUNCTION : Mis_GetTimeStep_Deleptonization
