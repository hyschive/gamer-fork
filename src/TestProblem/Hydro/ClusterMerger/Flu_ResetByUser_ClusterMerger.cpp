#include "GAMER.h"

#if ( MODEL == HYDRO  &&  defined GRAVITY )


extern int    Merger_Coll_NumHalos;
extern double eta, eps_f, eps_m, R_acc, R_dep; // parameters of jet feedback

extern double CM_Bondi_SinkMass[3];
extern double CM_Bondi_SinkMomX[3];
extern double CM_Bondi_SinkMomY[3];
extern double CM_Bondi_SinkMomZ[3];
extern double CM_Bondi_SinkMomXAbs[3];
extern double CM_Bondi_SinkMomYAbs[3];
extern double CM_Bondi_SinkMomZAbs[3];
extern double CM_Bondi_SinkE[3];
extern double CM_Bondi_SinkEk[3];
extern double CM_Bondi_SinkEt[3];
extern int    CM_Bondi_SinkNCell[3];

extern double Bondi_MassBH1;
extern double Bondi_MassBH2;
extern double Bondi_MassBH3;
extern double Mdot_BH1; // the accretion rate
extern double Mdot_BH2;
extern double Mdot_BH3;
extern double Jet_HalfHeight1;
extern double Jet_HalfHeight2;
extern double Jet_HalfHeight3;
extern double Jet_Radius1;
extern double Jet_Radius2;
extern double Jet_Radius3;
extern double Jet_Vec[3][3]; // jet direction  
extern double Mdot[3]; // the feedback injection rate
extern double Pdot[3];
extern double Edot[3];
extern double GasVel[3][3];  // gas velocity
extern double SoundSpeed[3]; 
extern double GasDens[3];
extern double RelativeVel[3]; // the relative velocity between BH and gas
extern double ClusterCen[3][3];
extern double BH_Pos[3][3]; // BH position (for updating ClusterCen)
extern double BH_Vel[3][3]; // BH velocity

double Jet_WaveK[3];  // jet wavenumber used in the sin() function to have smooth bidirectional jets
double Jet_HalfHeight[3];
double Jet_Radius[3];
double V_cyl[3]; // the volume of jet source
double M_inj[3], P_inj[3], E_inj[3]; // the injected density
double normalize_const[3];   // The exact normalization constant
       
// the variables that need to be recorded
double E_inj_exp[3] = { 0.0, 0.0, 0.0 };   // the expected amount of injected energy
double dt_base; 
double E_power_inj[3];   // the injection power

extern void GetClusterCenter( int lv, bool AdjustPos, bool AdjustVel, double Cen_old[][3], double Cen_new[][3], double Cen_Vel[][3] );

static bool FirstTime = true;
/*
double RandomNumber(RandomNumber_t *RNG, const double Min, const double Max )
{                   
// thread-private variables
#  ifdef OPENMP  
   const int TID = omp_get_thread_num();
#  else             
   const int TID = 0;
#  endif            
   return RNG->GetValue( TID, Min, Max );                                               
}                                               
static RandomNumber_t *RNG = NULL;
*/
extern int     JetDirection_NBin;     // number of bins of the jet direction table 
extern double *Time_table;            // the time table of jet direction 
extern double *Theta_table[3];        // the theta table of jet direction for 3 clusters
extern double *Phi_table[3];          // the phi table of jet direction for 3 clusters

extern bool   AdjustBHPos;
extern bool   AdjustBHVel;
extern double AdjustPeriod;
int AdjustCount = 0;   // count the number of adjustments
int merge_index = 0;   // record BH 1 merge BH 2 / BH 2 merge BH 1

//-------------------------------------------------------------------------------------------------------
// Function    :  Flu_ResetByUser_Func_ClusterMerger
// Description :  Function to reset the fluid field in the Bondi accretion problem
//
// Note        :  1. Invoked by Flu_ResetByUser_API_ClusterMerger() and Hydro_Init_ByFunction_AssignData() using the
//                   function pointer "Flu_ResetByUser_Func_Ptr"
//                   --> This function pointer is reset by Init_TestProb_Hydro_ClusterMerger()
//                   --> Hydro_Init_ByFunction_AssignData(): constructing initial condition
//                       Flu_ResetByUser_API_ClusterMerger()       : after each update
//                2. Input fluid[] stores the original values
//                3. Even when DUAL_ENERGY is adopted, one does NOT need to set the dual-energy variable here
//                   --> It will be set automatically
//                4. Enabled by the runtime option "OPT__RESET_FLUID"
//
// Parameter   :  fluid    : Fluid array storing both the input (origial) and reset values
//                           --> Including both active and passive variables
//                x/y/z    : Target physical coordinates
//                Time     : Target physical time
//                dt       : Time interval to advance solution
//                lv       : Target refinement level
//                AuxArray : Auxiliary array
//
// Return      :  true  : This cell has been reset
//                false : This cell has not been reset
//-------------------------------------------------------------------------------------------------------
int Flu_ResetByUser_Func_ClusterMerger( real fluid[], const double Emag, const double x, const double y, const double z, 
                                        const double Time, const double dt, const int lv, double AuxArray[] )
{
   const double Pos[3]  = { x, y, z };

// (1) SMBH Accretion

   double dr2[3][3], r2[3];
   const double V_dep = 4.0/3.0*M_PI*pow(R_dep,3.0); // the region to remove gas
// the density need to be removed
   double D_dep[3] = { Mdot_BH1*dt/V_dep, Mdot_BH2*dt/V_dep, Mdot_BH3*dt/V_dep };

   int iftrue = 0; // mark whether this cell is reset or not [0/1]

   for (int c=0; c<Merger_Coll_NumHalos; c++) {
      for (int d=0; d<3; d++)
      {
         dr2[c][d] = SQR( Pos[d] - ClusterCen[c][d] );
      }

//    update the conserved variables
      r2[c] = dr2[c][0] + dr2[c][1] + dr2[c][2];
      if ( r2[c] <= SQR(R_dep) )
      {
//         fluid[DENS] -= D_dep[c];                                                           
//         fluid[MOMX] -= D_dep[c]*GasVel[c][0];                                              
//         fluid[MOMY] -= D_dep[c]*GasVel[c][1];                                              
//         fluid[MOMZ] -= D_dep[c]*GasVel[c][2];                                              
//         fluid[ENGY] -= 0.5*D_dep[c]*( SQR(GasVel[c][0]) + SQR(GasVel[c][1]) + SQR(GasVel[c][2]) );

         iftrue = 1;
      }
   }
//   if ( iftrue==1 ){
//      return true;
//   }
//   else 
//      return false;


// (2) Jet Feedback (Recipe 3)

   double Jet_dr[3], Jet_dh[3], S, Area;
   double Dis_c2m, Dis_c2v, Dis_v2m, Vec_c2m[3][3], Vec_v2m[3];
   double TempVec[3]; 
   real   EngySin;
   int    status = 0;   // 0: not within any jets; 1: within jet 1; 2: within jet 2; 3: within both jet 1 and 2. 

   for (int c=0; c<Merger_Coll_NumHalos; c++)
   {
//    distance: jet center to mesh
      for (int d=0; d<3; d++)    Vec_c2m[c][d] = Pos[d] - ClusterCen[c][d];
      Dis_c2m = sqrt( SQR(Vec_c2m[c][0]) + SQR(Vec_c2m[c][1]) + SQR(Vec_c2m[c][2]) );

//    vectors for calculating the distance between cells and the jet sources
      for (int d=0; d<3; d++)    TempVec[d] = ClusterCen[c][d] + Jet_Vec[c][d];

//    distance: temporary vector to mesh
      for (int d=0; d<3; d++)    Vec_v2m[d] = Pos[d] - TempVec[d];
      Dis_v2m = sqrt( SQR(Vec_v2m[0]) + SQR(Vec_v2m[1]) + SQR(Vec_v2m[2]) );

//    distance: jet center to temporary vector
      Dis_c2v = sqrt( SQR(Jet_Vec[c][0]) + SQR(Jet_Vec[c][1]) + SQR(Jet_Vec[c][2]) );

//    check whether or not the target cell is within the jet source
      S      = 0.5*( Dis_c2m + Dis_v2m + Dis_c2v );
      Area   = sqrt( S*(S-Dis_c2m)*(S-Dis_v2m)*(S-Dis_c2v) );
      Jet_dr[c] = 2.0*Area/Dis_c2v;
      Jet_dh[c] = sqrt( Dis_c2m*Dis_c2m - Jet_dr[c]*Jet_dr[c] );

      if ( Jet_dh[c] <= Jet_HalfHeight[c]  &&  Jet_dr[c] <= Jet_Radius[c] )   status += c+1;
   }

   if ( status == 3 ){   // Consider only the jet with larger injection rate when overlap
      if ( Edot[0] >= Edot[1] )   status = 1;
      else   status = 2;
   }

   if ( status != 0 ) 
   {
//    Record the old momentum
      double MOMX_old = fluid[MOMX];
      double MOMY_old = fluid[MOMY];
      double MOMZ_old = fluid[MOMZ];

      fluid[DENS] += M_inj[status-1];      

//    Transfer into BH frame
      fluid[MOMX] -= BH_Vel[status-1][0]*fluid[DENS];
      fluid[MOMY] -= BH_Vel[status-1][1]*fluid[DENS];
      fluid[MOMZ] -= BH_Vel[status-1][2]*fluid[DENS]; 

//    use a sine function to make the velocity smooth within the jet from +Jet_Vec to -Jet_Vec
      EngySin = E_inj[status-1]*normalize_const[status-1]*sin( Jet_WaveK[status-1]*Jet_dh[status-1] );
      double P_SQR = SQR(fluid[MOMX])+SQR(fluid[MOMY])+SQR(fluid[MOMZ]);
      double tmp_dens = fluid[DENS]-M_inj[status-1];
//    the new momentum is calculated from the old density, new density, old momentum and injected energy
      double P_new = sqrt(2*fluid[DENS]*(EngySin+0.5*P_SQR/tmp_dens));
      P_new *= SIGN( Vec_c2m[status-1][0]*Jet_Vec[status-1][0] + Vec_c2m[status-1][1]*Jet_Vec[status-1][1] + Vec_c2m[status-1][2]*Jet_Vec[status-1][2] );
      fluid[MOMX] = P_new*Jet_Vec[status-1][0];
      fluid[MOMY] = P_new*Jet_Vec[status-1][1];
      fluid[MOMZ] = P_new*Jet_Vec[status-1][2];

//    Transfer back into the rest frame  
      fluid[MOMX] += BH_Vel[status-1][0]*fluid[DENS];
      fluid[MOMY] += BH_Vel[status-1][1]*fluid[DENS];
      fluid[MOMZ] += BH_Vel[status-1][2]*fluid[DENS]; 

      fluid[ENGY] += 0.5*((SQR(fluid[MOMX])+SQR(fluid[MOMY])+SQR(fluid[MOMZ]))/fluid[DENS]-(SQR(MOMX_old)+SQR(MOMY_old)+SQR(MOMZ_old))/tmp_dens);

   } // if ( status != 0 )  

   return status;


} // FUNCTION : Flu_ResetByUser_Func_ClusterMerger



//-------------------------------------------------------------------------------------------------------
// Function    :  Flu_ResetByUser_API_ClusterMerger
// Description :  API for resetting the fluid array in the Bondi accretion problem
//
// Note        :  1. Enabled by the runtime option "OPT__RESET_FLUID"
//                2. Invoked using the function pointer "Flu_ResetByUser_API_Ptr"
//                   --> This function pointer is reset by Init_TestProb_Hydro_ClusterMerger()
//                3. Currently does not work with "OPT__OVERLAP_MPI"
//                4. Invoke Flu_ResetByUser_Func_ClusterMerger() directly
//
// Parameter   :  lv    : Target refinement level
//                FluSg : Target fluid sandglass
//                TimeNew : Current physical time (system has been updated from TimeOld to TimeNew in EvolveLevel())
//                dt      : Time interval to advance solution (can be different from TimeNew-TimeOld in COMOVING)
//-------------------------------------------------------------------------------------------------------
void Flu_ResetByUser_API_ClusterMerger( const int lv, const int FluSg, const double TimeNew, const double dt )
{
/*
// TEMP!!! For restart.
   if (BH_Pos[0][0] == 0.0){                        
      BH_Pos[0][0] = 7.4718022e+00;
      BH_Pos[0][1] = 7.5023150e+00;
      BH_Pos[0][2] = 7.5038495e+00;
      BH_Pos[1][0] = 7.5331659e+00;
      BH_Pos[1][1] = 7.5056639e+00;
      BH_Pos[1][2] = 7.5041471e+00;
      ClusterCen[0][0] = 7.4718022e+00;
      ClusterCen[0][1] = 7.5023150e+00;
      ClusterCen[0][2] = 7.5038495e+00;
      ClusterCen[1][0] = 7.5331659e+00;
      ClusterCen[1][1] = 7.5056639e+00;
      ClusterCen[1][2] = 7.5041471e+00;
      BH_Vel[0][0] = 4.4017131e+02/UNIT_V*(Const_km/Const_s); 
      BH_Vel[0][1] = 2.1990993e+01/UNIT_V*(Const_km/Const_s);    
      BH_Vel[0][2] = -7.8597931e+00/UNIT_V*(Const_km/Const_s);
      BH_Vel[1][0] = -4.8891710e+02/UNIT_V*(Const_km/Const_s);
      BH_Vel[1][1] = -2.5195094e+01/UNIT_V*(Const_km/Const_s);         
      BH_Vel[1][2] = 5.7178019e+01/UNIT_V*(Const_km/Const_s);
      Bondi_MassBH1 = 7.1479449e+10/1e14;
      Bondi_MassBH2 = 7.1476139e+10/1e14;
      AdjustCount = 389; 
   } 
*/

   double RelativeBHPos[3] = { BH_Pos[0][0]-BH_Pos[1][0], BH_Pos[0][1]-BH_Pos[1][1], BH_Pos[0][2]-BH_Pos[1][2] };
   double RelativeBHVel[3] = { BH_Vel[0][0]-BH_Vel[1][0], BH_Vel[0][1]-BH_Vel[1][1], BH_Vel[0][2]-BH_Vel[1][2] };
   double AbsRelPos = sqrt( SQR(RelativeBHPos[0])+SQR(RelativeBHPos[1])+SQR(RelativeBHPos[2]) );
   double AbsRelVel = sqrt( SQR(RelativeBHVel[0])+SQR(RelativeBHVel[1])+SQR(RelativeBHVel[2]) );
   double escape_vel[2];
   double soften = amr->dh[MAX_LEVEL];
   if ( AbsRelPos > soften ){
      escape_vel[0] = sqrt(2*NEWTON_G*Bondi_MassBH2/AbsRelPos);
      escape_vel[1] = sqrt(2*NEWTON_G*Bondi_MassBH1/AbsRelPos);
   }
   else{   
      escape_vel[0] = sqrt(2*NEWTON_G*Bondi_MassBH2/soften);
      escape_vel[1] = sqrt(2*NEWTON_G*Bondi_MassBH1/soften);
   }

// Merge the two BHs if they are located within R_acc, and the relative velocity is small enough
   if ( Merger_Coll_NumHalos == 2 ){
      if ( AbsRelPos < R_acc  &&  ( AbsRelVel < 3*escape_vel[1]  ||  AbsRelVel < 3*escape_vel[0] ) ){
         Merger_Coll_NumHalos -= 1;
         if ( Bondi_MassBH1 >= Bondi_MassBH2 )   merge_index = 1;   // record BH 1 merge BH 2 / BH 2 merge BH 1
         else   merge_index = 2;
//       Relabel the BH particle being merged back to dark matter
         Bondi_MassBH1 += Bondi_MassBH2;
         Bondi_MassBH2 = 0.0;
         for (long p=0; p<amr->Par->NPar_AcPlusInac; p++) {                                  
            if ( amr->Par->Mass[p] >= (real)0.0  &&  amr->Par->Type[p] == real(PTYPE_CEN+1) ){       
               amr->Par->Type[p] = PTYPE_DARK_MATTER;      
            }
         }     
         Aux_Message( stdout, "Merge! In rank %d, TimeNew = %14.8e; merge_index = %d, BHPos1 = %14.8e, %14.8e, %14.8e; BHPos2 = %14.8e, %14.8e, %14.8e; BHVel1 = %14.8e, %14.8e, %14.8e; BHVel2 = %14.8e, %14.8e, %14.8e; AbsRelPos = %14.8e, AbsRelVel = %14.8e, escape_vel[0] = %14.8e, escape_vel[1] = %14.8e.\n", MPI_Rank, TimeNew, merge_index, BH_Pos[0][0], BH_Pos[0][1], BH_Pos[0][2], BH_Pos[1][0], BH_Pos[1][1], BH_Pos[1][2], BH_Vel[0][0], BH_Vel[0][1], BH_Vel[0][2], BH_Vel[1][0], BH_Vel[1][1], BH_Vel[1][2], AbsRelPos, AbsRelVel, escape_vel[0], escape_vel[1]);
      }  
   }

   const bool CurrentMaxLv = (  NPatchTotal[lv] > 0  &&  ( lv == MAX_LEVEL || NPatchTotal[lv+1] == 0 )  );

   double Mdot_BH[3] = { Mdot_BH1, Mdot_BH2, Mdot_BH3 };
   double Bondi_MassBH[3] = { Bondi_MassBH1, Bondi_MassBH2, Bondi_MassBH3 }; 

// Get the BH position and velocity and adjust them if needed
   bool AdjustPosNow = false;
   bool AdjustVelNow = false;
   if ( CurrentMaxLv  &&  AdjustCount < int(TimeNew/AdjustPeriod)){   // only adjust the BHs on the current maximum level 
      if ( AdjustBHPos == true )   AdjustPosNow = true;
      if ( AdjustBHVel == true )   AdjustVelNow = true;
      AdjustCount += 1;
   }
   GetClusterCenter( lv, AdjustPosNow, AdjustVelNow, BH_Pos, ClusterCen, BH_Vel );

   Jet_HalfHeight[0] = Jet_HalfHeight1;
   Jet_HalfHeight[1] = Jet_HalfHeight2;
   Jet_HalfHeight[2] = Jet_HalfHeight3;
   Jet_Radius[0] = Jet_Radius1;
   Jet_Radius[1] = Jet_Radius2;
   Jet_Radius[2] = Jet_Radius3;

// Set the jet direction vector
//   double Time_period = Time_table[JetDirection_NBin-1];
//   double Time_interpolate = fmod(TimeNew, Time_period);
   for (int c=0; c<Merger_Coll_NumHalos; c++) {
//      double theta = Mis_InterpolateFromTable( JetDirection_NBin, Time_table, Theta_table[c], Time_interpolate );
//      double phi   = Mis_InterpolateFromTable( JetDirection_NBin, Time_table, Phi_table[c], Time_interpolate );
//      double theta = 10.0*M_PI/180.0;
//      double phi = 2*M_PI*Time_interpolate/Time_period;
      Jet_Vec[c][0] = 1.0;   //cos(theta);   //sin(theta)*cos(phi);
      Jet_Vec[c][1] = 0.0;   //sin(theta)*cos(phi);   //sin(theta)*sin(phi);
      Jet_Vec[c][2] = 0.0;   //sin(theta)*sin(phi);   //cos(theta);
   }

   const double dh       = amr->dh[lv];
   const real   dv       = CUBE(dh);
#  if ( MODEL == HYDRO  ||  MODEL == MHD )
   const real   Gamma_m1 = GAMMA - (real)1.0;
   const real  _Gamma_m1 = (real)1.0 / Gamma_m1;
#  endif   

   int    Reset;
   real   fluid[NCOMP_TOTAL], fluid_bk[NCOMP_TOTAL], fluid_Bondi[NCOMP_TOTAL];
   double x, y, z, x0, y0, z0, x2, y2, z2, x02, y02, z02;
   double V_cyl_exacthalf[3] = { 0.0, 0.0, 0.0 };   // The exact volume of jet cylinder 
   double normalize[3]   = { 0.0, 0.0, 0.0 };   // For computing the correct normalization constant
   double V_cyl_exacthalf_sum[3], normalize_sum[3];   // for MPI_Allreduce()

// reset to 0 since we only want to record the number of void cells **for one sub-step**
   for (int c=0; c<Merger_Coll_NumHalos; c++) CM_Bondi_SinkNCell[c] = 0;

   for (int c=0; c<Merger_Coll_NumHalos; c++) { 

      Jet_WaveK[c] = 0.5*M_PI/Jet_HalfHeight[c];

//    reset gas velocity to zero
      for (int d=0; d<3; d++)  GasVel[c][d] = 0.0;

      const bool CheckMinPres_No = false;

      double rho = 0.0;  // the average density inside the accretion radius
      double Pres, tmp_Cs; // use for calculation of sound speed
      double Cs = 0.0;  // the average sound speed inside the accretion radius
      double gas_vel[3] = { 0.0, 0.0, 0.0 }; // average gas velocity
      double v = 0.0;  // the relative velocity between BH and gas
      int num = 0;  // the number of cells inside the accretion radius
      double ang_mom[3] = { 0.0, 0.0, 0.0 }; // total angular momentum inside the accretion radius


      for (int PID=0; PID<amr->NPatchComma[lv][1]; PID++)
      {
         x02 = amr->patch[0][lv][PID]->EdgeL[0] + 0.5*dh;
         y02 = amr->patch[0][lv][PID]->EdgeL[1] + 0.5*dh;
         z02 = amr->patch[0][lv][PID]->EdgeL[2] + 0.5*dh;
   
         for (int k=0; k<PS1; k++)  {  z2 = z02 + k*dh; 
         for (int j=0; j<PS1; j++)  {  y2 = y02 + j*dh; 
         for (int i=0; i<PS1; i++)  {  x2 = x02 + i*dh;
   
            for (int v=0; v<NCOMP_TOTAL; v++){
               fluid_Bondi[v] = amr->patch[FluSg][lv][PID]->fluid[v][k][j][i];
            }

#           ifdef MHD
            const real Emag = MHD_GetCellCenteredBEnergyInPatch( lv, PID, i, j, k, amr->MagSg[lv] );
#           else
            const real Emag = (real)0.0;
#           endif

//          calculate the average density, sound speed and gas velocity inside accretion radius
            if (SQR(x2-ClusterCen[c][0])+SQR(y2-ClusterCen[c][1])+SQR(z2-ClusterCen[c][2]) <= SQR(R_acc)){
               rho += fluid_Bondi[0]*dv;
               Pres = (real) Hydro_Con2Pres( fluid_Bondi[0], fluid_Bondi[1], fluid_Bondi[2], fluid_Bondi[3], fluid_Bondi[4], 
                                             fluid_Bondi+NCOMP_FLUID, true, MIN_PRES, Emag, EoS_DensEint2Pres_CPUPtr, 
                                             EoS_AuxArray_Flt, EoS_AuxArray_Int, h_EoS_Table, NULL );
               tmp_Cs = sqrt( EoS_DensPres2CSqr_CPUPtr( fluid_Bondi[0], Pres, NULL, EoS_AuxArray_Flt, EoS_AuxArray_Int,
                              h_EoS_Table ) );
               Cs += tmp_Cs;
               for (int d=0; d<3; d++)  gas_vel[d] += fluid_Bondi[d+1]*dv;
               num += 1;            
            }
            if (SQR(x2-ClusterCen[c][0])+SQR(y2-ClusterCen[c][1])+SQR(z2-ClusterCen[c][2]) <= SQR(R_acc)){
               double dr[3] = {x2-ClusterCen[c][0], y2-ClusterCen[c][1], z2-ClusterCen[c][2]};
               ang_mom[0] += dv*(dr[1]*fluid_Bondi[3]-dr[2]*fluid_Bondi[2]);
               ang_mom[1] += dv*(dr[2]*fluid_Bondi[1]-dr[0]*fluid_Bondi[3]);
               ang_mom[2] += dv*(dr[0]*fluid_Bondi[2]-dr[1]*fluid_Bondi[1]);
            }

//          Calculate the exact volume of jet cylinder and normalization
            if ( CurrentMaxLv ){
               double Jet_dr_2, Jet_dh_2, S_2, Area_2;
               double Dis_c2m_2, Dis_c2v_2, Dis_v2m_2, Vec_c2m_2[3], Vec_v2m_2[3];
               double TempVec_2[3]; 
               double Pos_2[3] = {x2, y2, z2};         
   
               for (int d=0; d<3; d++)    Vec_c2m_2[d] = Pos_2[d] - ClusterCen[c][d];
               Dis_c2m_2 = sqrt( SQR(Vec_c2m_2[0]) + SQR(Vec_c2m_2[1]) + SQR(Vec_c2m_2[2]) );
               for (int d=0; d<3; d++)    TempVec_2[d] = ClusterCen[c][d] + Jet_Vec[c][d];
               for (int d=0; d<3; d++)    Vec_v2m_2[d] = Pos_2[d] - TempVec_2[d];
               Dis_v2m_2 = sqrt( SQR(Vec_v2m_2[0]) + SQR(Vec_v2m_2[1]) + SQR(Vec_v2m_2[2]) );
               Dis_c2v_2 = sqrt( SQR(Jet_Vec[c][0]) + SQR(Jet_Vec[c][1]) + SQR(Jet_Vec[c][2]) );
         
               S_2      = 0.5*( Dis_c2m_2 + Dis_v2m_2 + Dis_c2v_2 );
               Area_2   = sqrt( S_2*(S_2-Dis_c2m_2)*(S_2-Dis_v2m_2)*(S_2-Dis_c2v_2) );
               Jet_dr_2 = 2.0*Area_2/Dis_c2v_2;
               Jet_dh_2 = sqrt( Dis_c2m_2*Dis_c2m_2 - Jet_dr_2*Jet_dr_2 );

               int sign = 1*SIGN( Vec_c2m_2[0]*Jet_Vec[c][0] + Vec_c2m_2[1]*Jet_Vec[c][1] + Vec_c2m_2[2]*Jet_Vec[c][2] ); 
               if ( sign > 0  &&  Jet_dh_2 <= Jet_HalfHeight[c]  &&  Jet_dr_2 <= Jet_Radius[c] )
               {
                  V_cyl_exacthalf[c] += dv;      
                  normalize[c] += sin( Jet_WaveK[c]*Jet_dh_2 )*dv;
               }
            }
         }}}
      } // for (int PID=0; PID<amr->NPatchComma[lv][1]; PID++)

      double rho_sum, Cs_sum, gas_vel_sum[3], ang_mom_sum[3];
      int num_sum;
      MPI_Allreduce( &num,     &num_sum,     1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
      MPI_Allreduce( &rho,     &rho_sum,     1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
      MPI_Allreduce( &Cs,      &Cs_sum,      1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
      MPI_Allreduce( gas_vel,  gas_vel_sum,  3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
      MPI_Allreduce( ang_mom,  ang_mom_sum,  3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
      MPI_Allreduce( &V_cyl_exacthalf[c], &V_cyl_exacthalf_sum[c], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );      
      MPI_Allreduce( &normalize[c],       &normalize_sum[c],       1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

      if (num_sum == 0){
         Mdot_BH[c] = 0.0;
         GasDens[c] = 0.0;
         SoundSpeed[c] = 0.0;
         RelativeVel[c] = 0.0;
      }
      else{
         for (int d=0; d<3; d++)  gas_vel_sum[d] /= rho_sum;
         rho_sum /= (4.0/3.0*M_PI*pow(R_acc,3));
         Cs_sum /= (double)num_sum;
         for (int d=0; d<3; d++)  BH_Pos[c][d] = ClusterCen[c][d];
//         for (int d=0; d<3; d++)  BH_Vel[c][d] = gas_vel_sum[d];
         for (int d=0; d<3; d++)  v += SQR(BH_Vel[c][d]-gas_vel_sum[d]);

//       calculate the accretion rate
         Mdot_BH[c] = 4.0*M_PI*SQR(NEWTON_G)*SQR(Bondi_MassBH[c])*rho_sum/pow(Cs_sum*Cs_sum+v,1.5);
         GasDens[c] = rho_sum;
         SoundSpeed[c] = Cs_sum;
         for (int d=0; d<3; d++)  GasVel[c][d] = gas_vel_sum[d];
         RelativeVel[c] = sqrt(v);

//       decide the jet direction by angular momentum
         double ang_mom_norm = sqrt(SQR(ang_mom_sum[0])+SQR(ang_mom_sum[1])+SQR(ang_mom_sum[2]));
         for (int d=0; d<3; d++)  Jet_Vec[c][d] = ang_mom_sum[d]/ang_mom_norm;
      }

      if (V_cyl_exacthalf_sum[c] != 0)   normalize_const[c] = V_cyl_exacthalf_sum[c]/normalize_sum[c];
      else                               normalize_const[c] = 0.5*M_PI;

   } // for (int c=0; c<Merger_Coll_NumHalos; c++)

   Mdot_BH1 = Mdot_BH[0];                            
   Mdot_BH2 = Mdot_BH[1];                            
   Mdot_BH3 = Mdot_BH[2];

// update BH mass    
   for (int c=0; c<Merger_Coll_NumHalos; c++) {
      if ( CurrentMaxLv ){
         Bondi_MassBH[c] += Mdot_BH[c]*dt;
      }
   }
   Bondi_MassBH1 = Bondi_MassBH[0];
   Bondi_MassBH2 = Bondi_MassBH[1];
   Bondi_MassBH3 = Bondi_MassBH[2];

// calculate the injection rate
   for (int c=0; c<Merger_Coll_NumHalos; c++){
      Mdot[c] = eta*Mdot_BH[c];
      Pdot[c] = sqrt(2*eta*eps_f*(1.0-eps_m))*Mdot_BH[c]*(Const_c/UNIT_V);
      Edot[c] = eps_f*Mdot_BH[c]*SQR(Const_c/UNIT_V);
      V_cyl[c] = M_PI*SQR(Jet_Radius[c])*2*Jet_HalfHeight[c];

//    calculate the density that need to be injected
      if ( CurrentMaxLv && V_cyl_exacthalf_sum[c] != 0){
         M_inj[c] = Mdot[c]*dt/(2*V_cyl_exacthalf_sum[c]);
         P_inj[c] = Pdot[c]*dt/(2*V_cyl_exacthalf_sum[c]);
         E_inj[c] = Edot[c]*dt/(2*V_cyl_exacthalf_sum[c]);
      }
      else{
         M_inj[c] = Mdot[c]*dt/V_cyl[c];
         P_inj[c] = Pdot[c]*dt/V_cyl[c];
         E_inj[c] = Edot[c]*dt/V_cyl[c]; 
      }
   }

   if ( CurrentMaxLv ){
      for (int c=0; c<Merger_Coll_NumHalos; c++) E_inj_exp[c] += Edot[c]*dt;
   }
   if ( lv == 0 )  dt_base = dt;

/*
// get the number of OpenMP threads
   int NT; 
#  ifdef OPENMP
#  pragma omp parallel
#  pragma omp master
   {  NT = omp_get_num_threads();  }
#  else
   {  NT = 1;                      }   
#  endif
 
// allocate RNG
   RNG = new RandomNumber_t( NT );
// set the random seed of each MPI rank
   for (int t=0; t<NT; t++) {
      RNG->SetSeed(t, MPI_Rank*1000+t);
   }   

   static double phi,theta; // angles which decide the jet direction 
 
// choose the jet direction 
   if ( FirstTime ){
      for (int c=0; c<Merger_Coll_NumHalos; c++) {
         phi = RandomNumber(RNG,0.0,2*M_PI); 
         theta = RandomNumber(RNG,0.0,M_PI); 
         Jet_Vec[c][0] = 1.0; //sin(theta)*cos(phi);
         Jet_Vec[c][1] = 0.0; //sin(theta)*sin(phi);
         Jet_Vec[c][2] = 0.0; //cos(theta);
      }
      FirstTime = false;
   }
*/


#  pragma omp parallel for private( Reset, fluid, fluid_bk, x, y, z, x0, y0, z0 ) schedule( runtime ) \
   reduction(+:CM_Bondi_SinkMass, CM_Bondi_SinkMomX, CM_Bondi_SinkMomY, CM_Bondi_SinkMomZ, CM_Bondi_SinkMomXAbs, CM_Bondi_SinkMomYAbs, CM_Bondi_SinkMomZAbs, CM_Bondi_SinkE, CM_Bondi_SinkEk, CM_Bondi_SinkEt, CM_Bondi_SinkNCell)


   for (int PID=0; PID<amr->NPatchComma[lv][1]; PID++)
   {
      x0 = amr->patch[0][lv][PID]->EdgeL[0] + 0.5*dh;
      y0 = amr->patch[0][lv][PID]->EdgeL[1] + 0.5*dh;
      z0 = amr->patch[0][lv][PID]->EdgeL[2] + 0.5*dh;

      for (int k=0; k<PS1; k++)  {  z = z0 + k*dh;
      for (int j=0; j<PS1; j++)  {  y = y0 + j*dh;
      for (int i=0; i<PS1; i++)  {  x = x0 + i*dh;

         for (int v=0; v<NCOMP_TOTAL; v++)
         {   
            fluid   [v] = amr->patch[FluSg][lv][PID]->fluid[v][k][j][i];
    
//          backup the unmodified values since we want to record the amount of sunk variables removed at the maximum level
            fluid_bk[v] = fluid[v];
         }   

#        ifdef MHD
         const real Emag = MHD_GetCellCenteredBEnergyInPatch( lv, PID, i, j, k, amr->MagSg[lv] );
#        else
         const real Emag = (real)0.0;
#        endif

//       reset this cell
         Reset = Flu_ResetByUser_Func_ClusterMerger( fluid, Emag, x, y, z, TimeNew, dt, lv, NULL );

//       operations necessary only when this cell has been reset
         if ( Reset != 0 )
         {
//          apply density and energy floors
#           if ( MODEL == HYDRO  ||  MODEL == MHD )
            fluid[DENS] = FMAX( fluid[DENS], (real)MIN_DENS );
            fluid[ENGY] = Hydro_CheckMinEintInEngy( fluid[DENS], fluid[MOMX], fluid[MOMY], fluid[MOMZ], fluid[ENGY],
                                                    (real)MIN_EINT, Emag );

//          calculate the dual-energy variable
#           ifdef DUAL_ENERGY
            fluid[DUAL] = Hydro_Con2Dual( fluid[DENS], fluid[MOMX], fluid[MOMY], fluid[MOMZ], fluid[ENGY], Emag,
                                          EoS_DensEint2Pres_CPUPtr, EoS_AuxArray_Flt, EoS_AuxArray_Int, h_EoS_Table );
#           endif

//          floor and normalize passive scalars
#           if ( NCOMP_PASSIVE > 0 )
            for (int v=NCOMP_FLUID; v<NCOMP_TOTAL; v++)  fluid[v] = FMAX( fluid[v], TINY_NUMBER );

            if ( OPT__NORMALIZE_PASSIVE )
               Hydro_NormalizePassive( fluid[DENS], fluid+NCOMP_FLUID, PassiveNorm_NVar, PassiveNorm_VarIdx );
#           endif
#           endif // if ( MODEL == HYDRO  ||  MODEL == MHD )

//          record the amount of sunk variables removed at the maximum level
            if ( CurrentMaxLv )
            {
               double Ek[3], Ek_new[3], Et[3], Et_new[3];
               for (int c=0; c<Merger_Coll_NumHalos; c++) { 
                  Ek[c] = (real)0.5*( SQR(fluid_bk[MOMX]) + SQR(fluid_bk[MOMY]) + SQR(fluid_bk[MOMZ]) ) / (fluid_bk[DENS]);
                  Ek_new[c] = (real)0.5*( SQR(fluid[MOMX]) + SQR(fluid[MOMY]) + SQR(fluid[MOMZ]) ) / fluid[DENS];
                  Et[c] = fluid_bk[ENGY] - Ek[c] - Emag;
                  Et_new[c] = fluid[ENGY] - Ek_new[c] - Emag;
               } 

               CM_Bondi_SinkMass[Reset-1]    += dv*(fluid[DENS]-fluid_bk[DENS]);
               CM_Bondi_SinkMomX[Reset-1]    += dv*(fluid[MOMX]-fluid_bk[MOMX]);
               CM_Bondi_SinkMomY[Reset-1]    += dv*(fluid[MOMY]-fluid_bk[MOMY]);
               CM_Bondi_SinkMomZ[Reset-1]    += dv*(fluid[MOMZ]-fluid_bk[MOMZ]);
               CM_Bondi_SinkMomXAbs[Reset-1] += dv*FABS( fluid[MOMX]-fluid_bk[MOMX] );
               CM_Bondi_SinkMomYAbs[Reset-1] += dv*FABS( fluid[MOMY]-fluid_bk[MOMY] );
               CM_Bondi_SinkMomZAbs[Reset-1] += dv*FABS( fluid[MOMZ]-fluid_bk[MOMZ] );
               CM_Bondi_SinkE[Reset-1]       += dv*(fluid[ENGY]-fluid_bk[ENGY]);
               CM_Bondi_SinkEk[Reset-1]      += dv*(Ek_new[Reset-1]-Ek[Reset-1]);
               CM_Bondi_SinkEt[Reset-1]      += dv*(Et_new[Reset-1]-Et[Reset-1]);
               CM_Bondi_SinkNCell[Reset-1]   ++;
            }
         } // if ( Reset != 0 )

//       store the reset values         
         for (int v=0; v<NCOMP_TOTAL; v++)   amr->patch[FluSg][lv][PID]->fluid[v][k][j][i] = fluid[v];

      }}} // i,j,k
   } // for (int PID=0; PID<amr->NPatchComma[lv][1]; PID++)

//   delete RNG;


} // FUNCTION : Flu_ResetByUser_API_ClusterMerger


#endif // #if ( MODEL == HYDRO  &&  defined GRAVITY )
