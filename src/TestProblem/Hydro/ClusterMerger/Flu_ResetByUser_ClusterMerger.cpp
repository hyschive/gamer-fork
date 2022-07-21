#include "GAMER.h"

#if ( MODEL == HYDRO  &&  defined GRAVITY )


extern double eta, eps_f, eps_m; // parameters of jet feedback

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

extern int    Merger_Coll_NumHalos;
extern double R_acc;  // the radius to compute the accretoin rate
extern double R_dep;  // the radius to deplete the accreted gas
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

// check the injection
double DENS_org[3];
double MOMX_org[3];
double MOMXabs_org[3];
double ENGY_org[3];

double ClusterCen[3][3] = {  { NULL_REAL, NULL_REAL, NULL_REAL }, // cluster center       
                             { NULL_REAL, NULL_REAL, NULL_REAL },
                             { NULL_REAL, NULL_REAL, NULL_REAL }  };  
double BH_Vel[3][3] = {  { NULL_REAL, NULL_REAL, NULL_REAL }, // BH velocity
                         { NULL_REAL, NULL_REAL, NULL_REAL },
                         { NULL_REAL, NULL_REAL, NULL_REAL }  }; 
extern void GetClusterCenter( double Cen[][3], double BH_Vel[][3] );

static bool FirstTime = true;
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

double Jet_WaveK[3];  // jet wavenumber used in the sin() function to have smooth bidirectional jets

double Jet_HalfHeight[3];
double Jet_Radius[3];
double V_cyl[3]; // the volume of jet source
double M_inj[3], P_inj[3], E_inj[3]; // the injected density

double E_inj_exp[3];
double dt_base;
double E_power_inj[3];


// A temporate parameter to choose the feedback recipe
const int Recipe = 3;

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
bool Flu_ResetByUser_Func_ClusterMerger( real fluid[], const double x, const double y, const double z, const double Time, 
                                         const double dt, const int lv, double AuxArray[] )
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


// (2) Jet Feedback

   double Jet_dr, Jet_dh, S, Area;
   double Dis_c2m, Dis_c2v, Dis_v2m, Vec_c2m[3], Vec_v2m[3];
   double TempVec[3]; 
   real   MomSin, EngySin;

   for (int c=0; c<Merger_Coll_NumHalos; c++)
   {
//    distance: jet center to mesh
      for (int d=0; d<3; d++)    Vec_c2m[d] = Pos[d] - ClusterCen[c][d];
      Dis_c2m = sqrt( SQR(Vec_c2m[0]) + SQR(Vec_c2m[1]) + SQR(Vec_c2m[2]) );

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
      Jet_dr = 2.0*Area/Dis_c2v;
      Jet_dh = sqrt( Dis_c2m*Dis_c2m - Jet_dr*Jet_dr );

      if ( Jet_dh <= Jet_HalfHeight[c]  &&  Jet_dr <= Jet_Radius[c] )
      {
//         fluid[MOMX] -= BH_Vel[c][0]*fluid[DENS];
//         fluid[MOMY] -= BH_Vel[c][1]*fluid[DENS];
//         fluid[MOMZ] -= BH_Vel[c][2]*fluid[DENS]; 

//       Record the old momentum
         double MOMX_old = fluid[MOMX];
         double MOMY_old = fluid[MOMY];
         double MOMZ_old = fluid[MOMZ];

//       reset the fluid variables within the jet source
         fluid[DENS] += M_inj[c];      

//       Transfer into BH frame
         fluid[MOMX] -= BH_Vel[c][0]*fluid[DENS];
         fluid[MOMY] -= BH_Vel[c][1]*fluid[DENS];
         fluid[MOMZ] -= BH_Vel[c][2]*fluid[DENS]; 


         if ( Recipe == 1 ){
            MomSin       = P_inj[c]*sqrt(2.0)*sin( Jet_WaveK[c]*Jet_dh );
            MomSin      *= SIGN( Vec_c2m[0]*Jet_Vec[c][0] + Vec_c2m[1]*Jet_Vec[c][1] + Vec_c2m[2]*Jet_Vec[c][2] );
            fluid[MOMX] += MomSin*Jet_Vec[c][0];
            fluid[MOMY] += MomSin*Jet_Vec[c][1];
            fluid[MOMZ] += MomSin*Jet_Vec[c][2];

            fluid[ENGY] += E_inj[c];
         }

         else if ( Recipe == 2 ){
            P_inj[c] = sqrt(2*E_inj[c]*fluid[DENS]); 

            MomSin       = P_inj[c]*sqrt(2.0)*sin( Jet_WaveK[c]*Jet_dh );
            MomSin      *= SIGN( Vec_c2m[0]*Jet_Vec[c][0] + Vec_c2m[1]*Jet_Vec[c][1] + Vec_c2m[2]*Jet_Vec[c][2] );
            fluid[MOMX] += MomSin*Jet_Vec[c][0];
            fluid[MOMY] += MomSin*Jet_Vec[c][1];
            fluid[MOMZ] += MomSin*Jet_Vec[c][2];

            fluid[ENGY] += 0.5*((SQR(fluid[MOMX])+SQR(fluid[MOMY])+SQR(fluid[MOMZ]))/fluid[DENS]-(SQR(fluid[MOMX]-MomSin*Jet_Vec[c][0])+SQR(fluid[MOMY]-MomSin*Jet_Vec[c][1])+SQR(fluid[MOMZ]-MomSin*Jet_Vec[c][2]))/(fluid[DENS]-M_inj[c]));
         }

         else if ( Recipe == 3 ){
//            P_inj[c] = -sqrt(SQR(fluid[MOMX])+SQR(fluid[MOMY])+SQR(fluid[MOMZ]))+sqrt((SQR(fluid[MOMX])+SQR(fluid[MOMY])+SQR(fluid[MOMZ]))+2*E_inj[c]*fluid[DENS]);
            EngySin = E_inj[c]*0.5*M_PI*sin( Jet_WaveK[c]*Jet_dh );

            double P_SQR = SQR(fluid[MOMX])+SQR(fluid[MOMY])+SQR(fluid[MOMZ]);
            double P_new = sqrt(2*fluid[DENS]*(EngySin+0.5*P_SQR/(fluid[DENS]-M_inj[c])));

//            MomSin       = P_inj[c]; //*0.5*M_PI*sin( Jet_WaveK[c]*Jet_dh ); //sqrt(2)
//            MomSin      *= SIGN( Vec_c2m[0]*Jet_Vec[c][0] + Vec_c2m[1]*Jet_Vec[c][1] + Vec_c2m[2]*Jet_Vec[c][2] );
            P_new *= SIGN( Vec_c2m[0]*Jet_Vec[c][0] + Vec_c2m[1]*Jet_Vec[c][1] + Vec_c2m[2]*Jet_Vec[c][2] );
//            fluid[MOMX] += MomSin*Jet_Vec[c][0];
//            fluid[MOMY] += MomSin*Jet_Vec[c][1];
//            fluid[MOMZ] += MomSin*Jet_Vec[c][2];
            fluid[MOMX] = P_new*Jet_Vec[c][0];
            fluid[MOMY] = P_new*Jet_Vec[c][1];
            fluid[MOMZ] = P_new*Jet_Vec[c][2];

//          Transfer back into the rest frame  
            fluid[MOMX] += BH_Vel[c][0]*fluid[DENS];
            fluid[MOMY] += BH_Vel[c][1]*fluid[DENS];
            fluid[MOMZ] += BH_Vel[c][2]*fluid[DENS]; 

//            fluid[ENGY] += 0.5*((SQR(fluid[MOMX])+SQR(fluid[MOMY])+SQR(fluid[MOMZ]))/fluid[DENS]-(SQR(fluid[MOMX]-MomSin*Jet_Vec[c][0])+SQR(fluid[MOMY]-MomSin*Jet_Vec[c][1])+SQR(fluid[MOMZ]-MomSin*Jet_Vec[c][2]))/(fluid[DENS]-M_inj[c]));
            fluid[ENGY] += 0.5*((SQR(fluid[MOMX])+SQR(fluid[MOMY])+SQR(fluid[MOMZ]))/fluid[DENS]-(SQR(MOMX_old)+SQR(MOMY_old)+SQR(MOMZ_old))/(fluid[DENS]-M_inj[c]));
//            fluid[ENGY] += E_inj[c];
         }

//            fluid[MOMX] += BH_Vel[c][0]*fluid[DENS];
//            fluid[MOMY] += BH_Vel[c][1]*fluid[DENS];
//            fluid[MOMZ] += BH_Vel[c][2]*fluid[DENS]; 


////       Transfer back into the rest frame 
//         fluid[MOMX] += GasVel[c][0]*fluid[DENS];
//         fluid[MOMY] += GasVel[c][1]*fluid[DENS];
//         fluid[MOMZ] += GasVel[c][2]*fluid[DENS]; 

//         P_inj[c] = sqrt(2*(fluid[ENGY]+E_inj[c])*fluid[DENS])-sqrt(2*fluid[ENGY]*fluid[DENS]);

//         double EngSin;
//         EngSin = E_inj[c]*0.5*M_PI*sin( Jet_WaveK[c]*Jet_dh );
//         EngSin *= SIGN( Vec_c2m[0]*Jet_Vec[c][0] + Vec_c2m[1]*Jet_Vec[c][1] + Vec_c2m[2]*Jet_Vec[c][2] );
//         P_inj[c] = sqrt(2*EngSin*fluid[DENS]);

//       use a sine function to make the velocity smooth within the jet from +Jet_Vec to -Jet_Vec
//         MomSin       = P_inj[c]*0.3*sin( Jet_WaveK[c]*Jet_dh );


//         double MOMX_old, MOMY_old, MOMZ_old;
//         MOMX_old = fluid[MOMX];
//         MOMY_old = fluid[MOMY];
//         MOMZ_old = fluid[MOMZ];
//
//         fluid[MOMX] = sqrt(2*(fluid[ENGY]+EngSin)*fluid[DENS])*Jet_Vec[c][0]*SIGN( Vec_c2m[0]*Jet_Vec[c][0] + Vec_c2m[1]*Jet_Vec[c][1] + Vec_c2m[2]*Jet_Vec[c][2] );
//         fluid[MOMY] = sqrt(2*(fluid[ENGY]+EngSin)*fluid[DENS])*Jet_Vec[c][1]*SIGN( Vec_c2m[0]*Jet_Vec[c][0] + Vec_c2m[1]*Jet_Vec[c][1] + Vec_c2m[2]*Jet_Vec[c][2] );
//         fluid[MOMZ] = sqrt(2*(fluid[ENGY]+EngSin)*fluid[DENS])*Jet_Vec[c][2]*SIGN( Vec_c2m[0]*Jet_Vec[c][0] + Vec_c2m[1]*Jet_Vec[c][1] + Vec_c2m[2]*Jet_Vec[c][2] );
//
//         fluid[ENGY] += 0.5*((SQR(fluid[MOMX])+SQR(fluid[MOMY])+SQR(fluid[MOMZ]))-(SQR(MOMX_old)+SQR(MOMY_old)+SQR(MOMZ_old)))/fluid[DENS];

//       return immediately since we do NOT allow different jet source to overlap
         return true;
      } // if (  Jet_dh <= Jet_HalfHeight[c]  &&  Jet_dr <= Jet_Radius[c] )
   } // for (int c=0; c<Merger_Coll_NumHalos; c++) 


   return false;


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

   double Mdot_BH[3] = { Mdot_BH1, Mdot_BH2, Mdot_BH3 };
   double Bondi_MassBH[3] = { Bondi_MassBH1, Bondi_MassBH2, Bondi_MassBH3 }; 

   GetClusterCenter( ClusterCen, BH_Vel );

   const double dh       = amr->dh[lv];
   const real   dv       = CUBE(dh);
#  if ( MODEL == HYDRO  ||  MODEL == MHD )
   const real   Gamma_m1 = GAMMA - (real)1.0;
   const real  _Gamma_m1 = (real)1.0 / Gamma_m1;
#  endif   

   bool   Reset;
   real   fluid[NCOMP_TOTAL], fluid_bk[NCOMP_TOTAL];
   double x, y, z, x0, y0, z0;

// reset to 0 since we only want to record the number of void cells **for one sub-step**
   for (int c=0; c<Merger_Coll_NumHalos; c++) CM_Bondi_SinkNCell[c] = 0;

   for (int c=0; c<Merger_Coll_NumHalos; c++) { 

//    reset gas velocity to zero
      for (int d=0; d<3; d++)  GasVel[c][d] = 0.0;

      const bool CheckMinPres_No = false;

      double rho = 0.0;  // the average density inside accretion radius
      double Pres; // use for calculation of sound speed
      double Cs = 0.0;  // the average sound speed inside accretion radius
      double v = 0.0;  // the relative velocity between BH and gas
      double num = 0.0;  // the number of cells inside accretion radius

      for (int PID=0; PID<amr->NPatchComma[lv][1]; PID++)
      {
         x0 = amr->patch[0][lv][PID]->EdgeL[0] + 0.5*dh;
         y0 = amr->patch[0][lv][PID]->EdgeL[1] + 0.5*dh;
         z0 = amr->patch[0][lv][PID]->EdgeL[2] + 0.5*dh;
   
         for (int k=0; k<PS1; k++)  {  z = z0 + k*dh; 
         for (int j=0; j<PS1; j++)  {  y = y0 + j*dh; 
         for (int i=0; i<PS1; i++)  {  x = x0 + i*dh;
   
            for (int v=0; v<NCOMP_TOTAL; v++){
               fluid[v] = amr->patch[FluSg][lv][PID]->fluid[v][k][j][i];
            }
//          calculate the average density, sound speed and gas velocity inside accretion radius
            if (SQR(x-ClusterCen[c][0])+SQR(y-ClusterCen[c][1])+SQR(z-ClusterCen[c][2]) <= SQR(R_acc)){
               rho += fluid[0]*dv;
               Pres = (real) Hydro_Con2Pres( fluid[0], fluid[1], fluid[2], fluid[3],
                                             fluid[4], fluid+NCOMP_FLUID,
                                             CheckMinPres_No, NULL_REAL, NULL_REAL,
                                             EoS_DensEint2Pres_CPUPtr, EoS_AuxArray_Flt,
                                             EoS_AuxArray_Int, h_EoS_Table, NULL );
               Cs += sqrt(  EoS_DensPres2CSqr_CPUPtr( fluid[0], Pres, NULL, EoS_AuxArray_Flt, EoS_AuxArray_Int,
                                                      h_EoS_Table )  );
               for (int d=0; d<3; d++)  GasVel[c][d] += fluid[d+1]*dv;
               num += 1.0;
            }
         }}}
      }
      if (num == 0.0){
         Mdot_BH[c] = 0.0;
         GasDens[c] = 0.0;
         SoundSpeed[c] = 0.0;
         RelativeVel[c] = 0.0;
      }
      else{
         for (int d=0; d<3; d++)  GasVel[c][d] /= rho;
//         Aux_Message( stdout, "Time = %g, lv = %d, Mass inside R_acc = %14.8e, dv = %14.8e\n", TimeNew, lv, rho*UNIT_M/Const_Msun, dv );
         rho /= (4.0/3.0*M_PI*pow(R_acc,3));
         Cs /= num;
         for (int d=0; d<3; d++)  v += SQR(BH_Vel[c][d]-GasVel[c][d]);

//       calculate the accretion rate
         Mdot_BH[c] = 100.0*4.0*M_PI*SQR(NEWTON_G)*SQR(Bondi_MassBH[c])*rho/pow(Cs*Cs+v,1.5);
         GasDens[c] = rho;
         SoundSpeed[c] = Cs;
         RelativeVel[c] = sqrt(v);
      }

   } // for (int c=0; c<Merger_Coll_NumHalos; c++)

   Mdot_BH1 = Mdot_BH[0];                            
   Mdot_BH2 = Mdot_BH[1];                            
   Mdot_BH3 = Mdot_BH[2];

// update BH mass    
   for (int c=0; c<Merger_Coll_NumHalos; c++) {
      if ( lv == MAX_LEVEL ){
         Bondi_MassBH[c] += Mdot_BH[c]*dt;
 //        Aux_Message( stdout, "Time = %g, dt = %14.8e, lv = %d, Mdot_BH = %14.8e\n", TimeNew, dt, lv, Mdot_BH[c]*UNIT_M/Const_Msun/UNIT_T*Const_yr);
      }
   }
   Bondi_MassBH1 = Bondi_MassBH[0];
   Bondi_MassBH2 = Bondi_MassBH[1];
   Bondi_MassBH3 = Bondi_MassBH[2];

   Jet_HalfHeight[0] = Jet_HalfHeight1;
   Jet_HalfHeight[1] = Jet_HalfHeight2;
   Jet_HalfHeight[2] = Jet_HalfHeight3;
   Jet_Radius[0] = Jet_Radius1;
   Jet_Radius[1] = Jet_Radius2;
   Jet_Radius[2] = Jet_Radius3;

// calculate the injection rate
   for (int c=0; c<Merger_Coll_NumHalos; c++){
      Mdot[c] = eta*Mdot_BH[c];
      Pdot[c] = sqrt(2*eta*eps_f*(1.0-eps_m))*Mdot_BH[c]*(Const_c/UNIT_V);
      Edot[c] = eps_f*Mdot_BH[c]*SQR(Const_c/UNIT_V);
      V_cyl[c] = M_PI*SQR(Jet_Radius[c])*2*Jet_HalfHeight[c];

//    calculate the density that need to be injected
      M_inj[c] = Mdot[c]*dt/V_cyl[c];
      P_inj[c] = Pdot[c]*dt/V_cyl[c];
      E_inj[c] = Edot[c]*dt/V_cyl[c];
//      P_inj[c] = sqrt(2*E_inj[c]*GasDens[c]);
      Jet_WaveK[c] = 0.5*M_PI/Jet_HalfHeight[c];
   } 

   if ( lv == 0 )  dt_base = dt;

//   Aux_Message( stdout, "Time = %g, lv = %d, GasDens = %14.8e\n", TimeNew, lv, GasDens[0]*UNIT_D/(Const_Msun/pow(Const_kpc,3)) );

   if ( lv == MAX_LEVEL ){

//      Aux_Message( stdout, "=============================================================================\n" );
//      Aux_Message( stdout, "  Time                 = %g\n",           TimeNew );
//      Aux_Message( stdout, "  dt                   = %g\n",           dt );
//      Aux_Message( stdout, "  Mdot                 = %14.8e\n",           Mdot[0] );
//      Aux_Message( stdout, "  Pdot                 = %14.8e\n",           Pdot[0] );
//      Aux_Message( stdout, "  Edot                 = %14.8e\n",           Edot[0] );
//      Aux_Message( stdout, "  V_cyl                 = %14.8e\n",           V_cyl[0] );
//      Aux_Message( stdout, "  M_inj                 = %14.8e\n",           M_inj[0] );
//      Aux_Message( stdout, "  P_inj                 = %14.8e\n",           P_inj[0] );
//      Aux_Message( stdout, "  E_inj                 = %14.8e\n",           E_inj[0] );
//      Aux_Message( stdout, "=============================================================================\n" );
   }   

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

  

#  pragma omp parallel for private( Reset, fluid, fluid_bk, x, y, z, x0, y0, z0 ) schedule( runtime ) \
   reduction(+:CM_Bondi_SinkMass[0], CM_Bondi_SinkMomX[0], CM_Bondi_SinkMomY[0], CM_Bondi_SinkMomZ[0], CM_Bondi_SinkMomXAbs[0], CM_Bondi_SinkMomYAbs[0], CM_Bondi_SinkMomZAbs[0], CM_Bondi_SinkE[0], CM_Bondi_SinkEk[0], CM_Bondi_SinkEt[0], CM_Bondi_SinkNCell[0], CM_Bondi_SinkMass[1], CM_Bondi_SinkMomX[1], CM_Bondi_SinkMomY[1], CM_Bondi_SinkMomZ[1], CM_Bondi_SinkMomXAbs[1], CM_Bondi_SinkMomYAbs[1], CM_Bondi_SinkMomZAbs[1], CM_Bondi_SinkE[1], CM_Bondi_SinkEk[1], CM_Bondi_SinkEt[1], CM_Bondi_SinkNCell[1])

// Reset the background conserved variables to zero
   for (int c=0; c<Merger_Coll_NumHalos; c++){
      DENS_org[c] = 0.0;
      MOMX_org[c] = 0.0;
      ENGY_org[c] = 0.0;
      MOMXabs_org[c] = 0.0;
   }

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

//            if (SQR(x-ClusterCen[0][0])+SQR(y-ClusterCen[0][1])+SQR(z-ClusterCen[0][2]) <= SQR(5*R_acc)){
//               fluid[DENS] = GasDens[0];
//            }
            fluid_bk[v] = fluid[v];
         }   

//       reset this cell
         Reset = Flu_ResetByUser_Func_ClusterMerger( fluid, x, y, z, TimeNew, dt, lv, NULL );

//       operations necessary only when this cell has been reset
         if ( Reset )
         {
//          apply density and energy floors
#           if ( MODEL == HYDRO  ||  MODEL == MHD )
#           ifdef MHD
            const real Emag = MHD_GetCellCenteredBEnergyInPatch( lv, PID, i, j, k, amr->MagSg[lv] );
#           else
            const real Emag = NULL_REAL;
#           endif

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
            if ( lv == MAX_LEVEL )
            {
               double Ek[3], Ek_new[3], Et[3], Et_new[3];

               for (int c=0; c<Merger_Coll_NumHalos; c++) { 
                  Ek[c] = (real)0.5*( SQR(fluid_bk[MOMX]) + SQR(fluid_bk[MOMY]) + SQR(fluid_bk[MOMZ]) ) / (fluid_bk[DENS]);
                  Ek_new[c] = (real)0.5*( SQR(fluid[MOMX]) + SQR(fluid[MOMY]) + SQR(fluid[MOMZ]) ) / fluid[DENS];
                  Et[c] = fluid_bk[ENGY] - Ek[c];
                  Et_new[c] = fluid[ENGY] - Ek_new[c];
   
                  CM_Bondi_SinkMass[c]    += dv*(fluid[DENS]-fluid_bk[DENS]);
                  CM_Bondi_SinkMomX[c]    += dv*(fluid[MOMX]-fluid_bk[MOMX]);
                  CM_Bondi_SinkMomY[c]    += dv*(fluid[MOMY]-fluid_bk[MOMY]);
                  CM_Bondi_SinkMomZ[c]    += dv*(fluid[MOMZ]-fluid_bk[MOMZ]);
                  CM_Bondi_SinkMomXAbs[c] += dv*FABS( fluid[MOMX]-fluid_bk[MOMX] );
                  CM_Bondi_SinkMomYAbs[c] += dv*FABS( fluid[MOMY]-fluid_bk[MOMY] );
                  CM_Bondi_SinkMomZAbs[c] += dv*FABS( fluid[MOMZ]-fluid_bk[MOMZ] );
                  CM_Bondi_SinkE[c]       += dv*(fluid[ENGY]-fluid_bk[ENGY]);
                  CM_Bondi_SinkEk[c]      += dv*(Ek_new[c]-Ek[c]);
                  CM_Bondi_SinkEt[c]      += dv*(Et_new[c]-Et[c]);
                  CM_Bondi_SinkNCell[c]   ++;

                  DENS_org[c] += dv*fluid_bk[DENS];
                  MOMX_org[c] += dv*fluid_bk[MOMX];
                  ENGY_org[c] += dv*fluid_bk[ENGY];
                  MOMXabs_org[c] += dv*FABS(fluid_bk[MOMX]);
               } // for (int c=0; c<Merger_Coll_NumHalos; c++)
            }
         } // if ( Reset )

//       store the reset values         
         for (int v=0; v<NCOMP_TOTAL; v++)   amr->patch[FluSg][lv][PID]->fluid[v][k][j][i] = fluid[v];

      }}} // i,j,k
   } // for (int PID=0; PID<amr->NPatchComma[lv][1]; PID++)

   delete RNG;

} // FUNCTION : Flu_ResetByUser_API_ClusterMerger


#endif // #if ( MODEL == HYDRO  &&  defined GRAVITY )
