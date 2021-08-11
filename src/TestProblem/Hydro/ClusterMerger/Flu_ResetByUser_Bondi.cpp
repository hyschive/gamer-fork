#include "GAMER.h"

#if ( MODEL == HYDRO  &&  defined GRAVITY )


extern double Bondi_SinkMass1;
extern double Bondi_SinkMomX1;
extern double Bondi_SinkMomY1;
extern double Bondi_SinkMomZ1;
extern double Bondi_SinkMomXAbs1;
extern double Bondi_SinkMomYAbs1;
extern double Bondi_SinkMomZAbs1;
extern double Bondi_SinkEk1;
extern double Bondi_SinkEt1;
extern int    Bondi_SinkNCell1;

extern double Bondi_SinkMass2; 
extern double Bondi_SinkMomX2; 
extern double Bondi_SinkMomY2; 
extern double Bondi_SinkMomZ2; 
extern double Bondi_SinkMomXAbs2;
extern double Bondi_SinkMomYAbs2;
extern double Bondi_SinkMomZAbs2;
extern double Bondi_SinkEk2;   
extern double Bondi_SinkEt2;   
extern int    Bondi_SinkNCell2;

extern int    Merger_Coll_NumHalos;
extern double R_acc;  // the radius to compute the accretoin rate
extern double R_dep;  // the radius to deplete the accreted gas
extern double Bondi_MassBH1;
extern double Bondi_MassBH2;
extern double Bondi_MassBH3;
extern double Mdot_BH1; // the accretion rate
extern double Mdot_BH2;
extern double Mdot_BH3;

//-------------------------------------------------------------------------------------------------------
// Function    :  Flu_ResetByUser_Func_Bondi
// Description :  Function to reset the fluid field in the Bondi accretion problem
//
// Note        :  1. Invoked by Flu_ResetByUser_API_Bondi() and Hydro_Init_ByFunction_AssignData() using the
//                   function pointer "Flu_ResetByUser_Func_Ptr"
//                   --> This function pointer is reset by Init_TestProb_Hydro_Bondi()
//                   --> Hydro_Init_ByFunction_AssignData(): constructing initial condition
//                       Flu_ResetByUser_API_Bondi()       : after each update
//                2. Input fluid[] stores the original values
//                3. Even when DUAL_ENERGY is adopted, one does NOT need to set the dual-energy variable here
//                   --> It will be set automatically
//                4. Enabled by the runtime option "OPT__RESET_FLUID"
//
// Parameter   :  fluid    : Fluid array storing both the input (origial) and reset values
//                           --> Including both active and passive variables
//                x/y/z    : Target physical coordinates
//                Time     : Target physical time
//                lv       : Target refinement level
//                AuxArray : Auxiliary array
//
// Return      :  true  : This cell has been reset
//                false : This cell has not been reset
//-------------------------------------------------------------------------------------------------------
bool Flu_ResetByUser_Func_Bondi( real fluid[], const double x, const double y, const double z, const double Time, 
                                 const int lv, double AuxArray[] )
{

   const double Pos[3]  = { x, y, z };
   double dr2[3][3], r2[3];
   const double V_dep = 4.0/3.0*M_PI*pow(R_dep,3.0); // the region to remove gas
// the density need to be removed
   const double D_dep[3] = { Mdot_BH1*dTime_AllLv[lv]/V_dep, Mdot_BH2*dTime_AllLv[lv]/V_dep, Mdot_BH3*dTime_AllLv[lv]/V_dep };

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
         fluid[DENS] -= D_dep[c];
         fluid[MOMX] -= D_dep[c]*GasVel[c][0];
         fluid[MOMY] -= D_dep[c]*GasVel[c][1];
         fluid[MOMZ] -= D_dep[c]*GasVel[c][2];
         fluid[ENGY] -= 0.5*D_dep[c]*( SQR(GasVel[c][0]) + SQR(GasVel[c][1]) + SQR(GasVel[c][2]) );

         return true;
         iftrue = 1;
      }
   }

   if ( iftrue==0 ){
      return false;
   }

} // FUNCTION : Flu_ResetByUser_Func_Bondi



//-------------------------------------------------------------------------------------------------------
// Function    :  Flu_ResetByUser_API_Bondi
// Description :  API for resetting the fluid array in the Bondi accretion problem
//
// Note        :  1. Enabled by the runtime option "OPT__RESET_FLUID"
//                2. Invoked using the function pointer "Flu_ResetByUser_API_Ptr"
//                   --> This function pointer is reset by Init_TestProb_Hydro_Bondi()
//                3. Currently does not work with "OPT__OVERLAP_MPI"
//                4. Invoke Flu_ResetByUser_Func_Bondi() directly
//
// Parameter   :  lv    : Target refinement level
//                FluSg : Target fluid sandglass
//                TTime : Target physical time
//-------------------------------------------------------------------------------------------------------
void Flu_ResetByUser_API_Bondi( const int lv, const int FluSg, const double TTime )
{
   double Bondi_MassBH[3] = { Bondi_MassBH1, Bondi_MassBH2, Bondi_MassBH3 }; 
   double Mdot_BH[3] = { Mdot_BH1, Mdot_BH2, Mdot_BH3 };
   double ClusterCen[3][3] = {  { NULL_REAL, NULL_REAL, NULL_REAL }, // cluster center                                  
                                { NULL_REAL, NULL_REAL, NULL_REAL },
                                { NULL_REAL, NULL_REAL, NULL_REAL }  };  
   double BH_Vel[3][3] = {  { NULL_REAL, NULL_REAL, NULL_REAL }, // BH velocity
                            { NULL_REAL, NULL_REAL, NULL_REAL },
                            { NULL_REAL, NULL_REAL, NULL_REAL }  };  
   GetClusterCenter( ClusterCen, BH_Vel );

   double GasVel[3][3] = {  { NULL_REAL, NULL_REAL, NULL_REAL }, // gas velocity
                            { NULL_REAL, NULL_REAL, NULL_REAL },
                            { NULL_REAL, NULL_REAL, NULL_REAL }  };

   const double dh       = amr->dh[lv];
   const real   dv       = CUBE(dh);
#  if ( MODEL == HYDRO  ||  MODEL == MHD )
   const real   Gamma_m1 = GAMMA - (real)1.0;
   const real  _Gamma_m1 = (real)1.0 / Gamma_m1;
#  endif   

   bool   Reset;
   real   fluid[NCOMP_TOTAL], fluid_bk[NCOMP_TOTAL];
   double x, y, z, x0, y0, z0;

   double SinkMass_OneSubStep_ThisRank1 = 0; // record the mass that will be added to the two BH
   double SinkMass_OneSubStep_ThisRank2 = 0;
   double SinkMass_OneSubStep_AllRank1;
   double SinkMass_OneSubStep_AllRank2;

// reset to 0 since we only want to record the number of void cells **for one sub-step**
   Bondi_SinkNCell = 0;

   for (int c=0; c<Merger_Coll_NumHalos; c++) { 

      double rho = 0.0;  // the average density inside accretion radius
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
   
   //       calculate the average density, sound speed and gas velocity inside accretion radius
            if (SQR(x-ClusterCen[c][0])+SQR(y-ClusterCen[c][1])+SQR(z-ClusterCen[c][2]) <= SQR(R_acc)){
               rho += fluid[0];
               Cs += sqrt(GAMMA*((GAMMA-1.0)*(fluid[4]-0.5*(SQR(fluid[1])+SQR(fluid[2])+SQR(fluid[3]))/fluid[0]))/fluid[0]);
               for (int d=0; d<3; d++)  GasVel[c][d] += fluid[d+1];
               num += 1.0;
            }
         }}}
      }
      for (int d=0; d<3; d++)  GasVel[c][d] /= rho;
      rho /= num;
      Cs /= num;
      for (int d=0; d<3; d++)  v += SQR(BH_Vel[c][d]-GasVel[c][d]);

   // calculate the accretion rate
      Mdot_BH[c] = 4.0*M_PI*Const_NewtonG*SQR(Bondi_MassBH[c])*rho/pow(Cs*Cs+v,1.5);
   } // for (int c=0; c<Merger_Coll_NumHalos; c++)

   Mdot_BH1 = Mdot_BH[0];                            
   Mdot_BH2 = Mdot_BH[1];                            
   Mdot_BH3 = Mdot_BH[2];
  
   
#  pragma omp parallel for private( Reset, fluid, fluid_bk, x, y, z, x0, y0, z0 ) schedule( runtime ) \
   reduction(+:Bondi_SinkMass1, Bondi_SinkMomX1, Bondi_SinkMomY1, Bondi_SinkMomZ1, Bondi_SinkMomXAbs1, Bondi_SinkMomYAbs1, \
               Bondi_SinkMomZAbs1, Bondi_SinkEk1, Bondi_SinkEt1, Bondi_SinkNCell1, SinkMass_OneSubStep_ThisRank1, \
               Bondi_SinkMass2, Bondi_SinkMomX2, Bondi_SinkMomY2, Bondi_SinkMomZ2, Bondi_SinkMomXAbs2, Bondi_SinkMomYAbs2, \
               Bondi_SinkMomZAbs2, Bondi_SinkEk2, Bondi_SinkEt2, Bondi_SinkNCell2, SinkMass_OneSubStep_ThisRank2)

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
//
//       reset this cell
         Reset = Flu_ResetByUser_Func_Bondi( fluid, x, y, z, TTime, lv, NULL );

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

//          calculate the dual-energy variable (entropy or internal energy)
#           if   ( DUAL_ENERGY == DE_ENPY )
            fluid[ENPY] = Hydro_Con2Entropy( fluid[DENS], fluid[MOMX], fluid[MOMY], fluid[MOMZ], fluid[ENGY], Emag,
                                             EoS_DensEint2Pres_CPUPtr, EoS_AuxArray_Flt, EoS_AuxArray_Int, h_EoS_Table );
#           elif ( DUAL_ENERGY == DE_EINT )
#           error : DE_EINT is NOT supported yet !!
#           endif

//          floor and normalize passive scalars
#           if ( NCOMP_PASSIVE > 0 )
            for (int v=NCOMP_FLUID; v<NCOMP_TOTAL; v++)  fluid[v] = FMAX( fluid[v], TINY_NUMBER );

            if ( OPT__NORMALIZE_PASSIVE )
               Hydro_NormalizePassive( fluid[DENS], fluid+NCOMP_FLUID, PassiveNorm_NVar, PassiveNorm_VarIdx );
#           endif
#           endif // if ( MODEL == HYDRO  ||  MODEL == MHD )

//          store the reset values
            for (int v=0; v<NCOMP_TOTAL; v++)   amr->patch[FluSg][lv][PID]->fluid[v][k][j][i] = fluid[v];


//          record the amount of sunk variables removed at the maximum level
            if ( lv == MAX_LEVEL )
            {
               if (SQR(x-ClusterCen[0][0])+SQR(y-ClusterCen[0][1])+SQR(z-ClusterCen[0][2]) < SQR(R_dep))
               {
                  real Ek1 = (real)0.5*( SQR(fluid_bk[MOMX]) + SQR(fluid_bk[MOMY]) + SQR(fluid_bk[MOMZ]) ) / fluid_bk[DENS];
                  real Ek_new1 = (real)0.5*( SQR(fluid[MOMX]) + SQR(fluid[MOMY]) + SQR(fluid[MOMZ]) ) / fluid[DENS];
                  real Et1 = fluid_bk[ENGY] - Ek1;
                  real Et_new1 = fluid[ENGY] - Ek_new1;
   
                  Bondi_SinkMass1    += dv*(fluid_bk[DENS]-fluid[DENS]);
                  Bondi_SinkMomX1    += dv*(fluid_bk[MOMX]-fluid[MOMX]);
                  Bondi_SinkMomY1    += dv*(fluid_bk[MOMY]-fluid[MOMY]);
                  Bondi_SinkMomZ1    += dv*(fluid_bk[MOMZ]-fluid[MOMZ]);
                  Bondi_SinkMomXAbs1 += dv*FABS( fluid_bk[MOMX]-fluid[MOMX] );
                  Bondi_SinkMomYAbs1 += dv*FABS( fluid_bk[MOMY]-fluid[MOMY] );
                  Bondi_SinkMomZAbs1 += dv*FABS( fluid_bk[MOMZ]-fluid[MOMZ] );
                  Bondi_SinkEk1      += dv*(Ek1-Ek_new1);
                  Bondi_SinkEt1      += dv*(Et1-Et_new1);
                  Bondi_SinkNCell1   ++;
   
                  SinkMass_OneSubStep_ThisRank1 += dv*(fluid_bk[DENS]-fluid[DENS]);
               }
               else if (SQR(x-ClusterCen[1][0])+SQR(y-ClusterCen[1][1])+SQR(z-ClusterCen[1][2]) < SQR(R_dep))
               {
                  real Ek2 = (real)0.5*( SQR(fluid_bk[MOMX]) + SQR(fluid_bk[MOMY]) + SQR(fluid_bk[MOMZ]) ) / fluid_bk[DENS];
                  real Ek_new2 = (real)0.5*( SQR(fluid[MOMX]) + SQR(fluid[MOMY]) + SQR(fluid[MOMZ]) ) / fluid[DENS];
                  real Et2 = fluid_bk[ENGY] - Ek2; 
                  real Et_new2 = fluid[ENGY] - Ek_new2;
   
                  Bondi_SinkMass2    += dv*(fluid_bk[DENS]-fluid[DENS]);
                  Bondi_SinkMomX2    += dv*(fluid_bk[MOMX]-fluid[MOMX]);
                  Bondi_SinkMomY2    += dv*(fluid_bk[MOMY]-fluid[MOMY]);
                  Bondi_SinkMomZ2    += dv*(fluid_bk[MOMZ]-fluid[MOMZ]);
                  Bondi_SinkMomXAbs2 += dv*FABS( fluid_bk[MOMX]-fluid[MOMX] );
                  Bondi_SinkMomYAbs2 += dv*FABS( fluid_bk[MOMY]-fluid[MOMY] );
                  Bondi_SinkMomZAbs2 += dv*FABS( fluid_bk[MOMZ]-fluid[MOMZ] );
                  Bondi_SinkEk2      += dv*(Ek2-Ek_new2);
                  Bondi_SinkEt2      += dv*(Et2-Et_new2);
                  Bondi_SinkNCell2   ++; 
   
                  SinkMass_OneSubStep_ThisRank2 += dv*(fluid_bk[DENS]-fluid[DENS]);
               }
            }
         } // if ( Reset )

      }}} // i,j,k
   } // for (int PID=0; PID<amr->NPatchComma[lv][1]; PID++)

// update BH mass
   MPI_Allreduce( &SinkMass_OneSubStep_ThisRank1, &SinkMass_OneSubStep_AllRank1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
   Bondi_MassBH1 += SinkMass_OneSubStep_AllRank1;
   MPI_Allreduce( &SinkMass_OneSubStep_ThisRank2, &SinkMass_OneSubStep_AllRank2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
   Bondi_MassBH2 += SinkMass_OneSubStep_AllRank2;


} // FUNCTION : Flu_ResetByUser_API_Bondi


#endif // #if ( MODEL == HYDRO  &&  defined GRAVITY )
