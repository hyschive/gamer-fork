#include "GAMER.h"

#if ( MODEL == HYDRO  &&  defined GRAVITY )


extern int    Merger_Coll_NumHalos;

extern double Jet_HalfHeight1;
extern double Jet_HalfHeight2;
extern double Jet_HalfHeight3;
extern double Jet_Radius1;
extern double Jet_Radius2;
extern double Jet_Radius3;
 
extern double Jet_Vec[3][3]; // jet direction  

/*
double ClusterCen_Flag[3][3] = {  { NULL_REAL, NULL_REAL, NULL_REAL }, // cluster center       
                                  { NULL_REAL, NULL_REAL, NULL_REAL },
                                  { NULL_REAL, NULL_REAL, NULL_REAL }  };
double BH_Vel_Flag[3][3] = {  { NULL_REAL, NULL_REAL, NULL_REAL }, // BH velocity
                              { NULL_REAL, NULL_REAL, NULL_REAL },
                              { NULL_REAL, NULL_REAL, NULL_REAL }  };  
extern void GetClusterCenter( double Cen[][3], double BH_Vel[][3] );
*/

extern double ClusterCen[3][3];


//-------------------------------------------------------------------------------------------------------
// Function    :  Flag_ClusterMerger
// Description :  Flag cells for refinement for the cluster merger test problem
//
// Note        :  1. Linked to the function pointer "Flag_User_Ptr" by Init_TestProb_Hydro_Bondi()
//                2. Please turn on the runtime option "OPT__FLAG_USER"
//
// Parameter   :  i,j,k     : Indices of the targeted element in the patch ptr[ amr->FluSg[lv] ][lv][PID]
//                lv        : Refinement level of the targeted patch
//                PID       : ID of the targeted patch
//                Threshold : User-provided threshold for the flag operation, which is loaded from the
//                            file "Input__Flag_User"
//
// Return      :  "true"  if the flag criteria are satisfied
//                "false" if the flag criteria are not satisfied
//-------------------------------------------------------------------------------------------------------
bool Flag_ClusterMerger( const int i, const int j, const int k, const int lv, const int PID, const double *Threshold )
{

   const double dh     = amr->dh[lv];
   const double Pos[3] = { amr->patch[0][lv][PID]->EdgeL[0] + (i+0.5)*dh,
                           amr->patch[0][lv][PID]->EdgeL[1] + (j+0.5)*dh,
                           amr->patch[0][lv][PID]->EdgeL[2] + (k+0.5)*dh  };

   bool Flag = false;

//   GetClusterCenter( ClusterCen_Flag, BH_Vel_Flag );
   double Jet_HalfHeight[3] = { Jet_HalfHeight1, Jet_HalfHeight2, Jet_HalfHeight3 };
   double Jet_Radius[3] = { Jet_Radius1, Jet_Radius2, Jet_Radius3 }; 

// Temp!!!
   for (int d=0; d<3; d++)   ClusterCen[0][d] = 5.0;
   Jet_Vec[0][0] = 1.0;
   for (int d=1; d<3; d++)   Jet_Vec[0][d] = 0.0;

// flag cells within the target radius
   double Jet_dr, Jet_dh, S, Area;
   double Dis_c2m, Dis_c2v, Dis_v2m, Vec_c2m[3], Vec_v2m[3];
   double TempVec[3]; 

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
 
      if ( Jet_dh <= Jet_HalfHeight[c]*Threshold[lv]  &&  Jet_dr <= Jet_Radius[c]*Threshold[lv]  &&  lv >= 5 ) {
         Flag = true;
         Aux_Message( stdout, "lv = %d, Threshold = %f\n", lv, Threshold[lv]);
//         Aux_Message( stdout, " Yes=======================================\n" );
         return Flag;
      }
   }

   return Flag;

} // FUNCTION : Flag_ClusterMerger



#endif // #if ( MODEL == HYDRO  &&  defined GRAVITY )
