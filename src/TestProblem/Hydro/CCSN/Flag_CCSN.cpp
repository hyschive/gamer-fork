#include "GAMER.h"


extern double CCSN_MaxRefine_RadFac;
extern double CCSN_CentralDens;



//-------------------------------------------------------------------------------------------------------
// Function    :  Flag_CoreCollapse
// Description :  Check if the element (i,j,k) of the input data satisfies the user-defined flag criteria
//
// Note        :  1. Invoked by "Flag_Check" using the function pointer "Flag_User_Ptr"
//                   --> The function pointer may be reset by various test problem initializers, in which case
//                       this function will become useless
//                2. Enabled by the runtime option "OPT__FLAG_USER"
//
// Parameter   :  i,j,k       : Indices of the target element in the patch ptr[ amr->FluSg[lv] ][lv][PID]
//                lv          : Refinement level of the target patch
//                PID         : ID of the target patch
//                Threshold   : User-provided threshold for the flag operation, which is loaded from the
//                              file "Input__Flag_User"
//                              In order of radius_min, radius_max, threshold_dens
//
// Return      :  "true"  if the flag criteria are satisfied
//                "false" if the flag criteria are not satisfied
//-------------------------------------------------------------------------------------------------------
bool Flag_CoreCollapse( const int i, const int j, const int k, const int lv, const int PID, const double *Threshold )
{

   bool Flag = false;
   bool MaxRefine = false;

   const double dh        = amr->dh[lv];
   const double Center[3] = { amr->BoxCenter[0], amr->BoxCenter[1], amr->BoxCenter[2] };
   const double Pos   [3] = { amr->patch[0][lv][PID]->EdgeL[0] + (i+0.5)*dh,
                              amr->patch[0][lv][PID]->EdgeL[1] + (j+0.5)*dh,
                              amr->patch[0][lv][PID]->EdgeL[2] + (k+0.5)*dh  };

   const double dx = Center[0] - Pos[0];
   const double dy = Center[1] - Pos[1];
   const double dz = Center[2] - Pos[2];
   const double r  = sqrt(  SQR( dx ) + SQR( dy ) + SQR( dz )  );

   const double CentralDens = CCSN_CentralDens * UNIT_D;

// (1) check if the allowed maximum level is reached
   if ( CentralDens < 1e11 * UNIT_D )
   {
      MaxRefine = dh * UNIT_L <= 2e5; // allowed finest resoultion of 2km
   }

   else if ( CentralDens < 1e12 * UNIT_D )
   {
      MaxRefine = dh * UNIT_L <= 1e5; // allowed finest resoultion of 1km
   }


// (2) always refined to highest level in the region with r < 30 km, if allowed
   if (  !MaxRefine  &&  ( r * UNIT_L < 3e6 )  )
      Flag = true;


   return Flag;

} // FUNCTION : Flag_CoreCollapse



//-------------------------------------------------------------------------------------------------------
// Function    :  Flag_Lightbulb
// Description :  Check if the element (i,j,k) of the input data satisfies the user-defined flag criteria
//
// Note        :  1. Invoked by "Flag_Check" using the function pointer "Flag_User_Ptr"
//                   --> The function pointer may be reset by various test problem initializers, in which case
//                       this function will become useless
//                2. Enabled by the runtime option "OPT__FLAG_USER"
//                3. For lightbulb test problem
//
// Parameter   :  i,j,k       : Indices of the target element in the patch ptr[ amr->FluSg[lv] ][lv][PID]
//                lv          : Refinement level of the target patch
//                PID         : ID of the target patch
//                Threshold   : User-provided threshold for the flag operation, which is loaded from the
//                              file "Input__Flag_User"
//                              In order of radius_min, radius_max, threshold_dens
//
// Return      :  "true"  if the flag criteria are satisfied
//                "false" if the flag criteria are not satisfied
//-------------------------------------------------------------------------------------------------------
bool Flag_Lightbulb( const int i, const int j, const int k, const int lv, const int PID, const double *Threshold )
{

   bool Flag = false;

   const double dh        = amr->dh[lv];
   const double Center[3] = { amr->BoxCenter[0], amr->BoxCenter[1], amr->BoxCenter[2] };
   const double Pos   [3] = { amr->patch[0][lv][PID]->EdgeL[0] + (i+0.5)*dh,
                              amr->patch[0][lv][PID]->EdgeL[1] + (j+0.5)*dh,
                              amr->patch[0][lv][PID]->EdgeL[2] + (k+0.5)*dh  };

   const double dx = Center[0] - Pos[0];
   const double dy = Center[1] - Pos[1];
   const double dz = Center[2] - Pos[2];
   const double r  = sqrt(  SQR( dx ) + SQR( dy ) + SQR( dz )  );

   const real (*Rho )[PS1][PS1] = amr->patch[ amr->FluSg[lv] ][lv][PID]->fluid[DENS];


// TODO: fine-tune the criteria
// (1) always refined to highest level in the region with r < 30 km
   if ( r * UNIT_L < 3e6 )
   {
      Flag = true;
   }

   else
   {
//    (2-a) density is larger than the threshold in Input__Flag_User
      if ( Rho[k][j][i] < Threshold[0] )   return false;

//    (2-b) the cell width at son level (lv+1) is larger than the threshold
      const double Min_CellWidth = r * CCSN_MaxRefine_RadFac;

      Flag = ( 0.5 * dh ) > Min_CellWidth;
   }


   return Flag;

} // FUNCTION : Flag_Lightbulb
