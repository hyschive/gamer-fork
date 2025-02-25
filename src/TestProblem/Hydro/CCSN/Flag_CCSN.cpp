#include "GAMER.h"


extern bool   CCSN_CC_MaxRefine_Flag1;
extern bool   CCSN_CC_MaxRefine_Flag2;
extern int    CCSN_CC_MaxRefine_LV1;
extern int    CCSN_CC_MaxRefine_LV2;
extern double CCSN_CC_MaxRefine_Dens1;
extern double CCSN_CC_MaxRefine_Dens2;
extern double CCSN_CentralDens;

extern bool   CCSN_Is_PostBounce;
extern double CCSN_REF_RBase;
extern double CCSN_Rsh_Max;
extern double CCSN_Rsh_Ave;

extern double CCSN_MaxRefine_Rad;
extern double CCSN_AngRes_Min;
extern double CCSN_AngRes_Max;



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

   bool Flag      = false;
   bool MaxRefine = false;

   const double dh     = amr->dh[lv];
   const double Pos[3] = { amr->patch[0][lv][PID]->EdgeL[0] + (i+0.5)*dh,
                           amr->patch[0][lv][PID]->EdgeL[1] + (j+0.5)*dh,
                           amr->patch[0][lv][PID]->EdgeL[2] + (k+0.5)*dh  };
#  ifdef GRAVITY
   const double dR [3] = { Pos[0]-GREP_Center[0],    Pos[1]-GREP_Center[1],    Pos[2]-GREP_Center[2]    };
#  else
   const double dR [3] = { Pos[0]-amr->BoxCenter[0], Pos[1]-amr->BoxCenter[1], Pos[2]-amr->BoxCenter[2] };
#  endif
   const double R      = sqrt( SQR(dR[0]) + SQR(dR[1]) + SQR(dR[2]) );

   const double CentralDens = CCSN_CentralDens / UNIT_D;


// (1) check if the allowed maximum level is reached
   if      ( CCSN_CC_MaxRefine_Flag1  &&  CentralDens < CCSN_CC_MaxRefine_Dens1 / UNIT_D )
      MaxRefine = lv >= CCSN_CC_MaxRefine_LV1;

   else if ( CCSN_CC_MaxRefine_Flag2  &&  CentralDens < CCSN_CC_MaxRefine_Dens2 / UNIT_D )
      MaxRefine = lv >= CCSN_CC_MaxRefine_LV2;


   if ( !MaxRefine ) {
//    (2) check if the minimum angular resolution is reached
      if ( CCSN_AngRes_Min > 0.0  &&  R * CCSN_AngRes_Min < dh )
         Flag = true;

//    (3) always refine to the highest level in the region within r < CCSN_MaxRefine_Rad
//    (3-a) always refine the innermost cells
      if ( R < amr->dh[lv] )
         Flag = true;

//    (3-b) refine the region within r < CCSN_MaxRefine_Rad
      if ( R * UNIT_L < CCSN_MaxRefine_Rad )
         Flag = true;
   }


   return Flag;

} // FUNCTION : Flag_CoreCollapse



//-------------------------------------------------------------------------------------------------------
// Function    :  Flag_PostBounce
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
bool Flag_PostBounce( const int i, const int j, const int k, const int lv, const int PID, const double *Threshold )
{

   bool Flag = false;

   const double dh     = amr->dh[lv];
   const double Pos[3] = { amr->patch[0][lv][PID]->EdgeL[0] + (i+0.5)*dh,
                           amr->patch[0][lv][PID]->EdgeL[1] + (j+0.5)*dh,
                           amr->patch[0][lv][PID]->EdgeL[2] + (k+0.5)*dh  };
#  ifdef GRAVITY
   const double dR [3] = { Pos[0]-GREP_Center[0],    Pos[1]-GREP_Center[1],    Pos[2]-GREP_Center[2]    };
#  else
   const double dR [3] = { Pos[0]-amr->BoxCenter[0], Pos[1]-amr->BoxCenter[1], Pos[2]-amr->BoxCenter[2] };
#  endif
   const double R = sqrt( SQR(dR[0]) + SQR(dR[1]) + SQR(dR[2]) );


// (1) always refine to the highest level in the region within r < CCSN_MaxRefine_Rad
// (1-a) always refine the innermost cells
   if ( R < amr->dh[lv] )
      Flag = true;

// (1-b) refine the region within r < CCSN_MaxRefine_Rad
   if ( R * UNIT_L < CCSN_MaxRefine_Rad )
      Flag = true;

// (2) check if the minimum angular resolution is reached
   if ( CCSN_AngRes_Min > 0.0  &&  R * CCSN_AngRes_Min < dh )
      Flag = true;


   return Flag;

} // FUNCTION : Flag_PostBounce



//-------------------------------------------------------------------------------------------------------
// Function    :  Flag_Region_CCSN
// Description :  Check if the element (i,j,k) of the input patch is within the regions allowed to be refined
//                for the CCSN test problem
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
bool Flag_Region_CCSN( const int i, const int j, const int k, const int lv, const int PID )
{

   const double dh     = amr->dh[lv];
   const double Pos[3] = { amr->patch[0][lv][PID]->EdgeL[0] + (i+0.5)*dh,
                           amr->patch[0][lv][PID]->EdgeL[1] + (j+0.5)*dh,
                           amr->patch[0][lv][PID]->EdgeL[2] + (k+0.5)*dh  };

   bool Within = true;


#  ifdef GRAVITY
   const double dR [3] = { Pos[0]-GREP_Center[0],    Pos[1]-GREP_Center[1],    Pos[2]-GREP_Center[2]    };
#  else
   const double dR [3] = { Pos[0]-amr->BoxCenter[0], Pos[1]-amr->BoxCenter[1], Pos[2]-amr->BoxCenter[2] };
#  endif
   const double R      = sqrt( SQR(dR[0]) + SQR(dR[1]) + SQR(dR[2]) );


// must check CCSN_MaxRefine_Rad before evaluating other criteria
   if ( R * UNIT_L <= CCSN_MaxRefine_Rad )
      return true;

// check the maximum allowed refinement level based on angular resolution
   if ( CCSN_AngRes_Max > 0.0  &&  2.0 * R * CCSN_AngRes_Max > dh )
      Within = false;

   if ( !Within )   return Within;


// check allowed maximum refine level based on distance to the center
   if ( CCSN_REF_RBase > 0.0 )
   {
      const double R_base = CCSN_REF_RBase;
            int    ratio  = (int) ( R / R_base );
            int    dlv    = 0;

      while ( ratio )   { dlv += 1; ratio >>= 1; }

      if ( lv + dlv >= MAX_LEVEL )  Within = false;
   }


   return Within;

} // FUNCTION : Flag_Region_CCSN
