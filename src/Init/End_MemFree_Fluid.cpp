#ifndef GPU

#include "GAMER.h"
#include "CUFLU.h"


#if ( FLU_SCHEME == MHM  ||  FLU_SCHEME == MHM_RP  ||  FLU_SCHEME == CTU )
extern real (*h_PriVar)      [NCOMP_LR            ][ CUBE(FLU_NXT)     ];
extern real (*h_Slope_PPM)[3][NCOMP_LR            ][ CUBE(N_SLOPE_PPM) ];
extern real (*h_FC_Var)   [6][NCOMP_TOTAL_PLUS_MAG][ CUBE(N_FC_VAR)    ];
extern real (*h_FC_Flux)  [3][NCOMP_TOTAL_PLUS_MAG][ CUBE(N_FC_FLUX)   ];
#ifdef MHD
extern real (*h_FC_Mag_Half)[NCOMP_MAG][ FLU_NXT_P1*SQR(FLU_NXT) ];
extern real (*h_EC_Ele     )[NCOMP_MAG][ CUBE(N_EC_ELE)          ];
#endif
#endif // FLU_SCHEME




//-------------------------------------------------------------------------------------------------------
// Function    :  End_MemFree_Fluid
// Description :  Free memory previously allocated by Init_MemAllocate_Fluid()
//
// Parameter   :  None
//-------------------------------------------------------------------------------------------------------
void End_MemFree_Fluid()
{

   for (int t=0; t<2; t++)
   {
      delete [] h_Flu_Array_F_In [t];  h_Flu_Array_F_In [t] = NULL;
      delete [] h_Flu_Array_F_Out[t];  h_Flu_Array_F_Out[t] = NULL;
      delete [] h_Flux_Array     [t];  h_Flux_Array     [t] = NULL;
#     ifdef UNSPLIT_GRAVITY
      delete [] h_Pot_Array_USG_F[t];  h_Pot_Array_USG_F[t] = NULL;
      delete [] h_Corner_Array_F [t];  h_Corner_Array_F [t] = NULL;
#     endif
      delete [] h_dt_Array_T     [t];  h_dt_Array_T     [t] = NULL;
      delete [] h_Flu_Array_T    [t];  h_Flu_Array_T    [t] = NULL;
#     ifdef DUAL_ENERGY
      delete [] h_DE_Array_F_Out [t];  h_DE_Array_F_Out [t] = NULL;
#     endif
      delete [] h_Flu_Array_S_In [t];  h_Flu_Array_S_In [t] = NULL;
      delete [] h_Flu_Array_S_Out[t];  h_Flu_Array_S_Out[t] = NULL;
      delete [] h_Corner_Array_S [t];  h_Corner_Array_S [t] = NULL;
#     ifdef MHD
      delete [] h_Mag_Array_F_In [t];  h_Mag_Array_F_In [t] = NULL;
      delete [] h_Mag_Array_F_Out[t];  h_Mag_Array_F_Out[t] = NULL;
      delete [] h_Ele_Array      [t];  h_Ele_Array      [t] = NULL;
      delete [] h_Mag_Array_T    [t];  h_Mag_Array_T    [t] = NULL;
      delete [] h_Mag_Array_S_In [t];  h_Mag_Array_S_In [t] = NULL;
#     endif
   } // for (int t=0; t<2; t++)

#  if ( FLU_SCHEME == MHM  ||  FLU_SCHEME == MHM_RP  ||  FLU_SCHEME == CTU )
   delete [] h_FC_Var;               h_FC_Var             = NULL;
   delete [] h_FC_Flux;              h_FC_Flux            = NULL;
   delete [] h_PriVar;               h_PriVar             = NULL;
#  if ( LR_SCHEME == PPM )
   delete [] h_Slope_PPM;            h_Slope_PPM          = NULL;
#  endif
#  ifdef MHD
   delete [] h_FC_Mag_Half;          h_FC_Mag_Half        = NULL;
   delete [] h_EC_Ele;               h_EC_Ele             = NULL;
#  endif
#  endif // FLU_SCHEME

#  if ( MODEL == HYDRO  &&  NEUTRINO_SCHEME == LEAKAGE )
   delete [] h_SrcLeakage_Radius;    h_SrcLeakage_Radius   = NULL;
   delete [] h_SrcLeakage_tau;       h_SrcLeakage_tau      = NULL;
   delete [] h_SrcLeakage_chi;       h_SrcLeakage_chi      = NULL;
   delete [] h_SrcLeakage_HeatFlux;  h_SrcLeakage_HeatFlux = NULL;
   delete [] h_SrcLeakage_HeatERms;  h_SrcLeakage_HeatERms = NULL;
   delete [] h_SrcLeakage_HeatEAve;  h_SrcLeakage_HeatEAve = NULL;

   SrcTerms.Leakage_Radius_DevPtr    = NULL;
   SrcTerms.Leakage_tau_DevPtr       = NULL;
   SrcTerms.Leakage_chi_DevPtr       = NULL;
   SrcTerms.Leakage_Heat_Flux_DevPtr = NULL;
   SrcTerms.Leakage_HeatE_Rms_DevPtr = NULL;
   SrcTerms.Leakage_HeatE_Ave_DevPtr = NULL;
#  endif

} // FUNCTION : End_MemFree_Fluid



#endif // #ifndef GPU
