#ifndef __CUFLU_DUALENERGY__
#define __CUFLU_DUALENERGY__



#include "CUFLU.h"

#if ( MODEL == HYDRO  &&  defined DUAL_ENERGY  &&  !defined SRHD )



// internal functions
#ifdef __CUDACC__
GPU_DEVICE
static real Hydro_DensPres2Dual( const real Dens, const real Pres, const real Gamma_m1 );
GPU_DEVICE
static real Hydro_DensDual2Pres( const real Dens, const real Dual, const real Gamma_m1,
                                 const bool CheckMinPres, const real MinPres );
#endif




//-------------------------------------------------------------------------------------------------------
// Function    :  Hydro_DualEnergyFix
// Description :  Correct the internal and total energies using the dual-energy formalism
//
// Note        :  1. Invoked by Hydro_FullStepUpdate(), InterpolateGhostZon(), ...
//                2. A floor value "MinPres" is applied to the corrected pressure if CheckMinPres is on
//                3  A floor value "TINY_NUMBER" is applied to the input dual-energy variable as well
//                4. Call-by-reference for "Etot, Dual, and DE_Status"
//                5. The dual-energy variable is determined by DUAL_ENERGY, which can be either
//                   DE_ENPY (entropy) or DE_EINT (internal energy)
//                   --> DE_ENPY: entropy = pressure / density^(Gamma-1)
//                       DE_EINT: internal_energy = pressure / (Gamma-1)
//                   --> Note that the entropy here is a monotonic function of entropy per volume
//                       instead of the real thermodynamic entropy (see Eqs. 48 and 49 in the Arepo code paper)
//                6. Fluid variables returned by this function are guaranteed to be consistent with each other
//                   --> It doesn't matter we use the dual-energy variable to correct Eint or vice versa,
//                       and it also holds even when the floor value is applied to pressure
//                7. Only support the Gamma-law EoS for now
//
// Parameter   :  Dens             : Mass density
//                MomX/Y/Z         : Momentum density
//                Etot             : Total energy density
//                Dual             : Dual-energy variable
//                DE_Status        : Assigned to (DE_UPDATED_BY_ETOT / DE_UPDATED_BY_DUAL / DE_UPDATED_BY_MIN_PRES)
//                                   to indicate whether this cell is updated by the total energy, dual-energy variable,
//                                   or pressure floor (MinPres)
//                Gamma_m1         : Adiabatic index - 1.0
//                _Gamma_m1        : 1.0/Gamma_m1
//                CheckMinPres     : Return Hydro_CheckMinPres()
//                                   --> In some cases we actually want to check if pressure becomes unphysical,
//                                       for which we don't want to enable this option
//                MinPres          : Minimum allowed pressure
//                PassiveFloor     : Bitwise flag to specify the passive scalars to be floored
//                DualEnergySwitch : if ( Eint/(Ekin+Emag) < DualEnergySwitch ) ==> correct Eint and Etot
//                                   else                                       ==> correct Dual
//                Emag             : Magnetic energy density (0.5*B^2) --> for MHD only
//
// Return      :  Etot, Dual, DE_Status
//-------------------------------------------------------------------------------------------------------
GPU_DEVICE
void Hydro_DualEnergyFix( const real Dens, const real MomX, const real MomY, const real MomZ,
                          real &Etot, real &Dual, char &DE_Status, const real Gamma_m1, const real _Gamma_m1,
                          const bool CheckMinPres, const real MinPres, const long PassiveFloor, const real DualEnergySwitch,
                          const real Emag )
{

   const bool CheckMinPres_No = false;
   const bool CheckMinEint_No = false;

// apply the dual-energy floor
   Dual = FMAX( Dual, TINY_NUMBER );


// calculate energies
// --> note that here Eint can even be negative due to numerical errors
// --> Enth (i.e., non-thermal energy) includes both kinetic and magnetic energies
   real Enth, Eint, Pres;

   Eint = Hydro_Con2Eint( Dens, MomX, MomY, MomZ, Etot, CheckMinEint_No, NULL_REAL, PassiveFloor, Emag,
                          NULL, NULL, NULL, NULL, NULL );
   Enth = Etot - Eint;


// determine whether or not to use the dual-energy variable to correct the total energy density
   if ( Eint/Enth < DualEnergySwitch )
   {
//    correct total energy
//    --> we will apply pressure floor later
      Pres      = Hydro_DensDual2Pres( Dens, Dual, Gamma_m1, CheckMinPres_No, NULL_REAL );
#     if   ( DUAL_ENERGY == DE_ENPY )
      Eint      = Pres*_Gamma_m1;
#     elif ( DUAL_ENERGY == DE_EINT )
      Eint      = Dual;
#     endif
      Etot      = Enth + Eint;
      DE_Status = DE_UPDATED_BY_DUAL;
   }

   else
   {
//    correct dual-energy variable
      Pres      = Eint*Gamma_m1;
#     if   ( DUAL_ENERGY == DE_ENPY )
      Dual      = Hydro_DensPres2Dual( Dens, Pres, Gamma_m1 );
#     elif ( DUAL_ENERGY == DE_EINT )
      Dual      = Eint;
#     endif
      DE_Status = DE_UPDATED_BY_ETOT;
   } // if ( Eint/Enth < DualEnergySwitch ) ... else ...


// apply pressure floor
   if ( CheckMinPres  &&  Pres < MinPres )
   {
      Pres = MinPres;
      Eint = Pres*_Gamma_m1;

//    ensure that both energy and dual-energy variable are consistent with the pressure floor
      Etot      = Enth + Eint;

#     if   ( DUAL_ENERGY == DE_ENPY )
      Dual      = Hydro_DensPres2Dual( Dens, Pres, Gamma_m1 );
#     elif ( DUAL_ENERGY == DE_EINT )
      Dual      = Eint;
#     endif
      DE_Status = DE_UPDATED_BY_MIN_PRES;
   }

} // FUNCTION : Hydro_DualEnergyFix



// Hydro_Con2Dual() is used by CPU only
#ifndef __CUDACC__
//-------------------------------------------------------------------------------------------------------
// Function    :  Hydro_Con2Dual
// Description :  Evaluate the dual-energy variable from the input fluid variables
//
// Note        :  1. Used by the dual-energy formalism
//                2. Invoked by Hydro_Init_ByFunction_AssignData(), Gra_Close(), Init_ByFile(), ...
//                3. Currently this function does NOT apply pressure floor when calling Hydro_Con2Pres()
//                   --> However, note that Hydro_DensPres2Dual() does apply a floor value (TINY_NUMBER) to the
//                       dual-energy variable
//
// Parameter   :  Dens              : Mass density
//                MomX/Y/Z          : Momentum density
//                Engy              : Total energy density
//                Emag              : Magnetic energy density (0.5*B^2) --> for MHD only
//                Passive           : Passive scalars
//                EoS_DensEint2Entr : EoS routine to compute the gas entropy
//                EoS_AuxArray_*    : Auxiliary arrays for EoS_DensEint2Entr()
//                EoS_Table         : EoS tables
//                PassiveFloor      : Bitwise flag to specify the passive scalars to be floored
//
// Return      :  Dual
//-------------------------------------------------------------------------------------------------------
real Hydro_Con2Dual( const real Dens, const real MomX, const real MomY, const real MomZ, const real Engy,
                     const real Emag, const real Passive[], const EoS_DE2S_t EoS_DensEint2Entr,
                     const double EoS_AuxArray_Flt[], const int EoS_AuxArray_Int[],
                     const real *const EoS_Table[EOS_NTABLE_MAX], const long PassiveFloor )
{

// calculate the dual-energy variable
   const bool CheckMin_No = false;
   real Dual;

#  if   ( DUAL_ENERGY == DE_ENPY )
   Dual = Hydro_Con2Entr( Dens, MomX, MomY, MomZ, Engy, Passive, CheckMin_No, NULL_REAL, PassiveFloor,
                          Emag, EoS_DensEint2Entr, EoS_AuxArray_Flt, EoS_AuxArray_Int, EoS_Table );
#  elif ( DUAL_ENERGY == DE_EINT )
   Dual = Hydro_Con2Eint( Dens, MomX, MomY, MomZ, Engy, CheckMin_No, NULL_REAL, PassiveFloor, Emag,
                          NULL, NULL, NULL, NULL, NULL );
#  endif

   return Dual;

} // FUNCTION : Hydro_Con2Dual
#endif // ifndef __CUDACC__



//-------------------------------------------------------------------------------------------------------
// Function    :  Hydro_DensPres2Dual
// Description :  Evaluate the dual-energy variable from the input density and pressure
//
// Note        :  1. Used by the dual-energy formalism
//                2. Invoked by Hydro_Con2Dual() and Hydro_DualEnergyFix()
//                   --> This function is invoked by both CPU and GPU codes
//                3. A floor value (TINY_NUMBER) is applied to the returned value
//
// Parameter   :  Dens     : Mass density
//                Pres     : Pressure
//                Gamma_m1 : Adiabatic index - 1.0
//
// Return      :  Dual
//-------------------------------------------------------------------------------------------------------
GPU_DEVICE
real Hydro_DensPres2Dual( const real Dens, const real Pres, const real Gamma_m1 )
{

   real Dual;

// calculate the dual-energy variable
#  if   ( DUAL_ENERGY == DE_ENPY )
   Dual = Pres*POW( Dens, -Gamma_m1 );
#  elif ( DUAL_ENERGY == DE_EINT )
   Dual = Pres / Gamma_m1;
#  endif

// apply a floor value
   Dual = FMAX( Dual, TINY_NUMBER );

   return Dual;

} // FUNCTION : Hydro_DensPres2Dual



//-------------------------------------------------------------------------------------------------------
// Function    :  Hydro_DensDual2Pres
// Description :  Evaluate the gas pressure from the input density and dual-energy variable
//
// Note        :  1. Used by the dual-energy formalism
//                2. Invoked by Hydro_DualEnergyFix(), Flu_Close(), Hydro_Aux_Check_Negative(), and Flu_FixUp()
//                   --> This function is invoked by both CPU and GPU codes
//                3. A floor value "MinPres" is applied to the returned pressure if CheckMinPres is on
//
// Parameter   :  Dens         : Mass density
//                Dual         : Dual-energy variable
//                Gamma_m1     : Adiabatic index - 1.0
//                CheckMinPres : Return Hydro_CheckMinPres()
//                               --> In some cases we actually want to check if pressure becomes unphysical,
//                                   for which we don't want to enable this option
//                MinPres      : Minimum allowed pressure
//
// Return      :  Pres
//-------------------------------------------------------------------------------------------------------
GPU_DEVICE
real Hydro_DensDual2Pres( const real Dens, const real Dual, const real Gamma_m1,
                          const bool CheckMinPres, const real MinPres )
{

   real Pres;

// calculate pressure
#  if   ( DUAL_ENERGY == DE_ENPY )
   Pres = Dual*POW( Dens, Gamma_m1 );
#  elif ( DUAL_ENERGY == DE_EINT )
   Pres = Dual*Gamma_m1;
#  endif

// apply a floor value
   if ( CheckMinPres )  Pres = Hydro_CheckMinPres( Pres, MinPres );

   return Pres;

} // FUNCTION : Hydro_DensDual2Pres



# if ( FLU_SCHEME == MHM_RP  &&  DUAL_ENERGY == DE_EINT )
//-------------------------------------------------------------------------------------------------------
// Function    : Hydro_DualEnergy_AdiabaticWork_HalfStep_MHM_RP
//
// Description : Add the adiabatic work term to update the dual energy for the half-step solution of MHM_RP
//
// Note        : 1. MHM should not use this function
//               2. Work w/ and w/o MHD
//               3. Invoked by Hydro_RiemannPredict()
//
// Reference   : [1] Bryan et al., ApJS 211, 19 (2012); doi:10.1088/0067-0049/211/2/19
//               [2] A simple dual implementation to track pressure accurately, S. Li, Astronum Proceeding, 385, 273 (2007)
//
// Parameter   : OneCell     : Single-cell fluid array to store the updated cell-centered dual energy
//               g_ConVar_In : Array storing the input conserved variables
//               g_Flux_Half : Array storing the input face-centered fluxes
//                             --> Accessed with the stride didx_flux
//               idx_in      : Index of accessing g_ConVar_In[]
//               didx_in     : Index increment of g_ConVar_In[]
//               idx_flux    : Index of accessing g_flux_Half[]
//               didx_flux   : Index increment of g_Flux_Half[]
//               dt_dh2      : 0.5 * dt / dh
//               EoS         : EoS object
//
// Return      : OneCell[DUAL]
//-------------------------------------------------------------------------------------------------------
GPU_DEVICE
void Hydro_DualEnergy_AdiabaticWork_HalfStep_MHM_RP( real OneCell[NCOMP_TOTAL_PLUS_MAG],
                                                     const real g_ConVar_In[][ CUBE(FLU_NXT) ],
                                                     const real g_Flux_Half[][NCOMP_TOTAL_PLUS_MAG][ CUBE(N_FC_FLUX) ],
                                                     const int idx_in, const int didx_in[3],
                                                     const int idx_flux, const int didx_flux[3],
                                                     const real dt_dh2, const EoS_t *EoS )
{
// 1. calculate the dual energy pressure
   real Passive[NCOMP_PASSIVE];
#  if ( NCOMP_PASSIVE > 0 )
   for (int v=0; v<NCOMP_PASSIVE; v++)  Passive[v] = g_ConVar_In[NCOMP_FLUID+v][idx_in];
#  endif

   const real pDual_old = EoS->DensEint2Pres_FuncPtr( g_ConVar_In[DENS][idx_in], g_ConVar_In[DUAL][idx_in], Passive,
                                                      EoS->AuxArrayDevPtr_Flt, EoS->AuxArrayDevPtr_Int, EoS->Table );


// 2. compute \div V using the upwind data; reference: [2]
   real div_V[3];

   for (int d=0; d<3; d++)
   {
#     ifdef MHD
      const real DensFlux_L = g_Flux_Half[d][DENS][ idx_flux - didx_flux[d] ];
      const real DensFlux_R = g_Flux_Half[d][DENS][ idx_flux                ];
#     else
      const real DensFlux_L = g_Flux_Half[d][DENS][ idx_flux                ];
      const real DensFlux_R = g_Flux_Half[d][DENS][ idx_flux + didx_flux[d] ];
#     endif

      div_V[d]  = ( DensFlux_R > (real)0.0 ) ?
                  ( DensFlux_R / g_ConVar_In[DENS][ idx_in              ] ) :
                  ( DensFlux_R / g_ConVar_In[DENS][ idx_in + didx_in[d] ] );

      div_V[d] -= ( DensFlux_L > (real)0.0 ) ?
                  ( DensFlux_L / g_ConVar_In[DENS][ idx_in - didx_in[d] ] ) :
                  ( DensFlux_L / g_ConVar_In[DENS][ idx_in              ] );
   } // for (int d=0; d<3; d++)


// 3. unconditionally update the dual energy
   OneCell[DUAL] -= pDual_old*dt_dh2*( div_V[0] + div_V[1] + div_V[2] );

} // FUNCTION : Hydro_DualEnergy_AdiabaticWork_HalfStep_MHM_RP
#endif // #if (  FLU_SCHEME == MHM_RP  &&  DUAL_ENERGY == DE_EINT  )



#if (  ( FLU_SCHEME == MHM_RP || FLU_SCHEME == MHM  || FLU_SCHEME == CTU )  &&  DUAL_ENERGY == DE_EINT  )
//-------------------------------------------------------------------------------------------------------
// Function    :  Hydro_DualEnergy_AdiabaticWork_FullStep
//
// Description :  Add the adiabatic work term to update the dual energy for the full-step solution of MHM_RP/MHM
//
// Note        :  1. Shared by both MHM and MHM_RP (but it hasn't been tested for MHM yet)
//                2. Work w/ and w/o MHD
//                3. Invoked by CPU/CUFLU_FluidSolver_MHM()
//
// Reference   :  [1] Bryan et al., ApJS 211, 19 (2012); doi:10.1088/0067-0049/211/2/19
//                [2] A simple dual implementation to track pressure accurately, S. Li, Astronum Proceeding, 385, 273 (2007)
//
// Parameter   :  Edual          : Dual energy to be updated
//                g_PriVar_Half  : Array storing the input cell-centered conserved variables
//                                 --> Accessed with the stride N_HF_VAR
//                                 --> Although its actually allocated size is FLU_NXT^3 since it points to g_PriVar_1PG[]
//                g_Flux         : Array storing the input face-centered fluxes
//                                 --> Accessed with the array stride N_FL_FLUX even though its actually
//                                     allocated size is N_FC_FLUX^3
//                g_FC_Var       : Array storing the input face-centered conserved variables
//                                 --> Accessed with the array stride N_FC_VAR^3
//                FracPassive    : true --> input passive scalars are mass fraction instead of density
//                NFrac          : Number of passive scalars for the option "FracPassive"
//                FracIdx        : Target variable indices for the option "FracPassive"
//                dt             : Time interval to advance solution
//                dh             : Cell size
//                EoS            : EoS object
//                idx_out        : Array index associated with Ecr
//
// Return      :  Edual
//-------------------------------------------------------------------------------------------------------
GPU_DEVICE
void Hydro_DualEnergy_AdiabaticWork_FullStep( real &Edual,
                                              const real g_PriVar_Half[][ CUBE(FLU_NXT) ],
                                              const real g_Flux[][NCOMP_TOTAL_PLUS_MAG][ CUBE(N_FC_FLUX) ],
                                              const real g_FC_Var[][NCOMP_TOTAL_PLUS_MAG][ CUBE(N_FC_VAR) ],
                                              const bool FracPassive, const int NFrac, const int FracIdx[],
                                              const real dt, const real dh, const EoS_t *EoS, const int idx_out )
{

   const int  size_ij      = SQR(PS2);
   const int  didx_flux[3] = { 1, N_FL_FLUX, SQR(N_FL_FLUX) };
   const int  didx_fc[3]   = { 1, N_FC_VAR,  SQR(N_FC_VAR)  };
   const real dt_dh        = dt/dh;

   {
//    index of the output array
      const int i_out    = idx_out % PS2;
      const int j_out    = idx_out % size_ij / PS2;
      const int k_out    = idx_out / size_ij;

//    index of the flux array
//    --> for MHD, one additional flux is evaluated along each transverse direction for computing the CT electric field
#     ifdef MHD
      const int i_flux   = i_out + 1;
      const int j_flux   = j_out + 1;
      const int k_flux   = k_out + 1;
#     else
      const int i_flux   = i_out;
      const int j_flux   = j_out;
      const int k_flux   = k_out;
#     endif
      const int idx_flux = IDX321( i_flux, j_flux, k_flux, N_FL_FLUX, N_FL_FLUX );

//    index of the g_PriVar_Half array
#     if ( FLU_SCHEME == CTU || FLU_SCHEME == MHM )
      const int i_hf     = i_out + FLU_GHOST_SIZE;
      const int j_hf     = j_out + FLU_GHOST_SIZE;
      const int k_hf     = k_out + FLU_GHOST_SIZE;
      const int idx_hf   = IDX321( i_hf, j_hf, k_hf, FLU_NXT, FLU_NXT );
#     else // MHM_RP
      const int i_hf     = i_out + (N_HF_VAR-PS2)/2;
      const int j_hf     = j_out + (N_HF_VAR-PS2)/2;
      const int k_hf     = k_out + (N_HF_VAR-PS2)/2;
      const int idx_hf   = IDX321( i_hf, j_hf, k_hf, N_HF_VAR, N_HF_VAR );
#     endif

//    index of the face-centered variables
      const int i_fc     = i_out + 1;
      const int j_fc     = j_out + 1;
      const int k_fc     = k_out + 1;
      const int idx_fc   = IDX321( i_fc, j_fc, k_fc, N_FC_VAR, N_FC_VAR );

//    1. calculate the pressure
      real Passive[NCOMP_PASSIVE];
#     if ( NCOMP_PASSIVE > 0 )
      for (int v=0; v<NCOMP_PASSIVE; v++)   Passive[v] = g_PriVar_Half[NCOMP_FLUID+v][idx_hf];
      if ( FracPassive )
         for (int v=0; v<NFrac; v++)   Passive[ FracIdx[v] ] *= g_PriVar_Half[DENS][idx_hf];
#     endif

      const real pDual_half = EoS->DensEint2Pres_FuncPtr( g_PriVar_Half[DENS][idx_hf], g_PriVar_Half[DUAL][idx_hf], Passive,
                                                          EoS->AuxArrayDevPtr_Flt, EoS->AuxArrayDevPtr_Int, EoS->Table );

//    2. compute \div V using the upwind data; reference: [2]
      real div_V[3];

      for (int d=0; d<3; d++)
      {
         const int faceL = 2*d;
         const int faceR = faceL+1;

#        ifdef MHD
         const real DensFlux_L = g_Flux[d][DENS][ idx_flux - didx_flux[d] ];
         const real DensFlux_R = g_Flux[d][DENS][ idx_flux                ];
#        else
         const real DensFlux_L = g_Flux[d][DENS][ idx_flux                ];
         const real DensFlux_R = g_Flux[d][DENS][ idx_flux + didx_flux[d] ];
#        endif

         div_V[d]  = ( DensFlux_R > (real)0.0 ) ?
                     ( DensFlux_R / g_FC_Var[faceR][DENS][ idx_fc              ] ) :
                     ( DensFlux_R / g_FC_Var[faceL][DENS][ idx_fc + didx_fc[d] ] );

         div_V[d] -= ( DensFlux_L > (real)0.0 ) ?
                     ( DensFlux_L / g_FC_Var[faceR][DENS][ idx_fc - didx_fc[d] ] ) :
                     ( DensFlux_L / g_FC_Var[faceL][DENS][ idx_fc              ] );
      } // for (int d=0; d<3; d++)


//    3. calculate the adiabatic work
      Edual -= pDual_half*dt_dh*( div_V[0] + div_V[1] + div_V[2] );

   }

} // FUNCTION : Hydro_DualEnergy_AdiabaticWork_FullStep
#endif // #if (  ( FLU_SCHEME == MHM_RP || FLU_SCHEME == MHM  || FLU_SCHEME == CTU )  &&  DUAL_ENERGY == DE_EINT  )



#endif // ( MODEL == HYDRO  &&  defined DUAL_ENERGY  &&  !defined SRHD )



#endif // #ifndef __CUFLU_DUALENERGY__
