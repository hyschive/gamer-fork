#include "CUFLU.h"
#include "NuclearEoS.h"
#include "Src_Leakage.h"



static const real ONETHIRD               = 1.0 / 3.0;
static const real ONESIXTH               = 1.0 / 6.0;
#ifdef FLOAT8
static const real EXP_OVERFLOW_THRESHOLD = 709.782712893384;
#else
static const real EXP_OVERFLOW_THRESHOLD = 88.722839;
#endif


#ifndef __CUDACC__

// not apply any correction to the leakage profiles for the stability issue
//#define LEAKAGE_PROF_CORR

extern double CCSN_Leakage_EAve [NType_Neutrino];
extern double CCSN_Leakage_RadNS[NType_Neutrino];

#endif


GPU_DEVICE
void Src_Leakage_ComputeLeak( const real Dens_Code, const real Temp_Kelv, const real Ye, const real chi[], const real tau[],
                              const real Heat_Flux[], const real *Heat_ERms, const real *Heat_EAve,
                              real *dEdt, real *dYedt, real *Lum, real *Heat, real *NetHeat,
                              const bool NuHeat, const real NuHeat_Fac, const real UNIT_D, const EoS_t *EoS );

GPU_DEVICE static
real Compute_FermiIntegral( const int Order, const real eta );




#ifndef __CUDACC__
//-------------------------------------------------------------------------------------------------------
// Function    :  Src_Leakage_ComputeTau
// Description :  Compute the neutrino optical depth using the Ruffert and Rosswog approaches,
//                as well as the neutrino flux and mean/RMS neutrino energy for neutrino heating
//
// Note        :  1. Invoked by Src_WorkBeforeMajorFunc_Leakage()
//                2. The stored Heat_Flux is scaled by Const_hc_MeVcm_CUBE to avoid overflow
//                3. Add "#ifndef __CUDACC__" since this routine is only useful on CPU
//                4. Ref: M. Rufferrt, H.-Th. Janka, G. Schaefer, 1996, A&A, 311, 532 (arXiv: 9509006)
//                        S. Rosswog & M. Liebendoerfer, 2003, MNRAS, 342, 673 (arXiv: 0302301)
//                        E. O'Connor & C. D. Ott, 2010, Class. Quantum Grav., 27, 114103 (arXiv: 0912.2393)
//
// Parameter   :  Ray       : Profile_t object array stores the profiles of all rays
//                            --> Density     : in code unit
//                            --> Temperature : in Kelvin
//                            --> Ye          : dimensionless
//                Edge      : Edge of bins in the radial direction in code unit
//                NRadius   : Number of bins in the radial    direction
//                NTheta    : Number of bins in the polar     direction
//                NPhi      : Number of bins in the azimuthal direction
//                tau_Ruff  : Array to store the opacity                          calculated via the Ruffert scheme
//                chi_Ross  : Array to store the energy-independent optical depth calculated via the Rosswog scheme
//                Heat_Flux : Array to store the      neutrino flux
//                Heat_ERms : Array to store the rms  neutrino energy at neutrino sphere
//                Heat_EAve : Array to store the mean neutrino energy at neutrino sphere
//
// Return      :  tau_Ruff, chi_Ross, Heat_Flux, Heat_ERms, Heat_EAve
//-------------------------------------------------------------------------------------------------------
void Src_Leakage_ComputeTau( Profile_t *Ray[], double *Edge,
                             const int NRadius, const int NTheta, const int NPhi,
                             real *tau_Ruff, real *chi_Ross, real *Heat_Flux,
                             real *Heat_ERms, real *Heat_EAve )
{

#  ifdef FLOAT8
   const double Tolerance_Leak = 1.0e-10;
#  else
   const double Tolerance_Leak = 1.0e-5;
#  endif

#  if ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )
   const real   EoS_TempMin_Kelv = POW( (real)10.0, h_EoS_Table[NUC_TAB_TORE     ][0] ) / Kelvin2MeV;
#  else
   const real   EoS_TempMin_Kelv = POW( (real)10.0, h_EoS_Table[NUC_TAB_EORT_MODE][0] ) / Kelvin2MeV;
#  endif

#  ifdef LEAKAGE_PROF_CORR
   const real   Dens_CorrThresh_Code = 1.0e3 / UNIT_D;
#  endif

   const bool   NuHeat     = SrcTerms.Leakage_NuHeat;
   const real   NuHeat_Fac = SrcTerms.Leakage_NuHeat_Fac;


// prepare the line element, surface area at bin center, and bin volume
   double Edge_CGS[NRadius+1], BinWidth[NRadius], BinSurfArea[NRadius], BinVol[NRadius];

   for (int i=0; i<NRadius+1; i++)   Edge_CGS[i] = Edge[i] * UNIT_L;

   for (int i=0; i<NRadius; i++)
   {
      BinWidth   [i] = Edge_CGS[i+1] - Edge_CGS[i];
      BinSurfArea[i] = 4.0 * M_PI * SQR(  0.5 * ( Edge_CGS[i] + Edge_CGS[i+1] )  ) / double( NTheta * NPhi );
      BinVol     [i] = 4.0 * M_PI * (  CUBE( Edge_CGS[i+1] ) - CUBE( Edge_CGS[i] )  )
                     / ( 3.0 * double( NTheta * NPhi ) );
   }


// allocate memory for the per-thread arrays
#  ifdef OPENMP
   const int NT = OMP_NTHREAD;
#  else
   const int NT = 1;
#  endif

   double EAve [NType_Neutrino] = { 0.0 };
   double RadNS[NType_Neutrino] = { 0.0 };

   long   **NCell=NULL;
   real   **Dens_Code=NULL, **Temp_Kelv=NULL, **Ye=NULL, **Eint_Code=NULL;
   real   **dEdt=NULL, **dYedt=NULL;
   double **Dens_CGS=NULL, **Temp_MeV=NULL;
   double **x_h=NULL, **x_n=NULL, **x_p=NULL, **abar=NULL, **zbar=NULL;
   double **eta_e=NULL, **eta_p=NULL, **eta_n=NULL, **eta_hat=NULL;
   double **kappa_scat_n_fac=NULL, **kappa_scat_p_fac=NULL, **kappa_abs_fac=NULL;
   double **OMP_EAve, **OMP_RadNS;

   double ***eta_nu=NULL, ***eta_nu_loc=NULL, ***kappa_tot=NULL, ***kappa_tot_old=NULL;
   double ***zeta=NULL, ***tau=NULL, ***chi=NULL;

   Aux_AllocateArray2D( NCell,            NT, NRadius );
   Aux_AllocateArray2D( Dens_Code,        NT, NRadius );
   Aux_AllocateArray2D( Temp_Kelv,        NT, NRadius );
   Aux_AllocateArray2D( Ye,               NT, NRadius );
   Aux_AllocateArray2D( Eint_Code,        NT, NRadius );
   Aux_AllocateArray2D( Dens_CGS,         NT, NRadius );
   Aux_AllocateArray2D( Temp_MeV,         NT, NRadius );
   Aux_AllocateArray2D( x_h,              NT, NRadius );
   Aux_AllocateArray2D( x_n,              NT, NRadius );
   Aux_AllocateArray2D( x_p,              NT, NRadius );
   Aux_AllocateArray2D( abar,             NT, NRadius );
   Aux_AllocateArray2D( zbar,             NT, NRadius );
   Aux_AllocateArray2D( eta_e,            NT, NRadius );
   Aux_AllocateArray2D( eta_p,            NT, NRadius );
   Aux_AllocateArray2D( eta_n,            NT, NRadius );
   Aux_AllocateArray2D( eta_hat,          NT, NRadius );
   Aux_AllocateArray2D( kappa_scat_n_fac, NT, NRadius );
   Aux_AllocateArray2D( kappa_scat_p_fac, NT, NRadius );
   Aux_AllocateArray2D( kappa_abs_fac,    NT, NRadius );
   Aux_AllocateArray2D( dEdt,             NT, NRadius );
   Aux_AllocateArray2D( dYedt,            NT, NRadius );

   Aux_AllocateArray2D( OMP_EAve,         NT, NType_Neutrino );
   Aux_AllocateArray2D( OMP_RadNS,        NT, NType_Neutrino );

   Aux_AllocateArray3D( eta_nu,           NT, NRadius, NType_Neutrino );
   Aux_AllocateArray3D( tau,              NT, NRadius, NType_Neutrino );
   Aux_AllocateArray3D( kappa_tot,        NT, NRadius, NType_Neutrino );
   Aux_AllocateArray3D( kappa_tot_old,    NT, NRadius, NType_Neutrino );
   Aux_AllocateArray3D( eta_nu_loc,       NT, NRadius, NType_Neutrino );
   Aux_AllocateArray3D( zeta,             NT, NRadius, NType_Neutrino );
   Aux_AllocateArray3D( chi,              NT, NRadius, NType_Neutrino );


// set up the start and end indices for each rank
   const int  NRay       = NTheta * NPhi;
   const int  NRay_Rank  = NRay / MPI_NRank;
   const int  NRay_Mod   = NRay % MPI_NRank;
   const int  IRay_start = MPI_Rank   * NRay_Rank + (  ( MPI_Rank < NRay_Mod ) ? MPI_Rank : NRay_Mod  );
   const int  IRay_end   = IRay_start + NRay_Rank + (  ( MPI_Rank < NRay_Mod ) ? 1        : 0         );


// recast leakage arrays and reset them to 0.0
   typedef real (*vla_2D)      [NType_Neutrino];
   typedef real (*vla_3D)[NRay][NType_Neutrino];

   vla_3D tau_Ruff_3D  = ( vla_3D ) tau_Ruff;
   vla_3D chi_Ross_3D  = ( vla_3D ) chi_Ross;
   vla_3D Heat_Flux_3D = ( vla_3D ) Heat_Flux;
   vla_2D Heat_ERms_2D = ( vla_2D ) Heat_ERms;
   vla_2D Heat_EAve_2D = ( vla_2D ) Heat_EAve;

   for (int i=0; i<NRadius; i++) {
   for (int j=0; j<NRay;    j++) {
//    skip leakage rays handled by this Rank
      if ( j >= IRay_start  &&  j < IRay_end )   continue;

      for (int k=0; k<NType_Neutrino; k++)
      {
         tau_Ruff_3D [i][j][k] = 0.0;
         chi_Ross_3D [i][j][k] = 0.0;
         Heat_Flux_3D[i][j][k] = 0.0;
         Heat_ERms_2D   [j][k] = 0.0;
         Heat_EAve_2D   [j][k] = 0.0;
      }
   }}


#  pragma omp parallel
   {
#     ifdef OPENMP
      const int TID = omp_get_thread_num();
#     else
      const int TID = 0;
#     endif

//    initialize arrays
      for (int k=0; k<NType_Neutrino; k++)
      {
         OMP_EAve [TID][k] = 0.0;
         OMP_RadNS[TID][k] = 0.0;
      }

//    initialize the last element in dEdt and dYedt to 0.0,
//    which is not updated in the following code, to avoid incorrect output in Leakage_Lum_*
      dEdt [TID][NRadius-1] = 0.0;
      dYedt[TID][NRadius-1] = 0.0;


#     pragma omp for schedule( runtime )
      for (int j=IRay_start; j<IRay_end; j++)
      {
//       (1-1) copy data
         for (int i=0; i<NRadius; i++)
         {
            const int Idx = i + j * NRadius;

            NCell    [TID][i] = Ray[0]->NCell[Idx];
            Dens_Code[TID][i] = Ray[0]->Data [Idx];
            Temp_Kelv[TID][i] = Ray[1]->Data [Idx];
            Ye       [TID][i] = Ray[2]->Data [Idx];

            Dens_CGS [TID][i] = Dens_Code[TID][i] * UNIT_D;
            Temp_MeV [TID][i] = Temp_Kelv[TID][i] * Kelvin2MeV;

            if ( NCell[TID][i] == 0L )
               Aux_Error( ERROR_INFO, "empty bin in Ray (%d) in %s !!\n", j, __FUNCTION__ );
         }

#        ifdef LEAKAGE_PROF_CORR
//       (1-2) fix potential undershoots near shock, adopted from FLASH
         for (int i=1; i<NRadius-1; i++)
         {
            if ( Dens_Code[TID][i] < Dens_CorrThresh_Code )
               Dens_Code[TID][i] = 0.5 * ( Dens_Code[TID][i-1] + Dens_Code[TID][i+1] );

//          make sure the temperature is not within a factor of 2 of the table lower bound
            if ( Temp_Kelv[TID][i] < 2.0 * EoS_TempMin_Kelv )
               Temp_Kelv[TID][i] = 0.5 * ( Temp_Kelv[TID][i-1] + Temp_Kelv[TID][i+1] );

//          make sure the temperature does not drop by more than 50% from one bin to the next
            if ( Temp_Kelv[TID][i] < 0.5 * Temp_Kelv[TID][i-1] )
               Temp_Kelv[TID][i] = 0.5 * ( Temp_Kelv[TID][i-1] + Temp_Kelv[TID][i+1] );

            if (  ( Ye[TID][i] < 0.0 )  ||  ( Ye[TID][i] > 0.53 )  )
               Ye[TID][i] = 0.5 * ( Ye[TID][i-1] + Ye[TID][i+1] );
         }
#        endif

//       (1-3) unit conversion
         for (int i=0; i<NRadius; i++)
         {
            Dens_CGS[TID][i] = Dens_Code[TID][i] * UNIT_D;
            Temp_MeV[TID][i] = Temp_Kelv[TID][i] * Kelvin2MeV;
         }


//       (2) invoke the nuclear EoS solver to obtain EoS variables
         const int  NTarget = 9;
               real In_Flt[4], Out[NTarget+1];
               int  In_Int[NTarget+1] = { NTarget, NUC_VAR_IDX_EORT,
                                          NUC_VAR_IDX_MUE, NUC_VAR_IDX_MUP, NUC_VAR_IDX_MUN,
                                          NUC_VAR_IDX_XH, NUC_VAR_IDX_XN, NUC_VAR_IDX_XP,
                                          NUC_VAR_IDX_ABAR, NUC_VAR_IDX_ZBAR };

         for (int i=0; i<NRadius; i++)
         {
            In_Flt[0] = Dens_Code[TID][i];
            In_Flt[1] = Temp_Kelv[TID][i];
            In_Flt[2] = Ye       [TID][i];
            In_Flt[3] = Temp_Kelv[TID][i];

            EoS_General_CPUPtr( NUC_MODE_TEMP, Out, In_Flt, In_Int, EoS_AuxArray_Flt, EoS_AuxArray_Int, h_EoS_Table );

//          check the obtained EoS variables are not NaN
            bool EOS_IsFinite = true;

            for (int n=0; n<NTarget; n++)
               if ( Out[n] != Out[n] )   { EOS_IsFinite = false;   break; }

            if ( ! EOS_IsFinite )
               Aux_Error( ERROR_INFO, "unphysical thermal variable in %s, Dens=%13.7e code units, Temp=%13.7e Kelvin, Ye=%13.7e, Mode %d\n",
                          __FUNCTION__, In_Flt[0], In_Flt[1], In_Flt[2], NUC_MODE_TEMP );

            Eint_Code[TID][i] = Out[0];
            eta_e    [TID][i] = Out[1] / Temp_MeV[TID][i];
            eta_p    [TID][i] = Out[2] / Temp_MeV[TID][i];
            eta_n    [TID][i] = Out[3] / Temp_MeV[TID][i];
            x_h      [TID][i] = Out[4];
            x_n      [TID][i] = Out[5];
            x_p      [TID][i] = Out[6];
            abar     [TID][i] = Out[7];
            zbar     [TID][i] = Out[8];
            eta_hat  [TID][i] = eta_n[TID][i] - eta_p[TID][i] - Const_Qnp / Temp_MeV[TID][i];

//          set to degeneracy parameter at chemical equilibrium with the stellar medium,
//          which will be updated after the Ruffert scheme
//          --> not need to include mass difference term (Q/T)
//              since we have rest masses in the chemical potentials
            eta_nu[TID][i][0] = eta_e[TID][i] - eta_n[TID][i] + eta_p[TID][i]; // (A5) in Ruffert et. al (1996)
            eta_nu[TID][i][1] = -eta_nu[TID][i][0];                            // (A5) in Ruffert et. al (1996)
            eta_nu[TID][i][2] = 0.0;
         } // for (int i=0; i<NRadius; i++)


//       (3) compute the optical depth using the Ruffert scheme
         double Yn, Yp, Ynp, Ypn, fac1, fac2;
         double kappa_abs_n, kappa_abs_p;
         double kappa_scat_n[NType_Neutrino], kappa_scat_p[NType_Neutrino];

//       (3-1) set up the initial guess of opacity and constant factors
         for (int i=0; i<NRadius; i++)
         {
            for (int k=0; k<NType_Neutrino; k++)
            {
               kappa_tot[TID][i][k] = 1.0e-5; // cm^-1
            }

            fac1 = Dens_CGS[TID][i] * SQR( Temp_MeV[TID][i] );

            kappa_scat_n_fac[TID][i] = Const_Ruffert_kappa_sn * fac1; // (A6)
            kappa_scat_p_fac[TID][i] = Const_Ruffert_kappa_sp * fac1; // (A6)
            kappa_abs_fac   [TID][i] = Const_Ruffert_kappa_a  * fac1; // (A11) and (A12)
         }

//       (3-2) iteratively obtain the converged optical depth
         const int  NIter_Max   = 200;
               int  NIter       = 0;
               bool IsConverged = false;

         while ( ! IsConverged  &&  NIter < NIter_Max )
         {
//          back up the used opacity
            for (int i=0; i<NRadius;        i++) {
            for (int k=0; k<NType_Neutrino; k++) {
               kappa_tot_old[TID][i][k] = kappa_tot[TID][i][k];
            }}

//          integrate opacity to get optical depth at the bin center; (A20)
//          --> assume the optical depth is zero at the right edge of outermost bin
            for (int k=0; k<NType_Neutrino; k++)
            {
               tau[TID][NRadius-1][k] = 0.5 * kappa_tot[TID][NRadius-1][k] * BinWidth[NRadius-1];

               for (int i=NRadius-2; i>=0; i--)
                  tau[TID][i][k] = tau[TID][i+1][k]
                                 + 0.5 * kappa_tot[TID][i  ][k] * BinWidth[i  ]
                                 + 0.5 * kappa_tot[TID][i+1][k] * BinWidth[i+1];
            }

//          use new optical depth to update opacity
            for (int i=0; i<NRadius; i++)
            {
//             compute neutrino degeneracy using the new optical depth
//             --> the eta^0_nu is set to 0.0
               eta_nu_loc[TID][i][0] = eta_nu[TID][i][0] * (  1.0 - exp( -tau[TID][i][0] )  ); // (A3)
               eta_nu_loc[TID][i][1] = eta_nu[TID][i][1] * (  1.0 - exp( -tau[TID][i][1] )  ); // (A4)
               eta_nu_loc[TID][i][2] = 0.0;                                                    // (A2)

//             number fraction with Pauli blocking effects (Y_NN), assumed completely dissociated
               Yn = ( 1.0 - Ye[TID][i] ) / (  1.0 + TwoThirds * fmax( 0.0, eta_n[TID][i] )  ); // (A8)
               Yp =         Ye[TID][i]   / (  1.0 + TwoThirds * fmax( 0.0, eta_p[TID][i] )  ); // (A8)

//             number fraction with Fermion blocking effects
               if ( x_h[TID][i] < 0.5 )
               {
                  fac1 = exp( -eta_hat[TID][i] );

                  Ynp = ( 2.0 * Ye[TID][i] - 1.0 ) / ( fac1 - 1.0 ); // (A13)
                  Ypn = fac1 * Ynp;                                  // (A14)
               }

               else
               {
                  Ynp = x_n[TID][i];
                  Ypn = x_p[TID][i];
               }

               Ynp = fmax( 0.0, Ynp );
               Ypn = fmax( 0.0, Ypn );

//             update opacity
//             --> electron neutrino (nu_e)
               fac1 = Compute_FermiIntegral( 53, eta_nu_loc[TID][i][0] );
               fac2 = 1.0 + exp(   eta_e[TID][i] - Compute_FermiIntegral( 54, eta_nu_loc[TID][i][0] )  ); // (A15)

               kappa_scat_n[0] = kappa_scat_n_fac[TID][i] * Yn  * fac1;        // (A6)
               kappa_scat_p[0] = kappa_scat_p_fac[TID][i] * Yp  * fac1;        // (A6)
               kappa_abs_n     = kappa_abs_fac   [TID][i] * Ynp * fac1 / fac2; // (A11)

//             --> electron anti-neutrino (nu_a)
               fac1 = Compute_FermiIntegral( 53, eta_nu_loc[TID][i][1] );
               fac2 = 1.0 + exp(  -eta_e[TID][i] - Compute_FermiIntegral( 54, eta_nu_loc[TID][i][1] )  ); // (A16)

               kappa_scat_n[1] = kappa_scat_n_fac[TID][i] * Yn  * fac1;        // (A6)
               kappa_scat_p[1] = kappa_scat_p_fac[TID][i] * Yp  * fac1;        // (A6)
               kappa_abs_p     = kappa_abs_fac   [TID][i] * Ypn * fac1 / fac2; // (A11)

//             --> heavy-lepton neutrino (nu_x)
               fac1 = Compute_FermiIntegral( 53, eta_nu_loc[TID][i][2] );

               kappa_scat_n[2] = kappa_scat_n_fac[TID][i] * Yn  * fac1; // (A6)
               kappa_scat_p[2] = kappa_scat_p_fac[TID][i] * Yp  * fac1; // (A6)

//             sum each contribution
               kappa_tot[TID][i][0] = kappa_scat_p[0] + kappa_scat_n[0] + kappa_abs_n; // nu_e; (A17)
               kappa_tot[TID][i][1] = kappa_scat_p[1] + kappa_scat_n[1] + kappa_abs_p; // nu_a; (A18)
               kappa_tot[TID][i][2] = kappa_scat_p[2] + kappa_scat_n[2];               // nu_x; (A19)
            } // for (int i=0; i<NRadius; i++)


//          check whether the opacity has converged
            IsConverged = true;

            for (int i=0; i<NRadius;        i++) {   if ( ! IsConverged )   break;
            for (int k=0; k<NType_Neutrino; k++) {   if ( ! IsConverged )   break;

               double rel_diff = fabs( kappa_tot[TID][i][k] / kappa_tot_old[TID][i][k] - 1.0 );

               if ( rel_diff > Tolerance_Leak )   IsConverged = false;
            }}

            NIter++;
         } // while ( ! IsConverged  &&  NIter < NIter_Max )

         if ( ! IsConverged )
         {
            Aux_Message( stderr, "\n#%6s %8s %14s %14s %14s %14s %14s %14s %14s %14s %14s\n",
                                 "Index", "NCell",
                                 "kappa_old[0]", "kappa_old[1]", "kappa_old[2]",
                                 "kappa_new[0]", "kappa_new[1]", "kappa_new[2]",
                                 "tau[0]", "tau[1]", "tau[2]" );

            for (int i=0; i<NRadius; i++)
               Aux_Message( stderr, "%7d %8ld %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e\n",
                                    i, NCell[TID][i],
                                    kappa_tot_old[TID][i][0], kappa_tot_old[TID][i][1], kappa_tot_old[TID][i][2],
                                    kappa_tot    [TID][i][0], kappa_tot    [TID][i][1], kappa_tot    [TID][i][2],
                                    tau          [TID][i][0], tau          [TID][i][1], tau          [TID][i][2] );

            Aux_Error( ERROR_INFO, "failed in finding converged opacity for Ray (%d) in %s !!\n",
                       j, __FUNCTION__ );
         }

//       update eta_nu and store tau
         for (int i=0; i<NRadius;        i++) {
         for (int k=0; k<NType_Neutrino; k++) {
            eta_nu     [TID][i][k] = eta_nu_loc[TID][i][k];
            tau_Ruff_3D[i]  [j][k] = tau       [TID][i][k];
         }}


//       (4) compute the energy-independent optical depth (chi) using the Rosswog scheme
//           --> use the neutrino degeneracy obtained from the Ruffert scheme
         double blocking_factor_nue, blocking_factor_nua, eta_pn, eta_np;
         double kappa_scat_fac_Rosswog, kappa_abs_fac_Rosswog;
         double kappa_tilde_scat_n[NType_Neutrino], kappa_tilde_scat_p[NType_Neutrino], kappa_tilde_scat_x[NType_Neutrino];
         double kappa_tilde_abs_n [NType_Neutrino], kappa_tilde_abs_p [NType_Neutrino], kappa_tilde_abs_x [NType_Neutrino];

//       (4-1) construct the energy-independent opacity (zeta)
         for (int i=0; i<NRadius; i++)
         {
//          neutrino nucleon scattering; (A17)
            kappa_scat_fac_Rosswog  = Const_Rosswog_kappa_s * Dens_CGS[TID][i];

            for (int k=0; k<NType_Neutrino; k++)
            {
               kappa_tilde_scat_n[k] = x_n[TID][i] * kappa_scat_fac_Rosswog;
               kappa_tilde_scat_p[k] = x_p[TID][i] * kappa_scat_fac_Rosswog;
            }

//          coherent neutrino nucleus scattering; (A18)
//          --> the power of Abar is 1 because the opacity multiples the number fraction, not mass fraction
            kappa_scat_fac_Rosswog *= 0.25 * abar[TID][i] * SQR( 1.0 - zbar[TID][i] / abar[TID][i] );

            for (int k=0; k<NType_Neutrino; k++)
            {
               kappa_tilde_scat_x[k] = x_h[TID][i] * kappa_scat_fac_Rosswog;
            }

//          neutrino absorption; (A19) and (A20)
//          --> the factor Const_NA is moved from eta_pn/eta_np to kappa_abs_fac_Rosswog
            kappa_abs_fac_Rosswog = Const_Rosswog_kappa_a;
            blocking_factor_nue   = 1.0 + exp(  eta_e[TID][i] - Compute_FermiIntegral( 54, eta_nu[TID][i][0] )  );
            blocking_factor_nua   = 1.0 + exp( -eta_e[TID][i] - Compute_FermiIntegral( 54, eta_nu[TID][i][1] )  );

            if ( Dens_CGS[TID][i] < 1.0e11 )
            {
//             non-degenerate state, use mass fraction as chemical potential
               eta_pn = Dens_CGS[TID][i] * x_p[TID][i];
               eta_np = Dens_CGS[TID][i] * x_n[TID][i];
            }

            else
            {
               fac1   = Dens_CGS[TID][i] * ( x_n[TID][i] - x_p[TID][i] );

               eta_pn =  fac1 / ( exp(  eta_hat[TID][i] ) - 1.0 ); // (A9)
               eta_np = -fac1 / ( exp( -eta_hat[TID][i] ) - 1.0 ); // (A9)
            }

            eta_pn = fmax( 0.0, eta_pn );
            eta_np = fmax( 0.0, eta_np );

            kappa_tilde_abs_n[0] = eta_np * kappa_abs_fac_Rosswog / blocking_factor_nue;
            kappa_tilde_abs_n[1] = 0.0; // no absorption of a-type neutrinos on neutrons
            kappa_tilde_abs_n[2] = 0.0; // no absorption of x-type neutrinos on neutrons
            kappa_tilde_abs_p[0] = 0.0; // no absorption of e-type neutrinos on protons
            kappa_tilde_abs_p[1] = eta_pn * kappa_abs_fac_Rosswog / blocking_factor_nua;
            kappa_tilde_abs_p[2] = 0.0; // no absorption of x-type neutrinos on protons
            kappa_tilde_abs_x[0] = 0.0; // no absorption on nuclei
            kappa_tilde_abs_x[1] = 0.0; // no absorption on nuclei
            kappa_tilde_abs_x[2] = 0.0; // no absorption on nuclei

//          sum each contribution to get zeta; (A21)
            for (int k=0; k<NType_Neutrino; k++)
            {
               zeta[TID][i][k] = kappa_tilde_scat_n[k] + kappa_tilde_abs_n[k]
                               + kappa_tilde_scat_p[k] + kappa_tilde_abs_p[k]
                               + kappa_tilde_scat_x[k] + kappa_tilde_abs_x[k];
            }
         } // for (int i=0; i<NRadius; i++)

//       (4-2) integrate zeta to get chi at the bin center; (A23)
//             --> assume chi is zero at the right edge of outermost bin
         for (int k=0; k<NType_Neutrino; k++)
         {
            chi[TID][NRadius-1][k] = 0.5 * zeta[TID][NRadius-1][k] * BinWidth[NRadius-1];

            for (int i=NRadius-2; i>=0; i--)
               chi[TID][i][k] = chi[TID][i+1][k]
                              + 0.5 * zeta[TID][i  ][k] * BinWidth[i  ]
                              + 0.5 * zeta[TID][i+1][k] * BinWidth[i+1];
         }

//       store chi
         for (int i=0; i<NRadius;        i++) {
         for (int k=0; k<NType_Neutrino; k++) {
            chi_Ross_3D[i][j][k] = chi[TID][i][k];
         }}


//       (5) neutrino heating in O'Connor & Ott (2010)
         double FermiInte_NS[2][NType_Neutrino];
         double NS_Rad         [NType_Neutrino] = { 0.0, 0.0, 0.0 };

//       (5-1) compute the rms and mean energy at neutrino sphere (tau = 2/3)
         double eta_nu_NS, Temp_MeV_NS;
         double Table_tau[2], Table_Data[2];

         for (int k=0; k<NType_Neutrino; k++)
         {
            int Idx_NS;

//          find the index that brackets tau = 2/3
            for (Idx_NS=0; Idx_NS<NRadius; Idx_NS++)
            {
               if ( tau[TID][Idx_NS][k] < TwoThirds )   break;
            }

//          compute the location of neutrino sphere
            Table_tau [0] = tau[TID][Idx_NS  ][k];
            Table_tau [1] = tau[TID][Idx_NS-1][k];
            Table_Data[0] = 0.5 * ( Edge_CGS[Idx_NS  ] + Edge_CGS[Idx_NS+1] );
            Table_Data[1] = 0.5 * ( Edge_CGS[Idx_NS-1] + Edge_CGS[Idx_NS  ] );

            NS_Rad[k] = Mis_InterpolateFromTable( 2, Table_tau, Table_Data, TwoThirds );

//          compute the eta and temperature at neutrino sphere via linear interpolation
            Table_Data[0] = eta_nu[TID][Idx_NS  ][k];
            Table_Data[1] = eta_nu[TID][Idx_NS-1][k];
            eta_nu_NS     = Mis_InterpolateFromTable( 2, Table_tau, Table_Data, TwoThirds );

            Table_Data[0] = Temp_MeV[TID][Idx_NS  ];
            Table_Data[1] = Temp_MeV[TID][Idx_NS-1];
            Temp_MeV_NS   = Mis_InterpolateFromTable( 2, Table_tau, Table_Data, TwoThirds );

//          compute the rms and mean neutrino energies
            FermiInte_NS[0][k] = Compute_FermiIntegral( 53, eta_nu_NS );
            FermiInte_NS[1][k] = Compute_FermiIntegral( 54, eta_nu_NS );

            Heat_ERms_2D[j][k] = Temp_MeV_NS * sqrt( FermiInte_NS[0][k] ); // description following eq. (30) in O'Connor & Ott (2010)
            Heat_EAve_2D[j][k] = Temp_MeV_NS *       FermiInte_NS[1][k];   // (A11) in Rosswog & Liebendoerfer (2003)

//          sum the rms and mean neutrino energies of each ray for recording
            OMP_EAve [TID][k] += Heat_EAve_2D[j][k];
            OMP_RadNS[TID][k] += NS_Rad[k];
         }

//       (5-2) leak along this ray to determine luminosity
         real   Lum            [NType_Neutrino];
         real   Heat           [NType_Neutrino];
         real   NetHeat        [NType_Neutrino];
         double lepton_blocking[NType_Neutrino];
         double LumAcc         [NType_Neutrino] = { 0.0 };

//       set the flux of the innermost bin to 0.0
         for (int k=0; k<NType_Neutrino-1; k++)   Heat_Flux_3D[0][j][k] = 0.0;


         for (int i=0; i<NRadius-1; i++)
         {
//          compute the blocking factor at neutrino sphere
            lepton_blocking[0] = 1.0 / (  1.0 + exp(  eta_e[TID][i] - FermiInte_NS[1][0] )  );
            lepton_blocking[1] = 1.0 / (  1.0 + exp( -eta_e[TID][i] - FermiInte_NS[1][1] )  );

//          compute the luminosity and corresponding flux
//          --> the returned Lum represents the luminosity per unit volume
            Src_Leakage_ComputeLeak( Dens_Code[TID][i], Temp_Kelv[TID][i], Ye[TID][i], chi_Ross_3D[i][j], tau_Ruff_3D[i][j],
                                     Heat_Flux_3D[i][j], Heat_ERms_2D[j], Heat_EAve_2D[j],
                                     dEdt[TID]+i, dYedt[TID]+i, Lum, Heat, NetHeat,
                                     NuHeat, NuHeat_Fac, UNIT_D, &EoS );

            for (int k=0; k<NType_Neutrino-1; k++)
            {
               LumAcc              [k] += BinVol[i] * Lum[k];
               Heat_Flux_3D[i+1][j][k]  = LumAcc[k] * lepton_blocking[k] / BinSurfArea[i+1];
            }
         } // for (int i=0; i<NRadius-1; i++)


//       (6) dump ray data for debugging
#        ifdef GAMER_DEBUG
         const int  NColumn = 16;
               char FileName[50];

         sprintf( FileName, "Leakage_Lum_%06d", j );
         FILE *File = fopen( FileName, "w" );

         const int Idx_Theta = j % NTheta;
         const int Idx_Phi   = int(j / NTheta);

//       metadata
         Aux_Message( File, "# Idx_Ray       : %6d\n",    j                  );
         Aux_Message( File, "# Idx_Theta     : %6d\n",    Idx_Theta          );
         Aux_Message( File, "# Idx_Phi       : %6d\n",    Idx_Phi            );
         Aux_Message( File, "# Rad_NS_Nue    : %14.7e\n", NS_Rad[0]          );
         Aux_Message( File, "# Rad_NS_Nua    : %14.7e\n", NS_Rad[1]          );
         Aux_Message( File, "# Rad_NS_Nux    : %14.7e\n", NS_Rad[2]          );
         Aux_Message( File, "# Heat_ERms_Nue : %14.7e\n", Heat_ERms_2D[j][0] );
         Aux_Message( File, "# Heat_ERms_Nua : %14.7e\n", Heat_ERms_2D[j][1] );
         Aux_Message( File, "# Heat_ERms_Nux : %14.7e\n", Heat_ERms_2D[j][2] );
         Aux_Message( File, "# Heat_EAve_Nue : %14.7e\n", Heat_EAve_2D[j][0] );
         Aux_Message( File, "# Heat_EAve_Nua : %14.7e\n", Heat_EAve_2D[j][1] );
         Aux_Message( File, "# Heat_EAve_Nux : %14.7e\n", Heat_EAve_2D[j][2] );
         Aux_Message( File, "# -------------------------------------------------------------------------------\n" );

         Aux_Message( File, "# %14s  %8s", "[ 1]", "[ 2]" );
         for (int c=2; c<NColumn; c++)   Aux_Message( File, "  %10s[%2d]", "", c+1 );
         Aux_Message( File, "\n" );

         Aux_Message( File, "# %14s  %8s",              "Radius", "NCell"                              );
         Aux_Message( File, "  %14s  %14s  %14s  %14s", "Dens", "Temp", "Ye", "sEint"                  );
         Aux_Message( File, "  %14s  %14s  %14s",       "tau_Ruff_Nue", "tau_Ruff_Nua", "tau_Ruff_Nux" );
         Aux_Message( File, "  %14s  %14s  %14s",       "chi_Ross_Nue", "chi_Ross_Nua", "chi_Ross_Nux" );
         Aux_Message( File, "  %14s  %14s",             "HeatFlux_Nue", "HeatFlux_Nua"                 );
         Aux_Message( File, "  %14s  %14s",             "dEdt", "dYedt"                                );
         Aux_Message( File, "\n" );

         Aux_Message( File, "# %14s  %8s",              "[cm]", "[1]"                               );
         Aux_Message( File, "  %14s  %14s  %14s  %14s", "[g/cm^3]", "[Kelvin]", "[1]", "[cm^2/s^2]" );
         Aux_Message( File, "  %14s  %14s  %14s",       "[1]", "[1]", "[1]"                         );
         Aux_Message( File, "  %14s  %14s  %14s",       "[1]", "[1]", "[1]"                         );
         Aux_Message( File, "  %14s  %14s",             "[erg/cm^2/s]", "[erg/cm^2/s]"              );
         Aux_Message( File, "  %14s  %14s",             "[erg/g/s]", "[1/s]"                        );
         Aux_Message( File, "\n" );

//       ray data
         for (int i=0; i<NRadius; i++)
         {
            Aux_Message( File, "  %14.7e  %8ld",                   0.5 * ( Edge_CGS[i] + Edge_CGS[i+1] ), NCell[TID][i]                                                 );
            Aux_Message( File, "  %14.7e  %14.7e  %14.7e  %14.7e", Dens_CGS[TID][i], Temp_Kelv[TID][i], Ye[TID][i], Eint_Code[TID][i] / Dens_Code[TID][i] * SQR(UNIT_V) );

            for (int k=0; k<NType_Neutrino;   k++)   Aux_Message( File, "  %14.7e", tau_Ruff_3D [i][j][k] );
            for (int k=0; k<NType_Neutrino;   k++)   Aux_Message( File, "  %14.7e", chi_Ross_3D [i][j][k] );
            for (int k=0; k<NType_Neutrino-1; k++)   Aux_Message( File, "  %14.7e", Heat_Flux_3D[i][j][k] / Const_hc_MeVcm_CUBE );

            Aux_Message( File, "  %14.7e  %14.7e", dEdt[TID][i], dYedt[TID][i] );
            Aux_Message( File, "\n" );
         }

         fclose( File );
#        endif // #ifdef GAMER_DEBUG
      } // for (int j=IRay_start; j<IRay_end; j++)
   } // #  pragma omp parallel for schedule( runtime )


// wait for unfinished jobs because they may be unevenly distributed (NRay % MPI_NRank != 0)
   MPI_Barrier( MPI_COMM_WORLD );


// sum over all OpenMP threads
   for (int t=0; t<NT;             t++) {
   for (int k=0; k<NType_Neutrino; k++) {
      EAve [k] += OMP_EAve [t][k];
      RadNS[k] += OMP_RadNS[t][k];
   }}


// collect data from all ranks
#  ifndef SERIAL
   MPI_Allreduce( MPI_IN_PLACE, tau_Ruff,  NRadius*NRay*NType_Neutrino, MPI_GAMER_REAL, MPI_SUM, MPI_COMM_WORLD );
   MPI_Allreduce( MPI_IN_PLACE, chi_Ross,  NRadius*NRay*NType_Neutrino, MPI_GAMER_REAL, MPI_SUM, MPI_COMM_WORLD );
   MPI_Allreduce( MPI_IN_PLACE, Heat_Flux, NRadius*NRay*NType_Neutrino, MPI_GAMER_REAL, MPI_SUM, MPI_COMM_WORLD );
   MPI_Allreduce( MPI_IN_PLACE, Heat_ERms,         NRay*NType_Neutrino, MPI_GAMER_REAL, MPI_SUM, MPI_COMM_WORLD );
   MPI_Allreduce( MPI_IN_PLACE, Heat_EAve,         NRay*NType_Neutrino, MPI_GAMER_REAL, MPI_SUM, MPI_COMM_WORLD );
   MPI_Allreduce( MPI_IN_PLACE, EAve,                   NType_Neutrino, MPI_DOUBLE,     MPI_SUM, MPI_COMM_WORLD );
   MPI_Allreduce( MPI_IN_PLACE, RadNS,                  NType_Neutrino, MPI_DOUBLE,     MPI_SUM, MPI_COMM_WORLD );
#  endif // ifndef SERIAL


// compute the averaged neutrino sphere and the mean neutrino energy across all rays
   for (int k=0; k<NType_Neutrino; k++)
   {
      CCSN_Leakage_EAve [k] = EAve [k] / NRay;
      CCSN_Leakage_RadNS[k] = RadNS[k] / NRay;
   }


// free allocated arrays
   Aux_DeallocateArray2D( NCell            );
   Aux_DeallocateArray2D( Dens_Code        );
   Aux_DeallocateArray2D( Temp_Kelv        );
   Aux_DeallocateArray2D( Ye               );
   Aux_DeallocateArray2D( Eint_Code        );
   Aux_DeallocateArray2D( Dens_CGS         );
   Aux_DeallocateArray2D( Temp_MeV         );
   Aux_DeallocateArray2D( x_h              );
   Aux_DeallocateArray2D( x_n              );
   Aux_DeallocateArray2D( x_p              );
   Aux_DeallocateArray2D( abar             );
   Aux_DeallocateArray2D( zbar             );
   Aux_DeallocateArray2D( eta_e            );
   Aux_DeallocateArray2D( eta_p            );
   Aux_DeallocateArray2D( eta_n            );
   Aux_DeallocateArray2D( eta_hat          );
   Aux_DeallocateArray2D( kappa_scat_n_fac );
   Aux_DeallocateArray2D( kappa_scat_p_fac );
   Aux_DeallocateArray2D( kappa_abs_fac    );
   Aux_DeallocateArray2D( dEdt             );
   Aux_DeallocateArray2D( dYedt            );
   Aux_DeallocateArray2D( OMP_EAve         );
   Aux_DeallocateArray2D( OMP_RadNS        );

   Aux_DeallocateArray3D( eta_nu           );
   Aux_DeallocateArray3D( tau              );
   Aux_DeallocateArray3D( kappa_tot        );
   Aux_DeallocateArray3D( kappa_tot_old    );
   Aux_DeallocateArray3D( eta_nu_loc       );
   Aux_DeallocateArray3D( zeta             );
   Aux_DeallocateArray3D( chi              );

} // FUNCTION : Src_Leakage_ComputeTau
#endif // #ifndef __CUDACC__



//-------------------------------------------------------------------------------------------------------
// Function    :  Src_Leakage_ComputeLeak
// Description :  Compute the heating rates for internal energy and electron fraction
//
// Note        :  1. Invoked by Src_Leakage_ComputeTau() and Src_Leakage()
//                2. The diffusion rate, production rate, input Heat_Flux, and output Lum,
//                   Heat and NetHeat are scaled by Const_hc_MeVcm_CUBE to avoid overflow
//                3. Ref: M. Rufferrt, H.-Th. Janka, G. Schaefer, 1996, A&A, 311, 532 (arXiv: 9509006)
//                        S. Rosswog & M. Liebendoerfer, 2003, MNRAS, 342, 673 (arXiv: 0302301)
//                        A. Burrows, S. Reddy, T. A. Thompson, 2006, Nuclear Physics A, 777, 356 (arXiv: 0404432)
//                        E. O'Connor & C. D. Ott, 2010, Class. Quantum Grav., 27, 114103 (arXiv: 0912.2393)
//
// Parameter   :  Dens_Code  : Density in code unit
//                Temp_Kelv  : Temperature in Kelvin
//                Ye         : Electron fraction
//                chi        : Energy-independent optical depth calculated using the Rosswog scheme
//                tau        : Optical depth                    calculated using the Ruffert scheme
//                Heat_Flux  : Neutrino flux
//                Heat_ERms  : Neutrino rms  energy at neutrino sphere
//                Heat_EAve  : Neutrino mean energy at neutrino sphere
//                dEdt       : Heating rate of internal energy   in erg/g/s
//                dYedt      : Heating rate of electron fraction in 1/s
//                Lum        : Local luminosity per unit volume
//                Heat       : Local neutrino heating rate per volume in erg/s/cm^3
//                NetHeat    : Local net      heating rate per volume in erg/s/cm^3
//                NuHeat     : Flag for including neutrino heating
//                NuHeat_Fac : Factor of the neutrino heating rate
//
// Return      :  dEdt, dYedt, Lum, Heat, NetHeat
//-------------------------------------------------------------------------------------------------------
GPU_DEVICE
void Src_Leakage_ComputeLeak( const real Dens_Code, const real Temp_Kelv, const real Ye, const real chi[], const real tau[],
                              const real Heat_Flux[], const real *Heat_ERms, const real *Heat_EAve,
                              real *dEdt, real *dYedt, real *Lum, real *Heat, real *NetHeat,
                              const bool NuHeat, const real NuHeat_Fac, const real UNIT_D, const EoS_t *EoS )
{

   const real Dens_CGS = Dens_Code * UNIT_D;
   const real Temp_MeV = Temp_Kelv * Kelvin2MeV;

// (1-1) invoke the nuclear EoS solver to obtain EoS variables
   const int  NTarget = 8;
         int  In_Int[NTarget+1] = { NTarget,
                                    NUC_VAR_IDX_MUE, NUC_VAR_IDX_MUP, NUC_VAR_IDX_MUN,
                                    NUC_VAR_IDX_XH, NUC_VAR_IDX_XN, NUC_VAR_IDX_XP,
                                    NUC_VAR_IDX_ABAR, NUC_VAR_IDX_ZBAR };
         real In_Flt[4]         = { Dens_Code, Temp_Kelv, Ye, Temp_Kelv };
         real Out[NTarget+1];

#  ifdef __CUDACC__
   EoS->General_FuncPtr( NUC_MODE_TEMP, Out, In_Flt, In_Int, EoS->AuxArrayDevPtr_Flt, EoS->AuxArrayDevPtr_Int, EoS->Table );
#  else
   EoS_General_CPUPtr  ( NUC_MODE_TEMP, Out, In_Flt, In_Int, EoS_AuxArray_Flt,        EoS_AuxArray_Int,        h_EoS_Table );
#  endif

   const real eta_e   = Out[0] / Temp_MeV;
   const real eta_p   = Out[1] / Temp_MeV;
   const real eta_n   = Out[2] / Temp_MeV;
   const real x_h     = Out[3];
   const real x_n     = Out[4];
   const real x_p     = Out[5];
   const real abar    = Out[6];
   const real zbar    = Out[7];
   const real eta_hat = eta_n - eta_p - Const_Qnp / Temp_MeV;
         real eta_nu[NType_Neutrino];

// not need to include mass difference term (Q/T) since we have rest masses in the chemical potentials
   eta_nu[0] = eta_e - eta_n + eta_p;
   eta_nu[1] = -eta_nu[0];
   eta_nu[2] = (real)0.0;

// (1-2) do nothing if outside the shock
   if (  ( x_h > (real)0.5 )  &&  ( Dens_CGS < (real)1.0e13 )  )
   {
      *dEdt  = (real)0.0;
      *dYedt = (real)0.0;

      for (int k=0; k<NType_Neutrino; k++)
      {
         Lum    [k] = (real)0.0;
         Heat   [k] = (real)0.0;
         NetHeat[k] = (real)0.0;
      }

      return;
   }

// (1-3) interpolate eta using the optical depth obtained from the Ruffert scheme
   for (int k=0; k<NType_Neutrino-1; k++)   eta_nu[k] *= (real)1.0 - EXP( -tau[k] );


// (2) compute the local energy-independent opacity (zeta) using the Rosswog scheme
   real zeta              [NType_Neutrino];
   real kappa_tilde_scat_n[NType_Neutrino]; // scatter, neutron
   real kappa_tilde_scat_p[NType_Neutrino]; // scatter, proton
   real kappa_tilde_scat_x[NType_Neutrino]; // scatter, nucleus
   real kappa_tilde_abs_n [NType_Neutrino]; // absorption, neutron
   real kappa_tilde_abs_p [NType_Neutrino]; // absorption, proton
   real kappa_tilde_abs_x [NType_Neutrino]; // absorption, nucleus
   real blocking_factor   [NType_Neutrino];
   real kappa_scat_fac_Rosswog, kappa_abs_fac_Rosswog, eta_pn, eta_np;

// (2-1) neutrino nucleon scattering; (A17)
   kappa_scat_fac_Rosswog  = (real)Const_Rosswog_kappa_s * Dens_CGS;

   for (int k=0; k<NType_Neutrino; k++)
   {
      kappa_tilde_scat_n[k] = x_n * kappa_scat_fac_Rosswog;
      kappa_tilde_scat_p[k] = x_p * kappa_scat_fac_Rosswog;
   }

// (2-2) coherent neutrino nucleus scattering; (A18)
//       --> the power of Abar is 1 because the opacity multiples the number fraction, not mass fraction
   kappa_scat_fac_Rosswog *= (real)0.25 * abar * SQR( (real)1.0 - zbar / abar );

   for (int k=0; k<NType_Neutrino; k++)
   {
      kappa_tilde_scat_x[k] = x_h * kappa_scat_fac_Rosswog;
   }

// (2-3) neutrino absorption; (A19) and (A20)
//       --> the factor Const_NA is moved from eta_pn/eta_np to kappa_abs_fac_Rosswog
   kappa_abs_fac_Rosswog = (real)Const_Rosswog_kappa_a;
   blocking_factor[0]    = (real)1.0 + EXP(  eta_e - Compute_FermiIntegral( 54, eta_nu[0] )  );
   blocking_factor[1]    = (real)1.0 + EXP( -eta_e - Compute_FermiIntegral( 54, eta_nu[1] )  );

   if ( Dens_CGS < (real)1.0e11 )
   {
//    non-degenerate state, use mass fraction as chemical potential
      eta_pn = Dens_CGS * x_p;
      eta_np = Dens_CGS * x_n;
   }

   else
   {
      const real factor = Dens_CGS * ( x_n - x_p );

      eta_pn =  factor / ( EXP(  eta_hat ) - (real)1.0 ); // (A9)
      eta_np = -factor / ( EXP( -eta_hat ) - (real)1.0 ); // (A9)
   }

   eta_pn = FMAX( (real)0.0, eta_pn );
   eta_np = FMAX( (real)0.0, eta_np );

   kappa_tilde_abs_n[0] = eta_np * kappa_abs_fac_Rosswog / blocking_factor[0];
   kappa_tilde_abs_n[1] = (real)0.0; // no absorption of a-type neutrinos on neutrons
   kappa_tilde_abs_n[2] = (real)0.0; // no absorption of x-type neutrinos on neutrons
   kappa_tilde_abs_p[0] = (real)0.0; // no absorption of e-type neutrinos on protons
   kappa_tilde_abs_p[1] = eta_pn * kappa_abs_fac_Rosswog / blocking_factor[1];
   kappa_tilde_abs_p[2] = (real)0.0; // no absorption of x-type neutrinos on protons
   kappa_tilde_abs_x[0] = (real)0.0; // no absorption on nuclei
   kappa_tilde_abs_x[1] = (real)0.0; // no absorption on nuclei
   kappa_tilde_abs_x[2] = (real)0.0; // no absorption on nuclei

// sum each contribution to get zeta; (A21)
   for (int k=0; k<NType_Neutrino; k++)
   {
      zeta[k] = kappa_tilde_scat_n[k] + kappa_tilde_abs_n[k]
              + kappa_tilde_scat_p[k] + kappa_tilde_abs_p[k]
              + kappa_tilde_scat_x[k] + kappa_tilde_abs_x[k];
   }


// (3) compute the diffusion rate in the Rosswog scheme
//     --> the diffusion time is increased by a factor of 2, as suggested by O'Connor & Ott (2010)
   real R_diff [NType_Neutrino];                                       // number diffusion rate per volume
   real Q_diff [NType_Neutrino];                                       // energy diffusion rate per volume
   real weight [NType_Neutrino] = { (real)1.0, (real)1.0, (real)4.0 }; // numerical multiplicity factor, g_nu

   for (int k=0; k<NType_Neutrino; k++)
   {
      const real factor_diff = (real)Const_leakage_diff * weight[k] * zeta[k] * Temp_MeV / SQR( chi[k] );

      R_diff[k] = factor_diff *            Compute_FermiIntegral( 0, eta_nu[k] ); // (A34)
      Q_diff[k] = factor_diff * Temp_MeV * Compute_FermiIntegral( 1, eta_nu[k] ); // (A35)
   }


// (4) compute the locally production rate in the Rosswog scheme
   real R_loc [NType_Neutrino]; // locally number production rate per volume
   real Q_loc [NType_Neutrino]; // locally energy production rate per volume

   const real Temp_MeV_Quartic = CUBE(Temp_MeV)   * Temp_MeV;
   const real Temp_MeV_Quintic = Temp_MeV_Quartic * Temp_MeV;
   const real Temp_MeV_Sextic  = Temp_MeV_Quintic * Temp_MeV;
   const real FermiInte_p[4]   = {  Compute_FermiIntegral(  3,  eta_e ),
                                    Compute_FermiIntegral(  4,  eta_e ),
                                    Compute_FermiIntegral(  5,  eta_e ),
                                    Compute_FermiIntegral( 43,  eta_e )  };
   const real FermiInte_m[4]   = {  Compute_FermiIntegral(  3, -eta_e ),
                                    Compute_FermiIntegral(  4, -eta_e ),
                                    Compute_FermiIntegral(  5, -eta_e ),
                                    Compute_FermiIntegral( 43, -eta_e )  };

// (4-1) electron capture and positron capture in the Rosswog scheme
//       --> the factor Const_NA is moved from eta_pn/eta_np to beta
   const real beta = (real)Const_leakage_beta; // (A7)

   R_loc[0] = beta * eta_pn * Temp_MeV_Quintic * FermiInte_p[1]; // (A6)
   Q_loc[0] = beta * eta_pn * Temp_MeV_Sextic  * FermiInte_p[2]; // (A10)
   R_loc[1] = beta * eta_np * Temp_MeV_Quintic * FermiInte_m[1]; // (A12)
   Q_loc[1] = beta * eta_np * Temp_MeV_Sextic  * FermiInte_m[2]; // (A13)
   R_loc[2] = (real)0.0;
   Q_loc[2] = (real)0.0;

// (4-2) electron-positron pair annihilation in the Ruffert scheme
   const real epsilon_m     = Temp_MeV_Quartic * FermiInte_p[0]; // (B5)
   const real epsilon_p     = Temp_MeV_Quartic * FermiInte_m[0]; // (B5)
   const real pair_R_factor = epsilon_m * epsilon_p;             // for (B8) and (B10), factored out the constants
   const real factor_pair   = (real)0.5 * ( FermiInte_p[3] + FermiInte_m[3] ); // (B16)
         real R_pair, Q_pair;

   for (int k=0; k<NType_Neutrino; k++)
   {
      blocking_factor[k] = (real)1.0 + EXP( eta_nu[k] - factor_pair ); // (B9)
   }

   R_pair = (real)Const_leakage_pair_ea * pair_R_factor / ( blocking_factor[0] * blocking_factor[1] ); // (B8)
   Q_pair = factor_pair * Temp_MeV * R_pair; // (B16)

   R_loc[0] += R_pair;
   Q_loc[0] += Q_pair;
   R_loc[1] += R_pair;
   Q_loc[1] += Q_pair;

   R_pair = (real)Const_leakage_pair_x  * pair_R_factor / ( blocking_factor[2] * blocking_factor[2] ); // (B10)
   Q_pair = factor_pair * Temp_MeV * R_pair; // (B16)

   R_loc[2] += R_pair;
   Q_loc[2] += Q_pair;

// (4-3) plasmon decay in the Ruffert scheme
   const real gamma          = Const_gamma_0 * SQRT(  SQR(M_PI) / (real)3.0 + SQR(eta_e)  );
   const real gamma_R_factor = SQR(Temp_MeV_Quartic) * CUBE( SQR(gamma) )
                             * EXP(-gamma) * ( (real)1.0 + gamma ); // (B11) and (B12), factored out the constants
   const real factor_gamma   = (real)1.0 + (real)0.5 * SQR(gamma) / ( (real)1.0 + gamma ); // (B17)
         real R_gamma, Q_gamma;

   for (int k=0; k<NType_Neutrino; k++)
   {
      blocking_factor[k] = (real)1.0 + EXP( eta_nu[k] - factor_gamma ); // (B13)
   }

   R_gamma = (real)Const_leakage_gamma_ea * gamma_R_factor / ( blocking_factor[0] * blocking_factor[1] ); // (B11)
   Q_gamma = factor_gamma * Temp_MeV * R_gamma; // (B17)

   R_loc[0] += R_gamma;
   Q_loc[0] += Q_gamma;
   R_loc[1] += R_gamma;
   Q_loc[1] += Q_gamma;

   R_gamma = (real)Const_leakage_gamma_x  * gamma_R_factor / ( blocking_factor[2] * blocking_factor[2] ); // (B12)
   Q_gamma = factor_gamma * Temp_MeV * R_gamma; // (B17)

   R_loc[2] += R_gamma;
   Q_loc[2] += Q_gamma;

// (4-4) nucleon-nucleon bremsstrahlung in Burrows et al. (2006)
   const real R_brem = (real)Const_leakage_brem * SQR(Dens_CGS) * Temp_MeV_Quartic * SQRT(Temp_MeV)
                     * ( SQR(x_n) + SQR(x_p) + (real)28.0 * x_n * x_p / (real)3.0 );
   const real Q_brem = (real)0.504 * R_brem * Temp_MeV / (real)0.231;

   for (int k=0; k<NType_Neutrino; k++)
   {
      R_loc[k] += weight[k] * R_brem;
      Q_loc[k] += weight[k] * Q_brem;
   }


// (5) compute the leakage using the Rosswog scheme
   real R_eff[NType_Neutrino]; // effective number production rate per volume
   real Q_eff[NType_Neutrino]; // effective energy production rate per volume

   for (int k=0; k<NType_Neutrino; k++)
   {
      R_eff[k] = R_loc[k] * R_diff[k] / ( R_diff[k] + R_loc[k] ); // (A1)
      Q_eff[k] = Q_loc[k] * Q_diff[k] / ( Q_diff[k] + Q_loc[k] ); // (A2)
   }


// (6) neutrino heating in O'Connor & Ott (2010)
   real Lum_tot                  = (real)0.0;
   real Heat_Eff[NType_Neutrino] = { (real)0.0, (real)0.0, (real)0.0 };

   if ( NuHeat )
   {
      const real heat_factor = (real)Const_leakage_heat * NuHeat_Fac; // cm^2/MeV^2/g; (29) and (30)
            real F[2]; // <1 / F_nu> * exp(-2 * tau_nu)

      for (int k=0; k<NType_Neutrino-1; k++)
      {
         F[k] = ( (real)4.275 * tau[k] + (real)1.15 ) * EXP( (real)-2.0 * tau[k] );
      }

//    local neutrino heating rate; (31)
      Heat_Eff[0] = heat_factor * Dens_CGS * x_n * Heat_Flux[0] * SQR( Heat_ERms[0] ) * F[0]; // erg/cm^3/s
      Heat_Eff[1] = heat_factor * Dens_CGS * x_p * Heat_Flux[1] * SQR( Heat_ERms[1] ) * F[1]; // erg/cm^3/s
   }


// (7) compute the heating rates for internal energy and electron fraction
   for (int k=0; k<NType_Neutrino; k++)
   {
      Lum    [k] = Q_eff[k] * Const_MeV - Heat_Eff[k]; // leakage scheme + neutrino heating, in erg/cm^3/s; (32)
      Heat   [k] = Heat_Eff[k];
      NetHeat[k] = FABS(  MIN( Lum[k], (real)0.0 )  );

      Lum_tot += Lum[k];
   }


// the factor Const_hc_MeVcm_CUBE is included here
   *dEdt  = -Lum_tot / ( Dens_CGS * Const_hc_MeVcm_CUBE );
   *dYedt = (  R_eff[1] - Heat_Eff[1] / ( Heat_EAve[1] * Const_MeV )
             - R_eff[0] + Heat_Eff[0] / ( Heat_EAve[0] * Const_MeV ) )
          * Const_mn / ( Dens_CGS * Const_hc_MeVcm_CUBE );

} // FUNCTION : Src_Leakage_ComputeLeak



//-------------------------------------------------------------------------------------------------------
// Function    :  Compute_FermiIntegral
// Description :  Evaluate the Fermi integral
//
// Note        :  1. Invoked by Src_Leakage_ComputeTau() and Src_Leakage_ComputeLeak()
//                2. Only support integer order <= 5
//                3. Compute the ratio of Fermi integrals of different orders algebraically
//                   to avoid underflow and zero division
//                4. Ref: K. Takahashi, M. F. El Eid, W. Hillebrandt, 1978, A&A, 67, 185
//
// Parameter   :  Order : Order of Fermi integral
//                eta   : Degeneracy
//
// Return      :  integral
//-------------------------------------------------------------------------------------------------------
GPU_DEVICE
real Compute_FermiIntegral( const int Order, const real eta )
{

   real integral = 0.0;

   if ( eta > (real)1.0e-3 )
   {
      const real eta_sqr = SQR(eta);

      switch ( Order )
      {
         case 0 :
            integral = ( eta > EXP_OVERFLOW_THRESHOLD )
                     ? eta
                     : LOG( (real)1.0 + EXP(eta) );
         break;

         case 1 :
            integral  = (real)0.5 * eta_sqr + (real)1.6449;
            integral /= (real)1.0 + EXP( (real)-1.6855 * eta );
         break;

         case 2 :
            integral  = ( ONETHIRD * eta_sqr + (real)3.2899 ) * eta;
            integral /= (real)1.0 - EXP( (real)-1.8246 * eta );
         break;

         case 3 :
            integral  = ( (real)0.25 * eta_sqr + (real)4.9348 ) * eta_sqr + (real)11.3644;
            integral /= (real)1.0 + EXP( (real)-1.9039 * eta );
         break;

         case 4 :
            integral  = (  ( (real)0.2 * eta_sqr + (real)6.5797 ) * eta_sqr + (real)45.4576  ) * eta;
            integral /= (real)1.0 - EXP( (real)-1.9484 * eta );
         break;

         case 5 :
            integral  = (  ( ONESIXTH * eta_sqr + (real)8.2247 ) * eta_sqr + (real)113.6439  ) * eta_sqr + (real)236.5323;
            integral /= (real)1.0 + EXP( (real)-1.9727 * eta );
         break;

         case 43: // case 4 / case 3
            integral  = (  ( (real)0.2  * eta_sqr + (real)6.5797 ) * eta_sqr + (real) 45.4576  ) * eta;
            integral /=    ( (real)0.25 * eta_sqr + (real)4.9348 ) * eta_sqr + (real) 11.3644;
            integral *= (  (real)1.0 + EXP( (real)-1.9039 * eta )  )
                      / (  (real)1.0 - EXP( (real)-1.9484 * eta )  );
         break;

         case 53: // case 5 / case 3
            integral  = (  ( ONESIXTH   * eta_sqr + (real)8.2247 ) * eta_sqr + (real)113.6439  ) * eta_sqr + (real)236.5323;
            integral /=    ( (real)0.25 * eta_sqr + (real)4.9348 ) * eta_sqr + (real) 11.3644;
            integral *= (  (real)1.0 + EXP( (real)-1.9039 * eta )  )
                      / (  (real)1.0 + EXP( (real)-1.9727 * eta )  );
         break;

         case 54: // case 5 / case 4
            integral  = (  ( ONESIXTH   * eta_sqr + (real)8.2247 ) * eta_sqr + (real)113.6439  ) * eta_sqr + (real)236.5323;
            integral /= (  ( (real)0.2  * eta_sqr + (real)6.5797 ) * eta_sqr + (real) 45.4576  ) * eta;
            integral *= (  (real)1.0 - EXP( (real)-1.9484 * eta )  )
                      / (  (real)1.0 + EXP( (real)-1.9727 * eta )  );
         break;
      } // switch ( Order )
   }

   else
   {
      switch ( Order )
      {
         case 0 :
            integral = LOG( (real)1.0 + EXP(eta) );
         break;

         case 1 :
            integral  = EXP(eta);
            integral /= (real)1.0 + (real)0.2159 * EXP( (real)0.8857 * eta );
         break;

         case 2 :
            integral  = (real)2.0 * EXP(eta);
            integral /= (real)1.0 + (real)0.1092 * EXP( (real)0.8908 * eta );
         break;

         case 3 :
            integral  = (real)6.0 * EXP(eta);
            integral /= (real)1.0 + (real)0.0559 * EXP( (real)0.9069 * eta );
         break;

         case 4 :
            integral  = (real)24.0 * EXP(eta);
            integral /= (real)1.0 + (real)0.0287 * EXP( (real)0.9257 * eta );
         break;

         case 5 :
            integral  = (real)120.0 * EXP(eta);
            integral /= (real)1.0 + (real)0.0147 * EXP( (real)0.9431 * eta );
         break;

         case 43: // case 4 / case 3
            integral  = (real)4.0  * (  (real)1.0 + (real)0.0559 * EXP( (real)0.9069 * eta )  )
                                   / (  (real)1.0 + (real)0.0287 * EXP( (real)0.9257 * eta )  );
         break;

         case 53: // case 5 / case 3
            integral  = (real)20.0 * (  (real)1.0 + (real)0.0559 * EXP( (real)0.9069 * eta )  )
                                   / (  (real)1.0 + (real)0.0147 * EXP( (real)0.9431 * eta )  );
         break;

         case 54: // case 5 / case 4
            integral  = (real)5.0  * (  (real)1.0 + (real)0.0287 * EXP( (real)0.9257 * eta )  )
                                   / (  (real)1.0 + (real)0.0147 * EXP( (real)0.9431 * eta )  );
         break;
      } // switch ( Order )
   } // if ( eta > (real)1.0e-3 ) ... else ...


   return integral;

} // FUNCTION : Compute_FermiIntegral
