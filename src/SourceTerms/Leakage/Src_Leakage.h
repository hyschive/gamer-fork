#ifndef __SRC_LEAKAGE_H__
#define __SRC_LEAKAGE_H__



#include "Macro.h"
#include "PhysicalConstant.h"


// auxiliary array indices
#define SRC_AUX_PNS_X         0     // AuxArray_Flt: x-coordinate of proto-neutron star
#define SRC_AUX_PNS_Y         1     // AuxArray_Flt: y-coordinate of proto-neutron star
#define SRC_AUX_PNS_Z         2     // AuxArray_Flt: z-coordinate of proto-neutron star
#define SRC_AUX_MAXRADIUS     3     // AuxArray_Flt: maximum radius
#define SRC_AUX_RADMIN_LOG    4     // AuxArray_Flt: minimum radius of logarithmic bins
#define SRC_AUX_DRAD          5     // AuxArray_Flt: spacing linear bins in the radial    direction
#define SRC_AUX_DTHETA        6     // AuxArray_Flt: spacing             in the polar     direction
#define SRC_AUX_DPHI          7     // AuxArray_Flt: spacing             in the azimuthal direction
#define SRC_AUX_YEMIN         8     // AuxArray_Flt: minimum allowed Ye after update in the leakge scheme
#define SRC_AUX_YEMAX         9     // AuxArray_Flt: maximum allowed Ye after update in the leakge scheme
#define SRC_AUX_VSQR2CODE    10     // AuxArray_Flt: convert velocity^2 to code unit

#define SRC_AUX_MODE          0     // AuxArray_Int: mode of the leakage scheme
#define SRC_AUX_NRAD_LIN      1     // AuxArray_Int: number of linear bin in the radial direction
#define SRC_AUX_STRIDE        2     // AuxArray_Int: stride for indexing
#define SRC_AUX_RECORD        3     // AuxArray_Int: record mode


// additional physical constants for the leakage scheme
const double Const_Qnp              = 1.293333;          // m_n - m_p, MeV
const double Const_Cv               = 0.5 + 2.0 * 0.23;  // vector coupling
const double Const_Ca               = 0.5;               // axial coupling
const double Const_alpha            = 1.23;              // gA
const double Const_me_MeV           = 5.10998950e-1;     // electron mass in MeV
const double Const_sigma0           = 1.76e-44;          // reference weak-interaction cross section in cm^2
const double Const_mn               = 1.67492749804e-24; // neutron mass in gram
const double Const_fsc              = 7.2973525693e-3;   // fine-structure constant


// derived constant
const double Const_gamma_0          = 5.56515284698977e-2;                             // 2.0 * sqrt(  Const_fsc / (3.0*M_PI) )
const double Const_hc_MeVcm         = 1.0e-6 * 2.0 * M_PI * Const_Planck_eV * Const_c; // h*c in MeV*cm
const double Const_alpha_SQR        = SQR ( Const_alpha    );
const double Const_me_MeV_SQR       = SQR ( Const_me_MeV   );
const double Const_hc_MeVcm_CUBE    = CUBE( Const_hc_MeVcm );


// Constant factor in the leakage scheme
// --> Ruffert scheme
const double Const_sn_0             = ( 1.0 + 5.0 * Const_alpha_SQR ) / 24.0;                       // cross section coefficients
const double Const_sp_0             = ( 4.0 * SQR(Const_Cv - 1.0) + 5.0 * Const_alpha_SQR ) / 24.0; // cross section coefficients
const double Const_Ruffert_kappa_sn = Const_sn_0 * Const_NA * Const_sigma0 / Const_me_MeV_SQR;      // constant coefficient of scattering process
const double Const_Ruffert_kappa_sp = Const_sp_0 * Const_NA * Const_sigma0 / Const_me_MeV_SQR;      // constant coefficient of scattering process
const double Const_Ruffert_kappa_a  = 0.25 * ( 1.0 + 3.0 * Const_alpha_SQR )
                                    * Const_NA * Const_sigma0 / Const_me_MeV_SQR;                   // constant coefficient of absorption process

// --> Rosswog scheme
const double Const_Rosswog_kappa_s  = 0.25 * Const_NA * Const_sigma0 / Const_me_MeV_SQR;            // constant coefficient of scattering process
const double Const_Rosswog_kappa_a  = Const_Ruffert_kappa_a;                                        // constant coefficient of absorption process

// --> leakage scheme (factored out 1.0 / Const_hc_MeVcm_CUBE to avoid overflow)
const double Const_leakage_diff     = 4.0 * M_PI * Const_c / 6.0;
const double Const_leakage_beta     = M_PI * Const_c * ( 1.0 + 3.0 * Const_alpha_SQR )
                                    * Const_NA * Const_sigma0 / Const_me_MeV_SQR;
const double Const_leakage_pair     = 64.0 * SQR( M_PI ) * Const_sigma0 * Const_c
                                    / ( Const_hc_MeVcm_CUBE * Const_me_MeV_SQR );
const double Const_leakage_pair_ea  = (  SQR( Const_Cv - Const_Ca ) + SQR( Const_Cv + Const_Ca       )  )
                                    * Const_leakage_pair / 36.0;
const double Const_leakage_pair_x   = (  SQR( Const_Cv - Const_Ca ) + SQR( Const_Cv + Const_Ca - 2.0 )  )
                                    * Const_leakage_pair /  9.0;
const double Const_leakage_gamma    = CUBE( M_PI ) * Const_sigma0 * Const_c
                                    / ( 3.0 * Const_fsc * Const_me_MeV_SQR * Const_hc_MeVcm_CUBE );
const double Const_leakage_gamma_ea = Const_leakage_gamma * SQR( Const_Cv       );
const double Const_leakage_gamma_x  = Const_leakage_gamma * SQR( Const_Cv - 1.0 ) * 4.0;
const double Const_leakage_brem     = 0.5 * 0.231 * ( 2.0778e2 / Const_MeV ) * Const_hc_MeVcm_CUBE;
const double Const_leakage_heat     = 0.25  * ( 1.0 + 3.0 * Const_alpha_SQR ) * Const_sigma0
                                    / ( Const_me_MeV_SQR * Const_mn );


// miscellaneous
const int    NType_Neutrino         = 3; // electron neutrino (e), electron anti-neutrino (a), heavy-lepton neutrino (x)
const double TwoThirds              = 2.0 / 3.0;
const double Kelvin2MeV             = 1.0e-6 * Const_kB_eV; // Kelvin to MeV
const double Erg2MeV                = 1.0    / Const_MeV;   // Erg to MeV


#endif // #ifndef __SRC_LEAKAGE_H__
