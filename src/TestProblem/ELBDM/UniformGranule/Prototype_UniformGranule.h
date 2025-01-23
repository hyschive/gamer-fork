#ifndef __PROTOTYPE_UNIFORMGRANULE_H__
#define __PROTOTYPE_UNIFORMGRANULE_H__

#include "GAMER.h"
#include "Profile_with_Sigma.h"

void Aux_ComputeProfile_with_Sigma( Profile_with_Sigma_t *Prof[], const double Center[], const double r_max_input, const double dr_min,
                                    const bool LogBin, const double LogBinRatio, const bool RemoveEmpty, const long TVarBitIdx[],
                                    const int NProf, const int MinLv, const int MaxLv, const PatchType_t PatchType,
                                    const double PrepTimeIn );

void Aux_ComputeCorrelation( Profile_t *Correlation[], const Profile_with_Sigma_t *prof_init[], const double Center[],
                             const double r_max_input, const double dr_min, const bool LogBin, const double LogBinRatio,
                             const bool RemoveEmpty, const long TVarBitIdx[], const int NProf, const int MinLv, const int MaxLv,
                             const PatchType_t PatchType, const double PrepTime, const double dr_min_prof );

#endif // #ifndef __PROTOTYPE_UNIFORMGRANULE_H__
