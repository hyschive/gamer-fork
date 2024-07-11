#ifndef __PROFILE_WITH_SIGMA_H__
#define __PROFILE_WITH_SIGMA_H__



//-------------------------------------------------------------------------------------------------------
// Structure   :  Profile_with_Sigma_t
// Description :  Data structure for computing a radial profile
//
// Data Member :  NBin        : Total number of radial bins
//                LogBin      : true/false --> log/linear bins
//                LogBinRatio : Ratio of adjacent log bins
//                MaxRadius   : Maximum radius
//                Center      : Target center coordinates
//                Allocated   : Whether or not member arrays such as Radius[] have been allocated
//                Radius      : Radial coordinate at each bin
//                Data        : Profile data at each bin
//                Data_Sigma  : Profile data at each bin
//                Weight      : Total weighting at each bin
//                NCell       : Number of cells at each bin
//
// Method      :  Profile_with_Sigma_t      : Constructor
//               ~Profile_with_Sigma_t      : Destructor
//                AllocateMemory : Allocate memory
//                FreeMemory     : Free memory
//-------------------------------------------------------------------------------------------------------
struct Profile_with_Sigma_t
{

// data members
// ===================================================================================
   int    NBin;
   bool   LogBin;
   double LogBinRatio;
   double MaxRadius;
   double Center[3];
   bool   Allocated;

   double *Radius;
   double *Data;
   double *Data_Sigma;
   double *Weight;
   long   *NCell;


   //===================================================================================
   // Constructor :  Profile_with_Sigma_t
   // Description :  Constructor of the structure "Profile_with_Sigma_t"
   //
   // Note        :  Initialize the data members
   //
   // Parameter   :  None
   //===================================================================================
   Profile_with_Sigma_t()
   {

      NBin       = -1;
      Allocated  = false;
      Radius     = NULL;
      Data       = NULL;
      Data_Sigma = NULL;
      Weight     = NULL;
      NCell      = NULL;

   } // METHOD : Profile_with_Sigma_t



   //===================================================================================
   // Destructor  :  ~Profile_with_Sigma_t
   // Description :  Destructor of the structure "Profile_with_Sigma_t"
   //
   // Note        :  Free memory
   //===================================================================================
   ~Profile_with_Sigma_t()
   {

      FreeMemory();

   } // METHOD : ~Profile_with_Sigma_t



   //===================================================================================
   // Method      :  AllocateMemory
   // Description :  Allocate member arrays
   //
   // Note        :  1. Invoked by Aux_ComputeProfile_with_Sigma()
   //                2. No data initialization is done here
   //
   // Parameter   :  None
   //
   // Return      :  Radius[], Data[], Weight[], NCell[], Allocated
   //===================================================================================
   void AllocateMemory()
   {

      if ( NBin < 0 )   Aux_Error( ERROR_INFO, "NBin (%d) < 0 !!\n", NBin );

//    free the previously allocated memory in case NBin has changed
//    --> allows the same Profile_with_Sigma_t object to be reused without the need to manually free memory first
      if ( Allocated )  FreeMemory();

      Radius       = new double [NBin];
      Data         = new double [NBin];
      Data_Sigma   = new double [NBin];
      Weight       = new double [NBin];
      NCell        = new long   [NBin];

      Allocated = true;

   } // METHOD : AllocateMemory



   //===================================================================================
   // Method      :  FreeMemory
   // Description :  Free the memory allocated by AllocateMemory()
   //
   // Note        :  Invoked by ~Profile_with_Sigma()
   //
   // Parameter   :  None
   //
   // Return      :  Radius[], Data[], Weight[], NCell[]
   //===================================================================================
   void FreeMemory()
   {

      delete [] Radius;     Radius       = NULL;
      delete [] Data;       Data         = NULL;
      delete [] Data_Sigma; Data_Sigma   = NULL;
      delete [] Weight;     Weight       = NULL;
      delete [] NCell;      NCell        = NULL;

      Allocated = false;

   } // METHOD : FreeMemory


}; // struct Profile_with_Sigma_t


#endif // #ifndef __PROFILE_WITH_SIGMA_H__
