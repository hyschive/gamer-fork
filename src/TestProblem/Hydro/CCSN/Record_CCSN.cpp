#include "GAMER.h"


       double CCSN_CentralDens;

       double CCSN_Rsh_Min = 0.0;
       double CCSN_Rsh_Max = 0.0;
       double CCSN_Rsh_Ave = 0.0;

       double CCSN_Leakage_NetHeatGain = 0.0;
       double CCSN_Leakage_Lum    [3]  = { 0.0 };
       double CCSN_Leakage_Heat   [3]  = { 0.0 };
       double CCSN_Leakage_NetHeat[3]  = { 0.0 };
       double CCSN_Leakage_EAve   [3]  = { 0.0 };
       double CCSN_Leakage_RadNS  [3]  = { 0.0 };

extern bool   CCSN_Is_PostBounce;

extern void Src_WorkBeforeMajorFunc_Leakage( const int lv, const double TimeNew, const double TimeOld, const double dt,
                                             double AuxArray_Flt[], int AuxArray_Int[] );




//-------------------------------------------------------------------------------------------------------
// Function    :  Record_CCSN_CentralQuant
// Description :  Record quantities at the center
//
// Note        :  1. Invoked by Record_CCSN()
//                2. The center here is defined as the cell with highest density
//-------------------------------------------------------------------------------------------------------
void Record_CCSN_CentralQuant()
{

   const char   filename_central_quant[] = "Record__CentralQuant";
   const double BoxCenter[3]             = { amr->BoxCenter[0], amr->BoxCenter[1], amr->BoxCenter[2] };

// allocate memory for per-thread arrays
#  ifdef OPENMP
   const int NT = OMP_NTHREAD;
#  else
   const int NT = 1;
#  endif

   const int      NData_Int = 6; // MPI_Rank, lv, PID, i, j, k
   const int      NData_Flt = 4; // dens, x, y, z
         int      Data_Int[NData_Int] = { 0 };
         double   Data_Flt[NData_Flt] = { -__DBL_MAX__ };
         int      OMP_Data_Int[NT][NData_Int];
         double **OMP_Data_Flt = NULL;

   Aux_AllocateArray2D( OMP_Data_Flt, NT, NData_Flt );


#  pragma omp parallel
   {
#     ifdef OPENMP
      const int TID = omp_get_thread_num();
#     else
      const int TID = 0;
#     endif

//    initialize arrays
      for (int b=0; b<NData_Int; b++)   OMP_Data_Int[TID][b] = -1;
      for (int b=0; b<NData_Flt; b++)   OMP_Data_Flt[TID][b] = -__DBL_MAX__;

      for (int lv=0; lv<NLEVEL; lv++)
      {
         const double dh = amr->dh[lv];

#        pragma omp for schedule( runtime )
         for (int PID=0; PID<amr->NPatchComma[lv][1]; PID++)
         {
            if ( amr->patch[0][lv][PID]->son != -1 )  continue;

            for (int k=0; k<PS1; k++)  {  const double z = amr->patch[0][lv][PID]->EdgeL[2] + (k+0.5)*dh;
            for (int j=0; j<PS1; j++)  {  const double y = amr->patch[0][lv][PID]->EdgeL[1] + (j+0.5)*dh;
            for (int i=0; i<PS1; i++)  {  const double x = amr->patch[0][lv][PID]->EdgeL[0] + (i+0.5)*dh;

               const double dens = amr->patch[ amr->FluSg[lv] ][lv][PID]->fluid[DENS][k][j][i];

               if ( dens > OMP_Data_Flt[TID][0] )
               {
                  OMP_Data_Int[TID][0] = MPI_Rank;
                  OMP_Data_Int[TID][1] = lv;
                  OMP_Data_Int[TID][2] = PID;
                  OMP_Data_Int[TID][3] = i;
                  OMP_Data_Int[TID][4] = j;
                  OMP_Data_Int[TID][5] = k;

                  OMP_Data_Flt[TID][0] = dens;
                  OMP_Data_Flt[TID][1] = x;
                  OMP_Data_Flt[TID][2] = y;
                  OMP_Data_Flt[TID][3] = z;
               }

            }}} // i,j,k
         } // for (int PID=0; PID<amr->NPatchComma[lv][1]; PID++)
      } // for (int lv=0; lv<NLEVEL; lv++)
   } // OpenMP parallel region


// find the maximum over all OpenMP threads
   for (int TID=0; TID<NT; TID++)
   {
      if ( OMP_Data_Flt[TID][0] > Data_Flt[0] )
      {
         for (int b=0; b<NData_Int; b++)   Data_Int[b] = OMP_Data_Int[TID][b];
         for (int b=0; b<NData_Flt; b++)   Data_Flt[b] = OMP_Data_Flt[TID][b];
      }
   }

// free per-thread arrays
   Aux_DeallocateArray2D( OMP_Data_Flt );


// collect data from all ranks
#  ifndef SERIAL
   {
      int    Data_Int_All[MPI_NRank * NData_Int];
      double Data_Flt_All[MPI_NRank * NData_Flt];

      MPI_Allgather( Data_Int, NData_Int, MPI_INT,    Data_Int_All, NData_Int, MPI_INT,    MPI_COMM_WORLD );
      MPI_Allgather( Data_Flt, NData_Flt, MPI_DOUBLE, Data_Flt_All, NData_Flt, MPI_DOUBLE, MPI_COMM_WORLD );

      for (int i=0; i<MPI_NRank; i++)
      {
         if ( Data_Flt_All[i * NData_Flt] >= Data_Flt[0] )
         {
            for (int b=0; b<NData_Int; b++)   Data_Int[b] = Data_Int_All[i * NData_Int + b];
            for (int b=0; b<NData_Flt; b++)   Data_Flt[b] = Data_Flt_All[i * NData_Flt + b];
         }
      }
   }
#  endif // ifndef SERIAL


// write to the file "Record__CentralQuant" by the MPI process which has the target patch
   if ( MPI_Rank == Data_Int[0] )
   {

      static bool FirstTime = true;

//    file header
      if ( FirstTime )
      {
         if ( Aux_CheckFileExist(filename_central_quant) )
         {
             Aux_Message( stderr, "WARNING : file \"%s\" already exists !!\n", filename_central_quant );
         }

         else
         {
             FILE *file_cent_quant = fopen( filename_central_quant, "w" );

             fprintf( file_cent_quant, "#%14s %12s %16s %16s %16s %16s %16s %16s %16s %16s",
                                       "1_Time [sec]", "2_Step", "3_PosX [cm]", "4_PosY [cm]", "5_PosZ [cm]",
                                       "6_Dens [g/cm^3]", "7_Ye", "8_Rsh_Min [cm]", "9_Rsh_Ave [cm]", "10_Rsh_Max [cm]" );

#            if ( defined NEUTRINO_SCHEME  &&  NEUTRINO_SCHEME == LEAKAGE )
             fprintf( file_cent_quant, "%29s %28s %28s %28s %28s %28s %28s %28s %28s %28s %28s %28s %28s %28s",
                                       "11_Leak_NetHeat_Gain [erg/s]", "12_Leak_Lum_Nue [erg/s]", "13_Leak_Lum_Nua [erg/s]", "14_Leak_Lum_Nux [erg/s]",
                                       "15_Leak_Heat_Nue [erg/s]", "16_Leak_Heat_Nua [erg/s]", "17_Leak_NetHeat_Nue [erg/s]", "18_Leak_NetHeat_Nua [erg/s]",
                                       "19_Leak_EAve_Nue [MeV]", "20_Leak_EAve_Nua [MeV]", "21_Leak_EAve_Nux [MeV]",
                                       "22_Leak_RadNS_Nue [cm]", "23_Leak_RadNS_Nua [cm]", "24_Leak_RadNS_Nux [cm]" );
#            endif

             fprintf( file_cent_quant, "\n" );

             fclose( file_cent_quant );
         }

         FirstTime = false;
      }

//    output data
      const int  lv  = Data_Int[1];
      const int  PID = Data_Int[2];
      const int  i   = Data_Int[3];
      const int  j   = Data_Int[4];
      const int  k   = Data_Int[5];
            real u[NCOMP_TOTAL];

      for (int v=0; v<NCOMP_TOTAL; v++)   u[v] = amr->patch[ amr->FluSg[lv] ][lv][PID]->fluid[v][k][j][i];

#     ifdef YE
      const real Ye = u[YE] / u[DENS];
#     else
      const real Ye = (real)0.0;
#     endif

      FILE *file_cent_quant = fopen( filename_central_quant, "a" );

      fprintf( file_cent_quant, "%15.7e %12ld %16.7e %16.7e %16.7e %16.7e %16.7e %16.7e %16.7e %16.7e",
               Time[0]*UNIT_T, Step, Data_Flt[1]*UNIT_L, Data_Flt[2]*UNIT_L, Data_Flt[3]*UNIT_L,
               u[DENS]*UNIT_D, Ye, CCSN_Rsh_Min*UNIT_L, CCSN_Rsh_Ave*UNIT_L, CCSN_Rsh_Max*UNIT_L );

#     if ( defined NEUTRINO_SCHEME  &&  NEUTRINO_SCHEME == LEAKAGE )
      fprintf( file_cent_quant, "%29.7e %28.7e %28.7e %28.7e %28.7e %28.7e %28.7e %28.7e %28.7e %28.7e %28.7e %28.7e %28.7e %28.7e",
               CCSN_Leakage_NetHeatGain, CCSN_Leakage_Lum[0], CCSN_Leakage_Lum[1], CCSN_Leakage_Lum[2],
               CCSN_Leakage_Heat[0], CCSN_Leakage_Heat[1], CCSN_Leakage_NetHeat[0], CCSN_Leakage_NetHeat[1],
               CCSN_Leakage_EAve[0], CCSN_Leakage_EAve[1], CCSN_Leakage_EAve[2],
               CCSN_Leakage_RadNS[0], CCSN_Leakage_RadNS[1], CCSN_Leakage_RadNS[2] );
#     endif

      fprintf( file_cent_quant, "\n" );

      fclose( file_cent_quant );

   } // if ( MPI_Rank == 0 )


// store the central density in cgs unit for detecting core bounces
   CCSN_CentralDens = Data_Flt[0] * UNIT_D;

} // FUNCTION : Record_CCSN_CentralQuant()



#if ( defined NEUTRINO_SCHEME  &&  NEUTRINO_SCHEME == LEAKAGE )
//-------------------------------------------------------------------------------------------------------
// Function    :  Record_CCSN_Leakage
// Description :  Compute the net heating in the gain region, and the total luminosity,
//                heating rate and net heating rate of the leakage scheme
//
// Note        :  1. Invoked by Record_CCSN_CentralQuant()
//-------------------------------------------------------------------------------------------------------
void Record_CCSN_Leakage()
{

   const int    NType_Neutrino = 3;
   const double Const_hc_MeVcm_CUBE = CUBE( 1.0e-6 * 2.0 * M_PI * Const_Planck_eV * Const_c );

// update leakage data at TimeNew
   const int    lv = 0;
   const double TimeNew = amr->FluSgTime[lv][ amr->FluSg[lv] ];

   Src_WorkBeforeMajorFunc_Leakage( lv, TimeNew, TimeNew, 0.0,
                                    Src_Leakage_AuxArray_Flt, Src_Leakage_AuxArray_Int );

// enable the record mode
   Src_Leakage_AuxArray_Int[4] = 1;

// allocate memory for per-thread arrays
#  ifdef OPENMP
   const int NT = OMP_NTHREAD;
#  else
   const int NT = 1;
#  endif

   double NetHeatGain = 0.0; // net heating rate in gain region
   double Lum    [NType_Neutrino] = { 0.0 };
   double Heat   [NType_Neutrino] = { 0.0 };
   double NetHeat[NType_Neutrino] = { 0.0 };

   double *OMP_NetHeatGain = new double [NT];
   double **OMP_Lum, **OMP_Heat, **OMP_NetHeat;

   Aux_AllocateArray2D( OMP_Lum,     NT, NType_Neutrino );
   Aux_AllocateArray2D( OMP_Heat,    NT, NType_Neutrino );
   Aux_AllocateArray2D( OMP_NetHeat, NT, NType_Neutrino );


#  pragma omp parallel
   {
#     ifdef OPENMP
      const int TID = omp_get_thread_num();
#     else
      const int TID = 0;
#     endif

//    initialize arrays
      OMP_NetHeatGain[TID] = 0.0;

      for (int n=0; n<NType_Neutrino; n++)
      {
         OMP_Lum    [TID][n] = 0.0;
         OMP_Heat   [TID][n] = 0.0;
         OMP_NetHeat[TID][n] = 0.0;
      }


//    loop over all levels
      for (int lv=0; lv<=MAX_LEVEL; lv++)
      {
         if ( NPatchTotal[lv] == 0 )   continue;

         const double dh     = amr->dh[lv];
         const double dv_CGS = CUBE( dh * UNIT_L );


#        pragma omp for schedule( static )
         for (int PID=0; PID<amr->NPatchComma[lv][1]; PID++)
         {
//          skip non-leaf patches
            if ( amr->patch[0][lv][PID]->son != -1 )   continue;

            const double z0 = amr->patch[0][lv][PID]->EdgeL[2];
            const double y0 = amr->patch[0][lv][PID]->EdgeL[1];
            const double x0 = amr->patch[0][lv][PID]->EdgeL[0];

            for (int k=0; k<PS1; k++)  {  const double z = z0 + (k+0.5)*dh;
            for (int j=0; j<PS1; j++)  {  const double y = y0 + (j+0.5)*dh;
            for (int i=0; i<PS1; i++)  {  const double x = x0 + (i+0.5)*dh;

//             get the input arrays
               real fluid[FLU_NIN_S];

               for (int v=0; v<FLU_NIN_S; v++)  fluid[v] = amr->patch[ amr->FluSg[lv] ][lv][PID]->fluid[v][k][j][i];

#              ifdef MHD
               real B[NCOMP_MAG];

               MHD_GetCellCenteredBFieldInPatch( B, lv, PID, i, j, k, amr->MagSg[lv] );
#              else
               real *B = NULL;
#              endif

               SrcTerms.Leakage_CPUPtr( fluid, B, &SrcTerms, 0.0, NULL_REAL, x, y, z, NULL_REAL, NULL_REAL,
                                        MIN_DENS, MIN_PRES, MIN_EINT, NULL,
                                        Src_Leakage_AuxArray_Flt, Src_Leakage_AuxArray_Int );

               if ( fluid[DENS] > 0.0 )
                  OMP_NetHeatGain[TID] += fluid[DENS] * dv_CGS;

               OMP_Lum    [TID][0] += fluid[MOMX] * dv_CGS;
               OMP_Lum    [TID][1] += fluid[MOMY] * dv_CGS;
               OMP_Lum    [TID][2] += fluid[MOMZ] * dv_CGS;
               OMP_Heat   [TID][0] += fluid[ENGY] * dv_CGS;
               OMP_Heat   [TID][1] += fluid[YE  ] * dv_CGS;
#              ifdef DYEDT_NU
               OMP_NetHeat[TID][0] += fluid[DEDT_NU ] * dv_CGS;
               OMP_NetHeat[TID][1] += fluid[DYEDT_NU] * dv_CGS;
#              endif

            }}} // i,j,k
         } // for (int PID=0; PID<amr->NPatchComma[lv][1]; PID++)
      } // for (int lv=0; lv<=MAX_LEVEL; lv++)
   } // OpenMP parallel region


// sum over all OpenMP threads
   for (int t=0; t<NT; t++)
   {
      NetHeatGain += OMP_NetHeatGain[t];

      for (int n=0; n<NType_Neutrino; n++)
      {
         Lum    [n] += OMP_Lum    [t][n];
         Heat   [n] += OMP_Heat   [t][n];
         NetHeat[n] += OMP_NetHeat[t][n];
      }
   }


// collect data from all ranks (in-place reduction)
#  ifndef SERIAL
   MPI_Allreduce( MPI_IN_PLACE, &NetHeatGain, 1,              MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
   MPI_Allreduce( MPI_IN_PLACE,  Lum,         NType_Neutrino, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
   MPI_Allreduce( MPI_IN_PLACE,  Heat,        NType_Neutrino, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
   MPI_Allreduce( MPI_IN_PLACE,  NetHeat,     NType_Neutrino, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
#  endif // ifndef SERIAL


// store the results with correction
   CCSN_Leakage_NetHeatGain = NetHeatGain;

   for (int n=0; n<NType_Neutrino; n++)
   {
      CCSN_Leakage_Lum    [n] = Lum    [n] / Const_hc_MeVcm_CUBE;
      CCSN_Leakage_Heat   [n] = Heat   [n] / Const_hc_MeVcm_CUBE;
      CCSN_Leakage_NetHeat[n] = NetHeat[n] / Const_hc_MeVcm_CUBE;
   }


// disable the record mode
   Src_Leakage_AuxArray_Int[4] = 0;


// free per-thread arrays
   delete [] OMP_NetHeatGain;

   Aux_DeallocateArray2D( OMP_Lum     );
   Aux_DeallocateArray2D( OMP_Heat    );
   Aux_DeallocateArray2D( OMP_NetHeat );

} // FUNCTION : Record_CCSN_Leakage()
#endif // if ( defined NEUTRINO_SCHEME  &&  NEUTRINO_SCHEME == LEAKAGE )



#ifdef GRAVITY
//-------------------------------------------------------------------------------------------------------
// Function    :  Record_CCSN_GWSignal
// Description :  Record the second-order time derivative of mass quadrupole moments
//
// Note        :  1. Invoked by Record_CCSN()
//                2. Ref: Kenichi Oohara, et al., 1997, PThPS, 128, 183 (arXiv: 1206.4724), sec. 2.1
//-------------------------------------------------------------------------------------------------------
void Record_CCSN_GWSignal()
{

   const char   filename_QuadMom_2nd[ ] = "Record__QuadMom_2nd";
   const double BoxCenter           [3] = { amr->BoxCenter[0], amr->BoxCenter[1], amr->BoxCenter[2] };

// allocate memory for per-thread arrays
#  ifdef OPENMP
   const int NT = OMP_NTHREAD;   // number of OpenMP threads
#  else
   const int NT = 1;
#  endif

   const int NData   = 6;
   const int ArrayID = 0;
   const int NPG_Max = POT_GPU_NPGROUP;

   double   QuadMom_2nd[NData] = { 0.0 };
   double **OMP_QuadMom_2nd    = NULL;
   Aux_AllocateArray2D( OMP_QuadMom_2nd, NT, NData );

   for (int TID=0; TID<NT; TID++) {
   for (int b=0; b<NData; b++)    {
      OMP_QuadMom_2nd[TID][b] = 0.0;
   }}


   for (int lv=0; lv<NLEVEL; lv++)
   {
      const double dh        = amr->dh[lv];
      const double dv        = CUBE( dh );
      const double TimeNew   = Time[lv];
      const int    NTotal    = amr->NPatchComma[lv][1] / 8;
            int   *PID0_List = new int [NTotal];

      for (int t=0; t<NTotal; t++)  PID0_List[t] = 8*t;

      for (int Disp=0; Disp<NTotal; Disp+=NPG_Max)
      {
//       prepare the potential file
         int NPG = ( NPG_Max < NTotal-Disp ) ? NPG_Max : NTotal-Disp;

         Prepare_PatchData( lv, TimeNew, &h_Pot_Array_P_Out[ArrayID][0][0][0][0], NULL,
                            GRA_GHOST_SIZE, NPG, PID0_List+Disp, _POTE, _NONE,
                            OPT__GRA_INT_SCHEME, INT_NONE, UNIT_PATCH, (GRA_GHOST_SIZE==0)?NSIDE_00:NSIDE_06, false,
                            OPT__BC_FLU, OPT__BC_POT, -1.0, -1.0, -1.0, -1.0, false );

#        pragma omp parallel for schedule( runtime )
         for (int PID_IDX=0; PID_IDX<8*NPG; PID_IDX++)
         {
#           ifdef OPENMP
            const int TID = omp_get_thread_num();
#           else
            const int TID = 0;
#           endif

            const int PID = 8*Disp + PID_IDX;

            if ( amr->patch[0][lv][PID]->son != -1 )  continue;

            for (int k=0; k<PS1; k++)  {  const double z = amr->patch[0][lv][PID]->EdgeL[2] + (k+0.5)*dh; const int kk = k + GRA_GHOST_SIZE;
            for (int j=0; j<PS1; j++)  {  const double y = amr->patch[0][lv][PID]->EdgeL[1] + (j+0.5)*dh; const int jj = j + GRA_GHOST_SIZE;
            for (int i=0; i<PS1; i++)  {  const double x = amr->patch[0][lv][PID]->EdgeL[0] + (i+0.5)*dh; const int ii = i + GRA_GHOST_SIZE;

               const double dx = x - BoxCenter[0];
               const double dy = y - BoxCenter[1];
               const double dz = z - BoxCenter[2];
               const double r  = sqrt(  SQR( dx ) + SQR( dy ) + SQR( dz )  );

               const double dens  = amr->patch[ amr->FluSg[lv] ][lv][PID]->fluid[DENS][k][j][i];
               const double momx  = amr->patch[ amr->FluSg[lv] ][lv][PID]->fluid[MOMX][k][j][i];
               const double momy  = amr->patch[ amr->FluSg[lv] ][lv][PID]->fluid[MOMY][k][j][i];
               const double momz  = amr->patch[ amr->FluSg[lv] ][lv][PID]->fluid[MOMZ][k][j][i];
               const double _dens = 1.0 / dens;

               const real (*PrepPotPtr)[GRA_NXT][GRA_NXT] = h_Pot_Array_P_Out[ArrayID][PID_IDX];

//             compute the potential gradient through central difference method
               const double dPhi_dx = ( PrepPotPtr[kk  ][jj  ][ii+1] - PrepPotPtr[kk  ][jj  ][ii-1] ) / (2.0 * dh);
               const double dPhi_dy = ( PrepPotPtr[kk  ][jj+1][ii  ] - PrepPotPtr[kk  ][jj-1][ii  ] ) / (2.0 * dh);
               const double dPhi_dz = ( PrepPotPtr[kk+1][jj  ][ii  ] - PrepPotPtr[kk-1][jj  ][ii  ] ) / (2.0 * dh);

               const double trace = _dens * ( SQR(momx) + SQR(momy) + SQR(momz) )
                                  -  dens * ( dx * dPhi_dx + dy * dPhi_dy + dz * dPhi_dz );

               OMP_QuadMom_2nd[TID][0] += dv * ( 2.0 * _dens * momx * momx - (2.0 / 3.0) * trace
                                               - 2.0 *  dens * dx * dPhi_dx                      );  // Ixx
               OMP_QuadMom_2nd[TID][1] += dv * ( 2.0 * _dens * momx * momy
                                               -        dens * ( dx * dPhi_dy + dy * dPhi_dx )   );  // Ixy
               OMP_QuadMom_2nd[TID][2] += dv * ( 2.0 * _dens * momx * momz
                                               -        dens * ( dx * dPhi_dz + dz * dPhi_dx )   );  // Ixz
               OMP_QuadMom_2nd[TID][3] += dv * ( 2.0 * _dens * momy * momy - (2.0 / 3.0) * trace
                                               - 2.0 *  dens * dy * dPhi_dy                      );  // Iyy
               OMP_QuadMom_2nd[TID][4] += dv * ( 2.0 * _dens * momy * momz
                                               -        dens * ( dy * dPhi_dz + dz * dPhi_dy )   );  // Iyz
               OMP_QuadMom_2nd[TID][5] += dv * ( 2.0 * _dens * momz * momz - (2.0 / 3.0) * trace
                                               - 2.0 *  dens * dz * dPhi_dz                      );  // Izz

            }}} // i,j,k
         } // for (int PID=0; PID<amr->NPatchComma[lv][1]; PID++)
      } // for (int Disp=0; Disp<NTotal; Disp+=NPG_Max)

      delete [] PID0_List;
   } // for (int lv=0; lv<NLEVEL; lv++)


// sum over all OpenMP threads
   for (int b=0; b<NData; b++) {
   for (int t=0; t<NT; t++)    {
      QuadMom_2nd[b] += OMP_QuadMom_2nd[t][b];
   }}

// free per-thread arrays
   Aux_DeallocateArray2D( OMP_QuadMom_2nd );


// collect data from all ranks (in-place reduction)
#  ifndef SERIAL
   if ( MPI_Rank == 0 )   MPI_Reduce( MPI_IN_PLACE, QuadMom_2nd, NData, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );
   else                   MPI_Reduce( QuadMom_2nd,  NULL,        NData, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );
#  endif // ifndef SERIAL


// multiply the coefficient (G / c^4) and unit
   const double QuadMom_fac = UNIT_M * SQR( UNIT_V ) * Const_NewtonG / pow( Const_c, 4.0 );

   for (int b=0; b<NData; b++)   QuadMom_2nd[b] *= QuadMom_fac;


// write to the file "Record__QuadMom_2nd"
   if ( MPI_Rank == 0 )
   {

      static bool FirstTime = true;

//    file header
      if ( FirstTime )
      {
         if ( Aux_CheckFileExist(filename_QuadMom_2nd) )
         {
             Aux_Message( stderr, "WARNING : file \"%s\" already exists !!\n", filename_QuadMom_2nd );
         }
         else
         {
             FILE *file_QuadMom_2nd = fopen( filename_QuadMom_2nd, "w" );
             fprintf( file_QuadMom_2nd, "#%14s %12s %16s %16s %16s %16s %16s %16s\n",
                                        "Time [sec]", "Step", "Ixx", "Ixy", "Ixz", "Iyy", "Iyz", "Izz" );
             fclose( file_QuadMom_2nd );
         }

         FirstTime = false;
      }

      FILE *file_QuadMom_2nd = fopen( filename_QuadMom_2nd, "a" );

                                    fprintf( file_QuadMom_2nd, "%15.7e %12ld", Time[0] * UNIT_T, Step );
      for (int b=0; b<NData; b++)   fprintf( file_QuadMom_2nd, "%17.7e", QuadMom_2nd[b] );
                                    fprintf( file_QuadMom_2nd, "\n" );

      fclose( file_QuadMom_2nd );

   } // if ( MPI_Rank == 0 )

} // FUNCTION : Record_CCSN_GWSignal()
#endif // ifdef GRAVITY



//-------------------------------------------------------------------------------------------------------
// Function    :  Detect_CoreBounce
// Description :  Check whether the core bounce occurs
//
// Note        :  1. Invoked by Record_CCSN()
//             :  2. Based on two criteria:
//                   --> (a) The central density is larger than 2e14
//                       (b) Any cells within 30km has entropy larger than 3
//-------------------------------------------------------------------------------------------------------
void Detect_CoreBounce()
{

// (1) criterion 1: central density is larger than 2e14
   if ( CCSN_CentralDens < 2e14 )   return;


// (2) criterion 2: any cells within 30km has entropy larger than 3
   const double BoxCenter[3] = { amr->BoxCenter[0], amr->BoxCenter[1], amr->BoxCenter[2] };

// allocate memory for per-thread arrays
#  ifdef OPENMP
   const int NT = OMP_NTHREAD;
#  else
   const int NT = 1;
#  endif

   real MaxEntr = -HUGE_NUMBER;
   real OMP_MaxEntr[NT];


#  pragma omp parallel
   {
#     ifdef OPENMP
      const int TID = omp_get_thread_num();
#     else
      const int TID = 0;
#     endif

//    initialize arrays
      OMP_MaxEntr[TID] = -HUGE_NUMBER;

      for (int lv=0; lv<NLEVEL; lv++)
      {
         const double dh = amr->dh[lv];

#        pragma omp for schedule( runtime )
         for (int PID=0; PID<amr->NPatchComma[lv][1]; PID++)
         {
            if ( amr->patch[0][lv][PID]->son != -1 )  continue;

            for (int k=0; k<PS1; k++)  {  const double z = amr->patch[0][lv][PID]->EdgeL[2] + (k+0.5)*dh;
            for (int j=0; j<PS1; j++)  {  const double y = amr->patch[0][lv][PID]->EdgeL[1] + (j+0.5)*dh;
            for (int i=0; i<PS1; i++)  {  const double x = amr->patch[0][lv][PID]->EdgeL[0] + (i+0.5)*dh;

               const double x0 = x - BoxCenter[0];
               const double y0 = y - BoxCenter[1];
               const double z0 = z - BoxCenter[2];
               const double r  = sqrt(  SQR( x0 ) + SQR( y0 ) + SQR( z0 )  );

//             ignore cells outside 30km
               if ( r * UNIT_L > 3.0e6 )   continue;

//             retrieve the entropy and store the maximum value
               real u[NCOMP_TOTAL], Entr, Emag=NULL_REAL;

               for (int v=0; v<NCOMP_TOTAL; v++)   u[v] = amr->patch[ amr->FluSg[lv] ][lv][PID]->fluid[v][k][j][i];

#              ifdef MHD
               Emag = MHD_GetCellCenteredBEnergyInPatch( lv, PID, i, j, k, amr->MagSg[lv] );
#              endif

               Entr = Hydro_Con2Entr( u[DENS], u[MOMX], u[MOMY], u[MOMZ], u[ENGY], u+NCOMP_FLUID,
                                      false, NULL_REAL, Emag, EoS_DensEint2Entr_CPUPtr,
                                      EoS_AuxArray_Flt, EoS_AuxArray_Int, h_EoS_Table );

               OMP_MaxEntr[TID] = FMAX( OMP_MaxEntr[TID], Entr );

            }}} // i,j,k
         } // for (int PID=0; PID<amr->NPatchComma[lv][1]; PID++)
      } // for (int lv=0; lv<NLEVEL; lv++)
   } // OpenMP parallel region


// find the maximum over all OpenMP threads
   for (int TID=0; TID<NT; TID++)
      MaxEntr = FMAX( MaxEntr, OMP_MaxEntr[TID] );


// collect data from all ranks
#  ifndef SERIAL
   MPI_Allreduce( MPI_IN_PLACE, &MaxEntr, 1, MPI_GAMER_REAL, MPI_MAX, MPI_COMM_WORLD );
#  endif // ifndef SERIAL


   if ( MaxEntr > 3.0 )   CCSN_Is_PostBounce = true;

} // FUNCTION : Detect_CoreBounce()



//-------------------------------------------------------------------------------------------------------
// Function    :  Detect_Shock
// Description :  Measure the minimum, maximum, and average shock radius if a shock is present
//
// Note        :  1. Invoked by Record_CCSN()
//                2. The average shock radius is weighted by the inverse cell volume
//                3. Ref: Balsara and Spicer, 1999, JCP, 149, 270, sec 2.2
//-------------------------------------------------------------------------------------------------------
void Detect_Shock()
{

#  ifdef OPENMP
   const int NT = OMP_NTHREAD;
#  else
   const int NT = 1;
#  endif

   const int NGhost  = 1;
   const int PS1P2   = PS1 + 2*NGhost;
   const int NPG_Max = FLU_GPU_NPGROUP;
   const int TVarCC  = _DENS | _PASSIVE | _VELX | _VELY | _VELZ | _PRES;
   const int Stride1 = CUBE( PS1P2 );
   const int Stride2 = NCOMP_TOTAL * Stride1;
   const int VELX    = NCOMP_PASSIVE + 1;
   const int VELY    = NCOMP_PASSIVE + 2;
   const int VELZ    = NCOMP_PASSIVE + 3;
   const int PRES    = NCOMP_PASSIVE + 4;

   const real   ThresholdFac_Pres = 0.5;
   const real   ThresholdFac_Vel  = 0.1;
   const double BoxCenter[3]      = { amr->BoxCenter[0], amr->BoxCenter[1], amr->BoxCenter[2] };


   double Shock_Min    =  HUGE_NUMBER;
   double Shock_Max    = -HUGE_NUMBER;
   double Shock_Ave    =  0.0;
   double Shock_Weight =  0.0;
   int    Shock_Found  =  false;

   double OMP_Shock_Min   [NT];
   double OMP_Shock_Max   [NT];
   double OMP_Shock_Ave   [NT];
   double OMP_Shock_Weight[NT];
   int    OMP_Shock_Found [NT];

   real *OMP_Fluid    = new real [ 8*NPG_Max*Stride2 ];
   real *OMP_Pres_Min = new real [ 8*NPG_Max*Stride1 ];
   real *OMP_Cs_Min   = new real [ 8*NPG_Max*Stride1 ];

   for (int t=0; t<NT; t++)
   {
      OMP_Shock_Min   [t] =  HUGE_NUMBER;
      OMP_Shock_Max   [t] = -HUGE_NUMBER;
      OMP_Shock_Ave   [t] = 0.0;
      OMP_Shock_Weight[t] = 0.0;
      OMP_Shock_Found [t] = false;
   }


   for (int lv=0; lv<NLEVEL; lv++)
   {
      const double dh        = amr->dh[lv];
      const double _dv       = 1.0 / CUBE( dh );
      const int    NTotal    = amr->NPatchComma[lv][1] / 8;
            int   *PID0_List = new int [NTotal];

      for (int t=0; t<NTotal; t++)  PID0_List[t] = 8*t;

      for (int Disp=0; Disp<NTotal; Disp+=NPG_Max)
      {
         const int NPG = MIN( NPG_Max, NTotal-Disp );

//       (1-a) prepare the primitive and passive variables
//             note that the data are prepared in order of density, Passive, velx, vely, velz, and pressure
         Prepare_PatchData( lv, Time[lv], OMP_Fluid, NULL,
                            NGhost, NPG, PID0_List+Disp, TVarCC, _NONE,
                            OPT__FLU_INT_SCHEME, INT_NONE, UNIT_PATCH, NSIDE_26, false,
                            OPT__BC_FLU, BC_POT_NONE, -1.0, -1.0, -1.0, -1.0, false );


#        pragma omp parallel for schedule( runtime )
         for (int PID_IDX=0; PID_IDX<8*NPG; PID_IDX++)
         {
#           ifdef OPENMP
            const int TID = omp_get_thread_num();
#           else
            const int TID = 0;
#           endif

            const int PID = 8*Disp + PID_IDX;

            if ( amr->patch[0][lv][PID]->son != -1 )  continue;

            real (*Fluid   )[PS1P2][PS1P2][PS1P2] = ( real(*)[PS1P2][PS1P2][PS1P2] ) ( OMP_Fluid    + PID_IDX*Stride2 );
            real (*Pres_Min)       [PS1P2][PS1P2] = ( real(*)       [PS1P2][PS1P2] ) ( OMP_Pres_Min + PID_IDX*Stride1 );
            real (*Cs_Min  )       [PS1P2][PS1P2] = ( real(*)       [PS1P2][PS1P2] ) ( OMP_Cs_Min   + PID_IDX*Stride1 );

//          (1-b) compute the sound speed and store in the density field
            for (int k=0; k<PS1P2; k++)  {
            for (int j=0; j<PS1P2; j++)  {
            for (int i=0; i<PS1P2; i++)  {

               real Passive[NCOMP_PASSIVE];
               real Dens = Fluid[DENS][k][j][i];
               real Pres = Fluid[PRES][k][j][i];

               for (int v=0; v<NCOMP_PASSIVE; v++)   Passive[v] = Fluid[v+1][k][j][i];

               const real CSqr = EoS_DensPres2CSqr_CPUPtr( Dens, Pres, Passive,
                                                           EoS_AuxArray_Flt, EoS_AuxArray_Int, h_EoS_Table );

               Fluid[DENS][k][j][i] = SQRT( CSqr );

            }}} // i, j, k


//          (2) use a naive method to find the minimum of pressure and sound speed in the local 3x3 subarray
            for (int k=NGhost; k<PS1+NGhost; k++)  {
            for (int j=NGhost; j<PS1+NGhost; j++)  {
            for (int i=NGhost; i<PS1+NGhost; i++)  {

               real Pres_Min_Loc = HUGE_NUMBER;
               real Cs_Min_Loc   = HUGE_NUMBER;

               for (int kk=k-NGhost; kk<=k+NGhost; kk++)  {
               for (int jj=j-NGhost; jj<=j+NGhost; jj++)  {
               for (int ii=i-NGhost; ii<=i+NGhost; ii++)  {
                  Pres_Min_Loc = FMIN( Pres_Min_Loc, Fluid[PRES][kk][jj][ii] );
                  Cs_Min_Loc   = FMIN( Cs_Min_Loc,   Fluid[DENS][kk][jj][ii] );
               }}}

               Pres_Min[k][j][i] = Pres_Min_Loc;
               Cs_Min  [k][j][i] = Cs_Min_Loc;

            }}} // i, j, k


            for (int k=0; k<PS1; k++)  {  const double z = amr->patch[0][lv][PID]->EdgeL[2] + (k+0.5)*dh; const int kk = k+NGhost;
            for (int j=0; j<PS1; j++)  {  const double y = amr->patch[0][lv][PID]->EdgeL[1] + (j+0.5)*dh; const int jj = j+NGhost;
            for (int i=0; i<PS1; i++)  {  const double x = amr->patch[0][lv][PID]->EdgeL[0] + (i+0.5)*dh; const int ii = i+NGhost;

               const double dx = x - BoxCenter[0];
               const double dy = y - BoxCenter[1];
               const double dz = z - BoxCenter[2];
               const double r  = sqrt(  SQR( dx ) + SQR( dy ) + SQR( dz )  );

//             (3) evaluate the undivided gradient of pressure and the undivided divergence of velocity
               real GradP = (real)0.5 * (   FABS( Fluid[PRES][kk+1][jj  ][ii  ] - Fluid[PRES][kk-1][jj  ][ii  ] )
                                          + FABS( Fluid[PRES][kk  ][jj+1][ii  ] - Fluid[PRES][kk  ][jj-1][ii  ] )
                                          + FABS( Fluid[PRES][kk  ][jj  ][ii+1] - Fluid[PRES][kk  ][jj  ][ii-1] )  );

               real DivV  = (real)0.5 * (       ( Fluid[VELZ][kk+1][jj  ][ii  ] - Fluid[VELZ][kk-1][jj  ][ii  ] )
                                          +     ( Fluid[VELY][kk  ][jj+1][ii  ] - Fluid[VELY][kk  ][jj-1][ii  ] )
                                          +     ( Fluid[VELX][kk  ][jj  ][ii+1] - Fluid[VELX][kk  ][jj  ][ii-1] )  );

//             (4) examine the criteria for detecting strong shock
               if (  ( GradP >=  ThresholdFac_Pres * Pres_Min[kk][jj][ii] )  &&
                     ( DivV  <= -ThresholdFac_Vel  * Cs_Min  [kk][jj][ii] )      )
               {
                  OMP_Shock_Min   [TID]  = fmin( r, OMP_Shock_Min[TID] );
                  OMP_Shock_Max   [TID]  = fmax( r, OMP_Shock_Max[TID] );
                  OMP_Shock_Ave   [TID] += r * _dv;
                  OMP_Shock_Weight[TID] += _dv;
                  OMP_Shock_Found [TID]  = true;
               }

            }}} // i,j,k
         } // for (int PID_IDX=0; PID_IDX<8*NPG; PID_IDX++)
      } // for (int Disp=0; Disp<NTotal; Disp+=NPG_Max)

      delete [] PID0_List;
   } // for (int lv=0; lv<NLEVEL; lv++)


// free memory
   delete [] OMP_Fluid;
   delete [] OMP_Pres_Min;
   delete [] OMP_Cs_Min;


// collect data over all OpenMP threads
   for (int t=0; t<NT; t++)
   {
      Shock_Min     = fmin( Shock_Min, OMP_Shock_Min[t] );
      Shock_Max     = fmax( Shock_Max, OMP_Shock_Max[t] );
      Shock_Ave    += OMP_Shock_Ave[t];
      Shock_Weight += OMP_Shock_Weight[t];
      Shock_Found  |= OMP_Shock_Found[t];
   }


// collect data from all ranks (in-place reduction)
#  ifndef SERIAL
   MPI_Allreduce( MPI_IN_PLACE, &Shock_Min,    1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD );
   MPI_Allreduce( MPI_IN_PLACE, &Shock_Max,    1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
   MPI_Allreduce( MPI_IN_PLACE, &Shock_Ave,    1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
   MPI_Allreduce( MPI_IN_PLACE, &Shock_Weight, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
   MPI_Allreduce( MPI_IN_PLACE, &Shock_Found,  1, MPI_INT,    MPI_BOR, MPI_COMM_WORLD );
#  endif // ifndef SERIAL


// update the shock radius
   CCSN_Rsh_Min = ( Shock_Found ) ? Shock_Min                : 0.0;
   CCSN_Rsh_Max = ( Shock_Found ) ? Shock_Max                : 0.0;
   CCSN_Rsh_Ave = ( Shock_Found ) ? Shock_Ave / Shock_Weight : 0.0;

} // FUNCTION : Detect_Shock()
