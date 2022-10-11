#include "GAMER.h"

#ifdef SUPPORT_HDF5
#include "hdf5.h"
#endif

#include <string>


// floating-point type in the input particle file
typedef double real_par_in;
//typedef float  real_par_in;

extern char    Merger_File_Par1[1000];
extern char    Merger_File_Par2[1000];
extern char    Merger_File_Par3[1000];
extern int     Merger_Coll_NumHalos;
extern double  Merger_Coll_PosX1;
extern double  Merger_Coll_PosY1;
extern double  Merger_Coll_PosX2;
extern double  Merger_Coll_PosY2;
extern double  Merger_Coll_PosX3;
extern double  Merger_Coll_PosY3;
extern double  Merger_Coll_VelX1;
extern double  Merger_Coll_VelY1;
extern double  Merger_Coll_VelX2;
extern double  Merger_Coll_VelY2;
extern double  Merger_Coll_VelX3;
extern double  Merger_Coll_VelY3;
extern bool    Merger_Coll_LabelCenter;


// variables that need to be record in Record__Center
// =======================================================================================
extern double  Bondi_MassBH1;
extern double  Bondi_MassBH2;
extern double  Bondi_MassBH3;
extern double  Mdot_BH1;
extern double  Mdot_BH2;
extern double  Mdot_BH3;

extern double CM_Bondi_SinkMass[3];
extern double CM_Bondi_SinkMomX[3];
extern double CM_Bondi_SinkMomY[3];
extern double CM_Bondi_SinkMomZ[3];
extern double CM_Bondi_SinkMomXAbs[3];
extern double CM_Bondi_SinkMomYAbs[3];
extern double CM_Bondi_SinkMomZAbs[3];
extern double CM_Bondi_SinkE[3];
extern double CM_Bondi_SinkEk[3];
extern double CM_Bondi_SinkEt[3];
extern int    CM_Bondi_SinkNCell[3];

extern double GasVel[3][3];  // gas velocity
extern double SoundSpeed[3];
extern double GasDens[3];
extern double RelativeVel[3];
extern double Jet_Vec[3][3]; // jet direction  
extern double Mdot[3]; // the feedback injection rate
extern double Pdot[3];
extern double Edot[3];
extern double E_inj_exp[3];
extern double dt_base;
extern double E_power_inj[3];
// =======================================================================================

#ifdef MASSIVE_PARTICLES
long Read_Particle_Number_ClusterMerger(std::string filename);
void Read_Particles_ClusterMerger(std::string filename, long offset, long num,
                                  real_par_in xpos[], real_par_in ypos[],
                                  real_par_in zpos[], real_par_in xvel[],
                                  real_par_in yvel[], real_par_in zvel[],
                                  real_par_in mass[], real_par_in ptype[]);
void GetClusterCenter( double Cen[][3], double BH_Vel[][3] );

//-------------------------------------------------------------------------------------------------------
// Function    :  Par_Init_ByFunction_ClusterMerger
// Description :  Initialize all particle attributes for the merging cluster test
//                --> Modified from "Par_Init_ByFile.cpp"
//
// Note        :  1. Invoked by Init_GAMER() using the function pointer "Par_Init_ByFunction_Ptr"
//                   --> This function pointer may be reset by various test problem initializers, in which case
//                       this funtion will become useless
//                2. Periodicity should be taken care of in this function
//                   --> No particles should lie outside the simulation box when the periodic BC is adopted
//                   --> However, if the non-periodic BC is adopted, particles are allowed to lie outside the box
//                       (more specifically, outside the "active" region defined by amr->Par->RemoveCell)
//                       in this function. They will later be removed automatically when calling Par_Aux_InitCheck()
//                       in Init_GAMER().
//                3. Particles set by this function are only temporarily stored in this MPI rank
//                   --> They will later be redistributed when calling Par_FindHomePatch_UniformGrid()
//                       and LB_Init_LoadBalance()
//                   --> Therefore, there is no constraint on which particles should be set by this function
//                4. File format: plain C binary in the format [Number of particles][Particle attributes]
//                   --> [Particle 0][Attribute 0], [Particle 0][Attribute 1], ...
//                   --> Note that it's different from the internal data format in the particle repository,
//                       which is [Particle attributes][Number of particles]
//                   --> Currently it only loads particle mass, position x/y/z, and velocity x/y/z
//                       (and exactly in this order)
//
// Parameter   :  NPar_ThisRank : Number of particles to be set by this MPI rank
//                NPar_AllRank  : Total Number of particles in all MPI ranks
//                ParMass       : Particle mass     array with the size of NPar_ThisRank
//                ParPosX/Y/Z   : Particle position array with the size of NPar_ThisRank
//                ParVelX/Y/Z   : Particle velocity array with the size of NPar_ThisRank
//                ParTime       : Particle time     array with the size of NPar_ThisRank
//                ParType       : Particle type     array with the size of NPar_ThisRank
//                AllAttribute  : Pointer array for all particle attributes
//                                --> Dimension = [PAR_NATT_TOTAL][NPar_ThisRank]
//                                --> Use the attribute indices defined in Field.h (e.g., Idx_ParCreTime)
//                                    to access the data
//
// Return      :  ParMass, ParPosX/Y/Z, ParVelX/Y/Z, ParTime, ParType, AllAttribute
//-------------------------------------------------------------------------------------------------------

void Par_Init_ByFunction_ClusterMerger( const long NPar_ThisRank, const long NPar_AllRank,
                                        real *ParMass, real *ParPosX, real *ParPosY, real *ParPosZ,
                                        real *ParVelX, real *ParVelY, real *ParVelZ, real *ParTime,
                                        real *ParType, real *AllAttribute[PAR_NATT_TOTAL] )
{

#ifdef SUPPORT_HDF5


   if ( MPI_Rank == 0 )    Aux_Message( stdout, "%s ...\n", __FUNCTION__ );

// check file existence
   if ( !Aux_CheckFileExist(Merger_File_Par1) )
      Aux_Error( ERROR_INFO, "file \"%s\" does not exist !!\n", Merger_File_Par1 );

   if ( Merger_Coll_NumHalos > 1  &&  !Aux_CheckFileExist(Merger_File_Par2) )
      Aux_Error( ERROR_INFO, "file \"%s\" does not exist !!\n", Merger_File_Par2 );

   if ( Merger_Coll_NumHalos > 2  &&  !Aux_CheckFileExist(Merger_File_Par3) )
      Aux_Error( ERROR_INFO, "file \"%s\" does not exist !!\n", Merger_File_Par3 );

   const std::string filename1(Merger_File_Par1);
   const std::string filename2(Merger_File_Par2);
   const std::string filename3(Merger_File_Par3);

// check file size
   long NPar_EachCluster[3] = {0,0,0};
   long NPar_AllCluster;

   if ( MPI_Rank == 0 ) {

      NPar_EachCluster[0] = Read_Particle_Number_ClusterMerger(filename1);

      Aux_Message( stdout, "   Number of particles in cluster 1 = %ld\n",
                   NPar_EachCluster[0] );

      if ( Merger_Coll_NumHalos > 1 ) {
         NPar_EachCluster[1] = Read_Particle_Number_ClusterMerger(filename2);
         Aux_Message( stdout, "   Number of particles in cluster 2 = %ld\n", NPar_EachCluster[1] );
      }

      if ( Merger_Coll_NumHalos > 2 ) {
         NPar_EachCluster[2] = Read_Particle_Number_ClusterMerger(filename3);
         Aux_Message( stdout, "   Number of particles in cluster 3 = %ld\n", NPar_EachCluster[2] );
      }

   }

   MPI_Bcast(NPar_EachCluster, 3, MPI_LONG, 0, MPI_COMM_WORLD);

   NPar_AllCluster = NPar_EachCluster[0] + NPar_EachCluster[1] + NPar_EachCluster[2];

   if ( NPar_AllCluster != NPar_AllRank )
      Aux_Error( ERROR_INFO, "total number of particles found in cluster [%ld] != expect [%ld] !!\n",
                 NPar_AllCluster, NPar_AllRank );

   // prepare to load data
   if ( MPI_Rank == 0 )    Aux_Message( stdout, "   Preparing to load data ... " );

   const int NCluster = Merger_Coll_NumHalos;
   long NPar_ThisRank_EachCluster[3]={0,0,0}, Offset[3];   // [0/1/2] --> cluster 1/2/3

   for (int c=0; c<NCluster; c++)
   {
      // get the number of particles loaded by each rank for each cluster
      long NPar_ThisCluster_EachRank[MPI_NRank];

      switch (c) {
      case 0:
         NPar_ThisRank_EachCluster[0] = NPar_EachCluster[0] / MPI_NRank + ( (MPI_Rank<NPar_EachCluster[0]%MPI_NRank)?1:0 );
         break;
      case 1:
         if (NCluster == 2)
            NPar_ThisRank_EachCluster[1] = NPar_ThisRank - NPar_ThisRank_EachCluster[0];
         else
            NPar_ThisRank_EachCluster[1] = NPar_EachCluster[1] / MPI_NRank + ( (MPI_Rank<NPar_EachCluster[1]%MPI_NRank)?1:0 );
         break;
      case 2:
         NPar_ThisRank_EachCluster[2] = NPar_ThisRank - NPar_ThisRank_EachCluster[0] - NPar_ThisRank_EachCluster[1];
         break;
      }

      MPI_Allgather( &NPar_ThisRank_EachCluster[c], 1, MPI_LONG, NPar_ThisCluster_EachRank, 1, MPI_LONG, MPI_COMM_WORLD );

      // check if the total number of particles is correct
      long NPar_Check = 0;
      for (int r=0; r<MPI_NRank; r++)
         NPar_Check += NPar_ThisCluster_EachRank[r];
         if ( NPar_Check != NPar_EachCluster[c] )
            Aux_Error( ERROR_INFO, "total number of particles in cluster %d: found (%ld) != expect (%ld) !!\n",
                       c, NPar_Check, NPar_EachCluster[c] );

      // set the file offset for this rank
      Offset[c] = 0;
      for (int r=0; r<MPI_Rank; r++)
         Offset[c] = Offset[c] + NPar_ThisCluster_EachRank[r];
   }

   if ( MPI_Rank == 0 )    Aux_Message( stdout, "done\n" );

   // load data to the particle repository

   const std::string filenames[3] = { Merger_File_Par1, Merger_File_Par2, Merger_File_Par3 };

   for ( int c=0; c<NCluster; c++ )
   {
      // load data
      if ( MPI_Rank == 0 )    Aux_Message( stdout, "   Loading cluster %d ... \n", c+1 );

      real_par_in *mass  = new real_par_in [NPar_ThisRank_EachCluster[c]];
      real_par_in *xpos  = new real_par_in [NPar_ThisRank_EachCluster[c]];
      real_par_in *ypos  = new real_par_in [NPar_ThisRank_EachCluster[c]];
      real_par_in *zpos  = new real_par_in [NPar_ThisRank_EachCluster[c]];
      real_par_in *xvel  = new real_par_in [NPar_ThisRank_EachCluster[c]];
      real_par_in *yvel  = new real_par_in [NPar_ThisRank_EachCluster[c]];
      real_par_in *zvel  = new real_par_in [NPar_ThisRank_EachCluster[c]];
      real_par_in *ptype = new real_par_in [NPar_ThisRank_EachCluster[c]];

      Read_Particles_ClusterMerger( filenames[c], Offset[c], NPar_ThisRank_EachCluster[c],
                                    xpos, ypos, zpos, xvel, yvel, zvel, mass, ptype );

#     ifndef TRACER
      for (long p=0; p<NPar_ThisRank_EachCluster[c]; p++) {
         if ( ptype[p] == PTYPE_TRACER )
            Aux_Error( ERROR_INFO,
"Tracer particles were found in the input data for cluster %d, but TRACER is not defined!\n",
                       c );
      }
#     endif

      if ( MPI_Rank == 0 ) Aux_Message( stdout, "done\n" );

      // store data to the particle repository
      if ( MPI_Rank == 0 )
         Aux_Message( stdout, "   Storing cluster %d to the particle repository ... \n", c+1 );

      // Compute offsets for assigning particles

      double coffset;
      switch (c) {
      case 0:
         coffset = 0;
         break;
      case 1:
         coffset = NPar_ThisRank_EachCluster[0];
         break;
      case 2:
         coffset = NPar_ThisRank_EachCluster[0]+NPar_ThisRank_EachCluster[1];
         break;
      }

      for (long p=0; p<NPar_ThisRank_EachCluster[c]; p++)
      {
         // particle index offset
         const long pp = p + coffset;

         // set the particle type
         ParType[pp] = real( ptype[p] );

         // --> convert to code unit before storing to the particle repository to avoid floating-point overflow
         // --> we have assumed that the loaded data are in cgs

         ParPosX[pp] = real( xpos[p] / UNIT_L );
         ParPosY[pp] = real( ypos[p] / UNIT_L );
         ParPosZ[pp] = real( zpos[p] / UNIT_L );

         if ( ptype[p] == PTYPE_TRACER ) {
            // tracer particles have zero mass
            // and their velocities will be set by
            // the grid later
            ParMass[pp] = 0.0;
            ParVelX[pp] = 0.0;
            ParVelY[pp] = 0.0;
            ParVelX[pp] = 0.0;
         } else {
            // For massive particles get their mass
            // and velocity
            ParMass[pp] = real( mass[p] / UNIT_M );
            ParVelX[pp] = real( xvel[p] / UNIT_V );
            ParVelY[pp] = real( yvel[p] / UNIT_V );
            ParVelZ[pp] = real( zvel[p] / UNIT_V );
         }

         // synchronize all particles to the physical time at the base level
         ParTime[pp] = Time[0];

      }

      delete [] mass;
      delete [] xpos;
      delete [] ypos;
      delete [] zpos;
      delete [] xvel;
      delete [] yvel;
      delete [] zvel;
      delete [] ptype;

   } // for (int c=0; c<NCluster; c++)

   if ( MPI_Rank == 0 )    Aux_Message( stdout, "done\n" );

   // shift center (assuming the center of loaded particles = [0,0,0])
   if ( MPI_Rank == 0 )    Aux_Message( stdout, "   Shifting particle center and adding bulk velocity ... " );

   real *ParPos[3] = { ParPosX, ParPosY, ParPosZ };

   const double ClusterCenter1[3]
      = { Merger_Coll_PosX1, Merger_Coll_PosY1, amr->BoxCenter[2] };
   const double ClusterCenter2[3]
      = { Merger_Coll_PosX2, Merger_Coll_PosY2, amr->BoxCenter[2] };
   const double ClusterCenter3[3]
      = { Merger_Coll_PosX3, Merger_Coll_PosY3, amr->BoxCenter[2] };

   for (long p=0; p<NPar_ThisRank_EachCluster[0]; p++) {
      if ( (int)ParType[p] != PTYPE_TRACER ) {
         ParVelX[p] += Merger_Coll_VelX1;
         ParVelY[p] += Merger_Coll_VelY1;
      }
      for (int d=0; d<3; d++)
         ParPos[d][p] += ClusterCenter1[d];
   }

   // NO reset particle mass 

   for (long p=NPar_ThisRank_EachCluster[0]; p<NPar_ThisRank_EachCluster[0]+NPar_ThisRank_EachCluster[1]; p++) {
      if ( (int)ParType[p] != PTYPE_TRACER ) {
         ParVelX[p] += Merger_Coll_VelX2;
         ParVelY[p] += Merger_Coll_VelY2;
      }
      for (int d=0; d<3; d++)
         ParPos[d][p] += ClusterCenter2[d];
   }

   // NO reset particle mass


   for (long p=NPar_ThisRank_EachCluster[0]+NPar_ThisRank_EachCluster[1]; p<NPar_ThisRank; p++) {
      if ( (int)ParType[p] != PTYPE_TRACER ) {
         ParVelX[p] += Merger_Coll_VelX3;
         ParVelY[p] += Merger_Coll_VelY3;
      }
      for (int d=0; d<3; d++)
         ParPos[d][p] += ClusterCenter3[d];
   }

   if ( MPI_Rank == 0 )    Aux_Message( stdout, "done\n" );


   // label cluster centers
   if ( Merger_Coll_LabelCenter ) {
      if ( MPI_Rank == 0 )    Aux_Message( stdout, "   Labeling cluster centers ... " );

      const double Centers[3][3] = {  { ClusterCenter1[0], ClusterCenter1[1], ClusterCenter1[2] },
                                      { ClusterCenter2[0], ClusterCenter2[1], ClusterCenter2[2] },
                                      { ClusterCenter3[0], ClusterCenter3[1], ClusterCenter3[2] }  };
      long pidx_offset = 0;

      for (int c=0; c<NCluster; c++) {
         long   min_pidx   = -1;
         real   min_pos[3] = { NULL_REAL, NULL_REAL, NULL_REAL };
         double min_r      = __DBL_MAX__;

         // get the particle in this rank closest to the cluster center
         for (long p=pidx_offset; p<pidx_offset+NPar_ThisRank_EachCluster[c]; p++) {
            const double r = SQR( ParPos[0][p] - Centers[c][0] ) +
                             SQR( ParPos[1][p] - Centers[c][1] ) +
                             SQR( ParPos[2][p] - Centers[c][2] );
            if ( r < min_r ) {
               min_pidx   = p;
               min_r      = r;
               min_pos[0] = ParPos[0][p];
               min_pos[1] = ParPos[1][p];
               min_pos[2] = ParPos[2][p];
            }
         }

         // collect data among all ranks
         double min_r_allrank;
         int    NFound_ThisRank=0, NFound_AllRank;
         MPI_Allreduce( &min_r, &min_r_allrank, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD );
         if ( min_r == min_r_allrank ) {
            ParType[min_pidx] = PTYPE_CEN + c;
            NFound_ThisRank = 1;
         }

         // check if one and only one particle is labeled
         MPI_Allreduce( &NFound_ThisRank, &NFound_AllRank, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
         if ( NFound_AllRank != 1 )
            Aux_Error( ERROR_INFO, "NFound_AllRank (%d) != 1 for cluster %d !!\n", NFound_AllRank, c );

         // update the particle index offset for the next cluster
         pidx_offset += NPar_ThisRank_EachCluster[c];
      } // for (int c=0; c<NCluster; c++)

      if ( MPI_Rank == 0 )    Aux_Message( stdout, "done\n" );
   } // if ( Merger_Coll_LabelCenter )


   if ( MPI_Rank == 0 )    Aux_Message( stdout, "%s ... done\n", __FUNCTION__ );

#endif // #ifdef SUPPORT_HDF5

} // FUNCTION : Par_Init_ByFunction_ClusterMerger

#ifdef SUPPORT_HDF5

long Read_Particle_Number_ClusterMerger(std::string filename)
{

   hid_t   file_id, dataset, dataspace;
   herr_t  status;
   hsize_t dims[1], maxdims[1];

   int rank;

   file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

   dataset   = H5Dopen(file_id, "particle_mass", H5P_DEFAULT);
   dataspace = H5Dget_space(dataset);
   rank      = H5Sget_simple_extent_dims(dataspace, dims, maxdims);

   H5Sclose(dataspace);
   H5Dclose(dataset);
   H5Fclose(file_id);

   return (long)dims[0];

} // FUNCTION : Read_Particle_Number_ClusterMerger

void Read_Particles_ClusterMerger( std::string filename, long offset, long num,
                                   real_par_in xpos[], real_par_in ypos[],
                                   real_par_in zpos[], real_par_in xvel[],
                                   real_par_in yvel[], real_par_in zvel[],
                                   real_par_in mass[], real_par_in ptype[] )
{

   hid_t   file_id, dataset, dataspace, memspace;
   herr_t  status;

   hsize_t start[2], stride[2], count[2], dims[2], maxdims[2];
   hsize_t start1d[1], stride1d[1], count1d[1], dims1d[1], maxdims1d[1];
   hsize_t start0[1];

   int rank;

   stride[0] = 1;
   stride[1] = 1;
   start[0] = (hsize_t)offset;

   file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

   dataset   = H5Dopen(file_id, "particle_position", H5P_DEFAULT);

   dataspace = H5Dget_space(dataset);

   rank      = H5Sget_simple_extent_dims(dataspace, dims, maxdims);

   count[0] = (hsize_t)num;
   count[1] = 1;

   dims[0] = count[0];
   dims[1] = 1;

   count1d[0] = (hsize_t)num;
   dims1d[0] = count1d[0];
   stride1d[0] = 1;
   start1d[0] = 0;
   start[1] = 0;

   status = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, start,
                                 stride, count, NULL);
   memspace = H5Screate_simple(1, dims1d, NULL);
   status = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, start1d,
                                 stride1d, count1d, NULL);
   status = H5Dread(dataset, H5T_NATIVE_DOUBLE, memspace, dataspace,
                     H5P_DEFAULT, xpos);

   if (status < 0) {
      Aux_Message(stderr, "Could not read particle x-position!!\n");
   }

   H5Sclose(memspace);
   H5Sclose(dataspace);
   dataspace = H5Dget_space(dataset);

   start[1] = 1;

   status = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, start,
                                 stride, count, NULL);
   memspace = H5Screate_simple(1, dims1d, NULL);
   status = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, start1d,
                                 stride1d, count1d, NULL);
   status = H5Dread(dataset, H5T_NATIVE_DOUBLE, memspace, dataspace,
                     H5P_DEFAULT, ypos);

   if (status < 0) {
      Aux_Message(stderr, "Could not read particle y-position!!\n");
   }

   H5Sclose(memspace);
   H5Sclose(dataspace);
   dataspace = H5Dget_space(dataset);

   start[1] = 2;

   status = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, start,
                                 stride, count, NULL);
   memspace = H5Screate_simple(1, dims1d, NULL);
   status = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, start1d,
                                 stride1d, count1d, NULL);
   status = H5Dread(dataset, H5T_NATIVE_DOUBLE, memspace, dataspace,
                     H5P_DEFAULT, zpos);

   if (status < 0) {
      Aux_Message(stderr, "Could not read particle z-position!!\n");
   }

   H5Sclose(memspace);
   H5Sclose(dataspace);
   H5Dclose(dataset);

   dataset   = H5Dopen(file_id, "particle_velocity", H5P_DEFAULT);

   dataspace = H5Dget_space(dataset);

   start[1] = 0;

   status = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, start,
                                 stride, count, NULL);
   memspace = H5Screate_simple(1, dims1d, NULL);
   status = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, start1d,
                                 stride1d, count1d, NULL);
   status = H5Dread(dataset, H5T_NATIVE_DOUBLE, memspace, dataspace,
                     H5P_DEFAULT, xvel);

   if (status < 0) {
      Aux_Message(stderr, "Could not read particle x-velocity!!\n");
   }

   H5Sclose(memspace);
   H5Sclose(dataspace);

   dataspace = H5Dget_space(dataset);

   start[1] = 1;

   status = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, start,
                                 stride, count, NULL);
   memspace = H5Screate_simple(1, dims1d, NULL);
   status = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, start1d,
                                 stride1d, count1d, NULL);
   status = H5Dread(dataset, H5T_NATIVE_DOUBLE, memspace, dataspace,
                     H5P_DEFAULT, yvel);

   if (status < 0) {
      Aux_Message(stderr, "Could not read particle y-velocity!!\n");
   }

   H5Sclose(memspace);
   H5Sclose(dataspace);

   dataspace = H5Dget_space(dataset);

   start[1] = 2;

   status = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, start,
                                 stride, count, NULL);
   memspace = H5Screate_simple(1, dims1d, NULL);
   status = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, start1d,
                                 stride1d, count1d, NULL);
   status = H5Dread(dataset, H5T_NATIVE_DOUBLE, memspace, dataspace,
                     H5P_DEFAULT, zvel);

   if (status < 0) {
      Aux_Message(stderr, "Could not read particle z-velocity!!\n");
   }

   H5Sclose(memspace);
   H5Sclose(dataspace);
   H5Dclose(dataset);

   dataset   = H5Dopen(file_id, "particle_mass", H5P_DEFAULT);

   dataspace = H5Dget_space(dataset);

   start1d[0] = (hsize_t)offset;
   start0[0] = 0;

   status = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, start1d,
                                 stride1d, count1d, NULL);
   rank      = H5Sget_simple_extent_dims(dataspace, dims1d, maxdims1d);
   memspace = H5Screate_simple(1, dims1d, NULL);
   status = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, start0,
                                 stride1d, count1d, NULL);
   status = H5Dread(dataset, H5T_NATIVE_DOUBLE, memspace, dataspace,
                     H5P_DEFAULT, mass);

   if (status < 0) {
      Aux_Message( stderr, "Could not read particle mass!!\n");
   }

   H5Sclose(memspace);
   H5Sclose(dataspace);
   H5Dclose(dataset);

   dataset   = H5Dopen(file_id, "particle_type", H5P_DEFAULT);

   dataspace = H5Dget_space(dataset);

   start1d[0] = (hsize_t)offset;
   start0[0] = 0;

   status = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, start1d,
                                 stride1d, count1d, NULL);
   rank      = H5Sget_simple_extent_dims(dataspace, dims1d, maxdims1d);
   memspace = H5Screate_simple(1, dims1d, NULL);
   status = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, start0,
                                 stride1d, count1d, NULL);
   status = H5Dread(dataset, H5T_NATIVE_DOUBLE, memspace, dataspace,
                     H5P_DEFAULT, ptype);

   if (status < 0) {
      Aux_Message( stderr, "Could not read particle type!!\n");
   }

   H5Sclose(memspace);
   H5Sclose(dataspace);
   H5Dclose(dataset);

   H5Fclose(file_id);

   return;

} // FUNCTION : Read_Particles_ClusterMerger

#endif // #ifdef SUPPORT_HDF5



//-------------------------------------------------------------------------------------------------------
// Function    :  Aux_Record_ClusterMerger
// Description :  Record the cluster centers
//
// Note        :  1. Invoked by main() using the function pointer "Aux_Record_User_Ptr",
//                   which must be set by a test problem initializer
//                2. Enabled by the runtime option "OPT__RECORD_USER"
//                3. This function will be called both during the program initialization and after each full update
//                4. Must enable Merger_Coll_LabelCenter
//
// Parameter   :  None
//-------------------------------------------------------------------------------------------------------
void Aux_Record_ClusterMerger()
{

   const char FileName[] = "Record__Center";
   static bool FirstTime = true;

   // header
   if ( FirstTime )
   {
      if ( MPI_Rank == 0 )
      {
         if ( Aux_CheckFileExist(FileName) )
            Aux_Message( stderr, "WARNING : file \"%s\" already exists !!\n", FileName );

         FILE *File_User = fopen( FileName, "a" );
         fprintf( File_User, "#%13s%14s",  "Time", "Step" );
         for (int c=0; c<Merger_Coll_NumHalos; c++)
            fprintf( File_User, " %13s%1d %13s%1d %13s%1d %13s%1d %13s%1d %13s%1d %13s%1d %13s%1d %13s%1d %13s%1d %13s%1d %13s%1d %13s%1d %13s%1d %13s%1d %13s%1d %13s%1d %13s%1d %13s%1d %13s%1d %13s%1d %13s%1d %13s%1d %13s%1d %13s%1d %13s%1d %13s%1d %13s%1d %13s%1d %13s%1d %13s%1d %13s%1d %13s%1d %13s%1d %13s%1d", "x", c, "y", c, "z", c, "BHVel_x[km/s]", c, "BHVel_y", c, "BHVel_z", c, "GasVel_x", c,"GasVel_y", c,"GasVel_z", c, "RelativeVel", c, "SoundSpeed", c, "GasDens", c, "mass_BH[Msun]", c, "Mdot[Msun/yr]", c, "NVoidCell", c, "MomXInj(cgs)", c, "MomYInj(cgs)", c,"MomZInj(cgs)", c, "MomXInjAbs", c, "MomYInjAbs", c, "MomZInjAbs", c,"MomXInj_err", c, "EInj_exp[erg]", c, "E_Inj[erg]", c, "E_Inj_err", c, "Ek_Inj[erg]", c, "Et_Inj[erg]", c, "PowerInj(cgs)", c, "MassInj[Msun]", c, "Mdot", c, "Pdot(cgs)", c, "Edot(cgs)", c, "Jet_Vec_x", c, "Jet_Vec_y", c, "Jet_Vec_z", c );
         fprintf( File_User, "\n" );
         fclose( File_User );
      }

      FirstTime = false;
   } // if ( FirstTime )


   // get cluster centers
   double Cen[3][3] = {  { NULL_REAL, NULL_REAL, NULL_REAL },
                         { NULL_REAL, NULL_REAL, NULL_REAL },
                         { NULL_REAL, NULL_REAL, NULL_REAL }  };
   double BH_Vel[3][3] = {  { NULL_REAL, NULL_REAL, NULL_REAL },
                            { NULL_REAL, NULL_REAL, NULL_REAL },
                            { NULL_REAL, NULL_REAL, NULL_REAL }  };
   GetClusterCenter( Cen, BH_Vel );

   double Bondi_MassBH[3] = { Bondi_MassBH1, Bondi_MassBH2, Bondi_MassBH3 };
   double Mdot_BH[3] = { Mdot_BH1, Mdot_BH2, Mdot_BH3 };

   for (int c=0; c<Merger_Coll_NumHalos; c++){
      for (int d=0; d<3; d++){
         BH_Vel[c][d] *= UNIT_V/(Const_km/Const_s);
      }
   }

   for (int c=0; c<Merger_Coll_NumHalos; c++) {
      Bondi_MassBH[c] *= UNIT_M/Const_Msun;
      Mdot_BH[c] *= (UNIT_M/UNIT_T)/(Const_Msun/Const_yr);
      Mdot[c] *= (UNIT_M/UNIT_T)/(Const_Msun/Const_yr);
      Pdot[c] *= UNIT_M*UNIT_V/UNIT_T;
      Edot[c] *= UNIT_E/UNIT_T;
      E_inj_exp[c]   *= UNIT_E; 
   }

// get the total number of cells within the void region
   int SinkNCell_Sum[3];
// get the total amount of sunk variables
   double Mass_Sum[3], MomX_Sum[3], MomY_Sum[3], MomZ_Sum[3], MomXAbs_Sum[3], MomYAbs_Sum[3], MomZAbs_Sum[3], E_Sum[3], Ek_Sum[3], Et_Sum[3];

   for (int c=0; c<Merger_Coll_NumHalos; c++){
      MPI_Reduce( &CM_Bondi_SinkNCell[c], &SinkNCell_Sum[c], 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD );
      MPI_Reduce( &CM_Bondi_SinkMass[c],    &Mass_Sum[c],    1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );
      MPI_Reduce( &CM_Bondi_SinkMomX[c],    &MomX_Sum[c],    1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );
      MPI_Reduce( &CM_Bondi_SinkMomY[c],    &MomY_Sum[c],    1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );
      MPI_Reduce( &CM_Bondi_SinkMomZ[c],    &MomZ_Sum[c],    1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );
      MPI_Reduce( &CM_Bondi_SinkMomXAbs[c], &MomXAbs_Sum[c], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );
      MPI_Reduce( &CM_Bondi_SinkMomYAbs[c], &MomYAbs_Sum[c], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );
      MPI_Reduce( &CM_Bondi_SinkMomZAbs[c], &MomZAbs_Sum[c], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );
      MPI_Reduce( &CM_Bondi_SinkE[c],       &E_Sum[c],      1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );
      MPI_Reduce( &CM_Bondi_SinkEk[c],      &Ek_Sum[c],      1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );
      MPI_Reduce( &CM_Bondi_SinkEt[c],      &Et_Sum[c],      1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );
   
      Mass_Sum[c]    *= UNIT_M/Const_Msun;
      MomX_Sum[c]    *= UNIT_M*UNIT_L/UNIT_T;
      MomY_Sum[c]    *= UNIT_M*UNIT_L/UNIT_T;
      MomZ_Sum[c]    *= UNIT_M*UNIT_L/UNIT_T;
      MomXAbs_Sum[c] *= UNIT_M*UNIT_L/UNIT_T;
      MomYAbs_Sum[c] *= UNIT_M*UNIT_L/UNIT_T;
      MomZAbs_Sum[c] *= UNIT_M*UNIT_L/UNIT_T;
      E_Sum[c]      *= UNIT_E;
      Ek_Sum[c]      *= UNIT_E;
      Et_Sum[c]      *= UNIT_E;
   }

   for (int c=0; c<Merger_Coll_NumHalos; c++) E_power_inj[c] = E_Sum[c]/(dt_base*UNIT_T);

   // output cluster centers' BH profiles
   if ( MPI_Rank == 0 )
   {
      FILE *File_User = fopen( FileName, "a" );
      fprintf( File_User, "%14.7e%14ld", Time[0], Step );
      for (int c=0; c<Merger_Coll_NumHalos; c++)
         fprintf( File_User, " %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e %14d %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e", Cen[c][0], Cen[c][1], Cen[c][2], BH_Vel[c][0], BH_Vel[c][1], BH_Vel[c][2], GasVel[c][0]*UNIT_V/(Const_km/Const_s), GasVel[c][1]*UNIT_V/(Const_km/Const_s), GasVel[c][2]*UNIT_V/(Const_km/Const_s), RelativeVel[c]*UNIT_V/(Const_km/Const_s), SoundSpeed[c]*UNIT_V/(Const_km/Const_s), GasDens[c]*UNIT_D/(Const_Msun/pow(Const_kpc,3)), Bondi_MassBH[c], Mdot_BH[c], SinkNCell_Sum[c], MomX_Sum[c], MomY_Sum[c], MomZ_Sum[c], MomXAbs_Sum[c], MomYAbs_Sum[c], MomZAbs_Sum[c], MomX_Sum[c]/MomXAbs_Sum[c], E_inj_exp[c], E_Sum[c], (E_Sum[c]-E_inj_exp[c])/E_inj_exp[c], Ek_Sum[c], Et_Sum[c], E_power_inj[c], Mass_Sum[c], Mdot[c], Pdot[c], Edot[c], Jet_Vec[c][0], Jet_Vec[c][1], Jet_Vec[c][2] );
      fprintf( File_User, "\n" );
      fclose( File_User );
   }

// reset the cumulative variables to zero
   for (int c=0; c<Merger_Coll_NumHalos; c++){
      CM_Bondi_SinkMass[c]    = 0.0;
      CM_Bondi_SinkMomX[c]    = 0.0;
      CM_Bondi_SinkMomY[c]    = 0.0;
      CM_Bondi_SinkMomZ[c]    = 0.0;
      CM_Bondi_SinkMomXAbs[c] = 0.0;
      CM_Bondi_SinkMomYAbs[c] = 0.0;
      CM_Bondi_SinkMomZAbs[c] = 0.0;
      CM_Bondi_SinkE[c]      = 0.0;
      CM_Bondi_SinkEk[c]      = 0.0;
      CM_Bondi_SinkEt[c]      = 0.0;
      E_inj_exp[c] = 0.0;
   }

} // FUNCTION : Aux_Record_ClusterMerger



//-------------------------------------------------------------------------------------------------------
// Function    :  GetClusterCenter
// Description :  Get the cluster centers
//
// Note        :  1. Must enable Merger_Coll_LabelCenter
//
// Parameter   :  Cen : Cluster centers
//
// Return      :  Cen[]
//-------------------------------------------------------------------------------------------------------
void GetClusterCenter( double Cen[][3], double BH_Vel[][3] )
{

   if ( ! Merger_Coll_LabelCenter  &&  MPI_Rank == 0 )
      Aux_Message( stderr, "WARNING : Merger_Coll_LabelCenter is disabled in %s !!\n", __FUNCTION__ );


   const real *ParPos[3] = { amr->Par->PosX, amr->Par->PosY, amr->Par->PosZ };
   const real *ParVel[3] = { amr->Par->VelX, amr->Par->VelY, amr->Par->VelZ };

   for (int c=0; c<Merger_Coll_NumHalos; c++) {
      double Cen_Tmp[3] = { -__FLT_MAX__, -__FLT_MAX__, -__FLT_MAX__ };   // set to -inf
      double Vel_Tmp[3] = { -__FLT_MAX__, -__FLT_MAX__, -__FLT_MAX__ };
      for (long p=0; p<amr->Par->NPar_AcPlusInac; p++) {
         if ( amr->Par->Type[p] == real(PTYPE_CEN+c) ) {
            for (int d=0; d<3; d++) Cen_Tmp[d] = ParPos[d][p];
            for (int d=0; d<3; d++) Vel_Tmp[d] = ParVel[d][p];
            break;
         }
      }
//    TEMP!!!
      for (int d=0; d<3; d++) Cen_Tmp[d] = 5.0;
      for (int d=0; d<3; d++) Vel_Tmp[d] = 0.0;

      // use MPI_MAX since Cen_Tmp[] is initialized as -inf
      MPI_Reduce( Cen_Tmp, Cen[c], 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD );
      MPI_Reduce( Vel_Tmp, BH_Vel[c], 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD );
   }

} // FUNCTION : GetClusterCenter

#endif // #ifdef MASSIVE_PARTICLES
