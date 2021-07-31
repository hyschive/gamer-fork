#include "GAMER.h"

#ifdef SUPPORT_HDF5
#include "hdf5.h"
#endif

#include <string>

#define PTYPE_CEN    100   // particle type for Merger_Coll_LabelCenter
                           // --> the particle closest to the center of x-th cluster is labeled as "PTYPE_CEN + x"

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

extern FieldIdx_t ParTypeIdx;

#ifdef PARTICLE
long Read_Particle_Number_ClusterMerger(std::string filename);
void Read_Particles_ClusterMerger(std::string filename, long offset, long num,
                                  real_par_in xpos[], real_par_in ypos[],
                                  real_par_in zpos[], real_par_in xvel[],
                                  real_par_in yvel[], real_par_in zvel[],
                                  real_par_in mass[], real_par_in ptype[]);

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
//                AllAttribute  : Pointer array for all particle attributes
//                                --> Dimension = [PAR_NATT_TOTAL][NPar_ThisRank]
//                                --> Use the attribute indices defined in Field.h (e.g., Idx_ParCreTime)
//                                    to access the data
//
// Return      :  ParMass, ParPosX/Y/Z, ParVelX/Y/Z, ParTime, AllAttribute
//-------------------------------------------------------------------------------------------------------

void Par_Init_ByFunction_ClusterMerger( const long NPar_ThisRank, const long NPar_AllRank,
                                        real *ParMass, real *ParPosX, real *ParPosY, real *ParPosZ,
                                        real *ParVelX, real *ParVelY, real *ParVelZ, real *ParTime,
                                        real *AllAttribute[PAR_NATT_TOTAL] )
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

         // --> convert to code unit before storing to the particle repository to avoid floating-point overflow
         // --> we have assumed that the loaded data are in cgs
         ParMass[pp] = real( mass[p] / UNIT_M );

         ParPosX[pp] = real( xpos[p] / UNIT_L );
         ParPosY[pp] = real( ypos[p] / UNIT_L );
         ParPosZ[pp] = real( zpos[p] / UNIT_L );

         ParVelX[pp] = real( xvel[p] / UNIT_V );
         ParVelY[pp] = real( yvel[p] / UNIT_V );
         ParVelZ[pp] = real( zvel[p] / UNIT_V );

         // synchronize all particles to the physical time at the base level
         ParTime[pp] = Time[0];

         // set the particle type
         AllAttribute[ParTypeIdx][pp] = real( ptype[p] );

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
      ParVelX[p] += Merger_Coll_VelX1;
      ParVelY[p] += Merger_Coll_VelY1;
      for (int d=0; d<3; d++)
         ParPos[d][p] += ClusterCenter1[d];
   }

   // reset particle mass outside the virial radius
   const double R_200_1 = 1194.326442*Const_kpc/UNIT_L;
   for (long p=0; p<NPar_ThisRank_EachCluster[0]; p++) {
   double r = pow(pow(ParPos[0][p]-ClusterCenter1[0],2.0)+pow(ParPos[1][p]-ClusterCenter1[1],2.0)+pow(ParPos[2][p]-ClusterCenter1[2],2.0),0.5);
	  if (r > R_200_1){
		 ParMass[p] *= exp(-(r-R_200_1)/(0.2*R_200_1));
	  }
   }	
//1722.516798

   for (long p=NPar_ThisRank_EachCluster[0]; p<NPar_ThisRank_EachCluster[0]+NPar_ThisRank_EachCluster[1]; p++) {
      ParVelX[p] += Merger_Coll_VelX2;
      ParVelY[p] += Merger_Coll_VelY2;
      for (int d=0; d<3; d++)
         ParPos[d][p] += ClusterCenter2[d];
   }

   // set particle mass to zero outside the virial radius
   for (long p=NPar_ThisRank_EachCluster[0]; p<NPar_ThisRank_EachCluster[0]+NPar_ThisRank_EachCluster[1]; p++) {
      if (pow(pow(ParPos[0][p]-ClusterCenter2[0],2.0)+pow(ParPos[1][p]-ClusterCenter2[1],2.0)+pow(ParPos[2][p]-ClusterCenter2[2],2.0),0.5)>1194.326442*Const_kpc/UNIT_L){
	     ParMass[p]=real(0.0);
      }
   }



   for (long p=NPar_ThisRank_EachCluster[0]+NPar_ThisRank_EachCluster[1]; p<NPar_ThisRank; p++) {
      ParVelX[p] += Merger_Coll_VelX3;
      ParVelY[p] += Merger_Coll_VelY3;
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
            AllAttribute[ParTypeIdx][min_pidx] = PTYPE_CEN + c;
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
         if ( ! Merger_Coll_LabelCenter )
            Aux_Message( stderr, "WARNING : Merger_Coll_LabelCenter is disabled in %s !!\n", __FUNCTION__ );

         if ( Aux_CheckFileExist(FileName) )
            Aux_Message( stderr, "WARNING : file \"%s\" already exists !!\n", FileName );

         FILE *File_User = fopen( FileName, "a" );
         fprintf( File_User, "#%13s%14s",  "Time", "Step" );
         for (int c=0; c<Merger_Coll_NumHalos; c++)
            fprintf( File_User, " %13s%1d %13s%1d %13s%1d", "x", c, "y", c, "z", c );
         fprintf( File_User, "\n" );
         fclose( File_User );
      }

      FirstTime = false;
   } // if ( FirstTime )


   // collect data from all ranks
   const real *ParPos[3] = { amr->Par->PosX, amr->Par->PosY, amr->Par->PosZ };
   double Cen[3][3] = {  { NULL_REAL, NULL_REAL, NULL_REAL },
                         { NULL_REAL, NULL_REAL, NULL_REAL },
                         { NULL_REAL, NULL_REAL, NULL_REAL }  };

   for (int c=0; c<Merger_Coll_NumHalos; c++) {
      double Cen_Tmp[3] = { -__FLT_MAX__, -__FLT_MAX__, -__FLT_MAX__ };   // set to -inf
      for (long p=0; p<amr->Par->NPar_AcPlusInac; p++) {
         if ( amr->Par->Attribute[ParTypeIdx][p] == real(PTYPE_CEN+c) ) {
            for (int d=0; d<3; d++) Cen_Tmp[d] = ParPos[d][p];
            break;
         }
      }
      // use MPI_MAX since Cen_Tmp[] is initialized as -inf
      MPI_Reduce( Cen_Tmp, Cen[c], 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD );
   }


   // output cluster centers
   if ( MPI_Rank == 0 )
   {
      FILE *File_User = fopen( FileName, "a" );
      fprintf( File_User, "%14.7e%14ld", Time[0], Step );
      for (int c=0; c<Merger_Coll_NumHalos; c++)
         fprintf( File_User, " %14.7e %14.7e %14.7e", Cen[c][0], Cen[c][1], Cen[c][2] );
      fprintf( File_User, "\n" );
      fclose( File_User );
   }

} // FUNCTION : Aux_Record_ClusterMerger

#endif // #ifdef PARTICLE
