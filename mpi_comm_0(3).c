/******************************************************************************
* FILE: mpi_comm_o.c
* DESCRIPTION:
*   A mpi program for passing a token around the processes:
*     Processes are organized as a 1D ring. At the beginning process 0 sets 
*     the token (integer) to 1, and then passes the token to process numprocs-1. 
*     When process numprocs-1 receives the token, it passes it to its immediate 
*     left neighbour, and so on as follows (numprocs = 5):
*       process 0 sends the token to process 4
*       process 4 sends the token to process 3
*       process 3 sends the token to process 2
*       process 2 sends the token to process 1
*       process 1 sends the token bake to process 0
* AUTHOR: Bing Bing Zhou
* LAST REVISED: 18/07/2022
******************************************************************************/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc , char **argv)
{
   MPI_Status status;

   int myid, numprocs;
   int token = 0; //token initialized to 0

   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   if (numprocs == 1){//only one process, no communication
      printf("I'm the lonely process with id = %d :(\n", myid);
      MPI_Finalize();
      return 0;
   } 

   if (myid == 0){
      token = 1; //set token to 1 and send
      printf("This is process %i from %i - I have set the token = %d.\n", myid, numprocs, token);
      MPI_Send(&token, 1, MPI_INT, numprocs - 1, 1, MPI_COMM_WORLD);
      MPI_Recv(&token, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, &status);
   }
   else{
      MPI_Recv(&token, 1, MPI_INT, (myid + 1) % numprocs, 1, MPI_COMM_WORLD, &status);
      printf("This is process %i from %i - I have received the token = %d.\n", myid, numprocs, token);
      MPI_Send(&token, 1, MPI_INT, myid - 1, 1, MPI_COMM_WORLD);
   }

   MPI_Finalize();

   return 0;
}


