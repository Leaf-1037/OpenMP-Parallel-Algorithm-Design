/******************************************************************************
* FILE: mpi_matmul_1.c
* DESCRIPTION:  
*   MPI Program - for parallel computation of matrix multiplication on
*   a distributed memory machine 
*   processes are organized into a 1D array/ring
*   MPI_Scatterv, MPI_Gatherv, and MPI_Sendrecv used for data communication
*   and also two pointers for send/recv communication buffers exchange
* AUTHOR: Bing Bing Zhou
* Last revised: 26/07/2023
******************************************************************************/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void mm_mul(int NRA, int NCA1, int NCA2, int NCB, double** A, double* B, double** C);
void print_matrix(double** T, int rows, int cols);

int main (int argc, char *argv[]) 
{
   double* A0; //auxiliary 1D for 2D matrix A
   double* B0; //auxiliary 1D for 2D matrix B
   double* C0; //auxiliary 1D for 2D matrix C
   double* D0; //auxiliary 1D for 2D matr//used for gatherv and scattervix D
   double** A; //the two-dimensional input matrix
   double** B; //the two-dimensional input matrix
   double** C; //the resulting matrix
   double** D; //the resulting matrix used for seq computation

   double* BUF00; //comm buffer
   double** BUF0;
   double* BUF10; //comm buffer
   double** BUF1;
   double *sd_p, *rv_p, *tmp_p; //pointers used for send/recv buffers exchange
   
   int N, L, M; //matrix sizes 
   int r, q, K, KM, dKM;
   int *displs, *displs1; //used for gatherv and scatterv
   int *srcounts, *srcounts1; //used for gatherv and scatterv
   int ct_s, ct_r;
   int i, j, l, l1, l2;
   double tt;
   double er;
   int cnt;

   int	myid, numprocs;

   MPI_Status status; //used for recv function

   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

 
   if (argc != 4){
      if (myid == 0){
         printf("Wrong number of arguments.\n");
	 printf("Please enter the command in the following format:\n");
	 printf("mpirun –np [proc num] %s [A rows N] [A colums L] [B columns M]\n", argv[0]);
	 printf("SAMPLE: mpirun –np 3 %s 20 25 20\n", argv[0]);
      }

      MPI_Finalize();
      return 0;             
   }

   //all processes will get the matrix size
   N = atoi(argv[1]); 
   L = atoi(argv[2]);
   M = atoi(argv[3]);

   if (myid == 0){
      printf("N = %d, L = %d, M = %d\n", N, L, M);
      printf("numprocs = %d\n\n", numprocs);
   }

   if (numprocs == 1){//only one process, no communication
      printf("I'm the lonely process with id = %d :(\n", myid);
      printf("Please use more than 1 process\n\n");
      MPI_Finalize();
      return 0;
   }

   if (myid == 0)
      printf("create and initialize matrices...\n\n");

   /*** create and initialize matrices ***/
   if (myid == 0) {//process 0
      //create matrix A of size N X L
      A0 = (double*) malloc(N * L * sizeof(double));
      A = (double**) malloc(N * sizeof(double *));
      for (i = 0; i < N; i++)
         A[i] = &A0[i*L];

      //create matrix B of size L X M
      B0 = (double*) malloc(L * M * sizeof(double));
      B = (double**) malloc(L * sizeof(double *));
      for (i = 0; i < L; i++)
         B[i] = &B0[i*M];

      //create matrix C of size N X M
      C0 = (double*) malloc(N * M * sizeof(double));
      C = (double**) malloc(N * sizeof(double *));
      for (i = 0; i < N; i++)
         C[i] = &C0[i*M];

      //create matrix D of size N X M
      D0 = (double*) malloc(N * M * sizeof(double));
      D = (double**) malloc(N * sizeof(double *));
      for (i = 0; i < N; i++)
         D[i] = &D0[i*M];

     srand(time(0)); // Seed the random number generator
      //initialize matrix A 
      for (i=0; i<N; i++)
         for (j=0; j<L; j++)
            A[i][j] = (double) rand() / RAND_MAX;
//            A[i][j] = 1.0 * (i+j);

//      printf ("matrix A from process %d:\n", myid);
//      print_matrix(A, N, L);

      //initialize matrix B 
      for (i=0; i<L; i++)
         for (j=0; j<M; j++){
            B[i][j] = (double) rand() / RAND_MAX;
//              B[i][j] = 1.0 * (i+j);
      }

//      printf ("matrix B from process %d:\n", myid);
//      print_matrix(B, L, M);

      //initialize matrix C to zero 
      for (i=0; i<N; i++)
         for (j=0; j<M; j++){
            C[i][j] = 0.0;
            D[i][j] = 0.0;
         }

//      printf ("matrix C from process %d:\n", myid);
//      print_matrix(C, N, M);

   }
   else { //all other processes
      //create a submatrix A of size K X L and C of size K X M
      q = N / numprocs;
      r = N % numprocs;
      if (myid < r)
         K = q+1;
      else
         K = q;

      //create matrix A of size K X L
      A0 = (double*) malloc(K * L * sizeof(double));
      A = (double**) malloc(K * sizeof(double *));
      for (i = 0; i < K; i++)
         A[i] = &A0[i*L];

      //create matrix C of size K X M
      C0 = (double*) malloc(K * M * sizeof(double));
      C = (double**) malloc(K * sizeof(double *));
      for (i = 0; i < K; i++)
         C[i] = &C0[i*M];

      //initialize matrix C to zero 
      for (i=0; i<K; i++)
         for (j=0; j<M; j++)
            C[i][j] = 0.0;
   }

   //create an additional communication buffers BUF0 and BUF1 of size Kmax by M
   //for local B and processe communication
   q = L / numprocs;
   r = L % numprocs;
   if (r > 0)
      K = q+1;
   else
      K = q;

   BUF00 = (double*) malloc(K * M * sizeof(double));
   BUF0 = (double**) malloc(K * sizeof(double *));
   for (i = 0; i < K; i++)
      BUF0[i] = &BUF00[i*M];

   BUF10 = (double*) malloc(K * M * sizeof(double));
   BUF1 = (double**) malloc(K * sizeof(double *));
   for (i = 0; i < K; i++)
      BUF1[i] = &BUF10[i*M];

   //initially send-pointer points to BUF0 and recv-pointer points to BUF1
   sd_p = BUF00;
   rv_p = BUF10;

   if (myid == 0)
      printf("distribute matrices...\n\n");

   /*** process 0 distributes matrics A and B to other processes ***/ 
   //distribute matrix B
   displs1 = (int*)malloc(numprocs*sizeof(int));
   srcounts1 = (int*)malloc(numprocs*sizeof(int));

   q = L / numprocs;
   r = L % numprocs;
   dKM = 0;
   for (i=0; i<numprocs; i++){//every process gets the same srcounts and displs
      if (i < r)
         K = q+1;
      else
         K = q;
      KM = K * M;      
      srcounts1[i] = KM;
      displs1[i] = dKM;
      dKM += KM;
   }

   KM = srcounts1[myid];   

   MPI_Scatterv(&B[0][0], srcounts1, displs1, MPI_DOUBLE, &BUF0[0][0], KM, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   //distribute matrix A
   displs = (int*)malloc(numprocs*sizeof(int));
   srcounts = (int*)malloc(numprocs*sizeof(int));

   q = N / numprocs;
   r = N % numprocs;
   dKM = 0;
   for (i=0; i<numprocs; i++){//every process gets the same srcounts and displs
      if (i < r)
         K = q+1;
      else
         K = q;
      KM = K * L;      
      srcounts[i] = KM;
      displs[i] = dKM;
      dKM += KM;
   }

   KM = srcounts[myid];

   MPI_Scatterv(&A[0][0], srcounts, displs, MPI_DOUBLE, &A[0][0], KM, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   if (myid == 0)
      printf("start parallel computation...\n\n");

   MPI_Barrier(MPI_COMM_WORLD);
   if (myid==0) tt = MPI_Wtime();

   K = KM / L;  //#rows in a local A

   l1 = srcounts1[myid] / M; //#rows in local B
   l2 = displs1[myid] / M; //the position of the starting row in global B

   mm_mul(K, l1, l2, M, A, sd_p, C); 

   int left = (myid-1+numprocs) % numprocs; //left process
   int right = (myid + 1) % numprocs; //right process

   for (i=1; i<numprocs; i++){
      ct_s = srcounts1[(myid+i-1)%numprocs]; //local B to send to left
      ct_r = srcounts1[(myid+i)%numprocs]; //B to receive from right

      MPI_Sendrecv(sd_p, ct_s, MPI_DOUBLE, left, 1, rv_p, ct_r, MPI_DOUBLE, right, 1, MPI_COMM_WORLD, &status);
      //exchange send and recv buffers using send and recv pointers
      tmp_p = sd_p; 
      sd_p = rv_p;
      rv_p = tmp_p;

      l1 = ct_r / M; //#rows in local B
      l2 = displs1[(myid+i)%numprocs] / M; //the position of the starting row in global B

        mm_mul(K, l1, l2, M, A, sd_p, C);
   }

  /*** other processes send partial results to process 0 ***/
   for (i=0; i<numprocs; i++){//recalculate displs and srcounts for C
      displs[i] = displs[i] / L * M;
      srcounts[i] = srcounts[i] / L * M;
   }

   KM = srcounts[myid]; //#rows in a local C

   MPI_Gatherv(&C[0][0], KM, MPI_DOUBLE, &C[0][0], srcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   MPI_Barrier(MPI_COMM_WORLD);
   if (myid==0) {
      tt = MPI_Wtime() - tt;
      printf("myid %d: It takes %f seconds to finish the parallel computation.\n\n", myid, tt); 

//      printf ("matrix C from process %d:\n", myid);
//      print_matrix(C, N, M);
   }


   /*** sequential computation ***/
   if (myid == 0){
      printf("start sequential computation...\n\n");
      tt = MPI_Wtime();
 
      for (i=0; i<N; i++)    
         for (l=0; l<L; l++)
         {
            double at = A[i][l]; 
            for (j=0; j<M; j++)     
               D[i][j] +=  at * B[l][j];
         } 

//      print_matrix(D, N, M);

      tt = MPI_Wtime() - tt;
      printf("myid %d: It takes %f seconds to finish the sequential computation.\n\n", myid, tt);       

      printf("check the correctness...\n\n");

      /*** check the correctness ***/
      cnt = 0;
      for (i=0; i<N; i++)
         for (j=0; j<M; j++){  
            er = (C[i][j] - D[i][j]) * (C[i][j] - D[i][j]);
         if (er > 1.0E-10)
            cnt += 1;
      }

      if (cnt)
         printf("results incorrect! differences = %d\n", cnt);
      else
         printf("results are the same!\n"); 

   }

   MPI_Finalize();
   return 0;
}

/*****************************************************************************
* This routine performs matrix multiplication
* A (NRA by NCA1) is one partion of global matrix A (NRA by NCA) and its starting
* column is NCA2
* B is NCA1 by NCB
* Since this program use a single pointer to point to the start of B, thus B
* is only a single pointer, but not double pointer
*This routine is only used for this program
******************************************************************************/
void mm_mul(int NRA, int NCA1, int NCA2, int NCB, double** A, double* B, double** C){
   int i, j, k;
   double at;

   if (NRA==0 || NCA1==0 || NCB==0){
      printf("matrix dimension NRA, or NCA1, or NCB is zero\n");
      return;
   } 


   for (i=0; i<NRA; i++)    
      for (k=0; k<NCA1; k++)
      {
         at = A[i][k+NCA2]; 
         for (j=0; j<NCB; j++)     
            C[i][j] +=  at * B[k*NCB+j];
      } 

   return;
} 

void print_matrix(double** T, int rows, int cols){
    for (int i=0; i < rows; i++){
        for (int j=0; j < cols; j++)
            printf("%.2f  ", T[i][j]);
        printf("\n");
    }
    printf("\n\n");
    return;
}


