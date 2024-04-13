/******************************************************************************
* FILE: omp_hello.c
* DESCRIPTION:
*   A simple "Hello World" openmp program for students to modify
* AUTHOR: Bing Bing Zhou
* LAST REVISED: 02/07/2022
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[])
{
    int tid, nthreads;
    /* Fork threads, each print out a "Hello World" statement */
    omp_set_num_threads(8);
#pragma omp parallel private(tid)
    {
        // Obtain and print thread id
        tid = omp_get_thread_num();
        printf("Hello World from thread = %d\n", tid);
        // Only master thread does this
        if (tid == 0) {
            nthreads = omp_get_num_threads();
            printf("Number of threads = %d\n", nthreads);
        }
    }
    
    /*All threads join master thread and terminate */
    return 0;

}