###########################################################################
#  Copyright 2023-24 Jose M. Badia <barrachi@uji.es> and                  #
#                    German Leon <leon@uji.es>                            #
#                                                                         #
#  mm_hybrid.c is part of mm_offloading                                   #
#                                                                         #
#  mm_offloading is free software: you can redistribute it and/or modify  #
#  it under the terms of the GNU General Public License as published by   #
#  the Free Software Foundation; either version 3 of the License, or      #
#  (at your option) any later version.                                    #
#                                                                         #
#  This file is distributed in the hope that it will be useful, but       #
#  WITHOUT ANY WARRANTY; without even the implied warranty of             #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU      #
#  General Public License for more details.                               #
#                                                                         #
#  You should have received a copy of the GNU General Public License      #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>   #
#                                                                         #
###########################################################################


// System includes
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "mm_utils.h"

// Transpose matrix in src of size nxn using nth_cpu threads
// Save the result in dst
void transpose(float *dst, float *src, int n, int nth_cpu) {
    #pragma omp parallel num_threads(nth_cpu) //collapse(2) 
    {
    for (int i = 0; i < n; i++) {
    #pragma omp parallel for 
//        #pragma omp for //collapse(2) 
        for (int j = 0; j < n; j++) {
            dst[j*n + i] = src[i*n + j];
        }
    }
    } // end pragma omp parallel
}

// Sequential matrix multiplication: C = A x B
// Square matrices of size nxn
void mm_seq(float *A, float *B, float *C, int n) {
  float aux;

  for (int i = 0; i < n; i++) 
     for (int j = 0; j < n; j++) {
        aux = 0.0f;
        for (int k = 0; k < n; k++) 
  	    aux += A[i*n+k] * B[k*n+j];
	C[i*n+j] = aux;
     }	

}

// Sequential matrix multiplication: C = A x B
// Square matrices of size nxn
// Version where B is first transposed to optimize memory accesses
void mm_seq_v1(float *A, float *B, float *C, int n) {
  float aux;

  float *Bt = (float *) malloc( n*n*sizeof(float) );
  transpose(Bt, B, n, 4);

  for (int i = 0; i < n; i++) 
     for (int j = 0; j < n; j++) {
        aux = 0.0f;
        for (int k = 0; k < n; k++) 
  	    aux += A[i*n+k] * Bt[j*n+k];
	C[i*n+j] = aux;
     }	

  free(Bt);  
}

// Parallel Matrix multiplication offloaded to the GPU. c = a x b
// Computes the first endGPU rows of c.
// c[0:endGPU,:] = a[0:endGPU,:] x b
// Uses n_teams teams of threads of size nth_per_team
// Collapses the first two loops to increase parallelism 
void mm_omp_gpu_part(float *a, float *b, float *c, int n, int endGPU, int n_teams, nth_per_team) {
   float aux;

   #pragma omp target teams device(0) num_teams(n_teams) thread_limit(nth_per_team)  \
        map(to:a[0:endGPU*n], b[0:n*n]) map(tofrom: c[0:endGPU*n]) \
        default(none) shared(a, b, c, n, endGPU, nth_per_team) 
   #pragma omp distribute parallel for num_threads(nth_per_team) \
        dist_schedule(static, n), collapse(2) \
        default(none) shared(a, b, c, n, endGPU) private(aux)
   for (int i = 0; i < endGPU; i++) {
      for (int j = 0; j < n; j++) {
         aux = 0.0f;
         for (int k = 0; k < n; k++) {
            aux += a[i*n+k] * b[k*n+j];
          } // end for k
          c[i*n+j] = aux;
      } // end for j
   } // end for i
}

// Parallel Matrix multiplication in the CPU. c = a x b
// Computes the last rows of c, from row endGPU
// c[endGPU:n,:] = a[endGPU:n,:] x b
// We first transpose b to improve the memory access pattern
void mm_omp_cpu_part(float *a, float *b, float *c, int n, int endGPU, int nth_cpu) {
   float aux;
   float *bt = (float *) malloc( n*n*sizeof(float) );
   transpose(bt, b, n, nth_cpu);
     
   #pragma omp parallel num_threads(nth_cpu) 
   { 
      for (int i = endGPU; i < n; i++) 
         #pragma omp for private(aux)
         for (int j = 0; j < n; j++) {
            aux = 0.0f;
            #pragma omp simd reduction(+:aux)
            for (int k = 0; k < n; k++) 
  	       aux += a[i*n+k] * bt[j*n+k];
      	    c[i*n+j] = aux;
         }		 
    }

    free(bt);
}

// Heterogeneous matrix multiplication distributed between GPU and CPU.
// First rows computation is offloaded to the GPU using OpenMP offloading
void mm_omp_hybrid(float *a, float *b, float *c, int n, int nth_cpu, int n_teams, int nth_per_team, float part, float *ti) {
  int endGPU = part*n; 
  float tini;

  tini = my_gettime();
  omp_set_max_active_levels(2);
  #pragma omp parallel num_threads(2) // CPU + GPU in parallel
  {
     // *********************    GPU   ******************************
     #pragma omp sections
     {
     #pragma omp section
     {
        if (endGPU > 0) {
           omp_set_num_threads(1);
	   mm_omp_gpu_part(a, b, c, n, endGPU, n_teams, nth_per_team);
        }
        ti[0] = 1000*(my_gettime() - tini);
     }

     // *********************    CPU   ******************************
     #pragma omp section
     {
        if (endGPU < n) {
           mm_omp_cpu_part(a, b, c, n, endGPU, nth_cpu);
        }
        ti[1] = 1000*(my_gettime() - tini);
     } 

     } // end omp pragama sections

  } // end pragma omp parallel

}

int main(int argc, char **argv) {
    float tini;
    float ti[2]; // Stores the GPU and CPU times
    int it;
    const int NITER = 3;

    if (argc != 6) {
        printf("ERROR. Usage:  mm_hybrid <n> <nth_cpu> <n_teams> <nth_per_team> <part>\n");
        exit(EXIT_SUCCESS);
    }


    int n = atoi(argv[1]);               // Size of the square matrices
    int nth_cpu = atoi(argv[2]);         // Number of OpenMP CPU threads
    int n_teams = atoi(argv[3]);         // Number of Teams
    int nth_per_team = atoi(argv[4]);    // Number threads per team
    float part = atof(argv[5]);          // Portion of the rows of C computed in the GPU [0..1]

    unsigned int size = n * n;
    unsigned int mem_size = sizeof(float) * size;
    double msecPerMatrixMul, gigaFlops;
    double flopsPerMatrixMul = 2.0*n*n*n;

    // Allocate matrices
    float *a = (float *) malloc(mem_size);
    float *b = (float *) malloc(mem_size);
    float *c = (float *) malloc(mem_size);

    int verify_product = 1; // Verifies the result with respect to the expected values
    int correct = 1;

    // Initialize matrices A and B with specific values
    // so that it is easy and efficiente to verify the result computed in C
    /*
     * A,B and C are of size nxn
     * A[i][j] = i*j       (i,j:1..n)
     * B[i][j] = 1/(i*j)
     * C = A x B
     * C[i][j] =  i*n/j
     *
    */
    init_matrices(a, b, n);

    tini = my_gettime();

    float tgpu = 0.0f;
    float tcpu = 0.0f;
    for (it = 0; it < NITER; it++) {
        mm_omp_hybrid(a, b, c, n, nth_cpu, n_teams, nth_per_team, part, ti);
	tgpu += ti[0];
	tcpu += ti[1];
    }

    // Compute and print the performance
    msecPerMatrixMul = (1000.0f * (my_gettime() - tini))/NITER;
    gigaFlops = (flopsPerMatrixMul * 1.0e-9f) /
                     (msecPerMatrixMul / 1000.0f);

    if (verify_product) {

       tini = my_gettime();
       correct = verify_golden(c, n);
       printf("correct: %d. Time to check: %.2f\n", correct, 1000.0f*(my_gettime() - tini) );
    }

    printf(
        "mm_omp_hybrid  (GPU %4.0f%%):  %s ; %.2f GFlop/s, Time= %.2f msec. GPU: %.2f, CPU: %.2f\n", 
        100.0f*part,
        correct ? "PASS" : "FAIL",
        gigaFlops, msecPerMatrixMul,
	tgpu/NITER, tcpu/NITER);

    // Clean up memory
    free(a);
    free(b);
    free(c);

    return correct;
}
