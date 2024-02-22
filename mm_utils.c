###########################################################################
#  Copyright 2023-24 Jose M. Badia <barrachi@uji.es> and                  #
#                    German Leon <leon@uji.es>                            #
#                                                                         #
#  mm_utils.c is part of mm_offloading                                    #
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
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

// returns time in seconds from system clock
float my_gettime() {
   struct timespec rt;
//   clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &rt);
   clock_gettime(CLOCK_MONOTONIC, &rt);
   return rt.tv_sec + 1.0e-9 * rt.tv_nsec;
}

// Init matrix to constant value "val"
void init_constant_matrix(float *data, int size, float val) {
    for (int i = 0; i < size; ++i) {
        data[i] = val;
    }
}

// Init matrix to random values
void init_random_matrix(float *data, int size) {
    srand(time(NULL));
    for (int i = 0; i < size; ++i) {
        data[i] = ((float)rand()/(float)(RAND_MAX*size*size));
    }
}

// Init matrices A and B to specific values, 
// so that result computed in C is easy to verify
/*
 * A,B and C are of size nxn
 * A[i][j] = i*j       (i,j:1..n)
 * B[i][j] = 1/(i*j)
 * C = A x B
 * C[i][j] =  i*n/j
 *
 */
void init_matrices(float *a, float *b, int n) {
    float *ptra = a;
    float *ptrb = b;
    for (int i = 1; i < (n+1); i++) 
        for (int j = 1; j < (n+1); j++) {
	    *ptra = i*j;
	    *ptrb = 1/(*ptra);
	    *ptra++;
	    *ptrb++;
        }	
}

void print_matrix(float *a, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
	    printf("%10.5f", a[i*n+j]);
        }	
	printf("\n");
     }
}


// test relative error by the formula
//     |<x, y> - <x,y>_golden|/<|x|, |y|>  < tol
int verify_golden(float *c, int n) {
    float *ptrc = c;
    float tol = 1.0e-06;
    float diff, aux;
    for (int i = 1; i < (n+1); i++) {
	aux = (float) (i*n);
        for (int j = 1; j < (n+1); j++) {
            diff = fabs(*ptrc - aux/j)/fabs(*ptrc)/n;
            if ( diff > tol  ) {
	        printf("**[%d, %d] c: %10.5f golden: %10.5f\n", i-1, j-1, *ptrc, ((float) (i*n)/(float) j));
//		printf("[%d, %d] %20.15f\n", i-1, j-1, diff);
                return 0;
	    }
            ptrc++;
	}
    }
    return 1;
}
