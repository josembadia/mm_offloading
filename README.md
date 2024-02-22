# Heterogeneous parallel matrix multiplication algorithm

C = A x B where A, B and C are square matrices with single precision float elements

This version of the matrix multiplication uses OpenMP offloading capability to distribute the computation
of the different rows of C between the CPU and GPU cores

Code is implemented in C and CUDA.

## Usage

mm_hybrid size num_cpu_threads num_teams num_threads_per_team part

where

  size = Number of rows and columns of the matrices

  num_cpu_threads = Number of OpenMP threads that compute the CPU part of matrix multiplication

  num_teams = Number of teams of threads launched in the GPU

  num_threads_per_team = Number of threads per team

  part = Percentage of rows computed in the GPU


Number of rows computed in the GPU is endGPU = size * part

Then, the GPU computes C[0:endGPU,:] = A[0:endGPU,:] * B

and the CPU computes C[endGPU:size,:] = A[endGPU:size,:] * B


