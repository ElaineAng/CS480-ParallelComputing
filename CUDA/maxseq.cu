#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <math.h>

long getmax(long *, long);

__global__ void getmaxcu(long * num, long size, long * max_ans){
  
  __device__ __shared__ long local_max_num;
  __device__ __shared__ long local_arr[1024];
    
  local_max_num = 10;
  local_arr[threadIdx.x] = num[blockDim.x * blockIdx.x + threadIdx.x];
  __syncthreads();

  int i;
  for (i=0; i<blockDim.x; i++){
      if (threadIdx.x == i && (local_arr[i] > local_max_num) && (local_arr[i] < size)){
	 local_max_num = local_arr[i];
      }
  }
  
  if (threadIdx.x == 0){
     	atomicMax((unsigned long long *) max_ans, 
  		(unsigned long long) local_max_num);
  }
   
}

int main(int argc, char *argv[]){
   long size = 0;  // The size of the array
   long i;  // loop index
   long * numbers; //pointer to the array
    
    if(argc !=2)
    {
       printf("usage: maxseq num\n");
       printf("num = size of the array\n");
       exit(1);
    }
   
    size = atol(argv[1]);

    numbers = (long *)malloc(size * sizeof(long));
    if( !numbers )
    {
       printf("Unable to allocate mem for an array of size %ld\n", size);
       exit(1);
    }    

    srand(time(NULL)); // setting a seed for the random number generator
    // Fill-up the array with random numbers from 0 to size-1 
    for( i = 0; i < size; i++)
       numbers[i] = rand() % size;    
   
    long * d_num;
    long * d_max;
    long h_max;
    h_max = 10;

    cudaMalloc(&d_num, size * sizeof(long));
    cudaMalloc(&d_max, sizeof(long));
    cudaMemcpy(d_num, numbers, size*sizeof(long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max, &h_max, sizeof(long), cudaMemcpyHostToDevice);

    cudaDeviceProp prop;
    int device;
    
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    int thrdPerBlk = 1024;
    int blkPerGrid = (int) ceil((float) size / thrdPerBlk);

    getmaxcu <<<blkPerGrid, thrdPerBlk>>> (d_num, size, d_max);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_max, d_max, sizeof(long), cudaMemcpyDeviceToHost);
    

    printf("The maximum number from GPU is %ld.\n", h_max);
    
    cudaFree(d_num);
    cudaFree(d_max);
    free(numbers);
    exit(0);
}


/*
   input: pointer to an array of long int
          number of elements in the array
   output: the maximum number of the array

long getmax(long num[], long size)
{
  long i;
  long max = num[0];

  for(i = 1; i < size; i++)
	if(num[i] > max)
	   max = num[i];

  return( max );

}
*/
