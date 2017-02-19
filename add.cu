// ver 20170219 by jian
// ref: http://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf
//cudaMalloc(), cudaFree(), cudaMemcpy()
//malloc(), free(), memcpy()
// concept of block, and thread
#include <stdio.h>
__global__ void add(int *a, int *b, int *c, int n) {
	//c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
	//c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	if (index<n) c[index] = a[index] + b[index];
}

#include <cstdlib>

void random_ints(int* a, int m){
   for (int i = 0; i < m; ++i) a[i] = rand();
}

//# define N (2048*2048)
# define N (32*32)
# define THREADS_PER_BLOCK 16

int main (void) {
	int *a,*b,*c;
	int *d_a,*d_b,*d_c;
	int size=N*sizeof(int);
	cudaMalloc((void **)&d_a,size);
	cudaMalloc((void **)&d_b,size);
	cudaMalloc((void **)&d_c,size);
	a = (int *)malloc(size); random_ints(a, N);
	b = (int *)malloc(size); random_ints(b, N);
	c = (int *)malloc(size);
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	//add<<<N,1>>>(d_a,d_b,d_c);
	//add<<<1,N>>>(d_a,d_b,d_c);
	//add<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_a,d_b,d_c,N);
	add<<<(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_a,d_b,d_c,N);
	cudaMemcpy(c,d_c,size,cudaMemcpyDeviceToHost);
	for (int i=0;i<N;i++) {
		printf("%d: %d + %d = %d\n",i,a[i],b[i],c[i]);
	}
	free(a);free(b);free(c);
	cudaFree(d_a);cudaFree(d_b);cudaFree(d_c);
	return 0;
}
