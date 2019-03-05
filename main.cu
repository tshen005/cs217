/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"
#include "support.cu"

#define  StreamN   3




int main (int argc, char *argv[])
{
    //set standard seed
    srand(217);

    Timer timer;
    
    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    float *A_h, *B_h, *C_h;
    size_t A_sz, B_sz, C_sz;
    unsigned int VecSize;
   

    if (argc == 1) {
        VecSize = 1000000;

    } else if (argc == 2) {
        VecSize = atoi(argv[1]);      
    }
  
    else {
        printf("\nOh no!\nUsage: ./vecAdd <Size>");
        exit(0);
    }

    cudaDeviceProp prop;
    int deviceID;
    cudaGetDevice(&deviceID);
    cudaGetDeviceProperties(&prop, deviceID);

    if(!prop.deviceOverlap){
        printf("No device will handle overlaps. so no speed up from stream.\n");
        return 0;
    }

    A_sz = VecSize;
    B_sz = VecSize;
    C_sz = VecSize;

    const unsigned int BLOCK_SIZE = 512;
    const unsigned int SegSize = VecSize / StreamN;
    unsigned int leftover = VecSize % (SegSize * StreamN);

    cudaStream_t stream0, stream1, stream2;
    cudaStreamCreateWithFlags(&stream0, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);


    cudaHostAlloc((void**)&A_h, A_sz*sizeof(float),cudaHostAllocDefault);
    for (unsigned int i=0; i < A_sz; i++) { A_h[i] = (rand()%100)/100.00; }

    cudaHostAlloc((void**)&B_h, B_sz*sizeof(float),cudaHostAllocDefault);
    for (unsigned int i=0; i < B_sz; i++) { B_h[i] = (rand()%100)/100.00; }

    cudaHostAlloc((void**)&C_h, C_sz*sizeof(float),cudaHostAllocDefault);

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    size Of vector: %u x %u\n  ", VecSize);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    float *A_d0, *B_d0, *C_d0;
    float *A_d1, *B_d1, *C_d1;
    float *A_d2, *B_d2, *C_d2;
   
    //INSERT CODE HERE

    cudaMalloc((void**)&A_d0, sizeof(float)*SegSize);
    cudaMalloc((void**)&B_d0, sizeof(float)*SegSize);
    cudaMalloc((void**)&C_d0, sizeof(float)*SegSize);
    cudaMalloc((void**)&A_d1, sizeof(float)*SegSize);
    cudaMalloc((void**)&B_d1, sizeof(float)*SegSize);
    cudaMalloc((void**)&C_d1, sizeof(float)*SegSize);
    cudaMalloc((void**)&A_d2, sizeof(float)*SegSize);
    cudaMalloc((void**)&B_d2, sizeof(float)*SegSize);
    cudaMalloc((void**)&C_d2, sizeof(float)*SegSize);



    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------


    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE

    unsigned int i = 0;

    for(i = 0;VecSize - i >= SegSize*StreamN; i += SegSize*StreamN){
        cudaMemcpyAsync(A_d0, A_h+i, SegSize*sizeof(float), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(B_d0, B_h+i, SegSize*sizeof(float), cudaMemcpyHostToDevice, stream0);

        cudaMemcpyAsync(A_d1, A_h+i+SegSize, SegSize*sizeof(float), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(B_d1, B_h+i+SegSize, SegSize*sizeof(float), cudaMemcpyHostToDevice, stream1);

        cudaMemcpyAsync(A_d2, A_h+i+2*SegSize, SegSize*sizeof(float), cudaMemcpyHostToDevice, stream2);
        cudaMemcpyAsync(B_d2, B_h+i+2*SegSize, SegSize*sizeof(float), cudaMemcpyHostToDevice, stream2);
        
        // Launch kernel  ---------------------------
        printf("Launching kernel..."); fflush(stdout);
        startTime(&timer);


        VecAdd<<<(SegSize - 1)/BLOCK_SIZE+1, BLOCK_SIZE, 0, stream0>>>(SegSize, A_d0, B_d0, C_d0);
        VecAdd<<<(SegSize - 1)/BLOCK_SIZE+1, BLOCK_SIZE, 0, stream1>>>(SegSize, A_d1, B_d1, C_d1);
        VecAdd<<<(SegSize - 1)/BLOCK_SIZE+1, BLOCK_SIZE, 0, stream2>>>(SegSize, A_d2, B_d2, C_d2);
        
        

        printf("Copying data from device to host..."); fflush(stdout);
        startTime(&timer);

        cudaMemcpyAsync(C_h+i, C_d0, SegSize*sizeof(float), cudaMemcpyDeviceToHost, stream0);
        cudaMemcpyAsync(C_h+i+SegSize, C_d1, SegSize*sizeof(float), cudaMemcpyDeviceToHost, stream1);
        cudaMemcpyAsync(C_h+i+2*SegSize, C_d2, SegSize*sizeof(float), cudaMemcpyDeviceToHost, stream2);

        stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    }

    

    cudaMalloc((void**)&A_d0, sizeof(float)*leftover);
    cudaMalloc((void**)&B_d0, sizeof(float)*leftover);
    cudaMalloc((void**)&C_d0, sizeof(float)*leftover);

    cudaMemcpyAsync(A_d0, A_h + i, leftover*sizeof(float), cudaMemcpyHostToDevice, stream0);
    cudaMemcpyAsync(B_d0, B_h + i, leftover*sizeof(float), cudaMemcpyHostToDevice, stream0);

    VecAdd<<<1, leftover, 0, stream0>>>(leftover, A_d0, B_d0, C_d0);

    cudaMemcpyAsync(C_h+i, C_d0, leftover*sizeof(float), cudaMemcpyDeviceToHost, stream0);


    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    

    verify(A_h, B_h, C_h, VecSize);


    // Free memory ------------------------------------------------------------

    cudaFreeHost(A_h);
    cudaFreeHost(B_h);
    cudaFreeHost(C_h);

    cudaFree(A_d0);
    cudaFree(B_d0);
    cudaFree(C_d0);

    cudaFree(A_d1);
    cudaFree(B_d1);
    cudaFree(C_d1);

    cudaFree(A_d2);
    cudaFree(B_d2);
    cudaFree(C_d2);

    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    //INSERT CODE HERE
    return 0;

}
