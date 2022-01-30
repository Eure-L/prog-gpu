#ifndef __CUDA_SAFE_CALL_H_

#include <cuda_runtime.h>
#include <unistd.h>
#include <stdio.h>

#define CUDA_SAFE_CALL( __call)                                         \
    do {                                                                \
        cudaError_t __err = __call;                                     \
        if( cudaSuccess != __err) {                                     \
            char __machinestr[64];                                      \
            int __dev;                                                  \
            cudaGetDevice(&__dev);                                      \
            gethostname(__machinestr, 64);                              \
            fprintf(stderr, "[Machine %s - Device %d] "       \
                    "Cuda error in file '%s' in line %i : %s (%s).\n",  \
                    __machinestr, __dev, __FILE__, __LINE__,    \
                    cudaGetErrorString( __err),                         \
                    cudaGetErrorName( __err) );                         \
            abort();                                                    \
        }                                                               \
    } while(0)


#endif // __CUDA_SAFE_CALL_H_

