#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void kernNaiveScanStep(int n, int offset, int* odata, const int* idata) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n) return;

            if (i >= offset) {
                odata[i] = idata[i] + idata[i - offset];
            } else {
                odata[i] = idata[i];
            }
            
        }

        __global__ void kernInclusiveToExclusive(int n, int* odata, const int* idata_inclusive) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n) return;

            if (i == 0) {
                odata[0] = 0;
            } else {
                odata[i] = idata_inclusive[i - 1];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            if (n <= 0) {
                timer().startGpuTimer();
                timer().endGpuTimer();
                return;
            }

            const int BLOCK_SIZE = 256;
            int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

            int* dev_ping = nullptr;
            int* dev_pong = nullptr;
            int* dev_out  = nullptr;

            cudaMalloc(&dev_ping, n * sizeof(int));
            cudaMalloc(&dev_pong, n * sizeof(int));
            cudaMalloc(&dev_out,  n * sizeof(int));

            cudaMemcpy(dev_ping, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            int rounds = ilog2ceil(n);
            for (int d = 0; d < rounds; d++) {
                int offset = 1 << d;
                kernNaiveScanStep<<<numBlocks, BLOCK_SIZE>>>(n, offset, dev_pong, dev_ping);
                checkCUDAError("kernNaiveScanStep");

                // swap ping/pong
                int* tmp = dev_ping;
                dev_ping = dev_pong;
                dev_pong = tmp;
            }

            // dev_ping currently holds inclusive scan; convert to exclusive
            kernInclusiveToExclusive<<<numBlocks, BLOCK_SIZE>>>(n, dev_out, dev_ping);
            checkCUDAError("kernInclusiveToExclusive");

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_out);
            cudaFree(dev_pong);
            cudaFree(dev_ping);
        }
    }
}
