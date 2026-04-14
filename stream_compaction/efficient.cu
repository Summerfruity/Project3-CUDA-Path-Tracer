#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

static const int BLOCK_SIZE = 256;

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        
        /**
         * Up-sweep phase of the scan operation. Builds a sum in place up the tree.
         * @param N      The number of elements in data. (Must be a power of 2. )
         * @param d      The depth of the tree (starting at 0).
         * @param data   The array of elements to sum.
         */
        __global__ void kernUpSweep(int N, int d, int* data) {
            int k = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = 1 << (d + 1); // 2^(d+1)
            int numThreads = N / stride; // number of threads needed at this depth
            if(k >= numThreads)
                return;
            
            int index = k * stride + stride - 1;
            int left = k * stride + (stride / 2) - 1;
            data[index] += data[left];
                     
        }

        /**
         * Down-sweep phase of the scan operation. Traverses down the tree building the scan in place.
         * @param N      The number of elements in data. (Must be a power of 2. )
         * @param d      The depth of the tree (starting at 0).
         * @param data   The array of elements to sum.
         */
        __global__ void kernDownSweep(int N, int d, int* data) {
            int k = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = 1 << (d + 1); // 2^(d+1)
            int numThreads = N / stride; // number of threads needed at this depth
            if(k >= numThreads)
                return;
            
            int temp = data[k * stride + (stride / 2) - 1];
            data[k * stride + (stride / 2) - 1] = data[k * stride + stride - 1];
            data[k * stride + stride - 1] += temp;
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

            // Work-efficient scan operates over a power-of-two-sized array.
            // For non-power-of-two n, pad to the next power of two with zeroes.
            int logN = ilog2ceil(n);
            int N = 1 << logN;

            int* dev_data = nullptr;
            cudaMalloc((void**)&dev_data, N * sizeof(int));
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            if (N > n) {
                cudaMemset(dev_data + n, 0, (N - n) * sizeof(int));
            }
            

            // upSweep in each depth
            timer().startGpuTimer();
            for (int d = 0; d < logN; d++) {
                int stride = 1 << (d + 1); // 2^(d+1)
                int numThreads = N / stride; // number of threads needed at this depth
                int numBlocks = (numThreads + BLOCK_SIZE - 1) / BLOCK_SIZE;

                kernUpSweep<<<numBlocks, BLOCK_SIZE>>>(N, d, dev_data);
                checkCUDAError("kernUpSweep");
            }

            // set the last element to 0 before downSweep
            cudaMemset(dev_data + N - 1, 0, sizeof(int));
            // downSweep in each depth
            for (int d = logN - 1; d >= 0; d--) {
                int stride = 1 << (d + 1); // 2^(d+1)
                int numThreads = N / stride; // number of threads needed at this depth
                int numBlocks = (numThreads + BLOCK_SIZE - 1) / BLOCK_SIZE;

                kernDownSweep<<<numBlocks, BLOCK_SIZE>>>(N, d, dev_data);
                checkCUDAError("kernDownSweep");
            }


            timer().endGpuTimer();
            // copy n (not N) results to host side
            cudaMemcpy(odata, dev_data, n*sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_data);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {

            if(n <= 0) {
                timer().startGpuTimer();
                timer().endGpuTimer();
                return 0;
            }

            // Allocate device memory
            int* dev_in;
            int* dev_bools;
            int* dev_indices; // length N (after padding)
            int* dev_out;

            int logN = ilog2ceil(n);
            int N = 1 << logN;
            cudaMalloc((void**)&dev_in, n * sizeof(int));
            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            cudaMalloc((void**)&dev_indices, N * sizeof(int));
            cudaMalloc((void**)&dev_out, n * sizeof(int));

            // Copy input data to device
            cudaMemcpy(dev_in, idata, n*sizeof(int), cudaMemcpyHostToDevice);
            cudaMemset(dev_out, 0, n*sizeof(int));

            timer().startGpuTimer();

            // Step 1: Map to boolean array
            int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            StreamCompaction::Common::kernMapToBoolean<<<numBlocks, BLOCK_SIZE>>>(n, dev_bools, dev_in);
            checkCUDAError("kernMapToBoolean");

            // Step 2: Scan
            // Note: can't use scan above since it is a host function. 
            cudaMemcpy(dev_indices, dev_bools, n*sizeof(int), cudaMemcpyDeviceToDevice);
            if(N > n) {
                cudaMemset(dev_indices + n, 0, (N-n)*sizeof(int));
            }

            // upSweep in each depth
            for (int d = 0; d < logN; d++) {
                int stride = 1 << (d + 1); // 2^(d+1)
                int numThreads = N / stride; // number of threads needed at this depth
                numBlocks = (numThreads + BLOCK_SIZE - 1) / BLOCK_SIZE;

                kernUpSweep<<<numBlocks, BLOCK_SIZE>>>(N, d, dev_indices);
                checkCUDAError("kernUpSweep");
            }

            // set the last element to 0 before downSweep
            cudaMemset(dev_indices + N - 1, 0, sizeof(int));
            // downSweep in each depth
            for (int d = logN - 1; d >= 0; d--) {
                int stride = 1 << (d + 1); // 2^(d+1)
                int numThreads = N / stride; // number of threads needed at this depth
                numBlocks = (numThreads + BLOCK_SIZE - 1) / BLOCK_SIZE;

                kernDownSweep<<<numBlocks, BLOCK_SIZE>>>(N, d, dev_indices);
                checkCUDAError("kernDownSweep");
            }


            // Step 3: Scatter
            // Note: scatter kernel will only process n elements, so it won't access the padded part of dev_indices.
            numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            
            Common::kernScatter<<<numBlocks, BLOCK_SIZE>>>(n, dev_out, dev_in, dev_bools, dev_indices);
            checkCUDAError("kernScatter");

            timer().endGpuTimer();

            // Copy compacted results back to host
            // The number of valid elements is given by the last element of dev_indices + the last element of dev_bools (if the last element of idata is non-zero).
            int compactedCount;
            cudaMemcpy(&compactedCount, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int lastBool;
            cudaMemcpy(&lastBool, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            compactedCount += lastBool; // add 1 if the last element is non-zero    
            cudaMemcpy(odata, dev_out, n*sizeof(int), cudaMemcpyDeviceToHost);

            // Free device memory
            cudaFree(dev_in);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            cudaFree(dev_out);
            
            return compactedCount;
        }
    }
}
