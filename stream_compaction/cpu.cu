#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            if (n <= 0) {
                timer().endCpuTimer();
                return;
            }

            odata[0] = 0;
            for (int i = 1; i < n; i++) {
                odata[i] = odata[i - 1] + idata[i - 1];
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            if (n <= 0) {
                timer().endCpuTimer();
                return 0;
            }

            int ptr = 0; // where to put the non-zero element

            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[ptr] = idata[i];
                    ptr++;
                }
            }
            timer().endCpuTimer();
            return ptr;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            if (n <= 0) {
                timer().endCpuTimer();
                return 0;
            }

            // Step 1: map to boolean
            int* bools = new int[n];
            for (int i = 0; i < n; i++) {
                bools[i] = (idata[i] != 0) ? 1 : 0;
            }

            // Step 2: exclusive scan
            int* indices = new int[n];
            indices[0] = 0;
            for (int i = 1; i < n; i++) {
                indices[i] = indices[i-1] + bools[i-1];
            }

            // Step 3: scatter
            for (int i = 0; i < n; i++) {
                if (bools[i] == 1) {
                    odata[indices[i]] = idata[i];
                }
            }

            int count = indices[n - 1] + bools[n - 1]; // total count of non-zero elements
            delete[] bools;
            delete[] indices;
            timer().endCpuTimer();
            return count;
        }
    }
}
