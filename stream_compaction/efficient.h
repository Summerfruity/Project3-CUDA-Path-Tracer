#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);

        // pre-allocate before rendering, free when done rendering
        void initScanDeviceBuffer(int maxN);
        void freeScanDeviceBuffer();

        void scanDevice(int n, int *dev_odata, const int *dev_idata);

        int compact(int n, int *odata, const int *idata);
    }
}
