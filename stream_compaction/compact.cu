#include <cuda.h>
#include <cuda_runtime.h>
#include "compact.hpp"
#include <cstdio>

namespace Compaction {

int blkSize = 128;

// perform reduction
__global__ void kernScanUp(int n, int dPow, int *data) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int j = 2*(k+1)*dPow-1;
  if (j < n)
    data[j] += data[j-dPow];
}

__global__ void kernScanDown(int n, int dPow, int *data) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int j = 2*(k+1)*dPow-1;

  if (j < n) {
    int t = data[j - dPow];
    data[j - dPow] = data[j];
    data[j] += t;
  }
};

int getPot(int n) {
  unsigned int pot = n;
  pot--;
  pot |= pot >> 1;
  pot |= pot >> 2;
  pot |= pot >> 4;
  pot |= pot >> 8;
  pot |= pot >> 16;
  pot++;

  return pot;
}

void devScanUtil(int n, int *devData) {
  int pot  = getPot(n);

  dim3 blkDim(blkSize);

  int dPow = 1;
  while (dPow < pot) {
    int m = pot/(2*dPow);
    dim3 blkCnt((m + blkDim.x - 1)/blkDim.x);
    kernScanUp<<<blkCnt,blkDim>>>(pot, dPow, devData);
    dPow *= 2;
  }
  cudaMemset(&devData[pot-1], 0, sizeof(int));

  while (dPow > 0) {
    int m = pot/(dPow);
    dim3 blkCnt((m + blkDim.x - 1)/blkDim.x);
    kernScanDown<<<blkCnt,blkDim>>>(pot, dPow, devData);
    dPow /= 2;
  }
}

}; // namespace Compaction

