#ifndef COMPACT_H
#define COMPACT_H

namespace Compaction {

extern int blkSize;

template <typename T>
__global__ void kernScatter(int n, T *out, const int *keep, const int *scan, const T *data) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n)
    return;

  if (keep[k]) {
    out[scan[k]] = data[k];
  }
}

int getPot(int n);
void devScanUtil(int n, int *devData);

template <typename T>
int compact(int n, T *devOut, const T *devIn, const int *devKeep) {
  int pot = getPot(n);

  dim3 blkDim(blkSize);
  dim3 blkCnt((n + blkDim.x - 1)/blkDim.x);

  // mark values to keep
  int *devScan;
  cudaMalloc((void**)&devScan, pot*sizeof(int));
  cudaMemset(devScan, 0, pot*sizeof(int));
  cudaMemcpy(devScan, devKeep, n*sizeof(int), cudaMemcpyDeviceToDevice);

  // scan boolean array
  devScanUtil(n, devScan);
  int nKeep;
  cudaMemcpy(&nKeep, &devScan[pot-1], sizeof(int), cudaMemcpyDeviceToHost);

  // scatter to output
  kernScatter<<<blkCnt,blkDim>>>(n, devOut, devKeep, devScan, devIn);

  cudaFree(devScan);

  return nKeep;
}

}; // namespace Compaction

#endif /* COMPACT_H */
