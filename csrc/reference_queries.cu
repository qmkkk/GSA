// Query kernels from FlashFPS/OpenPoints (MIT); see licenses/FlashFPS-MIT.txt.
// Extracted forward kernels; portable C exports added for the GSA package.
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
__global__ void ball_query_kernel_fast(int b, int n, int m, float radius,
                                       int nsample,
                                       const float *__restrict__ new_xyz,
                                       const float *__restrict__ xyz,
                                       int *__restrict__ idx) {
  // new_xyz: (B, M, 3)
  // xyz: (B, N, 3)
  // output:
  //      idx: (B, M, nsample)
  int bs_idx = blockIdx.y;
  int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (bs_idx >= b || pt_idx >= m)
    return;

  new_xyz += bs_idx * m * 3 + pt_idx * 3;
  xyz += bs_idx * n * 3;
  idx += bs_idx * m * nsample + pt_idx * nsample;

  float radius2 = radius * radius;
  float new_x = new_xyz[0];
  float new_y = new_xyz[1];
  float new_z = new_xyz[2];

  int cnt = 0;
  for (int k = 0; k < n; ++k) {
    float x = xyz[k * 3 + 0];
    float y = xyz[k * 3 + 1];
    float z = xyz[k * 3 + 2];
    float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
               (new_z - z) * (new_z - z);
    if (d2 < radius2) {
      if (cnt == 0) {
        for (int l = 0; l < nsample; ++l) {
          idx[l] = k;
        }
      }
      idx[cnt] = k;
      ++cnt;
      if (cnt >= nsample)
        break;
    }
  }
}

__global__ void three_nn_kernel_fast(int b, int n, int m,
                                     const float *__restrict__ unknown,
                                     const float *__restrict__ known,
                                     float *__restrict__ dist2,
                                     int *__restrict__ idx) {
  // unknown: (B, N, 3)
  // known: (B, M, 3)
  // output:
  //      dist2: (B, N, 3)
  //      idx: (B, N, 3)

  int bs_idx = blockIdx.y;
  int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (bs_idx >= b || pt_idx >= n)
    return;

  unknown += bs_idx * n * 3 + pt_idx * 3;
  known += bs_idx * m * 3;
  dist2 += bs_idx * n * 3 + pt_idx * 3;
  idx += bs_idx * n * 3 + pt_idx * 3;

  float ux = unknown[0];
  float uy = unknown[1];
  float uz = unknown[2];

  double best1 = 1e40, best2 = 1e40, best3 = 1e40;
  int besti1 = 0, besti2 = 0, besti3 = 0;
  for (int k = 0; k < m; ++k) {
    float x = known[k * 3 + 0];
    float y = known[k * 3 + 1];
    float z = known[k * 3 + 2];
    float d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);
    if (d < best1) {
      best3 = best2;
      besti3 = besti2;
      best2 = best1;
      besti2 = besti1;
      best1 = d;
      besti1 = k;
    } else if (d < best2) {
      best3 = best2;
      besti3 = besti2;
      best2 = d;
      besti2 = k;
    } else if (d < best3) {
      best3 = d;
      besti3 = k;
    }
  }
  dist2[0] = best1;
  dist2[1] = best2;
  dist2[2] = best3;
  idx[0] = besti1;
  idx[1] = besti2;
  idx[2] = besti3;
}

#ifdef _WIN32
#define API extern "C" __declspec(dllexport) int
#else
#define API extern "C" __attribute__((visibility("default"))) int
#endif
API ball(int b, int n, int m, float r, int ns, const float *q, const float *x,
         int *idx, void *s) {
  ball_query_kernel_fast<<<dim3((m + 255) / 256, b), 256, 0, (cudaStream_t)s>>>(
      b, n, m, r, ns, q, x, idx);
  return cudaGetLastError();
}
API nn3(int b, int n, int m, const float *q, const float *x, float *d, int *idx,
        void *s) {
  three_nn_kernel_fast<<<dim3((n + 255) / 256, b), 256, 0, (cudaStream_t)s>>>(
      b, n, m, q, x, d, idx);
  return cudaGetLastError();
}
