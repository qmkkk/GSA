// SPDX-License-Identifier: Apache-2.0
// Spatial median BVH construction uses NVIDIA cuBQL (Apache-2.0).
// Prefix metadata uses its published bottom-up refit pattern; query semantics
// explicitly preserve PointNet++ float32 arithmetic and ascending-ID ties.
#include <cuda_runtime.h>
#include <exception>
#include <limits.h>
#include <stdint.h>
#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
#include "cuBQL/bvh.h"
#include "cuBQL/builder/cuda.h"
using namespace cuBQL;
// A private allocator pool, identical for every policy. Geometry is never
// retained.
static cudaMemPool_t private_pool = nullptr;
static int pool_enabled = 1;
static cudaMemPool_t get_pool() {
  if (!private_pool) {
    int device;
    CUBQL_CUDA_CALL(GetDevice(&device));
    cudaMemPoolProps props = {};
    props.allocType = cudaMemAllocationTypePinned;
    props.handleTypes = cudaMemHandleTypeNone;
    props.location.type = cudaMemLocationTypeDevice;
    props.location.id = device;
    CUBQL_CUDA_CALL(MemPoolCreate(&private_pool, &props));
    uint64_t threshold = UINT64_MAX;
    CUBQL_CUDA_CALL(MemPoolSetAttribute(
        private_pool, cudaMemPoolAttrReleaseThreshold, &threshold));
  }
  return private_pool;
}
struct Resource : GpuMemoryResource {
  void malloc(void **p, size_t n, cudaStream_t s) override {
    if (pool_enabled) {
      CUBQL_CUDA_CALL(MallocFromPoolAsync(p, n, get_pool(), s));
    } else {
      CUBQL_CUDA_CALL(MallocAsync(p, n, s));
    }
  }
  void free(void *p, cudaStream_t s) override {
    CUBQL_CUDA_CALL(FreeAsync(p, s));
  }
};
struct Handle {
  bvh3f bvh;
  box3f *prefix = nullptr;
  int *minid = nullptr;
  const vec3f *x;
  int n;
  Resource resource;
};
__global__ void make_boxes(box3f *b, const vec3f *x, int n) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) {
    b[t].lower = x[t];
    b[t].upper = x[t];
  }
}
__global__ void prefix_init(bvh3f bvh, uint32_t *meta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 1 || i >= bvh.numNodes)
    return;
  if (i == 0)
    meta[0] = 0;
  auto a = bvh.nodes[i].admin;
  if (!a.count) {
    meta[a.offset] = i << 1;
    meta[a.offset + 1] = i << 1;
  }
}
__global__ void prefix_refit(bvh3f bvh, const vec3f *x, int n, uint32_t *meta,
                             box3f *levels, int *minids) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 1 || i >= bvh.numNodes)
    return;
  auto a = bvh.nodes[i].admin;
  if (!a.count)
    return;
  box3f box[4];
  for (int l = 0; l < 4; l++)
    box[l].set_empty();
  int minid = INT_MAX;
  for (int k = 0; k < a.count; k++) {
    int id = bvh.primIDs[a.offset + k];
    minid = min(minid, id);
    if (levels) {
      int m = n;
      for (int l = 0; l < 4; l++, m /= 4)
        if (id < m) {
          box[l].lower = min(box[l].lower, x[id]);
          box[l].upper = max(box[l].upper, x[id]);
        }
    }
  }
  int parent = meta[i] >> 1;
  while (true) {
    minids[i] = minid;
    if (levels)
      for (int l = 0; l < 4; l++)
        levels[l * bvh.numNodes + i] = box[l];
    __threadfence();
    if (i == 0)
      break;
    uint32_t bits = atomicAdd(meta + parent, 1u);
    if (!(bits & 1))
      break;
    i = parent;
    parent = bits >> 1;
    auto admin = bvh.nodes[i].admin;
    minid = min(minids[admin.offset], minids[admin.offset + 1]);
    if (levels)
      for (int l = 0; l < 4; l++) {
        auto left = levels[l * bvh.numNodes + admin.offset],
             right = levels[l * bvh.numNodes + admin.offset + 1];
        box[l].lower = min(left.lower, right.lower);
        box[l].upper = max(left.upper, right.upper);
      }
  }
}
__device__ float lower_distance(vec3f q, box3f b) {
  if (b.lower.x > b.upper.x)
    return INFINITY;
  float dx =
      fmaxf(0.f, fmaxf(__fsub_rd(b.lower.x, q.x), __fsub_rd(q.x, b.upper.x)));
  float dy =
      fmaxf(0.f, fmaxf(__fsub_rd(b.lower.y, q.y), __fsub_rd(q.y, b.upper.y)));
  float dz =
      fmaxf(0.f, fmaxf(__fsub_rd(b.lower.z, q.z), __fsub_rd(q.z, b.upper.z)));
  return __fadd_rd(__fadd_rd(__fmul_rd(dx, dx), __fmul_rd(dy, dy)),
                   __fmul_rd(dz, dz));
}
__device__ void insert_point(vec3f q, vec3f x, int id, float *d, int *ix) {
  float dx = q.x - x.x, dy = q.y - x.y, dz = q.z - x.z;
  float v = (dx * dx + dy * dy) + dz * dz;
  if (!isfinite(v))
    return;
  for (int k = 0; k < 3; k++)
    if (v < d[k] || (v == d[k] && id < ix[k])) {
      for (int j = 2; j > k; j--) {
        d[j] = d[j - 1];
        ix[j] = ix[j - 1];
      }
      d[k] = v;
      ix[k] = id;
      break;
    }
}
__global__ void query3(bvh3f bvh, const box3f *levels, const vec3f *x,
                       const vec3f *q, int nq, int m, int level, float *out,
                       int *idx, int *fallback) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= nq)
    return;
  vec3f p = q[t];
  float d[3] = {INFINITY, INFINITY, INFINITY};
  int ids[3] = {INT_MAX, INT_MAX, INT_MAX};
  int stack[64], top = 0;
  stack[top++] = 0;
  bool overflow = false;
  while (top) {
    int nodeid = stack[--top];
    auto node = bvh.nodes[nodeid];
    box3f box = level < 0 ? node.bounds : levels[level * bvh.numNodes + nodeid];
    if (lower_distance(p, box) > d[2])
      continue;
    if (node.admin.count) {
      for (int k = 0; k < node.admin.count; k++) {
        int id = bvh.primIDs[node.admin.offset + k];
        if (id < m)
          insert_point(p, x[id], id, d, ids);
      }
    } else {
      int left = node.admin.offset, right = left + 1;
      box3f a = level < 0 ? bvh.nodes[left].bounds
                          : levels[level * bvh.numNodes + left];
      box3f b = level < 0 ? bvh.nodes[right].bounds
                          : levels[level * bvh.numNodes + right];
      float da = lower_distance(p, a), db = lower_distance(p, b);
      if (da < db) {
        int tmp = left;
        left = right;
        right = tmp;
        float v = da;
        da = db;
        db = v;
      }
      if (top + 2 > 64) {
        overflow = true;
        break;
      }
      if (da <= d[2])
        stack[top++] = left;
      if (db <= d[2])
        stack[top++] = right;
    }
  }
  if (overflow) {
    atomicAdd(fallback, 1);
    for (int k = 0; k < 3; k++) {
      d[k] = INFINITY;
      ids[k] = INT_MAX;
    }
    for (int j = 0; j < m; j++)
      insert_point(p, x[j], j, d, ids);
  }
  for (int k = 0; k < 3; k++) {
    out[3 * t + k] = d[k];
    idx[3 * t + k] = ids[k] == INT_MAX ? 0 : ids[k];
  }
}
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif
API int build_bvh(const float *x, int n, int prefix, int leaf, void **out,
                  void *stream) {
  Handle *h = new Handle();
  h->x = (const vec3f *)x;
  h->n = n;
  cudaStream_t s = (cudaStream_t)stream;
  try {
    box3f *boxes;
    h->resource.malloc((void **)&boxes, n * sizeof(box3f), s);
    make_boxes<<<(n + 255) / 256, 256, 0, s>>>(boxes, h->x, n);
    gpuBuilder(h->bvh, boxes, n, BuildConfig(leaf), s, h->resource);
    h->resource.free(boxes, s);
    {
      uint32_t *meta;
      h->resource.malloc((void **)&meta, h->bvh.numNodes * sizeof(uint32_t), s);
      h->resource.malloc((void **)&h->minid, h->bvh.numNodes * sizeof(int), s);
      if (prefix)
        h->resource.malloc((void **)&h->prefix,
                           4 * h->bvh.numNodes * sizeof(box3f), s);
      prefix_init<<<(h->bvh.numNodes + 255) / 256, 256, 0, s>>>(h->bvh, meta);
      prefix_refit<<<(h->bvh.numNodes + 127) / 128, 128, 0, s>>>(
          h->bvh, h->x, n, meta, h->prefix, h->minid);
      h->resource.free(meta, s);
    }
    *out = h;
    return cudaGetLastError();
  } catch (const std::exception &e) {
    fprintf(stderr, "cuBQL build: %s\n", e.what());
    delete h;
    return -1;
  }
}
API int query_bvh(void *handle, const float *q, int nq, int m, int level,
                  float *d, int *ix, int *fallback, void *stream) {
  Handle *h = (Handle *)handle;
  if (m < 3 || m > h->n || level > 3)
    return -2;
  query3<<<(nq + 127) / 128, 128, 0, (cudaStream_t)stream>>>(
      h->bvh, h->prefix, h->x, (const vec3f *)q, nq, m, level, d, ix, fallback);
  return cudaGetLastError();
}
API int free_bvh(void *handle, int async_free, void *stream) {
  Handle *h = (Handle *)handle;
  cudaStream_t s = (cudaStream_t)stream;
  try {
    if (async_free) {
      h->resource.free(h->bvh.primIDs, s);
      h->resource.free(h->bvh.nodes, s);
    } else
      cuBQL::cuda::free(h->bvh, s, h->resource);
    if (h->prefix)
      h->resource.free(h->prefix, s);
    if (h->minid)
      h->resource.free(h->minid, s);
    delete h;
    return cudaGetLastError();
  } catch (...) {
    return -3;
  }
}

__device__ float point_distance(vec3f q, vec3f x) {
  float dx = q.x - x.x, dy = q.y - x.y, dz = q.z - x.z;
  return (dx * dx + dy * dy) + dz * dz;
}
__global__ void radius_ordered(bvh3f bvh, const box3f *levels,
                               const int *minids, const vec3f *x,
                               const vec3f *q, int nq, int m, int level,
                               float radius, int ns, int *out, int *fallback) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= nq)
    return;
  vec3f p = q[t];
  int ids[32];
  for (int k = 0; k < ns; k++)
    ids[k] = INT_MAX;
  float r2 = radius * radius;
  int stack[64], top = 0, count = 0;
  stack[top++] = 0;
  bool overflow = false;
  while (top) {
    int i = stack[--top];
    auto node = bvh.nodes[i];
    int mid = minids[i];
    if (mid >= m || (count == ns && mid >= ids[ns - 1]))
      continue;
    box3f box = level < 0 ? node.bounds : levels[level * bvh.numNodes + i];
    if (lower_distance(p, box) >= r2)
      continue;
    if (node.admin.count) {
      for (int k = 0; k < node.admin.count; k++) {
        int id = bvh.primIDs[node.admin.offset + k];
        if (id >= m || id >= ids[ns - 1] || !(point_distance(p, x[id]) < r2))
          continue;
        int pos = min(count, ns - 1);
        while (pos > 0 && ids[pos - 1] > id) {
          ids[pos] = ids[pos - 1];
          --pos;
        }
        ids[pos] = id;
        count = min(count + 1, ns);
      }
    } else {
      int left = node.admin.offset, right = left + 1;
      if (minids[left] < minids[right]) {
        int tmp = left;
        left = right;
        right = tmp;
      }
      if (top + 2 > 64) {
        overflow = true;
        break;
      }
      stack[top++] = left;
      stack[top++] = right;
    }
  }
  if (overflow) {
    atomicAdd(fallback, 1);
    count = 0;
    for (int id = 0; id < m && count < ns; id++)
      if (point_distance(p, x[id]) < r2)
        ids[count++] = id;
  }
  for (int k = 0; k < ns; k++)
    out[t * ns + k] = count ? (k < count ? ids[k] : ids[0]) : 0;
}
API int query_ball_bvh(void *handle, const float *q, int nq, int m, int level,
                       float radius, int ns, int *out, int *fallback,
                       void *stream) {
  Handle *h = (Handle *)handle;
  if (m < 1 || m > h->n || level > 3 || ns < 1 || ns > 32)
    return -2;
  radius_ordered<<<(nq + 127) / 128, 128, 0, (cudaStream_t)stream>>>(
      h->bvh, h->prefix, h->minid, h->x, (const vec3f *)q, nq, m, level, radius,
      ns, out, fallback);
  return cudaGetLastError();
}

API int set_pool_mode(int enabled) {
  pool_enabled = enabled;
  return 0;
}
API int pool_memory(uint64_t *used, uint64_t *reserved) {
  if (!private_pool) {
    *used = 0;
    *reserved = 0;
    return 0;
  }
  cudaError_t e =
      cudaMemPoolGetAttribute(private_pool, cudaMemPoolAttrUsedMemHigh, used);
  if (e)
    return e;
  return cudaMemPoolGetAttribute(private_pool, cudaMemPoolAttrReservedMemHigh,
                                 reserved);
}
API int pool_reset_peak() {
  if (!private_pool)
    return 0;
  uint64_t zero = 0;
  cudaError_t e =
      cudaMemPoolSetAttribute(private_pool, cudaMemPoolAttrUsedMemHigh, &zero);
  if (e)
    return e;
  return cudaMemPoolSetAttribute(private_pool, cudaMemPoolAttrReservedMemHigh,
                                 &zero);
}
