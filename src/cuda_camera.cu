#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <src/mesh/intersection.hpp>
#include <src/mesh/object.hpp>
#include <src/texture.hpp>
#include <src/camera.hpp>
#include <src/node.hpp>
#include <src/util.hpp>

#include <unordered_map>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/component_wise.hpp>
#include <glm/gtc/random.hpp>

#include <stream_compaction/compact.hpp>

#include <stb_image_write.h>

__global__ void kernRangeInit(int n, int *arr) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  arr[i] = i;
}

__global__ void kernRayInit(Ray *rayList, const CamDesc *cam) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;


  if (i >= cam->width || j >= cam->height || k >= cam->density2)
    return;

  float dx = 0, dy = 0;
  if (cam->useDOF && cam->density > 1) {
    float dr = float(k % cam->density) * cam->aperture / (cam->density - 1);
    float dth = float(k / cam->density) * 2.0f * M_PI / (cam->density - 1);
    dx = dr*cos(dth);
    dy = dr*sin(dth);
  }

  float x = (float(i)) / cam->width;
  float y = (float(j)) / cam->height;
  glm::vec3 pos = cam->pos + dx*cam->xAxis + dy*cam->yAxis;
  glm::vec3 pp = cam->pos + cam->fwd + (2.0f*x-1.0f)*cam->xAxis + (1.0f-2.0f*y)*cam->yAxis;
  glm::vec3 dir = glm::normalize(pp-pos);

  Ray &ray = rayList[k + cam->density*(i + cam->width*j)];
  ray = Ray(pos, dir, cam->rayCount, true);
  ray.pixIdx = i + j*cam->width;
  ray.dir = glm::normalize(ray.dir);
}

__device__ bool devShade(Ray &ray, Intersection &ix, rng_eng_t &eng, rng_dist_t &dist) {
  // if the ray didn't hit anything
  if(ix.t < 0.001) {
    ray.tx = glm::vec3(0);
    ix.t = -1;
    return false;
  }

  // if we hit a light, get its contribution and terminate the ray
  if(ix.mat->lEmit > 0.001) {
    ray.tx *= ix.mat->lEmit * ix.mat->diffCol;
    return false;
  }

  // Fresnel coefficients
  float n1,n2;
  if(ray.outside) {
    n1 = 1.0;
    n2 = ix.mat->ior;
  } else {
    n1 = ix.mat->ior;
    n2 = 1.0;
  }
  float nRatio = n1/n2;
  float cosTi = -glm::dot(ray.dir, ix.normal);
  float rSinTi2 = nRatio*nRatio*(1 - cosTi*cosTi);
  float rCoeff = 1.0, tCoeff = 0.0;
  float sq = std::sqrt(1 - rSinTi2);
  if(rSinTi2 < 1) {
    float rs = (nRatio*cosTi - sq) / (nRatio*cosTi + sq);
    float rp = (cosTi - nRatio*sq) / (cosTi + nRatio*sq);
    rCoeff = (rs*rs + rp*rp)/2;
    tCoeff = 1.0 - rCoeff;
  }

  glm::vec3 rflDir = glm::reflect(ray.dir, ix.normal);

  float pDiff = glm::compMax(ix.mat->diffCol);
  float pSpec = glm::compMax(ix.mat->specCol) + pDiff;
  float pRefl = ix.mat->mirr + pSpec;
  float pRefr = ix.mat->trans + pRefl;

  float p = dist(eng) * pRefr;
  if (p < pDiff) {
    glm::vec3 lmbCol;
    ray.dir = getCosineWeightedDirection(ix.normal, eng, dist);
    lmbCol = (1.f-ix.mat->trans)*ix.mat->diffCol*glm::clamp(glm::dot(ray.dir, ix.normal),0.0f,1.0f);
    ray.tx *= lmbCol;
  } else if (p < pSpec) {
    glm::vec3 lmbCol;
    lmbCol = ix.mat->specCol*glm::clamp(glm::dot(rflDir, ix.normal),0.0f,1.0f);
    ray.tx *= lmbCol;
    ray.dir = rflDir;
  } else if (p < pRefl) {
    ray.dir = rflDir;
    ray.tx *= rCoeff * ix.mat->mirr * ix.mat->specCol;
  } else if (p < pRefr) {
    if(rSinTi2 < 1) {
      ray.dir = glm::refract(rflDir, ix.normal, nRatio);
      ray.outside = !ray.outside;
    }
    ray.tx *= tCoeff * ix.mat->trans;
  }

  ray.p0 = ix.pos + .001f*ray.dir;
  return true;
}

__global__ void kernRayBounce(int nRays, Ray *rayList, int nObj, ObjDesc *objList, Intersection *ixList, int *hitList) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nRays)
    return;

  Ray &ray = rayList[i];
  Intersection &ix = ixList[i];

  float hitMin = FLT_MAX;
  Intersection ixTmp;
  hitList[i] = 0;
  for (int j = 0; j < nObj; j++) {
    bool hit = objIntersect(objList[j], ray, ixTmp);
    if (hit && ixTmp.t < hitMin) {
      hitMin = ixTmp.t;
      ix = ixTmp;
      hitList[i] = 1;
    }
  }
}

__global__ void kernShade(int nRays, Ray *rayList, Intersection *ixList, int *hitList, int iter) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nRays)
    return;

  rng_eng_t rng = makeSeededRandomEngine(iter, i, 0);
  rng_dist_t u01(0, 1);

  Ray &ray = rayList[i];
  Intersection &ix = ixList[i];
  if (hitList[i]) {
    if (!devShade(ray, ix, rng, u01))
      hitList[i] = 0;
    else
      hitList[i] = 1;
  }
}

__global__ void kernGetRayMat(int nRays, Intersection *ixList, void **rayMatList) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nRays)
    return;

  if (ixList[i].t > 0.0001)
    rayMatList[i] = ixList[i].mat;
  else
    rayMatList[i] = nullptr;
}

__global__ void kernImageUpdate(int nRays, Ray *rayList, Intersection *ixList, CamDesc *cam, glm::vec3 *data, int *hitList) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nRays)
    return;

  Ray &ray = rayList[i];
  if (hitList[i] == 0 && ixList[i].t > 0) {
    atomicAdd(&data[ray.pixIdx].x, ray.tx.x);
    atomicAdd(&data[ray.pixIdx].y, ray.tx.y);
    atomicAdd(&data[ray.pixIdx].z, ray.tx.z);
  }
}

__global__ void kernImageWrite(CamDesc *cam, glm::vec3 *data, uchar4 *out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= cam->width || j >= cam->height)
    return;

  glm::vec3 v = glm::clamp(data[i+cam->width*j] / float(cam->mcIter * cam->density),0.0f,1.0f) * 255.0f;
  out[i + cam->width*j].x = v.x;
  out[i + cam->width*j].y = v.y;
  out[i + cam->width*j].z = v.z;
  out[i + cam->width*j].w = 255;
}

void Camera::pathtrace(Node *root) {
  glm::vec3 *devData;
  cudaMalloc((void**)&devData, width*height*sizeof(glm::vec3));
  cudaMemset(devData, 0, width*height*sizeof(glm::vec3));

  float aRatio = (float)width/height;
  glm::vec3 xAxis = glm::cross(fwd, up);
  glm::vec3 yAxis = glm::cross(xAxis, fwd);
  float xFov = std::atan(std::tan(yFov) * aRatio);
  xAxis *= std::tan(xFov/2) * glm::length(fwd) / glm::length(xAxis);
  yAxis *= std::tan(yFov/2) * glm::length(fwd) / glm::length(yAxis);

  std::vector<Node*> nodeList;
  root->buildPreOrder(nodeList);

  // create the camera for cuda
  CamDesc cam, *devCam;
  cam.xAxis = xAxis;
  cam.yAxis = yAxis;
  cam.width = width;
  cam.height = height;
  cam.density = density;
  cam.fwd = fwd;
  cam.pos = pos;
  cam.rayCount = rayCount;
  cam.mcIter = mcIter;
  cam.useDOF = useDOF;
  cam.aperture = aperture;
  cam.density2 = density*density;
  cudaMalloc((void**)&devCam, sizeof(CamDesc));
  cudaMemcpy(devCam, &cam, sizeof(CamDesc), cudaMemcpyHostToDevice);

  // create the list of materials
  std::unordered_map<MatConf*, MatConf*> matPtrMap;
  for (Node *n : nodeList) {
    ObjDesc *obj = n->getDesc();
    matPtrMap[obj->mat] = nullptr;
  }

  // transfer material descriptions
  MatConf *devMatList;
  cudaMalloc((void**)&devMatList, matPtrMap.size()*sizeof(MatConf));
  int i = 0;
  for (auto &pair : matPtrMap) {
    cudaMemcpy(&devMatList[i], pair.first, sizeof(MatConf), cudaMemcpyHostToDevice);
    pair.second = &devMatList[i];
    i++;
  }

  // transfer object descriptions
  ObjDesc *devObjList;
  cudaMalloc((void**)&devObjList, nodeList.size()*sizeof(ObjDesc));
  i = 0;
  for (Node *n : nodeList) {
    ObjDesc obj = *n->getDesc();
    obj.mat = matPtrMap[obj.mat];
    cudaMemcpy(&devObjList[i], &obj, sizeof(ObjDesc), cudaMemcpyHostToDevice);
    i++;
  }

  // create the initial list of rays
  printf("ray create\n");
  Ray *devRayList[2];
  int nRays = width*height*density*density, nRaysOrig = nRays;
  cudaMalloc((void**)&devRayList[0], nRays*sizeof(Ray));
  cudaMalloc((void**)&devRayList[1], nRays*sizeof(Ray));
  checkCUDAError("cudaMalloc");

  // allocate the intersection list, hit list, and live list
  printf("ix/hit create\n");
  Intersection *devIxList[2], *devIxCache;
  int *devHitList[2], *devHitCache;
  int *devRayIdx;
  void **devRayMat;
  cudaMalloc((void**)&devIxList[0], nRays*sizeof(Intersection));
  cudaMalloc((void**)&devHitList[0], nRays*sizeof(int));
  if (useMatSort) {
    cudaMalloc((void**)&devIxList[1], nRays*sizeof(Intersection));
    cudaMalloc((void**)&devHitList[1], nRays*sizeof(int));
    cudaMalloc((void**)&devRayIdx, nRays*sizeof(int));
    cudaMalloc((void**)&devRayMat, nRays*sizeof(void*));
  }
  if (useIxCache) {
    cudaMalloc((void**)&devIxCache, nRays*sizeof(Intersection));
    cudaMalloc((void**)&devHitCache, nRays*sizeof(int));
  }
  checkCUDAError("cudaMalloc");

  // thrust device_ptr wrappers
  thrust::device_ptr<int> t_RayIdx(devRayIdx);
  thrust::device_ptr<int> t_RayMat((int*)devRayMat);
  thrust::device_ptr<Intersection> t_IxList[2];
  t_IxList[0] = thrust::device_ptr<Intersection>(devIxList[0]);
  t_IxList[1] = thrust::device_ptr<Intersection>(devIxList[1]);
  thrust::device_ptr<int> t_HitList[2];
  t_HitList[0] = thrust::device_ptr<int>(devHitList[0]);
  t_HitList[1] = thrust::device_ptr<int>(devHitList[1]);
  thrust::device_ptr<Ray> t_RayList[2];
  t_RayList[0] = thrust::device_ptr<Ray>(devRayList[0]);
  t_RayList[1] = thrust::device_ptr<Ray>(devRayList[1]);


  for (int j = 0; j < mcIter; j++) {
    // create the set of rays
    dim3 blkCnt2, blkSize2(8,8,4);
    blkCnt2.x = (width + blkSize2.x - 1) / blkSize2.x;
    blkCnt2.y = (height + blkSize2.y - 1) / blkSize2.y;
    blkCnt2.z = (density + blkSize2.z - 1) / blkSize2.z;
    kernRayInit<<<blkCnt2, blkSize2>>>(devRayList[0], devCam);
    nRays = nRaysOrig;

    dim3 blkSize(256), blkCnt;
    if (useMatSort) {
      for (int i = 0; i < rayCount; i++) {
        // perform one bounce, mark rays that are escaped or terminated
        if (useIxCache && i == 0 && j > 0) {
          cudaMemcpy(devIxList[0], devIxCache, nRaysOrig*sizeof(Intersection), cudaMemcpyDeviceToDevice);
          cudaMemcpy(devHitList[0], devHitCache, nRaysOrig*sizeof(int), cudaMemcpyDeviceToDevice);
        } else {
          blkCnt.x = (nRays + blkSize.x - 1)/ blkSize.x;
          kernRayBounce<<<blkCnt, blkSize>>>(nRays, devRayList[0], nodeList.size(), devObjList, devIxList[0], devHitList[0]);
          checkCUDAError("kernRayBounce");
        }

        if (useIxCache && i == 0 && j == 0) {
          cudaMemcpy(devIxCache, devIxList[0], nRays*sizeof(Intersection), cudaMemcpyDeviceToDevice);
          cudaMemcpy(devHitCache, devHitList[0], nRays*sizeof(int), cudaMemcpyDeviceToDevice);
        }

        // sort ray data by intersected material
        kernGetRayMat<<<blkCnt, blkSize>>>(nRays, devIxList[0], devRayMat);
        checkCUDAError("kernGetRayMat");
        kernRangeInit<<<blkCnt, blkSize>>>(nRays, devRayIdx);
        checkCUDAError("kernRangeInit");
        thrust::sort_by_key(t_RayMat, t_RayMat + nRays, t_RayIdx);
        thrust::gather(t_RayIdx, t_RayIdx + nRays, t_IxList[0], t_IxList[1]);
        thrust::gather(t_RayIdx, t_RayIdx + nRays, t_HitList[0], t_HitList[1]);
        thrust::gather(t_RayIdx, t_RayIdx + nRays, t_RayList[0], t_RayList[1]);
        checkCUDAError("matSort");

        // shade rays
        kernShade<<<blkCnt, blkSize>>>(nRays, devRayList[1], devIxList[1], devHitList[1], j);
        checkCUDAError("kernShade");

        // update values from rays that have hit lights
        blkCnt.x = (nRays + blkSize.x - 1)/ blkSize.x;
        kernImageUpdate<<<blkCnt, blkSize>>>(nRays, devRayList[1], devIxList[1], devCam, devData, devHitList[1]);
        checkCUDAError("kernImageUpdate");

        // discard rays that have terminated or escaped
        int hits = Compaction::compact(nRays, devRayList[0], devRayList[1], devHitList[1]);
        checkCUDAError("compact");
        nRays = hits;
      }
    }
    else {
      for (int i = 0; i < rayCount; i++) {
        // perform one bounce, mark rays that are escaped or terminated
        if (useIxCache && i == 0 && j > 0) {
          cudaMemcpy(devIxList[0], devIxCache, nRaysOrig*sizeof(Intersection), cudaMemcpyDeviceToDevice);
          cudaMemcpy(devHitList[0], devHitCache, nRaysOrig*sizeof(int), cudaMemcpyDeviceToDevice);
        } else {
          blkCnt.x = (nRays + blkSize.x - 1)/ blkSize.x;
          kernRayBounce<<<blkCnt, blkSize>>>(nRays, devRayList[0], nodeList.size(), devObjList, devIxList[0], devHitList[0]);
          checkCUDAError("kernRayBounce");
        }

        if (useIxCache && i == 0 && j == 0) {
          cudaMemcpy(devIxCache, devIxList[0], nRays*sizeof(Intersection), cudaMemcpyDeviceToDevice);
          cudaMemcpy(devHitCache, devHitList[0], nRays*sizeof(int), cudaMemcpyDeviceToDevice);
        }

        // shade rays
        kernShade<<<blkCnt, blkSize>>>(nRays, devRayList[0], devIxList[0], devHitList[0], j);
        checkCUDAError("kernShade");

        // update values from rays that have hit lights
        blkCnt.x = (nRays + blkSize.x - 1)/ blkSize.x;
        kernImageUpdate<<<blkCnt, blkSize>>>(nRays, devRayList[0], devIxList[0], devCam, devData, devHitList[0]);
        checkCUDAError("kernImageUpdate");

        // discard rays that have terminated or escaped
        int hits = Compaction::compact(nRays, devRayList[1], devRayList[0], devHitList[0]);
        checkCUDAError("compact");
        nRays = hits;

        std::swap(devRayList[0], devRayList[1]);
        if (nRays <= 0)
          break;
      }
    }
  }

  // copy and save image
  uchar4 *devCData;
  cudaMalloc((void**)&devCData, width*height*sizeof(uchar4));
  dim3 blkCnt2, blkSize2(16,16);
  blkCnt2.x = (width + blkSize2.x - 1) / blkSize2.x;
  blkCnt2.y = (height + blkSize2.y - 1) / blkSize2.y;
  kernImageWrite<<<blkCnt2, blkSize2>>>(devCam, devData, devCData);
  checkCUDAError("kernImageWrite");
  cudaFree(devData);

  unsigned char *data = new unsigned char[4*width*height];
  cudaMemcpy(data, devCData, width*height*sizeof(uchar4), cudaMemcpyDeviceToHost);
  stbi_write_png(outFile.c_str(), width, height, 4, data, 4*width);
  cudaFree(devCData);
  delete data;

  printf("free\n");
  cudaFree(devRayList[0]);
  cudaFree(devIxList[0]);
  cudaFree(devHitList[0]);
  cudaFree(devRayList[1]);
  cudaFree(devObjList);
  cudaFree(devMatList);
  cudaFree(devCam);
  if (useMatSort) {
    cudaFree(devIxList[1]);
    cudaFree(devHitList[1]);
    cudaFree(devRayMat);
    cudaFree(devRayIdx);
  }
  if (useIxCache) {
    cudaFree(devIxCache);
    cudaFree(devHitCache);
  }
}
