#ifndef CUDA_CAMERA_H
#define CUDA_CAMERA_H

#include <src/camera.hpp>
#include <src/mesh/intersection.hpp>

void rayInit(Ray *rayList, CamDesc *cam);
void rayBounce(int nRays, Ray *rayList, int nObj, ObjDesc *objList, Intersection *ixList);

#endif /* CUDA_CAMERA_H */
