#ifndef INTERSECTION_H
#define INTERSECTION_H

#include <glm/glm.hpp>
#include <src/conf.hpp>
#include <src/util.hpp>

struct ObjDesc;

struct Ray {
  CUDECL Ray(glm::vec3 p0, glm::vec3 dir, int iter, bool out) :
    p0(p0),
    dir(dir),
#ifndef __CUDA_ARCH__
    iter(iter),
#endif
    outside(out),
    tx(1,1,1) {}
  CUDECL Ray() :
    p0(0),
    dir(0),
#ifndef __CUDA_ARCH__
    iter(0),
#endif
    outside(true),
    tx(1,1,1) {}
  glm::vec3 p0, dir;
#ifndef __CUDA_ARCH__
  int iter;
#endif
  bool outside;
  glm::vec3 tx;
  int pixIdx;
};

struct Intersection {
  CUDECL Intersection() : t(-1), mat(NULL) {}
  // The parameter `t` along the ray which was used. (A negative value indicates no intersection.)
  float t;
  // The surface normal at the point of intersection. (Ignored if t < 0.)
  glm::vec3 normal;
  MatConf *mat;
  glm::vec3 pos;
};

CUDECL bool objIntersect(const ObjDesc &obj, const Ray &ray, Intersection &ix);

#endif /* INTERSECTION_H */
