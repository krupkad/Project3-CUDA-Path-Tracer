#include <src/mesh/intersection.hpp>
#include <src/mesh/object.hpp>

#include <glm/gtx/intersect.hpp>

static CUDECL bool sphereIntersect(const ObjDesc &obj, const glm::vec3 &x0, const glm::vec3 &dx, Intersection &ix) {
  ix.t = -1;
  float a = glm::dot(dx,dx);
  if(a < 0.0001)
    return false;

  float r = obj.bbMax.x;
  float b = 2.f * glm::dot(dx,x0);
  float c = glm::dot(x0,x0) - r*r;

  float d2 = b*b - 4*a*c;
  if(d2 < 0)
    return false;

  float d = sqrt(d2);
  float t1 = (-b - d)/(2*a);
  float t2 = (-b + d)/(2*a);
  if(t1 > 0.001)
    ix.t = t1;
  if(t2 > 0.001 && t2 < t1)
    ix.t = t2;

  if (ix.t < 0)
    return false;

  ix.normal = ix.t*dx + x0;
  ix.pos = ix.t*dx + x0;

  //printf("sphere: %f %f %f\n", ix.normal.x, ix.normal.y, ix.normal.z);

  return true;
}

static CUDECL bool cubeIntersect(const ObjDesc &obj, const glm::vec3 &x0, const glm::vec3 &dx, Intersection &ix) {
  ix.t = -1;

  float tFar = FLT_MAX;
  float tNear = -FLT_MAX;
  float len = obj.bbMax.x;
  for(int i = 0; i < 3; i++) {
    if(std::abs(dx[i]) < .001) {
      if(std::abs(x0[i]) > len)
        return false;
      else
        continue;
    }

    float t1 = (-len - x0[i])/dx[i];
    float t2 = (len - x0[i])/dx[i];

    float x = (t1 < t2) ? -len : len;
    if(t1 > t2) {
      float tmp = t1;
      t1 = t2;
      t2 = tmp;
    }

    if(t2 < tFar) {
      tFar = t2;
    }
    if(t1 > tNear) {
      ix.normal = glm::vec3();
      ix.normal[i] = x;
      tNear = t1;
    }
    if(tNear > tFar || tFar < 0)
      return false;
  }

  ix.t = tNear;
  ix.pos = x0 + dx*ix.t;

  return true;
}


static CUDECL bool cylinderIntersect(const ObjDesc &obj, const glm::vec3 &x0, const glm::vec3 &dx, Intersection &ix) {
  ix.t = -1;

  float zmin = obj.bbMax.z;
  float zmax = obj.bbMin.z;
  float radius = obj.bbMax.x;

  // find the t's that intersect sides
  float a = dx.x*dx.x + dx.z*dx.z;
  float b = 2.f * (x0.x*dx.x + x0.z*dx.z);
  float c = x0.x*x0.x + x0.z*x0.z - radius*radius;
  float d = b*b - 4*a*c;
  float t1 = FLT_MAX,t2 = FLT_MAX,t3 = FLT_MAX,t4 = FLT_MAX;
  if(a > 0.0001 && d > 0) {
    t1 = (-b - sqrt(d))/(2*a);
    t2 = (-b + sqrt(d))/(2*a);
  }
  if(std::abs(dx.y) > 0.0001) {
    t3 = (zmin-x0.y)/dx.y;
    t4 = (zmax-x0.y)/dx.y;
  }

  float t = FLT_MAX;
  glm::vec3 x1 = x0 + t1*dx;
  glm::vec3 x2 = x0 + t2*dx;
  glm::vec3 x3 = x0 + t3*dx;
  glm::vec3 x4 = x0 + t4*dx;
  if(t1 > 0.0001 && t1 < t && x1.y < zmax && x1.y > zmin) {
    ix.normal = glm::vec3(x1.x,0,x1.z);
    ix.t = t1;
    t = t1;
  }
  if(t2 > 0.0001 && t2 < t && x2.y < zmax && x2.y > zmin) {
    ix.normal = glm::vec3(x2.x,0,x2.z);
    ix.t = t2;
    t = t2;
  }
  if(t3 > 0.0001 && t3 < t && x3.z*x3.z + x3.x*x3.x < radius*radius) {
    ix.normal = glm::vec3(0,zmin,0);
    ix.t = t3;
    t = t3;
  }
  if(t4 > 0.0001 && t4 < t && x4.z*x4.z + x4.x*x4.x < radius*radius) {
    ix.normal = glm::vec3(0,zmax,0);
    ix.t = t4;
    t = t4;
  }
  ix.pos = ix.t*dx + x0;

  return true;
}

static CUDECL bool meshIntersect(const ObjDesc &obj, const glm::vec3 &x0, const glm::vec3 &dx, Intersection &ix) {
  ix.t = FLT_MAX;

  int N = obj.meshTriCnt;
  glm::vec3 bary;
  bool found = false;
  for (int i = 0; i < N; i++) {
    bool res = glm::intersectRayTriangle(x0, dx, obj.meshVtx[3*i],
        obj.meshVtx[3*i+1], obj.meshVtx[3*i+2], bary);
    found |= res;

    float u = bary[0], v = bary[1], w = 1.0f - u - v;

    if (res) {
      glm::vec3 p = u*obj.meshVtx[3*i] + v*obj.meshVtx[3*i+1] + w*obj.meshVtx[3*i+2];
      float t = glm::dot(p-x0, dx);
      glm::vec3 nrm = u*obj.meshNrm[3*i] + v*obj.meshNrm[3*i+1] + w*obj.meshNrm[3*i+2];

      if (t < ix.t) {
        ix.t = t;
        ix.pos = p;
        ix.normal = glm::normalize(nrm);
      }
    }
  }

  if (!found)
    ix.t = -1;
  return found;
}

CUDECL bool objIntersect(const ObjDesc &obj, const Ray &ray, Intersection &ix) {
  glm::vec3 dx(obj.localModelInv * glm::vec4(ray.dir, 0));
  glm::vec3 x0(obj.localModelInv * glm::vec4(ray.p0, 1));

  bool found = false;
  switch (obj.type) {
    case OBJ_CUBE:
      found = cubeIntersect(obj, x0, dx, ix);
      break;
    case OBJ_SPHERE:
      found = sphereIntersect(obj, x0, dx, ix);
      break;
    case OBJ_CYLINDER:
      found = cylinderIntersect(obj, x0, dx, ix);
      break;
    case OBJ_MESH:
      if (!obj.bbCull || cubeIntersect(obj, x0, dx, ix))
        found = meshIntersect(obj, x0, dx, ix);
      else {
        ix.t = -1;
        found = false;
      }
      break;
  }

  if (!found) {
    ix.t = -1;
    return false;
  }

  glm::vec4 nrm4 = glm::vec4(glm::normalize(ix.normal),0);
  ix.normal = glm::normalize(glm::vec3(obj.localModelInvTr*nrm4));
  ix.pos = glm::vec3(obj.localModel * glm::vec4(ix.pos,1));
  ix.mat = obj.mat;

  return true;
}
