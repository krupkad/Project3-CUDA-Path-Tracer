#define GLM_FORCE_RADIANS
#define GLM_SIMD_ENABLE_XYZW_UNION
#define GLM_FORCE_INLINE

#include <src/mesh/cube.hpp>
#include <src/util.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/component_wise.hpp>
#include <glm/gtx/associated_min_max.hpp>

#include <vector>
#include <GL/glew.h>
#include <iostream>

Cube::Cube() : len(.5) {
  buildObject();
  objDesc.type = OBJ_CUBE;
  objDesc.bbMin = glm::vec3(-len, -len, -len);
  objDesc.bbMax = glm::vec3(len, len, len);
}

Cube::~Cube() {
  vtx.clear();
  nrm.clear();
  idx.clear();
}

ObjDesc *Cube::getDesc() {
  return &objDesc;
}

void Cube::buildObject() {
  vtx.clear();
  nrm.clear();
  idx.clear();

  glm::vec3 vs[] = {
    glm::vec3(len,len,len),
    glm::vec3(len,-len,len),
    glm::vec3(len,-len,-len),
    glm::vec3(len,len,-len),
    glm::vec3(-len,len,-len),
    glm::vec3(-len,len,len),
    glm::vec3(-len,-len,len),
    glm::vec3(-len,-len,-len)
  };

  for(unsigned int i = 0; i < 8; i++) {
    glm::vec3 nvec = glm::vec3(0,0,0);
    for(int d = 0; d < 3; d++)
      if(vs[i][d] > 0)
        nvec[d] = 1;
      else
        nvec[d] = -1;
    vtx.push_back(vs[i]);
    nrm.push_back(nvec);
  }

  for(int i = 1; i < 7; i++) {
    unsigned int j = i%6 + 1;
    idx.push_back(0);
    idx.push_back(i);
    idx.push_back(j);

    idx.push_back(7);
    idx.push_back(i);
    idx.push_back(j);
  }

  tex.resize(vtx.size(),glm::vec2());

  uploadVBO();
}

bool Cube::doIntersect(const Ray& ray,Intersection& ix) {
  ix.t = -1;

  glm::vec3 x0 = ray.p0, dx = ray.dir;

  float tFar = FLT_MAX;
  float tNear = -FLT_MAX;
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

glm::vec3 Cube::sample() {
  int i = rand() % 6;
  float x = float(rand())/RAND_MAX - len, y = float(rand())/RAND_MAX - len, z = 2.0f*(float(i%2) - 1)*len;
  switch(i/2) {
    case 0:
      return glm::vec3(z,x,y);
    case 1:
      return glm::vec3(x,z,y);
    case 2:
      return glm::vec3(x,y,z);
    default:
      return glm::vec3();
  }
}
