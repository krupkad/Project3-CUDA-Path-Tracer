#include <src/kdtree.hpp>
#include <src/util.hpp>
#include <glm/glm.hpp>
#include <vector>
#include <iostream>
#include <stdexcept>

#include <glm/gtx/intersect.hpp>
#include <glm/gtx/component_wise.hpp>

KDNode::KDNode(const std::vector<unsigned int>& triIdx, const std::vector<glm::vec3>& vtx, const std::vector<glm::vec3>& nrm) {
  left = NULL;
  right = NULL;
  if(!triIdx.size())
    throw std::runtime_error("attempt zero size kdtree\n");

  vMin = glm::vec3(FLT_MAX);
  vMax = glm::vec3(-FLT_MAX);
  unsigned int N = triIdx.size() / 3;
  nTris = N;

  // get bounding box, tri midpoints, average midpoint
  std::vector<glm::vec3> mids;
  glm::vec3 avgMid(0.0);
  for(unsigned int i = 0; i < N; i++) {
    glm::vec3 tMin(FLT_MAX), tMax(-FLT_MAX);
    for(int a = 0; a < 3; a++) {
      triMin(vtx[triIdx[3*i+a]], vMin);
      triMax(vtx[triIdx[3*i+a]], vMax);
      triMin(vtx[triIdx[3*i+a]], tMin);
      triMax(vtx[triIdx[3*i+a]], tMax);
    }
    glm::vec3 mid = (tMin + tMax) * .5f;
    mids.push_back(mid);
    avgMid += mid/float(N);
    //std::cout << tMin << " " << tMax << "\n";
  }
  //std::cout << vMin << " " << vMax << "\n";

  // choose the largest axis of the bounding box
  glm::vec3 bSize = glm::abs(vMax-vMin);
  float maxSize = -FLT_MAX;
  int maxAxis = -1;
  for(int i = 0; i < 3; i++) {
    if(bSize[i] > maxSize) {
      maxAxis = i;
      maxSize = bSize[i];
    }
  }

  // split based on which side of the axis (rel. avg midpoint)
  std::vector<unsigned int> lTriIdx, rTriIdx;
  for(unsigned int i = 0; i < N; i++) {
    if(mids[i][maxAxis] <= avgMid[maxAxis]) {
      for(int d = 0; d < 3; d++)
        lTriIdx.push_back(triIdx[3*i+d]);
    }
    if(mids[i][maxAxis] > avgMid[maxAxis]) {
      for(int d = 0; d < 3; d++)
        rTriIdx.push_back(triIdx[3*i+d]);
    }
  }

  if(lTriIdx.size() == 0 && rTriIdx.size() > 0) {
    for(int i = 0; i < rTriIdx.size(); i++) {
      tris.push_back(vtx[rTriIdx[i]]);
      nrms.push_back(nrm[rTriIdx[i]]);
    }
  }
  else if(rTriIdx.size() == 0 && lTriIdx.size() > 0) {
    for(int i = 0; i < lTriIdx.size(); i++) {
      tris.push_back(vtx[lTriIdx[i]]);
      nrms.push_back(nrm[lTriIdx[i]]);
    }
  } else if(lTriIdx.size() > 0 && rTriIdx.size() > 0) {
    left = new KDNode(lTriIdx, vtx, nrm);
    right = new KDNode(rTriIdx, vtx, nrm);
  }
}

KDNode::~KDNode() {
  if(left)
    delete left;
  if(right)
    delete right;
}

void KDNode::triMin(const glm::vec3& a, glm::vec3& min) {
  for(int i = 0; i < 3; i++) {
    if(a[i] < min[i])
      min[i] = a[i];
  }
}

void KDNode::triMax(const glm::vec3& a, glm::vec3& max) {
  for(int i = 0; i < 3; i++) {
    if(a[i] > max[i])
      max[i] = a[i];
  }
}

bool KDNode::sphereIx(const Ray& ray) {
  glm::vec3 mCtr = (vMax + vMin)/2.0f;
  float r = glm::length(vMax - vMin)/2.0f;

  glm::vec3 L = mCtr - ray.p0;
  glm::vec3 D = glm::normalize(ray.dir);
  float tca = glm::dot(L,D);
  if (tca < 0)
    return false;

  float d2 = glm::dot(L,L) - tca*tca;
  if (d2 > r*r)
    return false;

  float thc = sqrt(r*r - d2);
  float t0 = tca - thc;
  float t1 = tca + thc;
  if (t0 < 0 && t1 < 0)
    return false;

  return true;
}

bool KDNode::boxIx(const Ray& ray) {
  glm::vec3 td1((vMin - ray.p0)/ray.dir);
  glm::vec3 td2((vMax - ray.p0)/ray.dir);

  glm::vec3 cMin(glm::min(td1, td2));
  glm::vec3 cMax(glm::max(td1, td2));

  float tmin = glm::compMax(cMin);
  float tmax = glm::compMin(cMax);

  /*
  float tx1 = (vMin[0] - x0[0])/dx[0];
  float tx2 = (vMax[0] - x0[0])/dx[0];

  float tmin = std::min(tx1, tx2);
  float tmax = std::max(tx1, tx2);

  float ty1 = (vMin[1] - x0[1])/dx[1];
  float ty2 = (vMax[1] - x0[1])/dx[1];

  tmin = std::max(tmin, std::min(ty1, ty2));
  tmax = std::min(tmax, std::max(ty1, ty2));

  float tz1 = (vMin[2] - x0[2])/dx[2];
  float tz2 = (vMax[2] - x0[2])/dx[2];

  tmin = std::max(tmin, std::min(tz1, tz2));
  tmax = std::min(tmax, std::max(tz1, tz2));
  */

  return tmax > std::max(0.0f, tmin);
}

bool KDNode::triIx(const Ray& ray, unsigned int i, Intersection &ix) {
  glm::vec3 p1 = tris[3*i], p2 = tris[3*i+1], p3 = tris[3*i+2];

  glm::vec3 dx = ray.dir;
  glm::vec3 x0 = ray.p0;

  glm::vec3 e1(p2-p1), e2(p3-p1);

  glm::vec3 N = glm::cross(e1,e2);
  float w = -glm::dot(p1,N);
  float k = glm::dot(dx,N);
  if(std::abs(k) < 0.0001)
    return false;
  float tt = -(glm::dot(x0,N) + w)/k;
  glm::vec3 pIx = x0 + tt*dx;

  glm::vec3 P = glm::cross(dx, e2);
  float det = glm::dot(e1,P);
  if(std::abs(det) < .0001)
    return false;

  glm::vec3 T = x0 - p1;
  float u = glm::dot(T,P) / det;
  if(u < 0 || u > 1)
    return false;

  glm::vec3 Q = glm::cross(T, e1);
  float v = glm::dot(dx, Q) / det;
  if(v < 0 || u + v > 1)
    return false;

  float t = glm::dot(e2,Q) / det;
  if(t < .0001)
    return false;
  ix.t = t;

  glm::vec3 n1 = nrms[3*i], n2 = nrms[3*i+1], n3 = nrms[3*i+2];
  //if(glm::any(glm::isnan(n1+n2+n3))) {
  // ix.normal = glm::normalize(N);
  //} else {
  ix.normal = glm::normalize(n1 + (n2-n1)*u + (n3-n1)*v);
  //}

  ix.pos = pIx;
  return true;
}

bool KDNode::hit(const Ray& ray, Intersection& ix) {
  Intersection ix_cur;

  if(!boxIx(ray))
    return false;

  bool found = false;
  if(tris.size() > 0) {
    for(int i = 0; i < tris.size()/3; i++) {
      triIx(ray, i, ix_cur);
      if(ix_cur.t > .001 && (ix.t < 0|| ix_cur.t < ix.t)) {
        found = true;
        ix = ix_cur;
      }
    }
    if(found)
      return true;
  }

  Intersection ixL,ixR;
  if(left)
    left->hit(ray,ixL);
  if(right)
    right->hit(ray,ixR);

  if(ixL.t > .001 && (ix.t < 0 || ixL.t < ix.t)) {
    ix = ixL;
    return true;
  }
  if(ixR.t > .001 && (ix.t < 0 || ixR.t < ix.t)) {
    ix = ixR;
    return true;
  }

  return false;
}


