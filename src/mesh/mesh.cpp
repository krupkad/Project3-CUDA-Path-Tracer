#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtx/vector_angle.hpp>

#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <sstream>

#include <src/mesh/mesh.hpp>
#include <src/util.hpp>

#include <cuda.h>

// read a vertex from an "f" line
static glm::ivec3 readFace(std::string str) {
  glm::ivec3 arr(-1,-1,-1);
  std::stringstream ss;

  int aidx = 0,tmp;
  for(unsigned int i = 0; i < str.size(); i++) {
    tmp = 0;
    if(str[i] == '/') {
      ss >> tmp;
      arr[aidx++] = tmp-1;
      ss.clear();
    } else {
      ss << str[i];
    }
  }
  if(aidx < 3) {
    ss >> tmp;
    arr[aidx++] = tmp-1;
  }
  return arr;
}

void Mesh::loadObj(std::string objFile) {
  // clear old data
  nrm.clear();
  vtx.clear();
  idx.clear();
  tex.clear();

  // open obj file
  std::ifstream in(objFile.c_str());
  if(!in) {
    std::cerr << "couldn't open obj " << objFile << "\n";
    return;
  }

  // obj data vectors
  std::vector<glm::vec2> objTex;
  std::vector<glm::vec3> objNrm, objVtx;
  std::vector<int> vtxIdx, nrmIdx, texIdx;

  // load the data
  std::string param;
  while(in >> param) {
    if(param == "#") {
      std::getline(in, param);
    }
    if(param == "v") {
      glm::vec3 v;
      in >> v;
      objVtx.push_back(v);
    }
    if(param == "vn") {
      glm::vec3 n;
      in >> n;
      objNrm.push_back(n);
    }
    if(param == "vt") {
      glm::vec2 t;
      in >> t;
      objTex.push_back(t);
    }
    if(param == "f") {
      std::string vStr;
      glm::ivec3 vVec;
      for(int i = 0; i < 3; i++) {
        in >> vStr;
        vVec = readFace(vStr);
        vtxIdx.push_back(vVec[0]);
        texIdx.push_back(vVec[1]);
        nrmIdx.push_back(vVec[2]);
      }
    }
  }

  std::vector<float> weight(objVtx.size());
  std::vector<glm::vec3> wNrm(objVtx.size());
  vMin = glm::vec3(FLT_MAX);
  vMax = glm::vec3(FLT_MIN);
  for(unsigned int i = 0; i < vtxIdx.size(); i++) {
    idx.push_back(i);

    int vIdx = vtxIdx[i];
    if(vIdx < 0 || vIdx >= (int)objVtx.size()) {
      std::cout << "obj: invalid vertex index " << vIdx << "\n";
      return;
    }
    vtx.push_back(objVtx[vIdx]);

    for(int d = 0; d < 3; d++) {
      if(objVtx[vIdx][d] > vMax[d])
        vMax[d] = objVtx[vIdx][d];
      if(objVtx[vIdx][d] < vMin[d])
        vMin[d] = objVtx[vIdx][d];
    }

    int tIdx = texIdx[i];
    if(tIdx >= (int)objTex.size() || (tIdx < 0 && objTex.size() > 0)) {
      std::cout << "obj: invalid texture index " << tIdx << " " << objTex.size() << "\n";
      return;
    }
    if(objTex.size() > 0)
      tex.push_back(objTex[tIdx]);

    int nIdx = nrmIdx[i];
    if(nIdx >= (int)objNrm.size()) {
      std::cout << "obj: invalid normal index" << "\n";
      return;
    }

    if(i % 3 == 2) {
      glm::vec3 crs = glm::cross(vtx[i-2]-vtx[i-1],vtx[i-1]-vtx[i]);
      float area = glm::length(crs)/2;
      areas.push_back(area);
      crs /= (2.f*area);

      glm::vec3 e01 = glm::normalize(vtx[i-2]-vtx[i-1]);
      glm::vec3 e12 = glm::normalize(vtx[i-1]-vtx[i]);
      glm::vec3 e20 = glm::normalize(vtx[i]-vtx[i-2]);
      float a0 = glm::angle(e01,e20);
      float a1 = glm::angle(e12,e01);
      float a2 = glm::angle(e12,e20);
      glm::vec3 angle(a0,a1,a2);

      glm::vec3 res;
      if(nIdx < 0) {
        objNrm.push_back(crs);
        res = crs;
      }
      else
        res = objNrm[nIdx];
      for(int j = 0; j < 3; j++) {
        float w = angle[j]*area;
        int k = vtxIdx[i-2+j];
        weight[k] += w;
        wNrm[k] += res*w;
      }
    }
  }

  for(unsigned int i = 0; i < vtxIdx.size(); i++) {
    int vIdx = vtxIdx[i];
    nrm.push_back(glm::normalize(wNrm[vIdx] / weight[vIdx]));
  }

  uploadVBO();
}

Mesh::Mesh(std::string objFile) {
  loadObj(objFile);
  objDesc.type = OBJ_MESH;
  objDesc.bbMin = vMin;
  objDesc.bbMax = vMax;

  int N = idx.size();
  cudaMalloc((void**)&objDesc.meshVtx, N*sizeof(glm::vec3));
  cudaMalloc((void**)&objDesc.meshNrm, N*sizeof(glm::vec3));
  int j = 0;
  for (int i : idx) {
    cudaMemcpy(&objDesc.meshVtx[j], &vtx[i], sizeof(glm::vec3), cudaMemcpyHostToDevice);
    cudaMemcpy(&objDesc.meshNrm[j], &nrm[i], sizeof(glm::vec3), cudaMemcpyHostToDevice);
    j++;
  }
  objDesc.meshTriCnt = N/3;
}

Mesh::~Mesh() {
  delete tree;
  cudaFree(objDesc.meshVtx);
  cudaFree(objDesc.meshNrm);
}

ObjDesc *Mesh::getDesc() {
  return &objDesc;
}

bool Mesh::triIx(const Ray& ray, unsigned int i, Intersection &ix) {
  unsigned int ti0 = idx[3*i], ti1 = idx[3*i+1], ti2 = idx[3*i+2];
  glm::vec3 p1 = vtx[ti0], p2 = vtx[ti1], p3 = vtx[ti2];

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

  glm::vec3 n1 = nrm[ti0], n2 = nrm[ti1], n3 = nrm[ti2];

  ix.normal = n1 + (n2-n1)*u + (n3-n1)*v;
  ix.pos = p1 + (p2-p1)*u + (p3-p1)*v;
  ix.normal = N;
  ix.pos = pIx;
  return true;
}

bool Mesh::sphereIx(const Ray& ray) {
  // dot(x0,x0) - r^2 + 2t dot(x0,dx) + dot(dx,dx) t^2 = 0
  // (x+tdx)^2 + (y+tdy)^2 + (z+tdz)^2 = r^2
  // x^2 + y^2 + z^2 + 2t(xdx+ydy+zdz) + t^2 (dx^2 + dy^2 + dz^2) - r^2

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

  return true;
}

bool Mesh::doIntersect(const Ray& ray,Intersection &ix) {
  /*
  Intersection cur;

  for(int i = 0; i < idx.size()/3; i++) {
    triIx(ray, i,cur);
    if(cur.t > .001 && (ix.t < 0 || cur.t < ix.t))
      ix = cur;
  }
  */
  //if(!sphereIx(ray))
  //  return false;
  return tree->hit(ray,ix);
}

glm::vec3 Mesh::sample() {
  int N = idx.size()/3;
  float p0 = float(rand())*totArea/RAND_MAX, p = 0.0;
  int i = 0;
  for(i = 0; i < N; i++) {
    p += areas[i];
    if(p > p0)
      break;
  }

  float u = float(rand())/RAND_MAX, v = float(rand())*(1.0-u)/RAND_MAX;
  unsigned int ti0 = idx[3*i], ti1 = idx[3*i+1], ti2 = idx[3*i+2];
  glm::vec3 p1 = vtx[ti0], p2 = vtx[ti1], p3 = vtx[ti2];
  return p1 + u*(p2-p1) + v*(p3-p1);
}

