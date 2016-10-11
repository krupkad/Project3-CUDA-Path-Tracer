#include <src/mesh/cylinder.hpp>
#include <src/util.hpp>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/random.hpp>

#include <vector>
#include <GL/glew.h>
#include <iostream>

#ifndef PI
#define PI 3.14169265358979f
#endif


// Creates a unit cylinder centered at (0, 0, 0)
Cylinder::Cylinder()
: radius_(0.5f), height_(1.0f)
{
  buildObject();
  objDesc.type = OBJ_CYLINDER;
  objDesc.bbMin = glm::vec3(-radius_, -radius_, -.5*height_);
  objDesc.bbMax = glm::vec3(radius_, radius_, .5*height_);
}

Cylinder::~Cylinder() {
    vtx.clear();
    nrm.clear();
    idx.clear();
}

ObjDesc *Cylinder::getDesc() {
  return &objDesc;
}

void Cylinder::buildObject()
{
    vtx.clear();
    nrm.clear();
    idx.clear();

    unsigned short subdiv = 20;
    float dtheta = 2.f * PI / subdiv;

    glm::vec4 point_top(0.0f, 0.5f * height_, radius_, 1.0f),
        point_bottom (0.0f, -0.5f * height_, radius_, 1.0f);
    std::vector<glm::vec3> cap_top, cap_bottom;

    // top and bottom cap vertices
    for (int i = 0; i < subdiv + 1; ++i) {
        glm::mat4 rotate = glm::rotate(glm::mat4(1.0f), i * dtheta, glm::vec3(0.f, 1.f, 0.f));

        cap_top.push_back(glm::vec3(rotate * point_top));
        cap_bottom.push_back(glm::vec3(rotate * point_bottom));
    }

    //Create top cap.
    for ( int i = 0; i < subdiv - 2; i++) {
        vtx.push_back(cap_top[0]);
        vtx.push_back(cap_top[i + 1]);
        vtx.push_back(cap_top[i + 2]);
    }
    //Create bottom cap.
    for (int i = 0; i < subdiv - 2; i++) {
        vtx.push_back(cap_bottom[0]);
        vtx.push_back(cap_bottom[i + 1]);
        vtx.push_back(cap_bottom[i + 2]);
    }
    //Create barrel
    for (int i = 0; i < subdiv; i++) {
        //Right-side up triangle
        vtx.push_back(cap_top[i]);
        vtx.push_back(cap_bottom[i + 1]);
        vtx.push_back(cap_bottom[i]);
        //Upside-down triangle
        vtx.push_back(cap_top[i]);
        vtx.push_back(cap_top[i + 1]);
        vtx.push_back(cap_bottom[i + 1]);
    }

    // create normals
    glm::vec3 top_centerpoint(0.0f , 0.5f * height_ , 0.0f),
        bottom_centerpoint(0.0f, -0.5f * height_, 0.0f);
    glm::vec3 normal(0, 1, 0);

    // Create top cap.
    for (int i = 0; i < subdiv - 2; i++) {
        nrm.push_back(normal);
        nrm.push_back(normal);
        nrm.push_back(normal);
    }
    // Create bottom cap.
    for (int i = 0; i < subdiv - 2; i++) {
        nrm.push_back(-normal);
        nrm.push_back(-normal);
        nrm.push_back(-normal);
    }

    // Create barrel
    for (int i = 0; i < subdiv; i++) {
        //Right-side up triangle
        nrm.push_back(glm::normalize(cap_top[i] - top_centerpoint));
        nrm.push_back(glm::normalize(cap_bottom[i + 1] - bottom_centerpoint));
        nrm.push_back(glm::normalize(cap_bottom[i] - bottom_centerpoint));
        //Upside-down triangle
        nrm.push_back(glm::normalize(cap_top[i] - top_centerpoint));
        nrm.push_back(glm::normalize(cap_top[i + 1] - top_centerpoint));
        nrm.push_back(glm::normalize(cap_bottom[i + 1] - bottom_centerpoint));
    }

    for (unsigned int i = 0; i < vtx.size(); ++i) {
        idx.push_back(i);
    }

    tex.resize(vtx.size(),glm::vec2());
    uploadVBO();
}


bool Cylinder::doIntersect(const Ray& ray,Intersection& ix) {
  // dot(x0,x0) - r^2 + 2t dot(x0,dx) + dot(dx,dx) t^2 = 0
  // (x-xc)^2 + (y-xc)^2 = r^2
  // 2(x-xc),2(y-yc),0

  glm::vec3 dx = ray.dir, x0 = ray.p0;

  ix.t = -1;

  float zmin = -.5*height_;
  float zmax = .5*height_;

  // find the t's that intersect sides
  float a = dx.x*dx.x + dx.z*dx.z;
  float b = 2.f * (x0.x*dx.x + x0.z*dx.z);
  float c = x0.x*x0.x + x0.z*x0.z - radius_*radius_;
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
  if(t3 > 0.0001 && t3 < t && x3.z*x3.z + x3.x*x3.x < radius_*radius_) {
    ix.normal = glm::vec3(0,zmin,0);
    ix.t = t3;
    t = t3;
  }
  if(t4 > 0.0001 && t4 < t && x4.z*x4.z + x4.x*x4.x < radius_*radius_) {
    ix.normal = glm::vec3(0,zmax,0);
    ix.t = t4;
    t = t4;
  }
  ix.pos = ix.t*dx + x0;

  return true;
}

glm::vec3 Cylinder::sample() {
  int i = rand() % 3;
  glm::vec2 cPos = glm::diskRand(radius_);
  switch(i) {
    case 0:
      return glm::vec3(cPos.x, cPos.y, .5*height_);
    case 1:
      return glm::vec3(cPos.x, cPos.y, -.5*height_);
    case 2:
      return glm::vec3(cPos.x, cPos.y, float(rand())*.5f*height_/RAND_MAX);
    default:
      return glm::vec3();
  }
}
