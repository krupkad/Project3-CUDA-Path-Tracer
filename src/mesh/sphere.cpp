#include <src/mesh/sphere.hpp>
#include <src/util.hpp>
#include <glm/gtc/random.hpp>

#include <vector>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/intersect.hpp>

#ifndef PI
#define PI 3.141592653589f
#endif

// Creates a unit sphere.
Sphere::Sphere() :
    center_(0.f, 0.f, 0.f),
    radius_(1.f)
{
    buildObject();
    objDesc.type = OBJ_SPHERE;
    objDesc.bbMin = glm::vec3(-radius_, -radius_, -radius_);
    objDesc.bbMax = glm::vec3(radius_, radius_, radius_);
}

Sphere::~Sphere() {
    vtx.clear();
    nrm.clear();
    idx.clear();
}

ObjDesc *Sphere::getDesc() {
  return &objDesc;
}

void Sphere::buildObject()
{
    vtx.clear();
    nrm.clear();
    idx.clear();

    // Find vertex positions for the sphere.
    unsigned int subdiv_axis = 16;      // vertical slices
    unsigned int subdiv_height = 16;        // horizontal slices
    float dphi = PI / subdiv_height;
    float dtheta = 2.0f * PI / subdiv_axis;
    float epsilon = 0.0001f;
    glm::vec3 color (0.6f, 0.6f, 0.6f);

    // North pole
    glm::vec3 point (0.0f, 1.0f, 0.0f);
    nrm.push_back(point);
    // scale by radius_ and translate by center_
    vtx.push_back(center_ + radius_ * point);

    for (float phi = dphi; phi < PI; phi += dphi) {
        for (float theta = dtheta; theta <= 2.0f * PI + epsilon; theta += dtheta) {
            float sin_phi = sin(phi);

            point[0] = sin_phi * sin(theta);
            point[1] = cos(phi);
            point[2] = sin_phi * cos(theta);

            nrm.push_back(point);
            vtx.push_back(center_ + radius_ * point);
        }
    }
    // South pole
    point = glm::vec3(0.0f, -1.0f, 0.0f);
    nrm.push_back(point);
    vtx.push_back(center_ + radius_ * point);

    // fill in index array.
    // top cap
    for (unsigned int i = 0; i < subdiv_axis - 1; ++i) {
        idx.push_back(0);
        idx.push_back(i + 1);
        idx.push_back(i + 2);
    }
    idx.push_back(0);
    idx.push_back(subdiv_axis);
    idx.push_back(1);

    // middle subdivs
    unsigned int index = 1;
    for (unsigned int i = 0; i < subdiv_height - 2; i++) {
        for (unsigned int j = 0; j < subdiv_axis - 1; j++) {
            // first triangle
            idx.push_back(index);
            idx.push_back(index + subdiv_axis);
            idx.push_back(index + subdiv_axis + 1);

            // second triangle
            idx.push_back(index);
            idx.push_back(index + subdiv_axis + 1);
            idx.push_back(index + 1);

            index++;
        }
        // reuse vertices from start and end point of subdiv_axis slice
        idx.push_back(index);
        idx.push_back(index + subdiv_axis);
        idx.push_back(index + 1);

        idx.push_back(index);
        idx.push_back(index + 1);
        idx.push_back(index + 1 - subdiv_axis);

        index++;
    }

    // end cap
    unsigned int bottom = (subdiv_height - 1) * subdiv_axis + 1;
    unsigned int offset = bottom - subdiv_axis;
    for (unsigned int i = 0; i < subdiv_axis - 1 ; ++i) {
        idx.push_back(bottom);
        idx.push_back(i + offset);
        idx.push_back(i + offset + 1);
    }
    idx.push_back(bottom);
    idx.push_back(bottom - 1);
    idx.push_back(offset);

    tex.resize(vtx.size(),glm::vec2());
    uploadVBO();
}

bool Sphere::doIntersect(const Ray& ray, Intersection& ix) {
  // dot(x0,x0) - r^2 + 2t dot(x0,dx) + dot(dx,dx) t^2 = 0
  // (x+tdx)^2 + (y+tdy)^2 + (z+tdz)^2 = r^2
  // x^2 + y^2 + z^2 + 2t(xdx+ydy+zdz) + t^2 (dx^2 + dy^2 + dz^2) - r^2

  ix.t = -1;

  glm::vec3 dx = ray.dir, x0 = ray.p0;

  float a = glm::dot(dx,dx);
  if(a < 0.0001)
    return false;

  float b = 2.f * glm::dot(dx,x0);
  float c = glm::dot(x0,x0) - radius_*radius_;

  float d2 = b*b - 4*a*c;
  if(d2 < 0)
    return false;

  float d = sqrt(d2);
  float t1 = (-b - d)/(2*a);
  float t2 = (-b + d)/(2*a);
  if(t1 > 0.001)
    ix.t = t1;
  else if(t2 > 0.001)
    ix.t = t2;

  // n = 2(x-xc),2(y-yc),2(z-zc) = 2x
  ix.normal = ix.t*dx + x0;
  ix.pos = ix.t*dx + x0;

  printf("sphere: %f %f %f\n", ix.normal.x, ix.normal.y, ix.normal.z);

  return true;
}


glm::vec3 Sphere::sample() {
  return glm::sphericalRand(radius_);
}
