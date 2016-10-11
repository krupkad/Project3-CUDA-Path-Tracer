#ifndef SPHERE_H
#define SPHERE_H

#include <src/mesh/object.hpp>

class Sphere : public Object {
  public:
    Sphere();
    virtual ~Sphere();

    virtual glm::vec3 sample();
    virtual ObjDesc *getDesc();

  private:
    virtual bool doIntersect(const Ray& ray,Intersection &ix);
    void buildObject();
    glm::vec3 center_;
    float radius_;
};

#endif
