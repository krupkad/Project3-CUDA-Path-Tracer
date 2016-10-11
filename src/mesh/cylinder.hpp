#ifndef CYLINDER_H
#define CYLINDER_H

#include <src/mesh/object.hpp>

class Cylinder : public Object {
  public:
    Cylinder();
    virtual ~Cylinder();
    virtual glm::vec3 sample();
    virtual ObjDesc *getDesc();

  private:
    virtual  bool doIntersect(const Ray& ray,Intersection& ix);
    void buildObject();
    float radius_;
    float height_;
};

#endif
