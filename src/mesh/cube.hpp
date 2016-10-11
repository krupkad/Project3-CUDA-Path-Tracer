#ifndef CUBE_H
#define CUBE_H

#include <src/mesh/object.hpp>

class Cube : public Object {
  public:
    Cube();
    virtual ~Cube();

    virtual glm::vec3 sample();
    virtual ObjDesc *getDesc();

  protected:
    virtual bool doIntersect(const Ray& ray,Intersection& ix);
    void buildObject();

    float len;
};

#endif
