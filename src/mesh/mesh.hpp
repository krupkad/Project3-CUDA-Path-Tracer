#ifndef MESH_H
#define MESH_H

#include <src/mesh/object.hpp>
#include <src/kdtree.hpp>

class Mesh : public Object {
  public:
    Mesh(std::string objFile);
    virtual ~Mesh();

    void loadObj(std::string objFile);
    virtual glm::vec3 sample();
    virtual ObjDesc *getDesc();
  protected:
    bool triIx(const Ray& ray, unsigned int i,Intersection& ix);
    virtual bool doIntersect(const Ray& ray, Intersection& ix);
    bool boxIx(Ray ray, const glm::mat4& T);
    bool sphereIx(const Ray& ray);
    glm::vec3 vMin,vMax;
    KDNode *tree;
    float totArea;
    std::vector<float> areas;
};

#endif /* MESH_H */
