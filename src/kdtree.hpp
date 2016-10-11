#ifndef KDNODE_H
#define KDNODE_H

#include <vector>
#include <glm/glm.hpp>
#include <src/mesh/object.hpp>

class KDNode {
  public:
    KDNode(const std::vector<unsigned int>& triIdx, const std::vector<glm::vec3>& vtx, const std::vector<glm::vec3>& nrm);
    ~KDNode();

    bool hit(const Ray& ray,Intersection& ix);

  private:
    void triMin(const glm::vec3& a, glm::vec3& min);
    void triMax(const glm::vec3& a, glm::vec3& max);
    bool triIx(const Ray& ray, unsigned int i,Intersection& ix);
    bool boxIx(const Ray& ray);
    bool sphereIx(const Ray& ray);

    glm::vec3 vMin, vMax;
    KDNode *left, *right;

    std::vector<glm::vec3> tris;
    std::vector<glm::vec3> nrms;

    unsigned int nTris;
};

#endif /* KDNODE_H */
