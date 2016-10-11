#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <vector>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>

#include <src/shader.hpp>
#include <src/conf.hpp>
#include <src/mesh/intersection.hpp>

enum ObjType {
  OBJ_CUBE,
  OBJ_SPHERE,
  OBJ_CYLINDER,
  OBJ_MESH,
  OBJ_NULL
};

struct ObjDesc {
  ObjType type;
  glm::vec3 bbMin, bbMax;
  glm::mat4 localModel, localModelInv, localModelInvTr;
  MatConf *mat;

  int meshTriCnt;
  glm::vec3 *meshVtx, *meshNrm;
  bool bbCull;
};

// A base class for objects in the scene graph.
class Object : public GLEntity {
  public:
    Object();
    virtual ~Object();

    virtual void draw(Shader* shader);
    virtual void bind(Shader *shader);
    virtual void unbind(Shader *shader);

    virtual ObjDesc *getDesc() = 0;

    virtual glm::vec3 sample() = 0;
    bool intersect(const glm::mat4& locModel, const glm::mat4& locModelInv,
                    const Ray& ray, Intersection& ix);

  protected:
    virtual bool doIntersect(const Ray& ray, Intersection &ix) = 0;
    void uploadVBO();

    std::vector<glm::vec3> vtx;        // vertex buffer
    std::vector<glm::vec3> nrm;         // normal buffer
    std::vector<glm::vec2> tex;
    std::vector<unsigned int> idx;      // index buffer
    float *data;

    unsigned int vboData, vboIdx;
    int sso_vtx,sso_nrm,sso_tex;

    ObjDesc objDesc;
};

#endif /* GEOMETRY_H */
