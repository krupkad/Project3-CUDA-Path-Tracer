#ifndef NODE_H
#define NODE_H

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>

#include <src/mesh/object.hpp>
#include <src/mesh/cube.hpp>
#include <src/mesh/cylinder.hpp>
#include <src/mesh/sphere.hpp>
#include <src/shader.hpp>
#include <src/conf.hpp>
#include <src/texture.hpp>

#include <string>
#include <iostream>
#include <deque>
#include <set>

class Scene;
class Node {
  public:
    Node(std::string name);
    Node(std::string name, Object *m);
    Node(const NodeConf& conf);
    Node(const NodeConf& conf, Object *m);
    ~Node();

    void addChild(Node *n);
    void delChild(Node *n);

    void addShader(Shader *s);
    void delShader(Shader *s);

    void bindTexture(Texture *tex);
    void unbindTexture();

    void bindMaterial(MatConf *mat);

    void draw(Shader *shader);
    void buildPreOrder(std::deque<std::string>& vec);
    void buildPreOrder(std::vector<Node*>& vec);

    void getObjDesc(std::vector<ObjDesc*> &objList);

    void translate(glm::vec3 dv);
    void rotate(glm::vec3 dv);
    void scale(glm::vec3 dv);
    void center(const glm::vec3& v);

    Intersection raytrace(const Ray& ray);

    void sample(Intersection& ix) const;

    ObjDesc *getDesc() const;

  protected:
    glm::mat4 getLocalModel() const;
    bool raytrace(const Ray& ray, Intersection& ix);
    void updateLocalModel();

    Node *parent;
    std::vector<Node*> children;
    Object *geometry;
    ObjDesc *objDesc;
    glm::vec3 tx,rot,sc,ctr;
    std::string name;

    std::set<Shader*> shaders;
    Texture *tex;
    MatConf *mat;

    bool bbCull;

    glm::mat4 localModel, localModelInv;
};

#endif /* NODE_H */
