#ifndef SCENE_H
#define SCENE_H

#include <src/mesh/object.hpp>
#include <src/light.hpp>
#include <src/shader.hpp>
#include <src/camera.hpp>
#include <src/conf.hpp>
#include <src/texture.hpp>

#include <map>
#include <string>
#include <deque>

class Node;
class Scene {
  public:
    Scene(const Config& conf);
    ~Scene();

    void draw(std::string sName);

    void resize(int width, int height);

    void addMesh(std::string name, std::string objFile);

    void addNode(const NodeConf& conf);
    void delNode(std::string name);

    void addMaterial(const MatConf& conf);
    void bindMaterial(std::string node, std::string mat);

    void addTexture(std::string name, std::string img);
    void delTexture(std::string name);

    void bindTexture(std::string nodeName, std::string tName);
    void unbindTexture(std::string nodeName, std::string tName);

    void addShader(std::string name, std::string pfx);
    void addShader(std::string name, std::string vSrc, std::string fSrc);
    void addShader(const ShaderConf& conf);
    void delShader(std::string name);

    void bindShader(std::string nodeName, std::string sName);
    void unbindShader(std::string nodeName, std::string sName);

    void moveLight(glm::vec3 dv);
    void moveMouse(float xpos, float ypos);
    void zoom(float dx);

    void translate(std::string name, const glm::vec3& dv);
    void rotate(std::string name, const glm::vec3& dv);
    void scale(std::string name, const glm::vec3& dv);
    void center(std::string name, const glm::vec3& v);
    std::deque<std::string> nodeList();

    void raytrace();

  private:

    void buildPreOrder();

    std::deque<std::string> preOrder;
    std::map<std::string,Node*> nodes;
    std::map<std::string,Shader*> shaders;
    std::map<std::string,Object*> geoms;
    std::map<std::string,Texture*> textures;

    Camera *camera;

    std::vector<Light*> lights;
    std::map<std::string,MatConf*> materials;

};

#endif /* SCENE_H */
