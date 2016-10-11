#ifndef CAMERA_H
#define CAMERA_H

#include <src/util.hpp>
#include <src/conf.hpp>
#include <src/node.hpp>
#include <src/shader.hpp>
#include <src/mesh/object.hpp>
#include <src/light.hpp>

#include <glm/glm.hpp>
#include <string>
#include <iostream>
#include <vector>

struct CamDesc {
  int width, height, density, density2;
  glm::vec3 pos, xAxis, yAxis, fwd;
  int rayCount, mcIter;
  bool useDOF;
  float aperture;
};

class Camera : public GLEntity {
  public:
    Camera(const CameraConf& conf);
    virtual ~Camera() {}
    void resize(int w, int h);
    void rotate(float xpos, float ypos, float dt);
    void zoom(float dx);

    virtual void bind(Shader *shader);
    void raytrace(Node *node, const std::vector<Light*>& lights);
    void pathtrace(Node *root);

  private:
    glm::vec3 rayIter(Ray ray, Node *root);
    glm::vec4 rayIter(Ray ray, Node *root, const std::vector<Light*>& lights);
    glm::vec4 doShadow(Node *root, const Intersection& ix, const std::vector<Light*>& lights);
    float yFov;
    glm::vec3 pos, up, fwd;
    unsigned int width, height;
    unsigned int density;
    std::string outFile;
    glm::vec3 ambient;
    unsigned int rayCount, lSamp, mcIter;

    bool doTrace, useMatSort, useIxCache, useDOF;

    float aperture;

    float yvAngle, xvAngle;
};

#endif /* CAMERA_H */
