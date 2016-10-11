#ifndef CONF_H
#define CONF_H

#include <glm/glm.hpp>
#include <src/util.hpp>

#include <vector>
#include <string>
#include <iostream>

#ifndef PI
#define PI 3.14169265358979f
#endif


struct NodeConf {
  NodeConf();
  glm::vec3 scale, tx, rot, ctr;
  std::string name, parentName, shape, objFile, texName, matName;
  bool bbCull;
};

struct TextureConf {
  std::string fName;
  std::string name;
};

struct CameraConf {
  CameraConf();
  glm::vec3 pos, fwd, up;
  float yFov;
  int width, height;
  unsigned int density;
  std::string outFile;
  glm::vec3 ambient;
  unsigned int rayIter, lSamp, mcIter;
  bool doTrace, useMatSort, useIxCache;

  bool useDOF;
  float aperture;
};

struct LightConf {
  LightConf();
  glm::vec3 pos, color, dir;
  float angle;
  float radius;
  int samp;
};

struct ShaderConf {
  std::string vertFile;
  std::string fragFile;
  std::string name;
};

struct MatConf {
  MatConf();
  glm::vec3 diffCol, specCol;
  float specExp, ior, mirr, trans, lEmit;
  std::string name;
};

struct Config {
  public:
    Config();
    Config(std::istream& inFile);
    ~Config();

    std::vector<NodeConf> nodes;
    std::vector<LightConf> lights;
    std::vector<ShaderConf> shaders;
    std::vector<TextureConf> textures;
    std::vector<MatConf> materials;
    CameraConf camera;

    void config(std::istream& inFile);

  private:
    std::string config(std::istream& inFile, LightConf& conf);
    std::string config(std::istream& inFile, CameraConf& conf);
    std::string config(std::istream& inFile, NodeConf& conf);
    std::string config(std::istream& inFile, ShaderConf& conf);
    std::string config(std::istream& inFile, TextureConf& conf);
    std::string config(std::istream& inFile, MatConf& conf);
};

#endif /* CONF_H */
