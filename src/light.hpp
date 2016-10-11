#ifndef LIGHT_HPP
#define LIGHT_HPP

#include <src/mesh/object.hpp>
#include <src/node.hpp>

class Light {
  public:
    virtual glm::vec3 color() const = 0;
    virtual glm::vec3 pos() const = 0;
};

class PointLight : public Light {
  public:
    PointLight(const LightConf& conf) : conf(conf) {}

    virtual glm::vec3 color() const {
      return conf.color;
    }

    virtual glm::vec3 pos() const {
      return conf.pos;
    }

  protected:
    LightConf conf;
};

class SolidLight : public Light {
  public:
     SolidLight(const Node* node) : node(node) {}


    virtual glm::vec3 color() const {
      Intersection ix;
      node->sample(ix);
      return ix.mat->diffCol;
    }

    virtual glm::vec3 pos() const {
      Intersection ix;
      node->sample(ix);
      return ix.pos;
    }

  protected:
    const Node* node;
};

#endif /* LIGHT_HPP */



