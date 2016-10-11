
#include <src/node.hpp>
#include <src/scene.hpp>
#include <src/mesh/object.hpp>
#include <src/mesh/cube.hpp>
#include <src/mesh/sphere.hpp>
#include <src/mesh/cylinder.hpp>
#include <src/mesh/mesh.hpp>
#include <src/shader.hpp>
#include <src/camera.hpp>
#include <src/texture.hpp>
#include <src/light.hpp>

#include <cstdio>

#include <map>
#include <string>
#include <deque>
#include <set>


class ContourShader : public Shader {
  public:
    ContourShader() : Shader() {}
    ContourShader(std::string pfx) : Shader(pfx) {}
    ContourShader(std::string vert, std::string frag) : Shader(vert,frag) {}

    virtual void draw(Object *obj) {
      use();

      // save context and set our stencil on the diffuse pass
      glPushAttrib(GL_ALL_ATTRIB_BITS);
      glClearStencil(0);
      glClear(GL_STENCIL_BUFFER_BIT);
      glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
      glEnable(GL_STENCIL_TEST);
      glStencilFunc(GL_ALWAYS, 1, -1);
      glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
      obj->draw(this);

      // use our stencil on the contour pass and restore context
      glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
      glStencilFunc(GL_NOTEQUAL, 1, -1);
      glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
      glLineWidth(2);
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
      glEnable(GL_LINE_SMOOTH);
      glColor3f(.75,.75,.75);
      obj->draw(this);
      glPopAttrib();
    }
};

Scene::Scene(const Config& conf) {
  nodes["null"] = new Node("null");

  geoms["cube"] = new Cube();
  geoms["sphere"] = new Sphere();
  geoms["null"] = NULL;
  geoms["cylinder"] = new Cylinder();

  shaders["diffuse"] = new Shader("shaders/diff");
  shaders["contour"] = new ContourShader("shaders/cntr");

  textures["null"] = NULL;

  materials["default"] = new MatConf();

  camera = new Camera(conf.camera);

  for(unsigned int i = 0; i < conf.shaders.size(); i++)
    addShader(conf.shaders[i]);

  for(unsigned int i = 0; i < conf.textures.size(); i++)
    addTexture(conf.textures[i].name, conf.textures[i].fName);

  for(unsigned int i = 0; i < conf.materials.size(); i++)
    addMaterial(conf.materials[i]);

  for(unsigned int i = 0; i < conf.nodes.size(); i++)
    addNode(conf.nodes[i]);

  for(unsigned int i = 0; i < conf.lights.size(); i++)
    lights.push_back(new PointLight(conf.lights[i]));

  buildPreOrder();
}

Scene::~Scene() {
  std::map<std::string,Object*>::iterator objItr;
  for(objItr = geoms.begin(); objItr != geoms.end(); objItr++) {
    if(objItr->second)
      delete objItr->second;
  }

  std::map<std::string,Shader*>::iterator sItr;
  for(sItr = shaders.begin(); sItr != shaders.end(); sItr++) {
    if(sItr->second)
      delete sItr->second;
  }

  std::map<std::string,MatConf*>::iterator matItr;
  for(matItr = materials.begin(); matItr != materials.end(); matItr++) {
    if(matItr->second)
      delete matItr->second;
  }

  delete camera;
  delete nodes["null"];
}

void Scene::resize(int width, int height) {
  camera->resize(width,height);
}

void Scene::addMesh(std::string name, std::string objFile) {
  if(geoms[name])
    delete geoms[name];
  geoms[name] = new Mesh(objFile);
}

void Scene::addNode(const NodeConf& conf) {
  std::string name = conf.name;
  std::string parent = conf.parentName;
  std::string shape = conf.shape;

  Node *node;
  if(shape != "mesh") {
    node = new Node(conf, geoms[shape]);
  } else {
    geoms[name] = new Mesh(conf.objFile);
    node = new Node(conf, geoms[name]);
  }
  node->addShader(shaders["diffuse"]);
  node->bindTexture(textures[conf.texName]);
  node->bindMaterial(materials[conf.matName]);

  if(materials[conf.matName] && materials[conf.matName]->lEmit > 0.001)
    lights.push_back(new SolidLight(node));

  if(nodes[name])
    delete nodes[name];
  nodes[name] = node;
  nodes[parent]->addChild(node);
}

void Scene::delNode(std::string name) {
  if(nodes[name]) {
    delete nodes[name];
    nodes.erase(name);
  }

  buildPreOrder();
}

void Scene::addMaterial(const MatConf& m) {
  std::string name = m.name;
  if(materials[name])
    delete materials[name];
  materials[name] = new MatConf();
  *materials[name] = m;
}

void Scene::addShader(std::string name, std::string vSrc, std::string fSrc) {
  if(shaders[name])
    delete shaders[name];
  shaders[name] = new Shader(vSrc, fSrc);
}

void Scene::addShader(std::string name, std::string pfx) {
  if(shaders[name])
    delete shaders[name];
  shaders[name] = new Shader(pfx);
}

void Scene::addShader(const ShaderConf& conf) {
  if(shaders[conf.name])
    delete shaders[conf.name];
  shaders[conf.name] = new Shader(conf);
}

void Scene::delShader(std::string name) {
  if(shaders[name]) {
    delete shaders[name];
    shaders.erase(name);
  }
}

std::deque<std::string> Scene::nodeList() {
  return preOrder;
}

void Scene::draw(std::string sName) {
  Shader *shader = shaders[sName];
  if(!shader) {
    std::cerr << "error: invalid shader " << sName << "\n";
    return;
  }

  shader->bind(camera);

  for(unsigned int i = 0; i < lights.size(); i++) {
    shader->setUniform("u_LightPos", lights[i]->pos());
    shader->setUniform("u_LightColor", lights[i]->color());
  }
  nodes["null"]->draw(shader);
}

void Scene::buildPreOrder() {
  preOrder.clear();
  nodes["null"]->buildPreOrder(preOrder);
}

void Scene::bindShader(std::string nName, std::string sName) {
  Node *n = nodes[nName];
  Shader *s = shaders[sName];
  if(!n) {
    std::cout << "error: no such node " << nName << std::endl;
    return;
  }
  if(!s) {
    std::cout << "error: no such shader " << sName << std::endl;
    return;
  }

  n->addShader(s);
}

void Scene::unbindShader(std::string nName, std::string sName) {
  Node *n = nodes[nName];
  Shader *s = shaders[sName];
  if(!n) {
    std::cout << "error: no such node " << nName << std::endl;
    return;
  }
  if(!s) {
    std::cout << "error: no such shader " << sName << std::endl;
    return;
  }

  n->delShader(s);
}

void Scene::translate(std::string name, const glm::vec3& dv) {
  Node *n = nodes[name];
  if(!n) {
    std::cout << "error: no such node " << name << std::endl;
    return;
  }
  n->translate(dv);
}

void Scene::rotate(std::string name, const glm::vec3& dv) {
  Node *n = nodes[name];
  if(!n) {
    std::cout << "error: no such node " << name << std::endl;
    return;
  }
  n->rotate(dv);
}

void Scene::scale(std::string name, const glm::vec3& dv) {
  Node *n = nodes[name];
  if(!n) {
    std::cout << "error: no such node " << name << std::endl;
    return;
  }
  n->scale(dv);
}

void Scene::center(std::string name, const glm::vec3& v) {
  Node *n = nodes[name];
  if(!n) {
    std::cout << "error: no such node " << name << std::endl;
    return;
  }
  n->center(v);
}

void Scene::addTexture(std::string name, std::string img) {
  if(!textures[name])
    delete textures[name];
  textures[name] = new Texture(img);
}

void Scene::delTexture(std::string name) {
  if(textures[name]) {
    delete textures[name];
    textures.erase(name);
  }
}

void Scene::bindMaterial(std::string nName, std::string mName) {
  Node *n = nodes[nName];
  MatConf *m = materials[mName];
  if(!n) {
    std::cout << "error: no such node " << nName << std::endl;
    return;
  }
  if(!m) {
    std::cout << "error: no such material " << mName << std::endl;
    return;
  }

  n->bindMaterial(m);
}

void Scene::bindTexture(std::string nName, std::string tName) {
  Node *n = nodes[nName];
  Texture *t = textures[tName];
  if(!n) {
    std::cout << "error: no such node " << nName << std::endl;
    return;
  }
  if(!t) {
    std::cout << "error: no such texture " << tName << std::endl;
    return;
  }

  n->bindTexture(t);
}

void Scene::unbindTexture(std::string nName, std::string tName) {
  Node *n = nodes[nName];
  Texture *t = textures[tName];
  if(!n) {
    std::cout << "error: no such node " << nName << std::endl;
    return;
  }
  if(!t) {
    std::cout << "error: no such texture " << tName << std::endl;
    return;
  }

  n->unbindTexture();
}

void Scene::moveLight(glm::vec3 dv) {
  for(unsigned int i = 0; i < lights.size(); i++)
    lights[i]->pos() += dv;
}

void Scene::moveMouse(float xpos, float ypos) {
  camera->rotate(xpos, ypos, .1);
}

void Scene::raytrace() {
  //camera->raytrace(nodes["null"],lights);
  camera->pathtrace(nodes["null"]);
}

void Scene::zoom(float dy) {
  camera->zoom(dy);
}

