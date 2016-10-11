#include <src/node.hpp>
#include <src/scene.hpp>

#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <string>
#include <iostream>

Node::Node(std::string name) :
  tx(0), rot(0), sc(1), parent(NULL),
  ctr(0),
  geometry(NULL),
  objDesc(NULL),
  mat(NULL),
  name(name),
  tex(NULL),
  bbCull(true)
{
  updateLocalModel();
}

Node::Node(std::string name,Object *m) :
  tx(0), rot(0), sc(1), parent(NULL),
  ctr(0),
  geometry(m),
  objDesc(NULL),
  mat(NULL),
  name(name),
  tex(NULL),
  bbCull(true)
{
  updateLocalModel();
}

Node::Node(const NodeConf& conf) :
  tx(conf.tx), rot(conf.rot), sc(conf.scale), parent(NULL),
  ctr(conf.ctr),
  geometry(NULL),
  objDesc(NULL),
  mat(NULL),
  name(conf.name),
  tex(NULL),
  bbCull(conf.bbCull)
{
  updateLocalModel();
}

Node::Node(const NodeConf& conf, Object *m) :
  tx(conf.tx), rot(conf.rot), sc(conf.scale), parent(NULL),
  ctr(conf.ctr),
  geometry(m),
  objDesc(NULL),
  mat(NULL),
  name(conf.name),
  tex(NULL),
  bbCull(conf.bbCull)
{
  updateLocalModel();
}

Node::~Node() {
  if(parent)
    parent->delChild(this);
  for(unsigned int i = 0; i < children.size(); i++)
    delete children[i];
  if (objDesc)
    delete objDesc;
}

void Node::center(const glm::vec3& v) {
  ctr = v;
}

void Node::updateLocalModel() {
  glm::mat4 ctrMat = glm::translate(-ctr);
  glm::mat4 ctrMatInv = glm::translate(ctr);

  glm::mat4 txMat = glm::translate(tx);

  glm::quat rotX = glm::quat(rot.x*glm::vec3(1,0,0));
  glm::quat rotY = glm::quat(rot.y*glm::vec3(0,1,0));
  glm::quat rotZ = glm::quat(rot.z*glm::vec3(0,0,1));
  glm::mat4 rotMat = glm::mat4_cast(rotZ*rotY*rotX);

  glm::mat4 scMat = glm::scale(sc);

  localModel = ctrMatInv*txMat*rotMat*scMat*ctrMat;
  if (parent)
    localModel = parent->localModel*localModel;
  localModelInv = glm::inverse(localModel);

  if (geometry) {
    if (!objDesc) {
      objDesc = new ObjDesc;
      *objDesc = *geometry->getDesc();
    }
    objDesc->localModel = localModel;
    objDesc->localModelInv = localModelInv;
    objDesc->localModelInvTr = glm::transpose(localModelInv);
    objDesc->bbCull = bbCull;
  }

  for(unsigned int i = 0; i < children.size(); i++)
    children[i]->updateLocalModel();
}

glm::mat4 Node::getLocalModel() const {
  return localModel;
}

void Node::draw(Shader *shader) {
  if(shaders.count(shader) && geometry) {
    // Set the 4x4 model transformation matrices
    // Also upload the inverse transpose for normal transformation
    shader->setUniform("u_Model", localModel);
    shader->setUniform("u_Color", mat->diffCol);
    glm::mat4 modelInvTr = glm::transpose(localModelInv);
    shader->setUniform("u_ModelInvTr", modelInvTr);

    // bind our texture
    if(tex) {
      shader->setUniform("u_hasTexture", true);
      shader->bind(tex);
    } else {
      shader->setUniform("u_hasTexture", false);
    }

    // draw our geometry
    shader->bind(geometry);
    shader->draw(geometry);
  }

  for(unsigned int i = 0; i < children.size(); i++)
    children[i]->draw(shader);
}


void Node::addShader(Shader *s) {
  shaders.insert(s);
}

void Node::delShader(Shader *s) {
  shaders.erase(s);
}

void Node::buildPreOrder(std::deque<std::string>& vec) {
  if(geometry)
    vec.push_back(name);
  for(unsigned int i = 0; i < children.size(); i++)
    children[i]->buildPreOrder(vec);
}

void Node::buildPreOrder(std::vector<Node*>& vec) {
  if(geometry)
    vec.push_back(this);
  for(unsigned int i = 0; i < children.size(); i++)
    children[i]->buildPreOrder(vec);
}

void Node::translate(glm::vec3 dv) {
  tx += dv;
  updateLocalModel();
}
void Node::rotate(glm::vec3 dv) {
  rot += dv;
  updateLocalModel();
}
void Node::scale(glm::vec3 dv) {
  sc += dv;
  updateLocalModel();
}

void Node::addChild(Node *n) {
  children.push_back(n);
  n->parent = this;
  n->updateLocalModel();
}

void Node::delChild(Node *n) {
  unsigned int i;
  for(i = 0; i < children.size(); i++) {
    if(children[i] == n)
      break;
  }

  if(i < children.size())
    children.erase(children.begin()+i);
}

void Node::bindTexture(Texture *t) {
  if(tex)
    unbindTexture();
  tex = t;
}


void Node::bindMaterial(MatConf *m) {
  mat = m;
  if (objDesc)
    objDesc->mat = m;
}

void Node::unbindTexture() {
  tex = NULL;
}

Intersection Node::raytrace(const Ray& ray) {
  Intersection ix;
  raytrace(ray, ix);
  if(ix.t > 0)
    ix.normal = glm::normalize(ix.normal);
  if(glm::dot(ray.dir,ix.normal) > 0)
    ix.normal *= -1;
  return ix;
}

bool Node::raytrace(const Ray& ray, Intersection& best) {
  if(geometry) {
    objIntersect(*objDesc, ray, best);
    best.mat = mat;
  } else {
    best.mat = NULL;
  }

  Intersection ix;
  for(unsigned int i = 0; i < children.size(); i++) {
    children[i]->raytrace(ray, ix);
    if(ix.t > .001 && (best.t < 0 || ix.t < best.t))
      best = ix;
  }

  return best.t > .001;
}

void Node::sample(Intersection& ix) const {
  if(geometry)
    ix.pos = glm::vec3(localModel*glm::vec4(geometry->sample(),1));
  else
    ix.pos = glm::vec3(localModel*glm::vec4(tx,1));
  ix.mat = mat;
}

ObjDesc *Node::getDesc() const {
  return objDesc;
}
