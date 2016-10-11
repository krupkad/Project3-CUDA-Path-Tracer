#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>

#include <src/mesh/object.hpp>
#include <src/util.hpp>


Object::Object() : data(NULL) {
  glGenBuffers(1, &vboData);
  glGenBuffers(1, &vboIdx);
}

Object::~Object() {
  if(data)
    delete data;
  glDeleteBuffers(1, &vboData);
  glDeleteBuffers(1, &vboIdx);
}

void Object::bind(Shader *shader) {
  shader->bindVertexData("vs_Position", vboData, sso_vtx);
  shader->bindVertexData("vs_Normal", vboData, sso_nrm);
  shader->bindVertexData("vs_UV", vboData, sso_tex);
}

void Object::unbind(Shader *shader) {
  shader->unbindVertexData("vs_Position");
  shader->unbindVertexData("vs_Normal");
  shader->unbindVertexData("vs_UV");
}

void Object::draw(Shader* shader) {
  // define our tris and draw
  shader->bindIndexData(vboIdx);
  glDrawElements(GL_TRIANGLES, idx.size(), GL_UNSIGNED_INT, 0);
}

void Object::uploadVBO() {
  // Sizes of the various array elements below.
  int tri_size = idx.size() * sizeof(unsigned int);
  int count = vtx.size();
  int stride = 3+3+2;

  sso_vtx = SHADER_SSO(3,stride,0);
  sso_nrm = SHADER_SSO(3,stride,3);
  sso_tex = SHADER_SSO(2,stride,6);

  tex.resize(vtx.size(), glm::vec2());
  nrm.resize(vtx.size(), glm::vec3());

  data = new float[stride*count];
  for(int i = 0; i < count; i++) {
    memcpy(&data[stride*i], glm::value_ptr(vtx[i]), 3*sizeof(float));
    memcpy(&data[stride*i + 3], glm::value_ptr(nrm[i]), 3*sizeof(float));
    memcpy(&data[stride*i + 6], glm::value_ptr(tex[i]), 2*sizeof(float));
  }

  int size = stride*count*sizeof(float);
  glBindBuffer(GL_ARRAY_BUFFER, vboData);
  glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);

  // Bind+upload the indices to the GL_ELEMENT_ARRAY_BUFFER.
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboIdx);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, tri_size, &idx[0], GL_STATIC_DRAW);
}

bool Object::intersect(const glm::mat4& locModel, const glm::mat4& locModelInv,
                        const Ray& ray, Intersection& ix) {
  glm::vec3 dx(locModelInv * glm::vec4(ray.dir,0));
  glm::vec3 x0(locModelInv * glm::vec4(ray.p0,1));
  Ray locRay(x0,dx,ray.iter,ray.outside);

  bool found = doIntersect(locRay,ix);

  if(found) {
    glm::vec4 nrm4 = glm::vec4(glm::normalize(ix.normal),0);
    glm::mat4 locModelInvTr = glm::transpose(locModelInv);
    ix.normal = glm::vec3(locModelInvTr*nrm4);
    ix.pos = glm::vec3(locModel * glm::vec4(ix.pos,1));
  }

  return found;
}
