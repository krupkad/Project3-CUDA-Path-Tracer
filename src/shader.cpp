#include <src/shader.hpp>
#include <src/mesh/object.hpp>

#include <map>
#include <string>
#include <iostream>
#include <fstream>
#include <cerrno>
#include <stdexcept>

#include <GL/glew.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

static std::string textFileRead(std::string filename);
static void printLinkInfoLog(int prog);
static void printShaderInfoLog(int shader);

Shader* Shader::inUse = NULL;

Shader::Shader() : hasVert(false), hasFrag(false), hasProgram(false) {}
Shader::~Shader() {
  if(hasVert)
    glDeleteShader(shadVert);
  if(hasFrag)
    glDeleteShader(shadFrag);
  if(hasProgram)
    glDeleteProgram(program);
}

Shader::Shader(const ShaderConf& conf) :  hasVert(false), hasFrag(false), hasProgram(false)  {
  setVertShader(conf.vertFile);
  setFragShader(conf.fragFile);
  recompile();
}

Shader::Shader(std::string vSrc, std::string fSrc) : hasVert(false), hasFrag(false), hasProgram(false)  {
  setVertShader(vSrc + ".vert.glsl");
  setFragShader(fSrc + ".frag.glsl");
  recompile();
}

Shader::Shader(std::string src) : hasVert(false), hasFrag(false), hasProgram(false)  {
  setVertShader(src + ".vert.glsl");
  setFragShader(src + ".frag.glsl");
  recompile();
}

void Shader::setVertShader(std::string src) {
  std::string vertSourceS = textFileRead(src);
  const char *vertSource = vertSourceS.c_str();
  if(hasVert)
    glDeleteShader(shadVert);
  shadVert = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(shadVert, 1, &vertSource, NULL);
  glCompileShader(shadVert);
  printShaderInfoLog(shadVert);

  hasVert = true;
}

void Shader::setFragShader(std::string src) {
  std::string fragSourceS = textFileRead(src);
  const char *fragSource = fragSourceS.c_str();
  if(hasFrag)
    glDeleteShader(shadFrag);
  shadFrag = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(shadFrag, 1, &fragSource, NULL);
  glCompileShader(shadFrag);
  printShaderInfoLog(shadFrag);

  hasFrag = true;
}

void Shader::setUniform(std::string name, const glm::vec3& v) {
  int loc = resolveUniform(name);
  if(loc < 0)
    return;
  glUniform3fv(loc, 1, glm::value_ptr(v));
}

void Shader::setUniform(std::string name, float x, float y, float z) {
  int loc = resolveUniform(name);
  if(loc < 0)
    return;
  glUniform3f(loc, x, y, z);
}

void Shader::setUniform(std::string name, bool v) {
  int loc = resolveUniform(name);
  if(loc < 0)
    return;
  glUniform1i(loc, v);
}

void Shader::setUniform(std::string name, int v) {
  int loc = resolveUniform(name);
  if(loc < 0)
    return;
  glUniform1i(loc, v);
}

void Shader::setUniform(std::string name, const glm::mat4& m) {
  int loc = resolveUniform(name);
  if(loc < 0)
    return;
  glUniformMatrix4fv(loc, 1, GL_FALSE, &m[0][0]);
}

void Shader::bindVertexData(std::string name, unsigned int vbo) {
  int loc = resolveAttribute(name);
  if(loc < 0)
    return;
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glVertexAttribPointer(loc, 3, GL_FLOAT, false, 0, NULL);
  glEnableVertexAttribArray(loc);
}


void Shader::bindVertexData(std::string name, unsigned int vbo, int sso) {
  int loc = resolveAttribute(name);
  if(loc < 0)
    return;
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  int size = SHADER_SSO_SIZE(sso);
  int stride = SHADER_SSO_STRIDE(sso) * sizeof(float);
  void *offset = (void*)(SHADER_SSO_OFFSET(sso) * sizeof(float));
  glVertexAttribPointer(loc, size, GL_FLOAT, false, stride, offset);
  glEnableVertexAttribArray(loc);
}

void Shader::bindIndexData(unsigned int vbo) {
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo);
}

void Shader::unbindVertexData(std::string name) {
  int loc = resolveAttribute(name);
  if(loc < 0)
    return;
  glDisableVertexAttribArray(loc);
}

void Shader::recompile() {
  if(hasProgram)
    glDeleteProgram(program);
  program = glCreateProgram();

  if(hasVert)
    glAttachShader(program, shadVert);
  if(hasFrag)
    glAttachShader(program, shadFrag);
  glLinkProgram(program);
  printLinkInfoLog(program);
  if(hasVert)
    glDetachShader(program, shadVert);
  if(hasFrag)
    glDetachShader(program, shadFrag);

  hasProgram = true;
}

int Shader::resolveUniform(std::string name) {
  use();

  std::map<std::string,unsigned int>::iterator itr = uMap.find(name);
  int loc;
  if(itr != uMap.end()) {
    loc = itr->second;
  } else {
    loc = glGetUniformLocation(program, name.c_str());
    if(loc < 0)
      std::cerr << "warning: couldn't resolve uniform '" << name << "', not binding...\n";
    uMap[name] = loc;
  }

  return loc;
}

int Shader::resolveAttribute(std::string name) {
  use();

  std::map<std::string,unsigned int>::iterator itr = aMap.find(name);
  int loc;
  if(itr != aMap.end()) {
    loc = itr->second;
  } else {
    loc = glGetAttribLocation(program, name.c_str());
    if(loc < 0)
      std::cerr << "warning: couldn't resolve attribute '" << name << "', not binding...\n";
    aMap[name] = loc;
  }

  return loc;
}

static std::string textFileRead(std::string filename)
{
    // http://insanecoding.blogspot.com/2011/11/how-to-read-in-file-in-c.html
    std::ifstream in(filename.c_str());
    if (!in) {
        std::cerr << "Error reading file " << filename << std::endl;
        throw (errno);
    }
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

static void printLinkInfoLog(int prog)
{
    GLint linked;
    glGetProgramiv(prog, GL_LINK_STATUS, &linked);
    if (linked == GL_TRUE) {
        return;
    }
    std::cerr << "GLSL LINK ERROR" << std::endl;

    int infoLogLen = 0;
    int charsWritten = 0;
    GLchar *infoLog;

    glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &infoLogLen);

    if (infoLogLen > 0) {
        infoLog = new GLchar[infoLogLen];
        // error check for fail to allocate memory omitted
        glGetProgramInfoLog(prog, infoLogLen, &charsWritten, infoLog);
        std::cerr << "InfoLog:" << std::endl << infoLog << std::endl;
        delete[] infoLog;
    }
    // Throwing here allows us to use the debugger to track down the error.
    throw;
}

static void printShaderInfoLog(int shader)
{
    GLint compiled;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (compiled == GL_TRUE) {
        return;
    }
    std::cerr << "GLSL COMPILE ERROR" << std::endl;

    int infoLogLen = 0;
    int charsWritten = 0;
    GLchar *infoLog;

    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLen);

    if (infoLogLen > 0) {
        infoLog = new GLchar[infoLogLen];
        // error check for fail to allocate memory omitted
        glGetShaderInfoLog(shader, infoLogLen, &charsWritten, infoLog);
        std::cerr << "InfoLog:" << std::endl << infoLog << std::endl;
        delete[] infoLog;
    }
    // Throwing here allows us to use the debugger to track down the error.
    throw;
}

void Shader::use() {
  if(inUse != this && hasProgram) {
    glUseProgram(program);
    inUse = this;
  }
}

void Shader::requireAttr(std::string name) {
  int l = resolveAttribute(name);
  if(l < 0) {
    std::string w = "shader missing required attribute '" + name + "'";
    throw std::runtime_error(w.c_str());
  }
}

void Shader::requireAttr(std::string name, int loc) {
  int l = resolveAttribute(name);
  if(l < 0 || l != loc) {
    std::string w = "shader missing required attribute '" + name + "'";
    throw std::runtime_error(w.c_str());
  }
}

void Shader::requireUniform(std::string name) {
  int l = resolveUniform(name);
  if(l < 0) {
    std::string w = "shader missing required uniform '" + name + "'";
    throw std::runtime_error(w.c_str());
  }
}

void Shader::requireUniform(std::string name, int loc) {
  int l = resolveUniform(name);
  if(l < 0 || l != loc) {
    std::string w = "shader missing required uniform '" + name + "'";
    throw std::runtime_error(w.c_str());
  }
}

void Shader::bind(GLEntity *entity) {
  entity->bind(this);
}

void Shader::draw(Object *obj) {
  obj->draw(this);
}
