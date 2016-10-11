#ifndef TEXTURE_H
#define TEXTURE_H

#include <src/shader.hpp>
#ifdef _WIN32
#include <windows.h>
#endif
#include <GL/gl.h>

// bmp loader adapted from
// http://www.opengl-tutorial.org/beginners-tutorials/tutorial-5-a-textured-cube/

class Texture : public GLEntity {
  public:
    Texture(std::string fname);
    virtual ~Texture();
    virtual void bind(Shader *s);
  private:
    GLuint texID;
};

#endif /* TEXTURE_H */
