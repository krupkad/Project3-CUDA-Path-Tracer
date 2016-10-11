#ifdef _WIN32
#include <windows.h>
#endif
#include <GL/glew.h>
#include <GL/gl.h>

#include <cstdio>
#include <iostream>

#include <src/texture.hpp>
#include <src/shader.hpp>
#include <src/util.hpp>

//#include <src/SOIL2/SOIL2.h>

Texture::Texture(std::string fname) {
  /*
  texID = SOIL_load_OGL_texture
    (
        fname.c_str(),
        SOIL_LOAD_AUTO,
        SOIL_CREATE_NEW_ID,
        SOIL_FLAG_MIPMAPS | SOIL_FLAG_INVERT_Y | SOIL_FLAG_NTSC_SAFE_RGB | SOIL_FLAG_COMPRESS_TO_DXT
    );
  if(texID == 0) {
    std::cerr << "SOIL loading error: " << SOIL_last_result() << "\n";
    return;
  }

  glBindTexture(GL_TEXTURE_2D, texID);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glBindTexture(GL_TEXTURE_2D, 0);
  */
}

Texture::~Texture() {
}

void Texture::bind(Shader *s) {
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texID);
  s->setUniform("u_texSampler", 0);
}


