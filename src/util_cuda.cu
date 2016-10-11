#include <src/util.hpp>
#include <GL/glew.h>
#include <iostream>
#include <glm/glm.hpp>

__host__ __device__
inline unsigned int utilhash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

__host__ __device__
rng_eng_t makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

__host__ __device__
glm::vec3 getCosineWeightedDirection(const glm::vec3& normal, rng_eng_t &eng, rng_dist_t &dist) {
  // Pick 2 random numbers in the range (0, 1)
  float xi1 = dist(eng);
  float xi2 = dist(eng);

  float up = sqrt(xi1);                       // cos(theta)
  float over = sqrt(1 - xi1); // sin(theta)
  float around = xi2 * 2.0f * M_PI;

  // Find a direction that is not the normal based off of whether or not the normal's components
  // are all equal to sqrt(1/3) or whether or not at least one component is less than sqrt(1/3).
  glm::vec3 upVec;
  if (fabs(fabs(normal.y) - 1) > 0.001)
    upVec = glm::vec3(0,1,0);
  else
    upVec = glm::vec3(0,0,1);

  //Use not-normal direction to generate two perpendicular directions
  glm::vec3 v1 = glm::normalize(glm::cross(normal, upVec));
  glm::vec3 v2 = glm::normalize(glm::cross(normal, v1));

  return (up * normal) + (float(cos(around)) * over * v1) + (float(sin(around)) * over * v2);
}

void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
    getchar();
#  endif
    exit(EXIT_FAILURE);
#endif
}

void printGLErrorLog()
{
    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        std::cerr << "OpenGL error " << error << ": ";
        const char *e =
            error == GL_INVALID_OPERATION             ? "GL_INVALID_OPERATION" :
            error == GL_INVALID_ENUM                  ? "GL_INVALID_ENUM" :
            error == GL_INVALID_VALUE                 ? "GL_INVALID_VALUE" :
            error == GL_INVALID_INDEX                 ? "GL_INVALID_INDEX" :
            "unknown";
        std::cerr << e << std::endl;

        // Throwing here allows us to use the debugger stack trace to track
        // down the error.
#ifndef __APPLE__
        // But don't do this on OS X. It might cause a premature crash.
        // http://lists.apple.com/archives/mac-opengl/2012/Jul/msg00038.html
        throw;
#endif
    }
}

