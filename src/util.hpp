#ifndef UTIL_H
#define UTIL_H

#include <string>
#include <iostream>
#include <glm/glm.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/random.h>

/* stuff useful for cuda */

#ifdef __CUDA_ARCH__
#define CUDECL __device__ __host__
#else
#define CUDECL
#endif

typedef thrust::default_random_engine rng_eng_t;
typedef thrust::uniform_real_distribution<float> rng_dist_t;

CUDECL inline unsigned int utilhash(unsigned int a);
CUDECL rng_eng_t makeSeededRandomEngine(int iter, int index, int depth);
CUDECL glm::vec3 getCosineWeightedDirection(const glm::vec3& normal, rng_eng_t &eng, rng_dist_t &dist);

#define ERRORCHECK 1
#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line);

/* iostream overloads for glm types */

#define vec2Type typename glm::tvec2<T,P>
#define vec3Type typename glm::tvec3<T,P>
#define vec4Type typename glm::tvec4<T,P>

template <typename T, glm::precision P>
std::ostream& operator<<(std::ostream& os, const vec3Type& vec) {
  os << "(" << vec[0] << ", " << vec[1] << ", " << vec[2] << ")";
  return os;
}

template <typename T, glm::precision P>
std::istream& operator>>(std::istream& is, vec3Type& vec) {
  is >> vec[0];
  is >> vec[1];
  is >> vec[2];
  return is;
}

template <typename T, glm::precision P>
std::ostream& operator<<(std::ostream& os, const vec4Type& vec) {
  os << "(" << vec[0] << ", " << vec[1] << ", " << vec[2] << ", " << vec[3] << ")";
  return os;
}

template <typename T, glm::precision P>
std::ostream& operator<<(std::ostream& os, const vec2Type& vec) {
  os << "(" << vec[0] << ", " << vec[1] << ")";
  return os;
}

template <typename T, glm::precision P>
std::istream& operator>>(std::istream& is, vec4Type& vec) {
  is >> vec[0];
  is >> vec[1];
  is >> vec[2];
  is >> vec[3];
  return is;
}

template <typename T, glm::precision P>
std::istream& operator>>(std::istream& is, vec2Type& vec) {
  is >> vec[0];
  is >> vec[1];
  return is;
}

template <typename T, glm::precision P>
vec3Type::bool_type operator>(const vec3Type& v1, T x) {
  return glm::greaterThan(v1,vec3Type(x));
}


template <typename T, glm::precision P>
vec3Type::bool_type operator<(const vec3Type& v1, T x) {
  return glm::lessThan(v1,vec3Type(x));
}

void printGLErrorLog();

#endif /* UTIL_H */
