#include <src/camera.hpp>
#include <src/util.hpp>
#include <src/shader.hpp>
#include <src/node.hpp>


#include <thrust/device_vector.h>
#include <thrust/sort.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/extented_min_max.hpp>
#include <glm/gtc/random.hpp>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <ctime>
#include <cstring>

#include <string>
#include <iostream>
#include <iomanip>

#include <GLFW/glfw3.h>
#include <stb_image_write.h>
#include <omp.h>

#ifndef PI
#define PI 3.14169265358979f
#endif

Camera::Camera(const CameraConf& conf) :
  yFov(conf.yFov),
  width(conf.width),
  height(conf.height),
  pos(conf.pos),
  up(conf.up),
  fwd(conf.fwd),
  density(conf.density),
  outFile(conf.outFile),
  ambient(conf.ambient),
  rayCount(conf.rayIter),
  lSamp(conf.lSamp),
  mcIter(conf.mcIter),
  doTrace(conf.doTrace),
  useMatSort(conf.useMatSort),
  useIxCache(conf.useIxCache),
  useDOF(conf.useDOF),
  aperture(conf.aperture)
{
  float len = glm::length(fwd);
  yvAngle = std::asin(fwd.y/len);
  xvAngle = std::atan2(fwd.x, fwd.z);
}

void Camera::resize(int w, int h) {
  glViewport(0, 0, w, h);
  width = w;
  height = h;
}

void Camera::zoom(float dy) {
  float r = 1.0 - 2.0*dy/height;
    yFov *= r;
  if(yFov > PI/3)
    yFov = PI/3;
  if(yFov < PI/6)
    yFov = PI/6;
  std::cout << std::setprecision(2) << std::fixed;
  std::cout << "zoom " << (PI/(4*yFov)) << "x\n";
}

void Camera::rotate(float xpos, float ypos, float dt) {

  float dx = float(xpos) * dt / width;
  float dy = float(ypos) * dt / height;
  xvAngle += dx;
  yvAngle += dy;

  printf("%f %f\n", dx, dy);

  if(xvAngle > PI + PI/3)
    xvAngle = PI + PI/3;
  if(xvAngle < PI - PI/3)
    xvAngle = PI - PI/3;
  if(yvAngle > PI/4)
    yvAngle = PI/4;
  if(yvAngle < -PI/4)
    yvAngle = -PI/4;

  printf("%f %f\n", xvAngle*180.0f/M_PI, yvAngle*180.0f/M_PI);

  float len = glm::length(fwd);
  fwd = len*glm::vec3(
    cos(yvAngle) * sin(xvAngle),
    sin(yvAngle),
    cos(yvAngle) * cos(xvAngle)
  );
  up = glm::vec3(0.0f,1.0f,0.0f);
  glm::vec3 right = glm::cross(right, glm::normalize(fwd));
}

void Camera::bind(Shader *shader) {

  glm::mat4 proj = glm::perspective(yFov, width / (float) height, 0.1f, 100.0f);
  glm::mat4 camera = glm::lookAt(pos, pos+fwd, up);

  shader->setUniform("u_CameraPos", pos);
  shader->setUniform("u_ViewProj", proj*camera);
}

glm::vec4 Camera::doShadow(Node *root, const Intersection& ix, const std::vector<Light*>& lights) {
  int numHit = 0, numTot = 0;
  glm::vec3 color;
  if(lSamp < 1)
    return glm::vec4(0);
  for(int i = 0; i < lights.size(); i++) {
    for(int j = 0; j < lSamp; j++) {
      glm::vec3 lPos = lights[i]->pos();
      glm::vec3 lCol = lights[i]->color();

      // light direction
      glm::vec3 L = glm::normalize(lPos - ix.pos);

      Ray lRay(ix.pos, (lPos-ix.pos), 1, true);
      Intersection lix = root->raytrace(lRay);
      numTot++;
      if(lix.t > 0.001 && lix.t < .999)
        continue;
      numHit++;

      // diffuse component
      float LN = glm::clamp(glm::dot(L,ix.normal),0.f,1.f);
      color += .5f*(1.f - ix.mat->trans)*lCol * LN * ix.mat->diffCol;
      //if(!glm::all(glm::abs(ix.normal - glm::vec3(0,1,0)) < .001f))
      //  std::cout << .5f*(1.f - ix.mat->trans)*lCol * LN * ix.mat->diffCol << "\n";

      // specular component
      glm::vec3 V = glm::normalize(pos - ix.pos);
      glm::vec3 R = glm::normalize(L - 2.f * ix.normal * glm::dot(ix.normal, L));
      float spec = std::pow(glm::dot(V,R), ix.mat->specExp);
      color += .5f*(1.f - ix.mat->mirr) * lCol * spec * ix.mat->specCol;
    }
  }
  color /= lSamp;

  return glm::vec4(color, float(numHit)/float(numTot));
}

glm::vec3 Camera::rayIter(Ray ray, Node *root) {

  // if the ray has done its maximum bounce count
  if(ray.iter <= 0)
    return glm::vec3(0);

  // if the ray has negligible transmittance, it wont contribute anyway
  if(glm::all(ray.tx < .001f))
    return glm::vec3(0);

  // if the ray didn't hit anything
  Intersection ix = root->raytrace(ray);
  if(ix.t < 0.001)
    return glm::vec3(0);

  // if we hit a light, get its contribution
  if(ix.mat->lEmit > 0.001)
    return ray.tx * ix.mat->lEmit * ix.mat->diffCol;

  // setup the bounced ray
  glm::vec3 inDir = ray.dir;
  glm::vec3 inTx = ray.tx;
  glm::vec3 rflDir = glm::reflect(inDir, ix.normal);
  ray.iter--;
  ray.p0 = ix.pos;

  // Fresnel coefficients
  float n1,n2;
  if(ray.outside) {
    n1 = 1.0;
    n2 = ix.mat->ior;
  } else {
    n1 = ix.mat->ior;
    n2 = 1.0;
  }
  float nRatio = n1/n2;
  float cosTi = -glm::dot(ray.dir, ix.normal);
  float rSinTi2 = nRatio*nRatio*(1 - cosTi*cosTi);
  float rCoeff = 1.0, tCoeff = 0.0;
  float sq = std::sqrt(1 - rSinTi2);
  if(rSinTi2 < 1) {
    float rs = (nRatio*cosTi - sq) / (nRatio*cosTi + sq);
    float rp = (cosTi - nRatio*sq) / (cosTi + nRatio*sq);
    rCoeff = (rs*rs + rp*rp)/2;
    tCoeff = 1.0 - rCoeff;
  }

  // handle reflection
  glm::vec3 rflCol;
  ray.dir = rflDir;
  ray.tx = inTx * ix.mat->specCol;
  if(ix.mat->mirr > .0001 && rCoeff > .0001)
    rflCol = rCoeff * ix.mat->mirr * rayIter(ray, root) * ix.mat->specCol;

  // handle refraction
  glm::vec3 rfrCol;
  ray.tx = inTx;
  if(ix.mat->trans > .0001 && tCoeff > .0001) {
    if(rSinTi2 < 1) {
      ray.dir = glm::refract(ray.dir, ix.normal, nRatio);
      ray.outside = !ray.outside;
    }
    //std::cout << "refr " << ray.p0 << " " << ray.dir << " " <<ray.iter<<"\n";
    rfrCol = tCoeff * ix.mat->trans * rayIter(ray, root);
  }

  // get the absorbance of the surface
  glm::vec3 diffCol = ix.mat->diffCol;
  float mAbs = 1.0f - std::max(diffCol.x,std::max(diffCol.y,diffCol.z));
  mAbs *= (1.f - ix.mat->mirr)*(1.f - ix.mat->trans);
  if(float(rand())/RAND_MAX < mAbs)
    return glm::vec3(0);

  thrust::default_random_engine rng = makeSeededRandomEngine(ray.iter, 0, 0);
  thrust::uniform_real_distribution<float> u01(0, 1);

  // do monte carlo
  glm::vec3 lmbCol;
  ray.dir = getCosineWeightedDirection(ix.normal, rng, u01);
  ray.tx = inTx*(1.f-ix.mat->trans)*ix.mat->diffCol/(1.0f - mAbs);
  lmbCol = rayIter(ray, root);

  return rflCol + rfrCol + lmbCol;
}

glm::vec4 Camera::rayIter(Ray ray, Node *root, const std::vector<Light*>& lights) {
  if(ray.iter <= 0)
    return glm::vec4(0);

  Intersection ix = root->raytrace(ray);
  if(ix.t < 0.001)
    return glm::vec4(0);

  // setup the bounced ray
  glm::vec3 inDir = ray.dir;
  glm::vec3 rflDir = glm::normalize(inDir - 2.f * ix.normal * glm::dot(ix.normal, inDir));
  ray.iter--;
  ray.p0 = ix.pos;

  // Fresnel coefficients
  float n1,n2;
  if(ray.outside) {
    n1 = 1.0;
    n2 = ix.mat->ior;
  } else {
    n1 = ix.mat->ior;
    n2 = 1.0;
  }
  float nRatio = n1/n2;
  float cosTi = -glm::dot(ray.dir, ix.normal);
  float rSinTi2 = nRatio*nRatio*(1 - cosTi*cosTi);
  float rCoeff = 1.0, tCoeff = 0.0;
  float sq = std::sqrt(1 - rSinTi2);
  if(rSinTi2 < 1) {
    float rs = (nRatio*cosTi - sq) / (nRatio*cosTi + sq);
    rs *= rs;
    float rp = (cosTi - nRatio*sq) / (cosTi + nRatio*sq);
    rp *= rp;
    rCoeff = (rs + rp)/2;
    tCoeff = 1.0 - rCoeff;
  }

  // handle reflection
  glm::vec4 rflCol;
  ray.dir = rflDir;
  if(ix.mat->mirr > .0001 && rCoeff > .0001)
    rflCol = rCoeff * ix.mat->mirr * rayIter(ray, root, lights) * glm::vec4(ix.mat->specCol,1);

  // handle refraction
  glm::vec4 rfrCol;
  if(ix.mat->trans > .0001 && tCoeff > .0001) {
    if(rSinTi2 < 1) {
      ray.dir = glm::normalize((nRatio*cosTi - sq)*ix.normal + nRatio*inDir);
      ray.outside = !ray.outside;
    }
    rfrCol = tCoeff * ix.mat->trans * rayIter(ray, root, lights);
  }

  glm::vec4 shdCol;
  if(ix.mat->mirr < 1 || ix.mat->trans < 1)
    shdCol = doShadow(root, ix, lights);

  return glm::vec4(glm::vec3(rflCol + rfrCol + shdCol), shdCol[3]);
}

void Camera::raytrace(Node *node, const std::vector<Light*>& lights) {
  unsigned char *data = new unsigned char[4*width*height];
  memset(data, 0, 4*width*height);

  float aRatio = (float)width/height;

  glm::vec3 xAxis = glm::cross(fwd, up);
  glm::vec3 yAxis = glm::cross(xAxis, fwd);

  float xFov = std::atan(std::tan(yFov) * aRatio);

  xAxis *= std::tan(xFov/2) * glm::length(fwd) / glm::length(xAxis);
  yAxis *= std::tan(yFov/2) * glm::length(fwd) / glm::length(yAxis);

  for(unsigned int j = 0; j < height; j++) {
    std::cout << "\rline " << j << std::flush;
    #pragma omp parallel for
    for(unsigned int i = 0; i < width; i++) {

      // indirect illumination
      glm::vec3 color(0);
      for(int d = 0; d < density; d++) {
        float x = (float(i) + float(rand())/RAND_MAX) / width;
        float y = (float(j) + float(rand())/RAND_MAX) / height;
        glm::vec3 dir = glm::normalize(fwd + (2*x-1)*xAxis + (1-2*y)*yAxis);
        Ray ray(pos, dir, rayCount, true);

        glm::vec4 dirCol = rayIter(ray, node, lights);
        glm::vec3 mcCol;
        if(dirCol[3] < 1 && mcIter > 0) {
          for(int w = 0; w < mcIter; w++)
            mcCol += rayIter(ray,node);
          mcCol /= float(mcIter);
        } else {
          dirCol[3] = 1;
        }
        color += (1.f - dirCol[3])*mcCol + dirCol[3]*glm::vec3(dirCol);
      }
      color /= float(density);


      data[4*(i + width*j) + 0] = 255.0*glm::clamp(color[0],0.f,1.f);
      data[4*(i + width*j) + 1] = 255.0*glm::clamp(color[1],0.f,1.f);
      data[4*(i + width*j) + 2] = 255.0*glm::clamp(color[2],0.f,1.f);
      data[4*(i + width*j) + 3] = 255;
    }
  }
  stbi_write_png(outFile.c_str(), width, height, 4, data, 4*width);
  delete data;
}


