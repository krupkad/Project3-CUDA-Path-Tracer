#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>

#include <GL/glew.h>
#ifdef _WIN32
#define GLFW_DLL
#endif
#include <GLFW/glfw3.h>

#include <fstream>
#include <iostream>
#include <string>
#include <deque>

#include <src/util.hpp>
#include <src/scene.hpp>

#include <pthread.h>

#ifndef PI
#define PI 3.14169265358979f
#endif

// Standard glut-based program functions
void init(void);
void resize(GLFWwindow*, int, int);
void keypress(GLFWwindow*, int, int, int, int);
void mousepos(GLFWwindow*, double, double);
void scroll(GLFWwindow*, double, double);

// scene graph
std::deque<std::string> nodeList;
Scene *scene;
int xSize = 640, ySize = 480;

pthread_t render_thread;
static bool render_allowed = true;
double render_t0, render_t1;
void *render_worker(void *data) {
  render_t0 = clock();
  scene->raytrace();
  render_t1 = clock();
  render_allowed = true;
  printf("render finished (%.3f s)\n", (render_t1-render_t0)/CLOCKS_PER_SEC);
}

int main(int argc, char** argv)
{
    if(argc < 2) {
      std::cerr << "usage: " << argv[0] << " " << "config\n";
      return EXIT_FAILURE;
    }

    if(!glfwInit())
      exit(EXIT_FAILURE);
    GLFWwindow* window = glfwCreateWindow(xSize, ySize, "Scene Graph", NULL, NULL);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
    glfwMakeContextCurrent(window);

    glewInit();

    // Set the color which clears the screen between frames
    glClearColor(0, 0, 0, 1);

    // Enable and clear the depth buffer
    glEnable(GL_DEPTH_TEST);
    glClearDepth(1.0);
    glDepthFunc(GL_LEQUAL);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // initialize the scene
    std::ifstream in(argv[1]);
    if(!in) {
      std::cout << "couldn't open config " << argv[1] << "\n";
      return EXIT_FAILURE;
    }
    Config sceneConf(in);
    scene = new Scene(sceneConf);
    in.close();

    // get the render order
    nodeList = scene->nodeList();
    if(nodeList.size() > 0) {
      std::string sel = nodeList.front();
      scene->bindShader(sel, "contour");
    }

    glfwSetWindowSizeCallback(window, resize);
    glfwSetKeyCallback(window, keypress);
    glfwSetCursorPosCallback(window, mousepos);
    glfwSetScrollCallback(window, scroll);

    int nbFrames = 0;
    double lastTime = glfwGetTime();
    while(!glfwWindowShouldClose(window)) {

      // Clear the screen so that we only see newly drawn images
      glfwSetCursorPos(window, xSize/2, ySize/2);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      scene->draw("diffuse");
      scene->draw("contour");

      // Move the rendering we just made onto the screen
      glfwSwapBuffers(window);
      glfwPollEvents();

      double currentTime = glfwGetTime();
      nbFrames++;
      if (currentTime - lastTime >= 1.0){
         std::cout << "\r" <<  1000.0/double(nbFrames) << std::flush;
         nbFrames = 0;
         lastTime += 1.0;
      }

      // Check for any GL errors that have happened recently
      printGLErrorLog();
    }

    pthread_join(render_thread, NULL);
    glfwDestroyWindow(window);
    glfwTerminate();
    delete scene;

    return EXIT_SUCCESS;
}

void keypress(GLFWwindow* window, int key, int code, int action, int mods) {
  if(!(mods & GLFW_MOD_SHIFT) && isalpha(key))
    key += 32;
  if(action != GLFW_PRESS && action != GLFW_REPEAT)
    return;
  std::string sel;
  switch(key) {
    case GLFW_KEY_ESCAPE:
    case 'q':
      exit(0);
      break;
    case 's':
    case 'p':
      if (render_allowed) {
        printf("attempt\n");
        pthread_create(&render_thread, NULL, render_worker, NULL);
        render_allowed = false;
      }
      break;
  }
}

void resize(GLFWwindow* window, int width, int height)
{
  xSize = width;
  ySize = height;
  scene->resize(width,height);
}

static double lastX, lastY;
void mousepos(GLFWwindow* window, double xpos, double ypos) {
  scene->moveMouse(xpos-lastX,ypos-lastY);
  lastX = xpos;
  lastY = ypos;
}

void scroll(GLFWwindow*, double dx, double dy) {
  scene->zoom(dy);
}
