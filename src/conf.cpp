#include <src/conf.hpp>
#include <glm/glm.hpp>

Config::Config() {}

Config::Config(std::istream& inFile) {
  config(inFile);
}

Config::~Config() {
  nodes.clear();
  lights.clear();
}

void Config::config(std::istream& inFile) {
  std::string param;
  inFile >> param;
  while(inFile) {
    if(param == "OBJECT") {
      NodeConf nc;
      param = config(inFile, nc);
      nodes.push_back(nc);
    }

    else if(param == "LIGHT") {
      LightConf lc;
      param = config(inFile, lc);
      lights.push_back(lc);
    }

    else if(param == "CAMERA") {
      param = config(inFile, camera);
    }

    else if(param == "SHADER") {
      ShaderConf sc;
      param = config(inFile, sc);
      shaders.push_back(sc);
    }

    else if(param == "TEXTURE") {
      TextureConf tc;
      param = config(inFile, tc);
      textures.push_back(tc);
    }

    else if(param == "MATERIAL") {
      MatConf mc;
      param = config(inFile, mc);
      materials.push_back(mc);
    }

    else break;
  }
}

LightConf::LightConf() :
  pos(5,3,1),
  color(1,1,1),
  angle(PI - .01),
  dir(0,-1,0),
  radius(0),
  samp(1)
{}

std::string Config::config(std::istream& inFile, LightConf& conf) {
  std::string param;
  while(inFile >> param) {
    if(param == "LPOS")
      inFile >> conf.pos;
    else if(param == "LCOL")
      inFile >> conf.color;
    else if(param == "ANGLE") {
      inFile >> conf.angle;
      conf.angle *= PI/180.f;
    }
    else if(param == "DIR") {
      inFile >> conf.dir;
      conf.dir = glm::normalize(conf.dir);
    }
    else if(param == "SIZE")
      inFile >> conf.radius;
    else if(param == "SAMP")
      inFile >> conf.samp;
    else
      break;
  }

  return param;
}

CameraConf::CameraConf() :
  pos(0,0,10),
  width(640),
  height(480),
  up(0,1,0),
  fwd(0,0,-10),
  yFov(PI/4),
  density(1),
  outFile("out.png"),
  ambient(.2,.2,.2),
  rayIter(5),
  lSamp(4),
  mcIter(50),
  doTrace(true),
  useMatSort(false),
  useIxCache(false)
{}

std::string Config::config(std::istream& inFile, CameraConf& conf) {
  std::string param;
  while(inFile >> param) {
    if(param == "RES") {
      inFile >> conf.width;
      inFile >> conf.height;
    }
    else if(param == "EYE")
      inFile >> conf.pos;
    else if(param == "LOOKAT")
      inFile >> conf.fwd;
    else if(param == "UP")
      inFile >> conf.up;
    else if(param == "FOVY") {
      inFile >> conf.yFov;
      conf.yFov *= PI/180.f;
    }
    else if(param == "DENSITY")
      inFile >> conf.density;
    else if(param == "FILE")
      inFile >> conf.outFile;
    else if(param == "AMBIENT")
      inFile >> conf.ambient;
    else if(param == "DEPTH")
      inFile >> conf.rayIter;
    else if(param == "LSAMP")
      inFile >> conf.lSamp;
    else if(param == "ITERATIONS")
      inFile >> conf.mcIter;
    else if(param == "RAYTRACE")
      inFile >> conf.doTrace;
    else if(param == "MATSORT")
      inFile >> conf.useMatSort;
    else if(param == "IXCACHE")
      inFile >> conf.useIxCache;
    else if(param == "APERTURE") {
      conf.useDOF = true;
      inFile >> conf.aperture;
    }
    else break;
  }

  return param;
}

NodeConf::NodeConf() :
  scale(1),
  tx(0),
  rot(0),
  ctr(0),
  matName("default"),
  parentName("null"),
  shape("cube"),
  texName("null"),
  bbCull(true)
{}

std::string Config::config(std::istream& inFile, NodeConf& conf) {
  inFile >> conf.name;

  std::string param;
  while(inFile >> param) {
    if(param == "PARENT") {
      inFile >> conf.parentName;
    } else if(param == "TRANS") {
      inFile >> conf.tx;
    } else if(param == "ROTAT") {
      inFile >> conf.rot;
      conf.rot *= PI/180.f;
    } else if(param == "SCALE") {
      inFile >> conf.scale;
    } else if(param == "MATERIAL") {
      inFile >> conf.matName;
    } else if(param == "CENTER") {
      inFile >> conf.ctr;
    } else if(param == "SHAPE") {
      inFile >> conf.shape;
    } else if(param == "FILE") {
      inFile >> conf.objFile;
    } else if(param == "TEXTURE") {
      inFile >> conf.texName;
    } else if(param == "BBCULL") {
      inFile >> conf.bbCull;
    } else break;
  }

  return param;
}

std::string Config::config(std::istream& inFile, ShaderConf& conf) {
  inFile >> conf.name;

  std::string param;
  while(inFile >> param) {
    if(param == "VERT") {
      inFile >> conf.vertFile;
    } else if(param == "FRAG") {
      inFile >> conf.fragFile;
    } else break;
  }

  return param;
}

std::string Config::config(std::istream& inFile, TextureConf& conf) {
  inFile >> conf.name;

  std::string param;
  while(inFile >> param) {
    if(param == "FILE") {
      inFile >> conf.fName;
    } else break;
  }

  return param;
}

MatConf::MatConf() :
  diffCol(.8,.8,.8),
  specCol(.8,.8,.8),
  specExp(10),
  ior(1.333),
  mirr(0),
  trans(0),
  lEmit(0)
{}

std::string Config::config(std::istream& inFile, MatConf& conf) {
  inFile >> conf.name;

  std::string param;
  while(inFile >> param) {
    if(param == "RGB") {
      inFile >> conf.diffCol;
    }

    else if(param == "SPECRGB") {
      inFile >> conf.specCol;
    }

    else if(param == "SPECX") {
      inFile >> conf.specExp;
    }

    else if(param == "REFRIOR") {
      inFile >> conf.ior;
    }

    else if(param == "REFL") {
      inFile >> conf.mirr;
    }

    else if(param == "REFR") {
      inFile >> conf.trans;
    }

    else if(param == "EMITTANCE") {
      inFile >> conf.lEmit;
    }

    else break;
  }

  std::cout << "confd " << conf.name << " " << conf.diffCol << "\n";
  return param;
}
