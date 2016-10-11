#version 130
// ^ Change this to version 130 if you have compatibility issues

//these are the interpolated values out of the rasterizer, so you can't know
//their specific values without knowing the vertices that contributed to them
varying vec4 fs_CameraVector;
varying vec4 fs_Normal;

void main() {
  float d = dot(fs_CameraVector, fs_Normal);
  float k = exp(-4*d*d);
  gl_FragColor = vec4(1,1,1,k);
}
