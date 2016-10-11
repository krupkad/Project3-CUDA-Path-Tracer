#version 130
// ^ Change this to version 130 if you have compatibility issues

//these are the interpolated values out of the rasterizer, so you can't know
//their specific values without knowing the vertices that contributed to them
varying vec4 fs_Normal;
varying vec4 fs_LightVector;
varying vec4 fs_Color;
varying vec4 fs_CameraVector;
varying vec2 fs_UV;

uniform sampler2D u_texSampler;
uniform bool u_hasTexture;
uniform vec3 u_LightColor;

void main() {
  // Material base color (before shading)
  vec4 diffuseColor;
  if(u_hasTexture) {
    vec4 tex = texture2D(u_texSampler, fs_UV);
    diffuseColor = vec4(tex.rgb, 1);
  } else {
    diffuseColor = fs_Color;
  }

  // Calculate the diffuse term
  float diffuseTerm = dot(fs_Normal, fs_LightVector);
  // Avoid negative lighting values
  diffuseTerm = clamp(diffuseTerm, 0, 1);

  // Fixed ambient term
  float ambientTerm = 0.3;

  // Phong specular reflections
  float specularTerm;
  vec4 r = reflect(-fs_LightVector, fs_Normal);
  float rv = dot(r, fs_CameraVector);
  if(rv < 0)
    specularTerm = 0;
  else
    specularTerm = pow(rv,10);

  float lightIntensity = diffuseTerm + specularTerm + ambientTerm;

  // Compute final shaded color
  gl_FragColor = vec4(lightIntensity * diffuseColor.rgb * u_LightColor, diffuseColor.a);
}
