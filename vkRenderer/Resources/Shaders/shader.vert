#version 450
#extension GL_ARB_separate_shader_objects : enable

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec2 aTexCoord;

layout (location = 0) out vec3 fragColor;
layout (location = 1) out vec2 texCoord;

layout (binding = 0) uniform UBO
{
  mat4 u_model;
  mat4 u_viewProj;
} ubo;

void main()
{
  gl_Position = ubo.u_viewProj * ubo.u_model * vec4(aPos, 1.0);
  fragColor = aColor;
  texCoord = aTexCoord;
}