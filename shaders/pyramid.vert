#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 fragTint;

layout(binding = 0) uniform UniformBufferObject {
    mat4 mvp;
    vec4 tint;   // цветовой множитель из UI
} ubo;

void main() {
    gl_Position = ubo.mvp * vec4(inPosition, 1.0);
    fragColor = inColor;
    fragTint  = ubo.tint.rgb;
}