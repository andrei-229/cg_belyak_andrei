#version 450
layout(location = 0) in vec3 inPosition;

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
    vec4 ambientColor;
    int numPointLights;
    int numSpotLights;
} scene;

layout(push_constant) uniform Push {
    mat4 model;
    vec4 color;
} push;

layout(location = 0) out vec3 fragColor;

void main() {
    gl_Position = scene.proj * scene.view * push.model * vec4(inPosition, 1.0);
    fragColor = push.color.rgb;
}