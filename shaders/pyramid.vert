#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragWorldPos;

layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    mat4 model;
    vec4 cameraPos;
    vec4 ambientColor;     // rgb = color, a = intensity
    vec4 dirLightDir;      // xyz = direction
    vec4 dirLightColor;    // rgb = color, a = intensity
    vec4 materialAlbedo;
    vec4 materialSpecular;
    float materialShininess;
} ubo;

void main() {
    vec4 worldPos = ubo.model * vec4(inPosition, 1.0);
    gl_Position = ubo.proj * ubo.view * worldPos;

    fragColor = inColor;
    // Перевод нормалей в мировое пространство (через обратную транспонированную матрицу модели)
    fragNormal = mat3(transpose(inverse(ubo.model))) * inNormal;
    fragWorldPos = worldPos.xyz;
}