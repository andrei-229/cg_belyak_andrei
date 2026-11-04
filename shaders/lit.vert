#version 450
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragWorldPos;

// Set 0: Данные сцены
layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
    vec4 ambientColor;
    int numPointLights;
    int numSpotLights;
} scene;

// Set 1: Данные модели
layout(set = 1, binding = 0) uniform ModelUBO {
    mat4 model;
    vec4 albedo;
    vec4 specular;
    float shininess;
} material;

void main() {
    vec4 worldPos = material.model * vec4(inPosition, 1.0);
    gl_Position = scene.proj * scene.view * worldPos;
    
    fragColor = inColor;
    fragNormal = mat3(transpose(inverse(material.model))) * inNormal;
    fragWorldPos = worldPos.xyz;
}