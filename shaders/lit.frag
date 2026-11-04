#version 450
layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragWorldPos;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
    vec4 ambientColor;
    int numPointLights;
    int numSpotLights;
} scene;

struct PointLight {
    vec4 posRadius;
    vec4 colorIntensity;
};
struct SpotLight {
    vec4 posRadius;
    vec4 dirInner;
    vec4 colorIntensity;
    vec4 outerAngle;
};

layout(std430, set = 0, binding = 1) readonly buffer PointLights { PointLight lights[]; } pointLights;
layout(std430, set = 0, binding = 2) readonly buffer SpotLights { SpotLight lights[]; } spotLights;

layout(set = 1, binding = 0) uniform ModelUBO {
    mat4 model;
    vec4 albedo;
    vec4 specular;
    float shininess;
} material;

void main() {
    vec3 N = normalize(fragNormal);
    vec3 V = normalize(scene.cameraPos.xyz - fragWorldPos);
    vec3 albedo = fragColor * material.albedo.rgb;
    
    vec3 ambient = scene.ambientColor.rgb * scene.ambientColor.a * albedo;
    vec3 finalColor = ambient;

    // Точечные источники
    for (int i = 0; i < scene.numPointLights; ++i) {
        vec3 lightPos = pointLights.lights[i].posRadius.xyz;
        float radius = pointLights.lights[i].posRadius.w;
        vec3 L = normalize(lightPos - fragWorldPos);
        float dist = max(length(lightPos - fragWorldPos), 0.1);
        
        float attenuation = 1.0 / (1.0 + dist * dist);
        if (dist > radius) attenuation = 0.0;
        
        vec3 lightCol = pointLights.lights[i].colorIntensity.rgb * pointLights.lights[i].colorIntensity.a;
        float diff = max(dot(N, L), 0.0);
        vec3 H = normalize(L + V);
        float spec = pow(max(dot(N, H), 0.0), material.shininess);
        
        finalColor += lightCol * (diff * albedo + material.specular.rgb * spec) * attenuation;
    }

    // Прожекторы
    for (int i = 0; i < scene.numSpotLights; ++i) {
        vec3 lightPos = spotLights.lights[i].posRadius.xyz;
        float radius = spotLights.lights[i].posRadius.w;
        vec3 lightDir = normalize(spotLights.lights[i].dirInner.xyz);
        float cosInner = spotLights.lights[i].dirInner.w;
        float cosOuter = spotLights.lights[i].outerAngle.x;

        vec3 L = normalize(lightPos - fragWorldPos);
        float dist = max(length(lightPos - fragWorldPos), 0.1);
        
        float attenuation = 1.0 / (1.0 + dist * dist);
        if (dist > radius) attenuation = 0.0;
        
        float theta = dot(normalize(-lightDir), L);
        float epsilon = cosInner - cosOuter;
        float spotIntensity = clamp((theta - cosOuter) / epsilon, 0.0, 1.0);
        attenuation *= spotIntensity;
        
        vec3 lightCol = spotLights.lights[i].colorIntensity.rgb * spotLights.lights[i].colorIntensity.a;
        float diff = max(dot(N, L), 0.0);
        vec3 H = normalize(L + V);
        float spec = pow(max(dot(N, H), 0.0), material.shininess);
        
        finalColor += lightCol * (diff * albedo + material.specular.rgb * spec) * attenuation;
    }

    // Защита от пересвета
    finalColor = clamp(finalColor, 0.0, 1.0);
    outColor = vec4(pow(finalColor, vec3(1.0 / 2.2)), 1.0);
}