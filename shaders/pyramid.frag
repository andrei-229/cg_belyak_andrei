#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragWorldPos;

layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    mat4 model;
    vec4 cameraPos;
    vec4 ambientColor;
    vec4 dirLightDir;
    vec4 dirLightColor;
    vec4 materialAlbedo;
    vec4 materialSpecular;
    float materialShininess;
} ubo;

void main() {
    // Нормализуем входные векторы
    vec3 N = normalize(fragNormal);
    vec3 V = normalize(ubo.cameraPos.xyz - fragWorldPos);
    vec3 L = normalize(-ubo.dirLightDir.xyz); // Свет направлен ОТ источника

    // Базовый цвет из Lab 1 (процедурный) умножаем на альбедо из UI
    vec3 albedo = fragColor * ubo.materialAlbedo.rgb;

    // 1. Ambient (Фоновый свет)
    vec3 ambient = ubo.ambientColor.rgb * ubo.ambientColor.a * albedo;

    // 2. Diffuse (Рассеянный свет)
    float diff = max(dot(N, L), 0.0);
    vec3 diffuse = ubo.dirLightColor.rgb * ubo.dirLightColor.a * diff * albedo;

    // 3. Specular (Блик по Блинну-Фонгу)
    vec3 H = normalize(L + V); // Полувектор
    float spec = pow(max(dot(N, H), 0.0), ubo.materialShininess);
    vec3 specular = ubo.dirLightColor.rgb * ubo.dirLightColor.a * spec * ubo.materialSpecular.rgb;

    // Итоговый цвет
    vec3 result = ambient + diffuse + specular;

    // Гамма-коррекция для более приятной картинки
    result = pow(result, vec3(1.0 / 2.2));

    outColor = vec4(result, 1.0);
}