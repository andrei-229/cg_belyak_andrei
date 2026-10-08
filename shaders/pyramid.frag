#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragTint;

layout(location = 0) out vec4 outColor;

void main() {
    // Процедурный цвет вершины × цвет из ImGui
    outColor = vec4(fragColor * fragTint, 1.0);
}