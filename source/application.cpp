#include "application.hpp"

#include <imgui.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace application
{

	namespace
	{

		struct Vertex
		{
			glm::vec3 pos;
			glm::vec3 color;
			glm::vec3 normal;
		};

		struct SceneUBO
		{
			glm::mat4 view;
			glm::mat4 proj;
			glm::vec4 cameraPos;
			glm::vec4 ambientColor;
			int32_t numPointLights;
			int32_t numSpotLights;
			float padding[2];
		};

		struct ModelUBO
		{
			glm::mat4 model;
			glm::vec4 albedo;
			glm::vec4 specular;
			float shininess;
			float padding[3];
		};

		struct PointLight
		{
			glm::vec4 posRadius;
			glm::vec4 colorIntensity;
		};

		struct SpotLight
		{
			glm::vec4 posRadius;
			glm::vec4 dirInner;
			glm::vec4 colorIntensity;
			glm::vec4 outerAngle;
		};

		// Структура для Push Constants
		struct VizPushConstants
		{
			glm::mat4 model;
			glm::vec4 color;
		};

		constexpr uint32_t MAX_LIGHTS = 32;

		// --- Генерация геометрии ---
		std::vector<Vertex> buildPyramidVertices()
		{
			std::vector<Vertex> verts;
			glm::vec3 p0 = {-0.5f, 0.0f, -0.5f}, p1 = {0.5f, 0.0f, -0.5f};
			glm::vec3 p2 = {0.5f, 0.0f, 0.5f}, p3 = {-0.5f, 0.0f, 0.5f};
			glm::vec3 top = {0.0f, 1.0f, 0.0f};

			auto addTri = [&](glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 color)
			{
				glm::vec3 n = glm::normalize(glm::cross(c - a, b - a));
				verts.push_back({a, color, n});
				verts.push_back({b, color, n});
				verts.push_back({c, color, n});
			};
			glm::vec3 cBase = {0.8f, 0.8f, 0.8f}, cSide = {0.6f, 0.6f, 0.6f};
			addTri(p0, p2, p1, cBase);
			addTri(p0, p3, p2, cBase);
			addTri(p0, p1, top, cSide);
			addTri(p1, p2, top, cSide);
			addTri(p2, p3, top, cSide);
			addTri(p3, p0, top, cSide);
			return verts;
		}

		std::vector<Vertex> buildCubeVertices()
		{
			std::vector<Vertex> verts;
			float s = 0.5f;
			glm::vec3 p[] = {
				{-s, -s, -s}, {s, -s, -s}, {s, s, -s}, {-s, s, -s}, {-s, -s, s}, {s, -s, s}, {s, s, s}, {-s, s, s}};
			auto addFace = [&](int a, int b, int c, int d)
			{
				glm::vec3 n = glm::normalize(glm::cross(p[c] - p[a], p[b] - p[a]));
				verts.push_back({p[a], {1, 1, 1}, n});
				verts.push_back({p[b], {1, 1, 1}, n});
				verts.push_back({p[c], {1, 1, 1}, n});
				verts.push_back({p[a], {1, 1, 1}, n});
				verts.push_back({p[c], {1, 1, 1}, n});
				verts.push_back({p[d], {1, 1, 1}, n});
			};
			addFace(0, 1, 2, 3);
			addFace(5, 4, 7, 6);
			addFace(4, 0, 3, 7);
			addFace(1, 5, 6, 2);
			addFace(4, 5, 1, 0);
			addFace(3, 2, 6, 7);
			return verts;
		}

		std::vector<Vertex> buildConeLineVertices()
		{
			std::vector<Vertex> verts;
			int segments = 32;
			float angleStep = glm::radians(360.0f) / segments;
			for (int i = 0; i < segments; ++i)
			{
				float a1 = i * angleStep;
				float a2 = (i + 1) * angleStep;
				verts.push_back({{0, 0, 0}, {1, 1, 1}, {0, 1, 0}});
				verts.push_back({{cos(a1), sin(a1), -1}, {1, 1, 1}, {0, 1, 0}});
				verts.push_back({{cos(a1), sin(a1), -1}, {1, 1, 1}, {0, 1, 0}});
				verts.push_back({{cos(a2), sin(a2), -1}, {1, 1, 1}, {0, 1, 0}});
			}
			return verts;
		}

		const std::vector<Vertex> pyramidVertices = buildPyramidVertices();
		const std::vector<Vertex> cubeVertices = buildCubeVertices();
		const std::vector<Vertex> coneLineVertices = buildConeLineVertices();

		// --- Vulkan Объекты ---
		VkBuffer pyramidBuffer, cubeBuffer, coneLineBuffer, sceneUboBuffer, modelUboBuffer, pointSsboBuffer, spotSsboBuffer;
		VmaAllocation pyramidAlloc, cubeAlloc, coneLineAlloc, sceneUboAlloc, modelUboAlloc, pointSsboAlloc, spotSsboAlloc;

		VkDescriptorSetLayout sceneLayout, modelLayout;
		VkDescriptorPool descPool;
		VkDescriptorSet sceneSet, modelSet;

		VkPipelineLayout mainPipeLayout, vizPipeLayout;
		VkPipeline litPipeline, unlitPipeline, linePipeline;

		// --- Состояние приложения ---
		float camYaw = 45.0f, camPitch = 30.0f, camDistance = 5.0f, fovDeg = 60.0f;
		float modelRotation = 0.0f;
		glm::vec3 matAlbedo = glm::vec3(1.0f);
		glm::vec3 matSpecular = glm::vec3(1.0f);
		float matShininess = 32.0f;
		glm::vec3 ambientColor = glm::vec3(0.1f, 0.1f, 0.15f);
		float ambientIntensity = 1.0f;

		std::vector<PointLight> pointLights;
		std::vector<SpotLight> spotLights;
		bool showVisualizers = true;

		// --- Хелперы ---
		VkShaderModule loadShaderModule(const std::string &path)
		{
			std::ifstream file(path, std::ios::ate | std::ios::binary);
			if (!file.is_open())
			{
				std::cerr << "Failed to open: " << path << '\n';
				return VK_NULL_HANDLE;
			}
			size_t size = static_cast<size_t>(file.tellg());
			std::vector<char> buffer(size);
			file.seekg(0);
			file.read(buffer.data(), static_cast<std::streamsize>(size));
			file.close();
			VkShaderModuleCreateInfo info{};
			info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
			info.codeSize = size;
			info.pCode = reinterpret_cast<const uint32_t *>(buffer.data());
			VkShaderModule module = VK_NULL_HANDLE;
			vkCreateShaderModule(graphics::internal::context.device, &info, nullptr, &module);
			return module;
		}

		VkBuffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VmaAllocation &outAlloc)
		{
			VkBufferCreateInfo bInfo{};
			bInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			bInfo.size = size;
			bInfo.usage = usage;
			bInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
			VmaAllocationCreateInfo aInfo{};
			aInfo.usage = VMA_MEMORY_USAGE_AUTO;
			aInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
			VkBuffer buf = VK_NULL_HANDLE;
			vmaCreateBuffer(graphics::internal::context.allocator, &bInfo, &aInfo, &buf, &outAlloc, nullptr);
			return buf;
		}

		void copyToBuffer(VmaAllocation alloc, const void *data, VkDeviceSize size)
		{
			void *mapped = nullptr;
			vmaMapMemory(graphics::internal::context.allocator, alloc, &mapped);
			std::memcpy(mapped, data, static_cast<size_t>(size));
			vmaUnmapMemory(graphics::internal::context.allocator, alloc);
		}

		VkPipeline createGraphicsPipeline(VkShaderModule vert, VkShaderModule frag, VkPipelineLayout layout, VkPrimitiveTopology topology, bool isLit)
		{
			VkPipelineShaderStageCreateInfo stages[2]{};
			stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vert, "main"};
			stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, frag, "main"};

			VkVertexInputBindingDescription bindingDesc = {0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
			VkVertexInputAttributeDescription attrDesc[3] = {
				{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)},
				{1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color)},
				{2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)}};
			VkPipelineVertexInputStateCreateInfo vInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
			vInput.vertexBindingDescriptionCount = 1;
			vInput.pVertexBindingDescriptions = &bindingDesc;
			vInput.vertexAttributeDescriptionCount = 3;
			vInput.pVertexAttributeDescriptions = attrDesc;

			VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
			ia.topology = topology;
			VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
			vp.viewportCount = 1;
			vp.scissorCount = 1;

			VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
			rs.polygonMode = VK_POLYGON_MODE_FILL;
			rs.cullMode = isLit ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE;
			rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
			rs.lineWidth = 1.0f;

			VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
			ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
			VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
			ds.depthTestEnable = VK_TRUE;
			ds.depthWriteEnable = VK_TRUE;
			ds.depthCompareOp = VK_COMPARE_OP_LESS;
			VkPipelineColorBlendAttachmentState cba{};
			cba.colorWriteMask = 0xF;
			VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
			cb.attachmentCount = 1;
			cb.pAttachments = &cba;
			VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
			VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
			dy.dynamicStateCount = 2;
			dy.pDynamicStates = dyn;

			VkGraphicsPipelineCreateInfo pi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
			pi.stageCount = 2;
			pi.pStages = stages;
			pi.pVertexInputState = &vInput;
			pi.pInputAssemblyState = &ia;
			pi.pViewportState = &vp;
			pi.pRasterizationState = &rs;
			pi.pMultisampleState = &ms;
			pi.pDepthStencilState = &ds;
			pi.pColorBlendState = &cb;
			pi.pDynamicState = &dy;
			pi.layout = layout;
			pi.renderPass = graphics::internal::context.render_pass;

			VkPipeline pipe;
			vkCreateGraphicsPipelines(graphics::internal::context.device, VK_NULL_HANDLE, 1, &pi, nullptr, &pipe);
			return pipe;
		}

	} // namespace

	bool initialize()
	{
		auto &ctx = graphics::internal::context;

		pyramidBuffer = createBuffer(sizeof(Vertex) * pyramidVertices.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, pyramidAlloc);
		copyToBuffer(pyramidAlloc, pyramidVertices.data(), sizeof(Vertex) * pyramidVertices.size());
		cubeBuffer = createBuffer(sizeof(Vertex) * cubeVertices.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, cubeAlloc);
		copyToBuffer(cubeAlloc, cubeVertices.data(), sizeof(Vertex) * cubeVertices.size());
		coneLineBuffer = createBuffer(sizeof(Vertex) * coneLineVertices.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, coneLineAlloc);
		copyToBuffer(coneLineAlloc, coneLineVertices.data(), sizeof(Vertex) * coneLineVertices.size());

		sceneUboBuffer = createBuffer(sizeof(SceneUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sceneUboAlloc);
		modelUboBuffer = createBuffer(sizeof(ModelUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, modelUboAlloc);
		pointSsboBuffer = createBuffer(sizeof(PointLight) * MAX_LIGHTS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, pointSsboAlloc);
		spotSsboBuffer = createBuffer(sizeof(SpotLight) * MAX_LIGHTS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, spotSsboAlloc);

		VkDescriptorSetLayoutBinding sBindings[3]{};
		sBindings[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
		sBindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
		sBindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
		VkDescriptorSetLayoutCreateInfo sl0{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
		sl0.bindingCount = 3;
		sl0.pBindings = sBindings;
		vkCreateDescriptorSetLayout(ctx.device, &sl0, nullptr, &sceneLayout);

		VkDescriptorSetLayoutBinding mBindings[1]{};
		mBindings[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
		VkDescriptorSetLayoutCreateInfo sl1{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
		sl1.bindingCount = 1;
		sl1.pBindings = mBindings;
		vkCreateDescriptorSetLayout(ctx.device, &sl1, nullptr, &modelLayout);

		VkDescriptorPoolSize poolSizes[] = {
			{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2},
			{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2}};
		VkDescriptorPoolCreateInfo pInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
		pInfo.maxSets = 2;
		pInfo.poolSizeCount = 2;
		pInfo.pPoolSizes = poolSizes;
		vkCreateDescriptorPool(ctx.device, &pInfo, nullptr, &descPool);

		VkDescriptorSetAllocateInfo alloc0{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
		alloc0.descriptorPool = descPool;
		alloc0.descriptorSetCount = 1;
		alloc0.pSetLayouts = &sceneLayout;
		vkAllocateDescriptorSets(ctx.device, &alloc0, &sceneSet);

		VkDescriptorSetAllocateInfo alloc1{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
		alloc1.descriptorPool = descPool;
		alloc1.descriptorSetCount = 1;
		alloc1.pSetLayouts = &modelLayout;
		vkAllocateDescriptorSets(ctx.device, &alloc1, &modelSet);

		VkDescriptorBufferInfo sceneBi{sceneUboBuffer, 0, sizeof(SceneUBO)};
		VkDescriptorBufferInfo pointBi{pointSsboBuffer, 0, sizeof(PointLight) * MAX_LIGHTS};
		VkDescriptorBufferInfo spotBi{spotSsboBuffer, 0, sizeof(SpotLight) * MAX_LIGHTS};
		VkWriteDescriptorSet writes[3]{};
		writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sceneSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &sceneBi, nullptr};
		writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sceneSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &pointBi, nullptr};
		writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sceneSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &spotBi, nullptr};
		vkUpdateDescriptorSets(ctx.device, 3, writes, 0, nullptr);

		VkDescriptorBufferInfo modelBi{modelUboBuffer, 0, sizeof(ModelUBO)};
		VkWriteDescriptorSet write1{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, modelSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &modelBi, nullptr};
		vkUpdateDescriptorSets(ctx.device, 1, &write1, 0, nullptr);

		VkDescriptorSetLayout mainSetLayouts[] = {sceneLayout, modelLayout};
		VkPipelineLayoutCreateInfo mainPlInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
		mainPlInfo.setLayoutCount = 2;
		mainPlInfo.pSetLayouts = mainSetLayouts;
		vkCreatePipelineLayout(ctx.device, &mainPlInfo, nullptr, &mainPipeLayout);

		VkPushConstantRange pushRange{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(VizPushConstants)};
		VkPipelineLayoutCreateInfo vizPlInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
		vizPlInfo.setLayoutCount = 1;
		vizPlInfo.pSetLayouts = &sceneLayout;
		vizPlInfo.pushConstantRangeCount = 1;
		vizPlInfo.pPushConstantRanges = &pushRange;
		vkCreatePipelineLayout(ctx.device, &vizPlInfo, nullptr, &vizPipeLayout);

		VkShaderModule litV = loadShaderModule("shaders/lit.vert.spv");
		VkShaderModule litF = loadShaderModule("shaders/lit.frag.spv");
		VkShaderModule unlitV = loadShaderModule("shaders/unlit.vert.spv");
		VkShaderModule unlitF = loadShaderModule("shaders/unlit.frag.spv");

		litPipeline = createGraphicsPipeline(litV, litF, mainPipeLayout, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, true);
		unlitPipeline = createGraphicsPipeline(unlitV, unlitF, vizPipeLayout, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, false);
		linePipeline = createGraphicsPipeline(unlitV, unlitF, vizPipeLayout, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, false);

		vkDestroyShaderModule(ctx.device, litV, nullptr);
		vkDestroyShaderModule(ctx.device, litF, nullptr);
		vkDestroyShaderModule(ctx.device, unlitV, nullptr);
		vkDestroyShaderModule(ctx.device, unlitF, nullptr);

		pointLights.push_back({{1.0f, 1.5f, 1.0f, 10.0f}, {1.0f, 0.2f, 0.2f, 1.0f}});
		spotLights.push_back({{0.0f, 3.0f, 0.0f, 15.0f},
							  {0.0f, -1.0f, 0.0f, glm::cos(glm::radians(20.0f))},
							  {0.2f, 0.5f, 1.0f, 2.0f},
							  {glm::cos(glm::radians(30.0f)), 0.0f, 0.0f, 0.0f}});

		return true;
	}

	void shutdown()
	{
		auto &ctx = graphics::internal::context;
		vkQueueWaitIdle(ctx.graphics_queue);
		vkDestroyPipeline(ctx.device, litPipeline, nullptr);
		vkDestroyPipeline(ctx.device, unlitPipeline, nullptr);
		vkDestroyPipeline(ctx.device, linePipeline, nullptr);
		vkDestroyPipelineLayout(ctx.device, mainPipeLayout, nullptr);
		vkDestroyPipelineLayout(ctx.device, vizPipeLayout, nullptr);
		vkDestroyDescriptorPool(ctx.device, descPool, nullptr);
		vkDestroyDescriptorSetLayout(ctx.device, sceneLayout, nullptr);
		vkDestroyDescriptorSetLayout(ctx.device, modelLayout, nullptr);
		vmaDestroyBuffer(ctx.allocator, pyramidBuffer, pyramidAlloc);
		vmaDestroyBuffer(ctx.allocator, cubeBuffer, cubeAlloc);
		vmaDestroyBuffer(ctx.allocator, coneLineBuffer, coneLineAlloc);
		vmaDestroyBuffer(ctx.allocator, sceneUboBuffer, sceneUboAlloc);
		vmaDestroyBuffer(ctx.allocator, modelUboBuffer, modelUboAlloc);
		vmaDestroyBuffer(ctx.allocator, pointSsboBuffer, pointSsboAlloc);
		vmaDestroyBuffer(ctx.allocator, spotSsboBuffer, spotSsboAlloc);
	}

	void update([[maybe_unused]] double time)
	{
		auto &ctx = graphics::internal::context;

		ImGui::Begin("Advanced Lighting");

		if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen))
		{
			ImGui::SliderFloat("Yaw", &camYaw, -180.0f, 180.0f);
			ImGui::SliderFloat("Pitch", &camPitch, -89.0f, 89.0f);
			ImGui::SliderFloat("Distance", &camDistance, 1.0f, 20.0f);
			ImGui::SliderFloat("FOV", &fovDeg, 10.0f, 120.0f);
		}

		if (ImGui::CollapsingHeader("Object", ImGuiTreeNodeFlags_DefaultOpen))
		{
			ImGui::SliderFloat("Rotation", &modelRotation, -180.0f, 180.0f);
			ImGui::ColorEdit3("Albedo", &matAlbedo.x);
			ImGui::ColorEdit3("Specular", &matSpecular.x);
			ImGui::SliderFloat("Shininess", &matShininess, 2.0f, 256.0f);
		}

		ImGui::Checkbox("Show Visualizers", &showVisualizers);
		ImGui::ColorEdit3("Ambient Color", &ambientColor.x);
		ImGui::SliderFloat("Ambient Intensity", &ambientIntensity, 0.0f, 2.0f);

		if (ImGui::CollapsingHeader("Point Lights", ImGuiTreeNodeFlags_DefaultOpen))
		{
			if (ImGui::Button("Add Point Light") && pointLights.size() < MAX_LIGHTS)
			{
				// Спавним в случайном месте, чтобы они не перекрывали друг друга
				float rx = (rand() % 10 - 5) * 0.4f;
				float rz = (rand() % 10 - 5) * 0.4f;
				pointLights.push_back({{rx, 2.0f, rz, 10.0f}, {1.0f, 1.0f, 1.0f, 1.0f}});
			}
			for (size_t i = 0; i < pointLights.size(); ++i)
			{
				ImGui::PushID(i);
				ImGui::Text("Point Light %d", (int)i);
				ImGui::DragFloat3("Pos", &pointLights[i].posRadius.x, 0.1f);
				ImGui::SliderFloat("Radius", &pointLights[i].posRadius.w, 0.1f, 50.0f);
				ImGui::ColorEdit3("Color", &pointLights[i].colorIntensity.x);
				ImGui::SliderFloat("Intensity", &pointLights[i].colorIntensity.w, 0.0f, 3.0f);
				if (ImGui::Button("Remove"))
				{
					pointLights.erase(pointLights.begin() + i);
					ImGui::PopID();
					break;
				}
				ImGui::Separator();
				ImGui::PopID();
			}
		}

		if (ImGui::CollapsingHeader("Spot Lights", ImGuiTreeNodeFlags_DefaultOpen))
		{
			if (ImGui::Button("Add Spot Light") && spotLights.size() < MAX_LIGHTS)
			{
				float rx = (rand() % 10 - 5) * 0.4f;
				float rz = (rand() % 10 - 5) * 0.4f;
				spotLights.push_back({{rx, 5.0f, rz, 15.0f},
									  {0.0f, -1.0f, 0.0f, glm::cos(glm::radians(20.0f))},
									  {1.0f, 1.0f, 1.0f, 1.0f},
									  {glm::cos(glm::radians(30.0f)), 0.0f, 0.0f, 0.0f}});
			}
			for (size_t i = 0; i < spotLights.size(); ++i)
			{
				ImGui::PushID(i + 1000);
				ImGui::Text("Spot Light %d", (int)i);
				ImGui::DragFloat3("Pos", &spotLights[i].posRadius.x, 0.1f);
				ImGui::SliderFloat("Radius", &spotLights[i].posRadius.w, 0.1f, 50.0f);
				ImGui::DragFloat3("Dir", &spotLights[i].dirInner.x, 0.1f);
				float innerDeg = glm::degrees(glm::acos(glm::clamp(spotLights[i].dirInner.w, -1.0f, 1.0f)));
				float outerDeg = glm::degrees(glm::acos(glm::clamp(spotLights[i].outerAngle.x, -1.0f, 1.0f)));
				if (ImGui::SliderFloat("Inner Angle", &innerDeg, 1.0f, 89.0f))
					spotLights[i].dirInner.w = glm::cos(glm::radians(innerDeg));
				if (ImGui::SliderFloat("Outer Angle", &outerDeg, 1.0f, 90.0f))
					spotLights[i].outerAngle.x = glm::cos(glm::radians(outerDeg));
				ImGui::ColorEdit3("Color", &spotLights[i].colorIntensity.x);
				ImGui::SliderFloat("Intensity", &spotLights[i].colorIntensity.w, 0.0f, 3.0f);
				if (ImGui::Button("Remove"))
				{
					spotLights.erase(spotLights.begin() + i);
					ImGui::PopID();
					break;
				}
				ImGui::Separator();
				ImGui::PopID();
			}
		}

		ImGui::End();

		float aspect = (float)ctx.swapchain_extent.width / ctx.swapchain_extent.height;
		float radYaw = glm::radians(camYaw), radPitch = glm::radians(camPitch);
		glm::vec3 camPos = {camDistance * cos(radPitch) * sin(radYaw), camDistance * sin(radPitch), camDistance * cos(radPitch) * cos(radYaw)};

		SceneUBO sUbo{};
		sUbo.view = glm::lookAt(camPos, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		sUbo.proj = glm::perspective(glm::radians(fovDeg), aspect, 0.1f, 100.0f);
		sUbo.cameraPos = glm::vec4(camPos, 1.0f);
		sUbo.ambientColor = glm::vec4(ambientColor, ambientIntensity);
		sUbo.numPointLights = (int)pointLights.size();
		sUbo.numSpotLights = (int)spotLights.size();
		copyToBuffer(sceneUboAlloc, &sUbo, sizeof(SceneUBO));

		if (!pointLights.empty())
			copyToBuffer(pointSsboAlloc, pointLights.data(), pointLights.size() * sizeof(PointLight));
		if (!spotLights.empty())
			copyToBuffer(spotSsboAlloc, spotLights.data(), spotLights.size() * sizeof(SpotLight));

		ModelUBO mUbo{};
		mUbo.model = glm::rotate(glm::mat4(1.0f), glm::radians(modelRotation), glm::vec3(0.0f, 1.0f, 0.0f));
		mUbo.albedo = glm::vec4(matAlbedo, 1.0f);
		mUbo.specular = glm::vec4(matSpecular, 1.0f);
		mUbo.shininess = matShininess;
		copyToBuffer(modelUboAlloc, &mUbo, sizeof(ModelUBO));
	}

	void render(const graphics::internal::FrameData &fd)
	{
		auto &ctx = graphics::internal::context;
		vkResetCommandBuffer(fd.command_buffer, 0);
		VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(fd.command_buffer, &beginInfo);

		VkClearValue clearValues[2]{};
		clearValues[0].color = {{0.02f, 0.02f, 0.03f, 1.0f}};
		clearValues[1].depthStencil = {1.0f, 0};
		VkRenderPassBeginInfo rpBegin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
		rpBegin.renderPass = ctx.render_pass;
		rpBegin.framebuffer = fd.framebuffer;
		rpBegin.renderArea = {{0, 0}, ctx.swapchain_extent};
		rpBegin.clearValueCount = 2;
		rpBegin.pClearValues = clearValues;
		vkCmdBeginRenderPass(fd.command_buffer, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);

		VkViewport viewport{0, 0, (float)ctx.swapchain_extent.width, (float)ctx.swapchain_extent.height, 0, 1};
		vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
		VkRect2D scissor{{0, 0}, ctx.swapchain_extent};
		vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

		VkDeviceSize offset = 0;

		// 1. Отрисовка пирамиды
		VkDescriptorSet mainSets[] = {sceneSet, modelSet};
		vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, litPipeline);
		vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, mainPipeLayout, 0, 2, mainSets, 0, nullptr);
		vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &pyramidBuffer, &offset);
		vkCmdDraw(fd.command_buffer, (uint32_t)pyramidVertices.size(), 1, 0, 0);

		// 2. Отрисовка визуализаторов (Push Constants)
		if (showVisualizers)
		{
			VizPushConstants pc{};

			// Кубики
			vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, unlitPipeline);
			vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vizPipeLayout, 0, 1, &sceneSet, 0, nullptr);
			vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &cubeBuffer, &offset);

			for (const auto &pl : pointLights)
			{
				pc.model = glm::translate(glm::mat4(1.0f), glm::vec3(pl.posRadius)) * glm::scale(glm::mat4(1.0f), glm::vec3(0.05f));
				pc.color = pl.colorIntensity;
				vkCmdPushConstants(fd.command_buffer, vizPipeLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
				vkCmdDraw(fd.command_buffer, (uint32_t)cubeVertices.size(), 1, 0, 0);
			}

			for (const auto &sl : spotLights)
			{
				pc.model = glm::translate(glm::mat4(1.0f), glm::vec3(sl.posRadius)) * glm::scale(glm::mat4(1.0f), glm::vec3(0.05f));
				pc.color = sl.colorIntensity;
				vkCmdPushConstants(fd.command_buffer, vizPipeLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
				vkCmdDraw(fd.command_buffer, (uint32_t)cubeVertices.size(), 1, 0, 0);
			}

			// Конусы
			vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, linePipeline);
			vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &coneLineBuffer, &offset);

			for (const auto &sl : spotLights)
			{
				glm::vec3 pos = glm::vec3(sl.posRadius);
				glm::vec3 dir = glm::normalize(glm::vec3(sl.dirInner));
				float range = sl.posRadius.w;
				float outerAngle = glm::acos(glm::clamp(sl.outerAngle.x, -1.0f, 1.0f));
				float baseRadius = range * glm::tan(outerAngle);

				glm::vec3 up = glm::vec3(0, 1, 0);
				if (glm::abs(glm::dot(dir, up)) > 0.99f)
					up = glm::vec3(1, 0, 0);
				glm::vec3 right = glm::normalize(glm::cross(dir, up));
				up = glm::normalize(glm::cross(right, dir));

				glm::mat4 rot(1.0f);
				rot[0] = glm::vec4(right, 0.0f);
				rot[1] = glm::vec4(up, 0.0f);
				rot[2] = glm::vec4(-dir, 0.0f);

				pc.model = glm::translate(glm::mat4(1.0f), pos) * rot * glm::scale(glm::mat4(1.0f), glm::vec3(baseRadius, baseRadius, range));
				pc.color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
				vkCmdPushConstants(fd.command_buffer, vizPipeLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
				vkCmdDraw(fd.command_buffer, (uint32_t)coneLineVertices.size(), 1, 0, 0);
			}
		}

		vkCmdEndRenderPass(fd.command_buffer);
		vkEndCommandBuffer(fd.command_buffer);
	}

} // namespace application