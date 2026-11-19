#include "application.hpp"

#include <imgui.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <random>

namespace application
{

	namespace
	{

		// --- Структуры данных ---
		struct Vertex
		{
			glm::vec3 pos;
			glm::vec3 color;
			glm::vec3 normal;
			glm::vec2 uv;
		};

		struct SceneUBO
		{
			glm::mat4 view;
			glm::mat4 proj;
			glm::vec4 cameraPos;
			glm::vec4 ambientColor;
			int32_t numPointLights;
			int32_t numSpotLights;
			float time;
			float padding[1];
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

		struct ModelPushConstants
		{
			int32_t useNontrivialSampling;
		};

		struct VizPushConstants
		{
			glm::mat4 model;
			glm::vec4 color;
		};

		struct Texture
		{
			VkImage image = VK_NULL_HANDLE;
			VkImageView view = VK_NULL_HANDLE;
			VmaAllocation alloc = VK_NULL_HANDLE;
		};

		constexpr uint32_t MAX_LIGHTS = 32;
		constexpr uint32_t MAX_MODELS = 4;

		// --- Геометрия ---
		std::vector<Vertex> buildPyramidVertices()
		{
			std::vector<Vertex> verts;
			glm::vec3 p0 = {-0.5f, 0.0f, -0.5f}, p1 = {0.5f, 0.0f, -0.5f};
			glm::vec3 p2 = {0.5f, 0.0f, 0.5f}, p3 = {-0.5f, 0.0f, 0.5f};
			glm::vec3 top = {0.0f, 1.0f, 0.0f};

			auto addTri = [&](glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 color, glm::vec2 ua, glm::vec2 ub, glm::vec2 uc)
			{
				glm::vec3 n = glm::normalize(glm::cross(c - a, b - a));
				verts.push_back({a, color, n, ua});
				verts.push_back({b, color, n, ub});
				verts.push_back({c, color, n, uc});
			};
			glm::vec3 cBase = {0.8f, 0.8f, 0.8f}, cSide = {0.6f, 0.6f, 0.6f};
			addTri(p0, p2, p1, cBase, {0, 0}, {1, 1}, {1, 0});
			addTri(p0, p3, p2, cBase, {0, 0}, {0, 1}, {1, 1});
			addTri(p0, p1, top, cSide, {0, 0}, {1, 0}, {0.5f, 1});
			addTri(p1, p2, top, cSide, {0, 0}, {1, 0}, {0.5f, 1});
			addTri(p2, p3, top, cSide, {0, 0}, {1, 0}, {0.5f, 1});
			addTri(p3, p0, top, cSide, {0, 0}, {1, 0}, {0.5f, 1});
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
				verts.push_back({p[a], {1, 1, 1}, n, {0, 0}});
				verts.push_back({p[b], {1, 1, 1}, n, {0, 0}});
				verts.push_back({p[c], {1, 1, 1}, n, {0, 0}});
				verts.push_back({p[a], {1, 1, 1}, n, {0, 0}});
				verts.push_back({p[c], {1, 1, 1}, n, {0, 0}});
				verts.push_back({p[d], {1, 1, 1}, n, {0, 0}});
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
				verts.push_back({{0, 0, 0}, {1, 1, 1}, {0, 1, 0}, {0, 0}});
				verts.push_back({{cos(a1), sin(a1), -1}, {1, 1, 1}, {0, 1, 0}, {0, 0}});
				verts.push_back({{cos(a1), sin(a1), -1}, {1, 1, 1}, {0, 1, 0}, {0, 0}});
				verts.push_back({{cos(a2), sin(a2), -1}, {1, 1, 1}, {0, 1, 0}, {0, 0}});
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
		VkSampler textureSampler;
		VkCommandPool tempCommandPool;

		Texture texSpecular, texEmissive;
		int selectedAlbedoIndex = 0;
		std::vector<Texture> albedoTextures;

		size_t minUboAlignment;
		size_t modelUboSize;
		size_t dynamicAlignment;

		// --- Состояние ---
		float camYaw = 45.0f, camPitch = 30.0f, camDistance = 5.0f, fovDeg = 60.0f;
		float modelRotation = 0.0f;
		glm::vec3 matAlbedoColor = glm::vec3(1.0f);
		float matShininess = 32.0f;
		glm::vec3 ambientColor = glm::vec3(0.1f, 0.1f, 0.15f);
		float ambientIntensity = 1.0f;
		bool useNontrivialSampling = false;
		float globalTime = 0.0f;
		bool showVisualizers = true;

		std::vector<PointLight> pointLights;
		std::vector<SpotLight> spotLights;

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

		void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VmaMemoryUsage memUsage, VkBuffer &buffer, VmaAllocation &alloc)
		{
			VkBufferCreateInfo bInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
			bInfo.size = size;
			bInfo.usage = usage;
			bInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
			VmaAllocationCreateInfo aInfo{};
			aInfo.usage = memUsage;
			if (memUsage == VMA_MEMORY_USAGE_AUTO_PREFER_HOST)
			{
				aInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
			}
			vmaCreateBuffer(graphics::internal::context.allocator, &bInfo, &aInfo, &buffer, &alloc, nullptr);
		}

		void copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size)
		{
			auto &ctx = graphics::internal::context;
			VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
			allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			allocInfo.commandPool = tempCommandPool;
			allocInfo.commandBufferCount = 1;
			VkCommandBuffer cmd;
			vkAllocateCommandBuffers(ctx.device, &allocInfo, &cmd);

			VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
			beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			vkBeginCommandBuffer(cmd, &beginInfo);
			VkBufferCopy copyRegion{};
			copyRegion.size = size;
			vkCmdCopyBuffer(cmd, src, dst, 1, &copyRegion);
			vkEndCommandBuffer(cmd);

			VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &cmd;
			vkQueueSubmit(ctx.graphics_queue, 1, &submitInfo, VK_NULL_HANDLE);
			vkQueueWaitIdle(ctx.graphics_queue);
			vkFreeCommandBuffers(ctx.device, tempCommandPool, 1, &cmd);
		}

		void uploadMeshToVRAM(VkBuffer &gpuBuffer, VmaAllocation &gpuAlloc, const std::vector<Vertex> &vertices)
		{
			VkDeviceSize size = sizeof(Vertex) * vertices.size();
			VkBuffer stagingBuffer;
			VmaAllocation stagingAlloc;
			createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, stagingBuffer, stagingAlloc);
			void *data;
			vmaMapMemory(graphics::internal::context.allocator, stagingAlloc, &data);
			std::memcpy(data, vertices.data(), (size_t)size);
			vmaUnmapMemory(graphics::internal::context.allocator, stagingAlloc);

			createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, gpuBuffer, gpuAlloc);
			copyBuffer(stagingBuffer, gpuBuffer, size);
			vmaDestroyBuffer(graphics::internal::context.allocator, stagingBuffer, stagingAlloc);
		}

		Texture createTexture(int width, int height, const void *pixels, VkFormat format)
		{
			Texture tex;
			auto &ctx = graphics::internal::context;
			VkDeviceSize imageSize = width * height * 4;

			VkBuffer stagingBuffer;
			VmaAllocation stagingAlloc;
			createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, stagingBuffer, stagingAlloc);
			void *data;
			vmaMapMemory(ctx.allocator, stagingAlloc, &data);
			std::memcpy(data, pixels, (size_t)imageSize);
			vmaUnmapMemory(ctx.allocator, stagingAlloc);

			VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
			imageInfo.imageType = VK_IMAGE_TYPE_2D;
			imageInfo.extent = {(uint32_t)width, (uint32_t)height, 1};
			imageInfo.mipLevels = 1;
			imageInfo.arrayLayers = 1;
			imageInfo.format = format;
			imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
			imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
			imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
			imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

			VmaAllocationCreateInfo allocInfo{};
			allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
			vmaCreateImage(ctx.allocator, &imageInfo, &allocInfo, &tex.image, &tex.alloc, nullptr);

			VkCommandBuffer cmd;
			VkCommandBufferAllocateInfo cbInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
			cbInfo.commandPool = tempCommandPool;
			cbInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			cbInfo.commandBufferCount = 1;
			vkAllocateCommandBuffers(ctx.device, &cbInfo, &cmd);

			VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
			beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			vkBeginCommandBuffer(cmd, &beginInfo);

			VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
			barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.image = tex.image;
			barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

			VkBufferImageCopy region{};
			region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
			region.imageExtent = {(uint32_t)width, (uint32_t)height, 1};
			vkCmdCopyBufferToImage(cmd, stagingBuffer, tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
			vkEndCommandBuffer(cmd);

			VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &cmd;
			vkQueueSubmit(ctx.graphics_queue, 1, &submitInfo, VK_NULL_HANDLE);
			vkQueueWaitIdle(ctx.graphics_queue);
			vkFreeCommandBuffers(ctx.device, tempCommandPool, 1, &cmd);
			vmaDestroyBuffer(ctx.allocator, stagingBuffer, stagingAlloc);

			VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
			viewInfo.image = tex.image;
			viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			viewInfo.format = format;
			viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
			vkCreateImageView(ctx.device, &viewInfo, nullptr, &tex.view);
			return tex;
		}

		VkPipeline createGraphicsPipeline(VkShaderModule vert, VkShaderModule frag, VkPipelineLayout layout, VkPrimitiveTopology topology, bool isLit)
		{
			VkPipelineShaderStageCreateInfo stages[2]{};
			stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vert, "main"};
			stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, frag, "main"};

			VkVertexInputBindingDescription bindingDesc = {0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
			VkVertexInputAttributeDescription attrDesc[4] = {
				{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)},
				{1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color)},
				{2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)},
				{3, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)}};
			VkPipelineVertexInputStateCreateInfo vInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
			vInput.vertexBindingDescriptionCount = 1;
			vInput.pVertexBindingDescriptions = &bindingDesc;
			vInput.vertexAttributeDescriptionCount = 4;
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

		VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
		poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		poolInfo.queueFamilyIndex = ctx.graphics_queue_index;
		vkCreateCommandPool(ctx.device, &poolInfo, nullptr, &tempCommandPool);

		VkPhysicalDeviceProperties props;
		vkGetPhysicalDeviceProperties(ctx.physical_device, &props);
		minUboAlignment = props.limits.minUniformBufferOffsetAlignment;
		modelUboSize = sizeof(ModelUBO);
		dynamicAlignment = (modelUboSize + minUboAlignment - 1) & ~(minUboAlignment - 1);

		uploadMeshToVRAM(pyramidBuffer, pyramidAlloc, pyramidVertices);
		uploadMeshToVRAM(cubeBuffer, cubeAlloc, cubeVertices);
		uploadMeshToVRAM(coneLineBuffer, coneLineAlloc, coneLineVertices);

		createBuffer(sizeof(SceneUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, sceneUboBuffer, sceneUboAlloc);
		createBuffer(dynamicAlignment * MAX_MODELS, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, modelUboBuffer, modelUboAlloc);
		createBuffer(sizeof(PointLight) * MAX_LIGHTS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, pointSsboBuffer, pointSsboAlloc);
		createBuffer(sizeof(SpotLight) * MAX_LIGHTS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, spotSsboBuffer, spotSsboAlloc);

		VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		vkCreateSampler(ctx.device, &samplerInfo, nullptr, &textureSampler);

		int texW = 512, texH = 512;
		std::vector<uint8_t> checker(texW * texH * 4);
		std::vector<uint8_t> noise(texW * texH * 4);
		std::mt19937 rng(1337);
		for (int i = 0; i < texW * texH; ++i)
		{
			int x = i % texW, y = i / texW;
			bool isWhite = ((x / 32) % 2) == ((y / 32) % 2);
			checker[i * 4 + 0] = isWhite ? 255 : 50;
			checker[i * 4 + 1] = isWhite ? 255 : 50;
			checker[i * 4 + 2] = isWhite ? 255 : 50;
			checker[i * 4 + 3] = 255;
			uint8_t val = rng() % 256;
			noise[i * 4 + 0] = val;
			noise[i * 4 + 1] = val;
			noise[i * 4 + 2] = val;
			noise[i * 4 + 3] = 255;
		}

		albedoTextures.push_back(createTexture(texW, texH, checker.data(), VK_FORMAT_R8G8B8A8_SRGB));
		albedoTextures.push_back(createTexture(texW, texH, noise.data(), VK_FORMAT_R8G8B8A8_SRGB));

		std::vector<uint8_t> white(texW * texH * 4, 255);
		texSpecular = createTexture(texW, texH, white.data(), VK_FORMAT_R8G8B8A8_UNORM);
		std::vector<uint8_t> black(texW * texH * 4, 0);
		texEmissive = createTexture(texW, texH, black.data(), VK_FORMAT_R8G8B8A8_UNORM);

		// Set 0
		VkDescriptorSetLayoutBinding sBindings[3]{};
		sBindings[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
		sBindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
		sBindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
		VkDescriptorSetLayoutCreateInfo sl0{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
		sl0.bindingCount = 3;
		sl0.pBindings = sBindings;
		vkCreateDescriptorSetLayout(ctx.device, &sl0, nullptr, &sceneLayout);

		// Set 1
		VkDescriptorSetLayoutBinding mBindings[4]{};
		mBindings[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
		mBindings[1] = {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
		mBindings[2] = {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
		mBindings[3] = {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
		VkDescriptorSetLayoutCreateInfo sl1{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
		sl1.bindingCount = 4;
		sl1.pBindings = mBindings;
		vkCreateDescriptorSetLayout(ctx.device, &sl1, nullptr, &modelLayout);

		VkDescriptorPoolSize poolSizes[] = {
			{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}, {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2}, {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3}};
		VkDescriptorPoolCreateInfo pInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
		pInfo.maxSets = 2;
		pInfo.poolSizeCount = 4;
		pInfo.pPoolSizes = poolSizes;
		vkCreateDescriptorPool(ctx.device, &pInfo, nullptr, &descPool);

		VkDescriptorSetAllocateInfo alloc0{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descPool, 1, &sceneLayout};
		vkAllocateDescriptorSets(ctx.device, &alloc0, &sceneSet);
		VkDescriptorSetAllocateInfo alloc1{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descPool, 1, &modelLayout};
		vkAllocateDescriptorSets(ctx.device, &alloc1, &modelSet);

		VkDescriptorBufferInfo sceneBi{sceneUboBuffer, 0, sizeof(SceneUBO)};
		VkDescriptorBufferInfo pointBi{pointSsboBuffer, 0, sizeof(PointLight) * MAX_LIGHTS};
		VkDescriptorBufferInfo spotBi{spotSsboBuffer, 0, sizeof(SpotLight) * MAX_LIGHTS};
		VkWriteDescriptorSet w0[3]{};
		w0[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sceneSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &sceneBi, nullptr};
		w0[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sceneSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &pointBi, nullptr};
		w0[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sceneSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &spotBi, nullptr};
		vkUpdateDescriptorSets(ctx.device, 3, w0, 0, nullptr);

		VkDescriptorBufferInfo modelBi{modelUboBuffer, 0, sizeof(ModelUBO)};
		VkDescriptorImageInfo imgAlbedo{textureSampler, albedoTextures[0].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
		VkDescriptorImageInfo imgSpecular{textureSampler, texSpecular.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
		VkDescriptorImageInfo imgEmissive{textureSampler, texEmissive.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

		VkWriteDescriptorSet w1[4]{};
		w1[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, modelSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, nullptr, &modelBi, nullptr};
		w1[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, modelSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &imgAlbedo, nullptr, nullptr};
		w1[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, modelSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &imgSpecular, nullptr, nullptr};
		w1[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, modelSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &imgEmissive, nullptr, nullptr};
		vkUpdateDescriptorSets(ctx.device, 4, w1, 0, nullptr);

		VkDescriptorSetLayout mainSetLayouts[] = {sceneLayout, modelLayout};
		VkPushConstantRange mainPushRange{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ModelPushConstants)};
		VkPipelineLayoutCreateInfo mainPlInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
		mainPlInfo.setLayoutCount = 2;
		mainPlInfo.pSetLayouts = mainSetLayouts;
		mainPlInfo.pushConstantRangeCount = 1;
		mainPlInfo.pPushConstantRanges = &mainPushRange;
		vkCreatePipelineLayout(ctx.device, &mainPlInfo, nullptr, &mainPipeLayout);

		// Визуализаторы используют только Set 0 и свои Push Constants
		VkPushConstantRange vizPushRange{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(VizPushConstants)};
		VkPipelineLayoutCreateInfo vizPlInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
		vizPlInfo.setLayoutCount = 1;
		vizPlInfo.pSetLayouts = &sceneLayout;
		vizPlInfo.pushConstantRangeCount = 1;
		vizPlInfo.pPushConstantRanges = &vizPushRange;
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
		spotLights.push_back({{0.0f, 3.0f, 0.0f, 15.0f}, {0.0f, -1.0f, 0.0f, glm::cos(glm::radians(20.0f))}, {0.2f, 0.5f, 1.0f, 2.0f}, {glm::cos(glm::radians(30.0f)), 0.0f, 0.0f, 0.0f}});

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

		vkDestroySampler(ctx.device, textureSampler, nullptr);

		auto destroyTex = [&](Texture &t)
		{
			if (t.image != VK_NULL_HANDLE)
			{
				vkDestroyImageView(ctx.device, t.view, nullptr);
				vmaDestroyImage(ctx.allocator, t.image, t.alloc);
			}
		};
		for (auto &t : albedoTextures)
			destroyTex(t);
		destroyTex(texSpecular);
		destroyTex(texEmissive);

		vkDestroyCommandPool(ctx.device, tempCommandPool, nullptr);
	}

	void update(double time)
	{
		auto &ctx = graphics::internal::context;
		globalTime = (float)time;

		ImGui::Begin("Advanced Lighting & Textures");
		if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen))
		{
			ImGui::SliderFloat("Yaw", &camYaw, -180.0f, 180.0f);
			ImGui::SliderFloat("Pitch", &camPitch, -89.0f, 89.0f);
			ImGui::SliderFloat("Distance", &camDistance, 1.0f, 20.0f);
			ImGui::SliderFloat("FOV", &fovDeg, 10.0f, 120.0f);
		}
		if (ImGui::CollapsingHeader("Model", ImGuiTreeNodeFlags_DefaultOpen))
		{
			ImGui::SliderFloat("Rotation", &modelRotation, -180.0f, 180.0f);
			ImGui::ColorEdit3("Albedo Color", &matAlbedoColor.x);
			ImGui::SliderFloat("Shininess", &matShininess, 2.0f, 256.0f);
			ImGui::Checkbox("Nontrivial Sampling", &useNontrivialSampling);

			const char *texNames[] = {"Checker", "Noise"};
			if (ImGui::Combo("Albedo Texture", &selectedAlbedoIndex, texNames, IM_ARRAYSIZE(texNames)))
			{
				VkDescriptorImageInfo imgInfo{textureSampler, albedoTextures[selectedAlbedoIndex].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
				VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, modelSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &imgInfo, nullptr, nullptr};
				vkUpdateDescriptorSets(ctx.device, 1, &w, 0, nullptr);
			}
		}

		ImGui::Checkbox("Show Visualizers", &showVisualizers);
		if (ImGui::CollapsingHeader("Ambient", ImGuiTreeNodeFlags_DefaultOpen))
		{
			ImGui::ColorEdit3("Color", &ambientColor.x);
			ImGui::SliderFloat("Intensity", &ambientIntensity, 0.0f, 2.0f);
		}

		// --- UI Точечных источников ---
		if (ImGui::CollapsingHeader("Point Lights", ImGuiTreeNodeFlags_DefaultOpen))
		{
			if (ImGui::Button("Add Point Light") && pointLights.size() < MAX_LIGHTS)
			{
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

		// --- UI Прожекторных источников ---
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
		sUbo.time = globalTime;

		void *data;
		vmaMapMemory(ctx.allocator, sceneUboAlloc, &data);
		std::memcpy(data, &sUbo, sizeof(SceneUBO));
		vmaUnmapMemory(ctx.allocator, sceneUboAlloc);

		if (!pointLights.empty())
		{
			vmaMapMemory(ctx.allocator, pointSsboAlloc, &data);
			std::memcpy(data, pointLights.data(), pointLights.size() * sizeof(PointLight));
			vmaUnmapMemory(ctx.allocator, pointSsboAlloc);
		}
		if (!spotLights.empty())
		{
			vmaMapMemory(ctx.allocator, spotSsboAlloc, &data);
			std::memcpy(data, spotLights.data(), spotLights.size() * sizeof(SpotLight));
			vmaUnmapMemory(ctx.allocator, spotSsboAlloc);
		}

		ModelUBO mUbo{};
		mUbo.model = glm::rotate(glm::mat4(1.0f), glm::radians(modelRotation), glm::vec3(0.0f, 1.0f, 0.0f));
		mUbo.albedo = glm::vec4(matAlbedoColor, 1.0f);
		mUbo.specular = glm::vec4(1.0f);
		mUbo.shininess = matShininess;

		vmaMapMemory(ctx.allocator, modelUboAlloc, &data);
		std::memcpy(data, &mUbo, sizeof(ModelUBO));
		vmaUnmapMemory(ctx.allocator, modelUboAlloc);
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

		// 1. Рисуем пирамиду
		vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, litPipeline);
		uint32_t dynamicOffset = 0;
		VkDescriptorSet sets[] = {sceneSet, modelSet};
		vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, mainPipeLayout, 0, 2, sets, 1, &dynamicOffset);

		ModelPushConstants pc{};
		pc.useNontrivialSampling = useNontrivialSampling ? 1 : 0;
		vkCmdPushConstants(fd.command_buffer, mainPipeLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

		vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &pyramidBuffer, &offset);
		vkCmdDraw(fd.command_buffer, (uint32_t)pyramidVertices.size(), 1, 0, 0);

		// 2. Рисуем визуализаторы света
		if (showVisualizers)
		{
			VizPushConstants vpc{};

			vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, unlitPipeline);
			vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vizPipeLayout, 0, 1, &sceneSet, 0, nullptr);
			vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &cubeBuffer, &offset);

			for (const auto &pl : pointLights)
			{
				vpc.model = glm::translate(glm::mat4(1.0f), glm::vec3(pl.posRadius)) * glm::scale(glm::mat4(1.0f), glm::vec3(0.1f));
				vpc.color = pl.colorIntensity;
				vkCmdPushConstants(fd.command_buffer, vizPipeLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(vpc), &vpc);
				vkCmdDraw(fd.command_buffer, (uint32_t)cubeVertices.size(), 1, 0, 0);
			}

			for (const auto &sl : spotLights)
			{
				vpc.model = glm::translate(glm::mat4(1.0f), glm::vec3(sl.posRadius)) * glm::scale(glm::mat4(1.0f), glm::vec3(0.1f));
				vpc.color = sl.colorIntensity;
				vkCmdPushConstants(fd.command_buffer, vizPipeLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(vpc), &vpc);
				vkCmdDraw(fd.command_buffer, (uint32_t)cubeVertices.size(), 1, 0, 0);
			}

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

				vpc.model = glm::translate(glm::mat4(1.0f), pos) * rot * glm::scale(glm::mat4(1.0f), glm::vec3(baseRadius, baseRadius, range));
				vpc.color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
				vkCmdPushConstants(fd.command_buffer, vizPipeLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(vpc), &vpc);
				vkCmdDraw(fd.command_buffer, (uint32_t)coneLineVertices.size(), 1, 0, 0);
			}
		}

		vkCmdEndRenderPass(fd.command_buffer);
		vkEndCommandBuffer(fd.command_buffer);
	}

} // namespace application