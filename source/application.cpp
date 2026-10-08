#include "application.hpp"

#include <imgui.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <cmath>
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
		};

		struct UniformData
		{
			glm::mat4 mvp;
			glm::vec4 tint; // xyz — множитель цвета, w — запас
		};

		// ---------- Процедурные вершины пирамиды ----------
		// Цвет каждой вершины вычисляется из её локальной позиции:
		//   R = x + 0.5  (слева → справа)
		//   G = z + 0.5  (спереди → назад)
		//   B = y        (основание → вершина)
		std::vector<Vertex> buildPyramidVertices()
		{
			const glm::vec3 positions[] = {
				{-0.5f, 0.0f, -0.5f},
				{0.5f, 0.0f, -0.5f},
				{0.5f, 0.0f, 0.5f},
				{-0.5f, 0.0f, 0.5f},
				{0.0f, 1.0f, 0.0f},
			};
			std::vector<Vertex> verts;
			verts.reserve(5);
			for (const auto &p : positions)
			{
				Vertex v;
				v.pos = p;
				// Небольшой сдвиг, чтобы не было чисто чёрных углов
				v.color = glm::vec3(
					0.2f + 0.8f * (p.x + 0.5f),
					0.2f + 0.8f * (p.z + 0.5f),
					0.2f + 0.8f * p.y);
				verts.push_back(v);
			}
			return verts;
		}

		const std::vector<Vertex> pyramidVertices = buildPyramidVertices();

		const std::vector<uint16_t> pyramidIndices = {
			0,
			1,
			2,
			0,
			2,
			3, // основание
			0,
			4,
			1, // боковая грань 1
			1,
			4,
			2, // боковая грань 2
			2,
			4,
			3, // боковая грань 3
			3,
			4,
			0, // боковая грань 4
		};

		// ---------- Vulkan-объекты ----------
		VkBuffer vertexBuffer = VK_NULL_HANDLE;
		VmaAllocation vertexAllocation = nullptr;
		VkBuffer indexBuffer = VK_NULL_HANDLE;
		VmaAllocation indexAllocation = nullptr;
		VkBuffer uniformBuffer = VK_NULL_HANDLE;
		VmaAllocation uniformAllocation = nullptr;

		VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
		VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
		VkDescriptorSet descriptorSet = VK_NULL_HANDLE;

		VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
		VkPipeline pipeline = VK_NULL_HANDLE;

		// ---------- Состояние UI / анимации ----------
		// Проекция
		bool useOrthographic = false;
		float fovDeg = 45.0f;
		float orthoSize = 3.0f;

		// Трансформации (пользовательские, поверх анимации)
		glm::vec3 userPosition = glm::vec3(0.0f);
		glm::vec3 userRotation = glm::vec3(0.0f); // в градусах
		glm::vec3 userScale = glm::vec3(1.0f);

		// Анимация
		bool animationPlaying = true;
		double animationTime = 0.0;
		double lastUpdateTime = -1.0;
		float animSpeed = 1.0f;	   // множитель угловой скорости орбиты
		float orbitRadius = 1.5f;  // радиус круговой траектории
		float spinSpeed = 90.0f;   // град/сек собственного вращения
		float bobAmplitude = 0.3f; // амплитуда подпрыгивания по Y
		float bobFrequency = 2.0f; // частота подпрыгивания

		// Цвет
		glm::vec3 tint = glm::vec3(1.0f);

		// ---------- Хелперы ----------
		VkShaderModule loadShaderModule(const std::string &path)
		{
			std::ifstream file(path, std::ios::ate | std::ios::binary);
			if (!file.is_open())
			{
				std::cerr << "Failed to open shader: " << path << '\n';
				return VK_NULL_HANDLE;
			}
			const size_t size = static_cast<size_t>(file.tellg());
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

		VkBuffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VmaAllocation &outAllocation)
		{
			VkBufferCreateInfo bufferInfo{};
			bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			bufferInfo.size = size;
			bufferInfo.usage = usage;
			bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

			VmaAllocationCreateInfo allocInfo{};
			allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
			allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
							  VMA_ALLOCATION_CREATE_MAPPED_BIT;

			VkBuffer buffer = VK_NULL_HANDLE;
			vmaCreateBuffer(graphics::internal::context.allocator, &bufferInfo, &allocInfo,
							&buffer, &outAllocation, nullptr);
			return buffer;
		}

		void copyToBuffer(VmaAllocation allocation, const void *data, VkDeviceSize size)
		{
			void *mapped = nullptr;
			vmaMapMemory(graphics::internal::context.allocator, allocation, &mapped);
			std::memcpy(mapped, data, static_cast<size_t>(size));
			vmaUnmapMemory(graphics::internal::context.allocator, allocation);
		}

	} // namespace

	// ============================================================
	// INITIALIZE
	// ============================================================
	bool initialize()
	{
		auto &ctx = graphics::internal::context;

		// --- Вершинный буфер ---
		const VkDeviceSize vbSize = sizeof(Vertex) * pyramidVertices.size();
		vertexBuffer = createBuffer(vbSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexAllocation);
		copyToBuffer(vertexAllocation, pyramidVertices.data(), vbSize);

		// --- Индексный буфер ---
		const VkDeviceSize ibSize = sizeof(uint16_t) * pyramidIndices.size();
		indexBuffer = createBuffer(ibSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indexAllocation);
		copyToBuffer(indexAllocation, pyramidIndices.data(), ibSize);

		// --- Uniform-буфер ---
		uniformBuffer = createBuffer(sizeof(UniformData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, uniformAllocation);

		// --- Descriptor set layout ---
		VkDescriptorSetLayoutBinding uboBinding{};
		uboBinding.binding = 0;
		uboBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uboBinding.descriptorCount = 1;
		uboBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = 1;
		layoutInfo.pBindings = &uboBinding;

		if (vkCreateDescriptorSetLayout(ctx.device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS)
		{
			std::cerr << "Failed to create descriptor set layout\n";
			return false;
		}

		// --- Descriptor pool ---
		VkDescriptorPoolSize poolSize{};
		poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSize.descriptorCount = 1;

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = 1;
		poolInfo.pPoolSizes = &poolSize;
		poolInfo.maxSets = 1;

		if (vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
		{
			std::cerr << "Failed to create descriptor pool\n";
			return false;
		}

		// --- Descriptor set ---
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = descriptorPool;
		allocInfo.descriptorSetCount = 1;
		allocInfo.pSetLayouts = &descriptorSetLayout;

		if (vkAllocateDescriptorSets(ctx.device, &allocInfo, &descriptorSet) != VK_SUCCESS)
		{
			std::cerr << "Failed to allocate descriptor set\n";
			return false;
		}

		VkDescriptorBufferInfo bufferInfo{};
		bufferInfo.buffer = uniformBuffer;
		bufferInfo.offset = 0;
		bufferInfo.range = sizeof(UniformData);

		VkWriteDescriptorSet write{};
		write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		write.dstSet = descriptorSet;
		write.dstBinding = 0;
		write.descriptorCount = 1;
		write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		write.pBufferInfo = &bufferInfo;

		vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);

		// --- Pipeline layout ---
		VkPipelineLayoutCreateInfo plInfo{};
		plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		plInfo.setLayoutCount = 1;
		plInfo.pSetLayouts = &descriptorSetLayout;

		if (vkCreatePipelineLayout(ctx.device, &plInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
		{
			std::cerr << "Failed to create pipeline layout\n";
			return false;
		}

		// --- Шейдеры ---
		VkShaderModule vertModule = loadShaderModule("shaders/pyramid.vert.spv");
		VkShaderModule fragModule = loadShaderModule("shaders/pyramid.frag.spv");
		if (vertModule == VK_NULL_HANDLE || fragModule == VK_NULL_HANDLE)
		{
			return false;
		}

		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vertModule;
		stages[0].pName = "main";
		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = fragModule;
		stages[1].pName = "main";

		// --- Vertex input ---
		VkVertexInputBindingDescription bindingDesc{};
		bindingDesc.binding = 0;
		bindingDesc.stride = sizeof(Vertex);
		bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		VkVertexInputAttributeDescription attrDesc[2]{};
		attrDesc[0].location = 0;
		attrDesc[0].binding = 0;
		attrDesc[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attrDesc[0].offset = offsetof(Vertex, pos);
		attrDesc[1].location = 1;
		attrDesc[1].binding = 0;
		attrDesc[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attrDesc[1].offset = offsetof(Vertex, color);

		VkPipelineVertexInputStateCreateInfo vertexInput{};
		vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInput.vertexBindingDescriptionCount = 1;
		vertexInput.pVertexBindingDescriptions = &bindingDesc;
		vertexInput.vertexAttributeDescriptionCount = 2;
		vertexInput.pVertexAttributeDescriptions = attrDesc;

		VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

		VkPipelineViewportStateCreateInfo viewportState{};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.scissorCount = 1;

		VkPipelineRasterizationStateCreateInfo raster{};
		raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		raster.polygonMode = VK_POLYGON_MODE_FILL;
		raster.cullMode = VK_CULL_MODE_NONE;
		raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		raster.lineWidth = 1.0f;

		VkPipelineMultisampleStateCreateInfo multisample{};
		multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineDepthStencilStateCreateInfo depthStencil{};
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil.depthTestEnable = VK_TRUE;
		depthStencil.depthWriteEnable = VK_TRUE;
		depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

		VkPipelineColorBlendAttachmentState blendAttach{};
		blendAttach.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
									 VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		VkPipelineColorBlendStateCreateInfo blend{};
		blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blend.attachmentCount = 1;
		blend.pAttachments = &blendAttach;

		VkDynamicState dynStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
		VkPipelineDynamicStateCreateInfo dynamicState{};
		dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicState.dynamicStateCount = 2;
		dynamicState.pDynamicStates = dynStates;

		VkGraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = 2;
		pipelineInfo.pStages = stages;
		pipelineInfo.pVertexInputState = &vertexInput;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &raster;
		pipelineInfo.pMultisampleState = &multisample;
		pipelineInfo.pDepthStencilState = &depthStencil;
		pipelineInfo.pColorBlendState = &blend;
		pipelineInfo.pDynamicState = &dynamicState;
		pipelineInfo.layout = pipelineLayout;
		pipelineInfo.renderPass = ctx.render_pass;
		pipelineInfo.subpass = 0;

		const VkResult result = vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);

		vkDestroyShaderModule(ctx.device, vertModule, nullptr);
		vkDestroyShaderModule(ctx.device, fragModule, nullptr);

		if (result != VK_SUCCESS)
		{
			std::cerr << "Failed to create graphics pipeline\n";
			return false;
		}

		return true;
	}

	// ============================================================
	// SHUTDOWN
	// ============================================================
	void shutdown()
	{
		auto &ctx = graphics::internal::context;
		vkQueueWaitIdle(ctx.graphics_queue);

		if (pipeline != VK_NULL_HANDLE)
		{
			vkDestroyPipeline(ctx.device, pipeline, nullptr);
			pipeline = VK_NULL_HANDLE;
		}
		if (pipelineLayout != VK_NULL_HANDLE)
		{
			vkDestroyPipelineLayout(ctx.device, pipelineLayout, nullptr);
			pipelineLayout = VK_NULL_HANDLE;
		}
		if (descriptorPool != VK_NULL_HANDLE)
		{
			vkDestroyDescriptorPool(ctx.device, descriptorPool, nullptr);
			descriptorPool = VK_NULL_HANDLE;
		}
		if (descriptorSetLayout != VK_NULL_HANDLE)
		{
			vkDestroyDescriptorSetLayout(ctx.device, descriptorSetLayout, nullptr);
			descriptorSetLayout = VK_NULL_HANDLE;
		}
		if (uniformBuffer != VK_NULL_HANDLE)
		{
			vmaDestroyBuffer(ctx.allocator, uniformBuffer, uniformAllocation);
			uniformBuffer = VK_NULL_HANDLE;
			uniformAllocation = nullptr;
		}
		if (indexBuffer != VK_NULL_HANDLE)
		{
			vmaDestroyBuffer(ctx.allocator, indexBuffer, indexAllocation);
			indexBuffer = VK_NULL_HANDLE;
			indexAllocation = nullptr;
		}
		if (vertexBuffer != VK_NULL_HANDLE)
		{
			vmaDestroyBuffer(ctx.allocator, vertexBuffer, vertexAllocation);
			vertexBuffer = VK_NULL_HANDLE;
			vertexAllocation = nullptr;
		}
	}

	// ============================================================
	// UPDATE
	// ============================================================
	void update(double time)
	{
		auto &ctx = graphics::internal::context;

		// ----- Время анимации -----
		if (lastUpdateTime < 0.0)
		{
			lastUpdateTime = time;
		}
		double dt = time - lastUpdateTime;
		lastUpdateTime = time;
		if (dt > 0.1)
			dt = 0.1; // защита от больших скачков
		if (animationPlaying)
		{
			animationTime += dt;
		}

		// ----- Интерфейс -----
		ImGui::Begin("Pyramid Controls");

		// (1) Проекция
		ImGui::SeparatorText("Projection");
		ImGui::Checkbox("Orthographic", &useOrthographic);
		if (useOrthographic)
		{
			ImGui::SliderFloat("Ortho size", &orthoSize, 0.5f, 10.0f);
		}
		else
		{
			ImGui::SliderFloat("FOV", &fovDeg, 10.0f, 120.0f);
		}

		// (2) Трансформации
		ImGui::SeparatorText("Transform");
		ImGui::DragFloat3("Position", &userPosition.x, 0.01f);
		ImGui::SliderFloat3("Rotation", &userRotation.x, -180.0f, 180.0f, "%.1f deg");
		ImGui::DragFloat3("Scale", &userScale.x, 0.01f, 0.01f, 10.0f);

		// (3) Анимация
		ImGui::SeparatorText("Animation");
		ImGui::Checkbox("Playing", &animationPlaying);
		ImGui::SameLine();
		if (ImGui::Button("Reset"))
		{
			animationTime = 0.0;
		}
		ImGui::SliderFloat("Speed", &animSpeed, 0.0f, 5.0f);
		ImGui::SliderFloat("Orbit radius", &orbitRadius, 0.0f, 5.0f);
		ImGui::SliderFloat("Spin speed", &spinSpeed, 0.0f, 720.0f, "%.0f deg/s");
		ImGui::SliderFloat("Bob amplitude", &bobAmplitude, 0.0f, 2.0f);
		ImGui::SliderFloat("Bob frequency", &bobFrequency, 0.0f, 10.0f);

		// (4) Цвет
		ImGui::SeparatorText("Color");
		ImGui::ColorEdit3("Tint", &tint.x);

		ImGui::End();

		// ----- Расчёт матрицы -----
		const float aspect = static_cast<float>(ctx.swapchain_extent.width) /
							 static_cast<float>(ctx.swapchain_extent.height);

		// Позиция по сложной траектории
		const float phase = static_cast<float>(animationTime) * animSpeed;
		const float animX = orbitRadius * std::cos(phase);
		const float animZ = orbitRadius * std::sin(phase);
		const float animY = bobAmplitude * std::sin(phase * bobFrequency);
		const float spinAngle = static_cast<float>(animationTime) * spinSpeed;

		const glm::vec3 animatedPos = glm::vec3(animX, animY, animZ);

		// Модельная матрица
		glm::mat4 model = glm::mat4(1.0f);
		model = glm::translate(model, animatedPos + userPosition);
		model = glm::rotate(model, glm::radians(userRotation.x), glm::vec3(1, 0, 0));
		model = glm::rotate(model, glm::radians(userRotation.y + spinAngle), glm::vec3(0, 1, 0));
		model = glm::rotate(model, glm::radians(userRotation.z), glm::vec3(0, 0, 1));
		model = glm::scale(model, userScale);

		// Вид
		const glm::mat4 view = glm::lookAt(
			glm::vec3(0.0f, 2.5f, 6.0f),
			glm::vec3(0.0f, 0.5f, 0.0f),
			glm::vec3(0.0f, 1.0f, 0.0f));

		// Проекция
		glm::mat4 proj;
		if (useOrthographic)
		{
			proj = glm::ortho(
				-orthoSize * aspect, orthoSize * aspect,
				-orthoSize, orthoSize,
				-50.0f, 50.0f);
		}
		else
		{
			proj = glm::perspective(glm::radians(fovDeg), aspect, 0.1f, 100.0f);
		}

		// ----- Загрузка UBO -----
		UniformData ubo{};
		ubo.mvp = proj * view * model;
		ubo.tint = glm::vec4(tint, 1.0f);

		copyToBuffer(uniformAllocation, &ubo, sizeof(ubo));
	}

	// ============================================================
	// RENDER
	// ============================================================
	void render(const graphics::internal::FrameData &fd)
	{
		auto &ctx = graphics::internal::context;

		vkResetCommandBuffer(fd.command_buffer, 0);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(fd.command_buffer, &beginInfo);

		VkClearValue clearValues[2]{};
		clearValues[0].color = {{0.1f, 0.1f, 0.15f, 1.0f}};
		clearValues[1].depthStencil = {1.0f, 0};

		VkRenderPassBeginInfo rpBegin{};
		rpBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		rpBegin.renderPass = ctx.render_pass;
		rpBegin.framebuffer = fd.framebuffer;
		rpBegin.renderArea = {{0, 0}, ctx.swapchain_extent};
		rpBegin.clearValueCount = 2;
		rpBegin.pClearValues = clearValues;

		vkCmdBeginRenderPass(fd.command_buffer, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);

		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = static_cast<float>(ctx.swapchain_extent.width);
		viewport.height = static_cast<float>(ctx.swapchain_extent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);

		VkRect2D scissor{{0, 0}, ctx.swapchain_extent};
		vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

		vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

		VkDeviceSize offset = 0;
		vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertexBuffer, &offset);
		vkCmdBindIndexBuffer(fd.command_buffer, indexBuffer, 0, VK_INDEX_TYPE_UINT16);
		vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
								pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

		vkCmdDrawIndexed(fd.command_buffer, static_cast<uint32_t>(pyramidIndices.size()), 1, 0, 0, 0);

		vkCmdEndRenderPass(fd.command_buffer);
		vkEndCommandBuffer(fd.command_buffer);
	}

} // namespace application