#define TINYGLTF3_IMPLEMENTATION
#define TINYGLTF3_ENABLE_FS          // Enable filesystem support (optional)

#include "tiny_gltf_v3.h" 

#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/quaternion.hpp> // For handling rotations if needed
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp> // For glm::value_ptr

#include <fstream>
#include <iostream>
#include <string>
#include <filesystem> // C++17 filesystem library for path manipulation
#include <limits>
#include <cstdint> // For fixed-width integer types like uint32_t
#include <algorithm>
#include <cmath>
#include <stdexcept>

using namespace std;

namespace
{
    struct AccessorView
    {
        const uint8_t* base = nullptr; // Pointer to the start of the buffer data
        uint64_t count = 0; // Number of elements in the accessor（e.g., number of vertices)
        int32_t componentType = 0; // GLTF component type (e.g., GL_FLOAT, GL_UNSIGNED_SHORT)
        int32_t type = 0; // GLTF type (e.g., VEC3, SCALAR)
        int32_t stride = 0; // Byte stride between elements (0 means tightly packed)
    };

    [[noreturn]] void FailSceneLoad(const std::string& message)
    {
        throw std::runtime_error(message);
    }

    Material MakeFallbackMaterial()
    {
        Material fallback{};
        fallback.color = glm::vec3(0.8f);
        fallback.specular.exponent = 0.0f;
        fallback.specular.color = glm::vec3(1.0f);
        fallback.hasReflective = 0.0f;
        fallback.hasRefractive = 0.0f;
        fallback.indexOfRefraction = 1.0f;
        fallback.emittance = 0.0f;
        return fallback;
    }

    int EnsureFallbackMaterial(std::vector<Material>& materials)
    {
        materials.push_back(MakeFallbackMaterial());
        return static_cast<int>(materials.size()) - 1;
    }

    Material ConvertGLTFMaterial(const tg3_material& gltfMaterial)
    {
        Material material = MakeFallbackMaterial();

        const glm::vec3 baseColor(
            static_cast<float>(gltfMaterial.pbr_metallic_roughness.base_color_factor[0]),
            static_cast<float>(gltfMaterial.pbr_metallic_roughness.base_color_factor[1]),
            static_cast<float>(gltfMaterial.pbr_metallic_roughness.base_color_factor[2]));
        const glm::vec3 emissiveColor(
            static_cast<float>(gltfMaterial.emissive_factor[0]),
            static_cast<float>(gltfMaterial.emissive_factor[1]),
            static_cast<float>(gltfMaterial.emissive_factor[2]));

        const float emissiveStrength = std::max(emissiveColor.r, std::max(emissiveColor.g, emissiveColor.b));

        material.color = baseColor;
        material.specular.color = baseColor;
        material.specular.exponent = static_cast<float>(gltfMaterial.pbr_metallic_roughness.roughness_factor);

        if (emissiveStrength > 0.0f)
        {
            material.color = emissiveColor / emissiveStrength;
            material.emittance = emissiveStrength;
        }

        return material;
    }

    struct SceneBounds
    {
        bool valid = false;
        glm::vec3 min = glm::vec3(0.0f);
        glm::vec3 max = glm::vec3(0.0f);

        void expand(const glm::vec3& point)
        {
            if (!valid)
            {
                min = point;
                max = point;
                valid = true;
                return;
            }

            min = glm::min(min, point);
            max = glm::max(max, point);
        }

        void expand(const glm::vec3& boundsMin, const glm::vec3& boundsMax)
        {
            expand(boundsMin);
            expand(boundsMax);
        }

        glm::vec3 center() const
        {
            return valid ? 0.5f * (min + max) : glm::vec3(0.0f);
        }

        glm::vec3 size() const
        {
            return valid ? (max - min) : glm::vec3(0.0f);
        }

        float radius() const
        {
            return valid ? std::max(glm::length(size()) * 0.5f, 0.5f) : 1.0f;
        }
    };

    struct GLTFNodeTransform
    {
        int32_t nodeIndex = -1;
        glm::mat4 world = glm::mat4(1.0f);
    };

    struct ImportedCameraSpec
    {
        bool valid = false;
        glm::vec3 position = glm::vec3(0.0f);
        glm::vec3 lookAt = glm::vec3(0.0f, 0.0f, -1.0f);
        glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
        float fovyDegrees = 45.0f;
        float aspectRatio = 1.0f;
    };

    std::string TG3ToStdString(const tg3_str& value)
    {
        if (!value.data || value.len == 0)
        {
            return std::string();
        }
        return std::string(value.data, value.len);
    }

    void DumpGLTFErrors(const tg3_error_stack& errors)
    {
        const uint32_t count = tg3_errors_count(&errors);
        for (uint32_t i = 0; i < count; ++i)
        {
            const tg3_error_entry* entry = tg3_errors_get(&errors, i);
            if (!entry)
            {
                continue;
            }

            cerr << "GLTF error: "
                 << (entry->message ? entry->message : "(no message)");
            if (entry->json_path)
            {
                cerr << " at " << entry->json_path;
            }
            cerr << endl;
        }
    }

    bool ParseGLTFFile(const std::string& gltfPath, tg3_model& model, tg3_error_stack& errors)
    {
        model = tg3_model{};
        model.default_scene = -1;

        errors = tg3_error_stack{};
        tg3_error_stack_init(&errors);

        tg3_parse_options options{};
        tg3_parse_options_init(&options);

        const tg3_error_code parseCode = tg3_parse_file(
            &model,
            &errors,
            gltfPath.c_str(),
            static_cast<uint32_t>(gltfPath.size()),
            &options);

        if (parseCode != TG3_OK || tg3_errors_has_error(&errors))
        {
            cerr << "Failed to parse GLTF file: " << gltfPath << " (code " << parseCode << ")" << endl;
            DumpGLTFErrors(errors);
            return false;
        }

        if (model.nodes_count == 0)
        {
            cerr << "GLTF has no nodes: " << gltfPath << endl;
            return false;
        }

        return true;
    }

    Material MakeEmissiveMaterial(const glm::vec3& color, float emittance)
    {
        Material material = MakeFallbackMaterial();
        material.color = color;
        material.specular.color = color;
        material.emittance = emittance;
        return material;
    }
    
}

/**
 * Helper function to build an AccessorView from a GLTF model and accessor index. 
 * This function reads the accessor's properties and sets up the view accordingly. 
 * It returns true on success, false on failure 
 * (e.g., if the accessor index is invalid or if the buffer data cannot be accessed).
 * @param model The GLTF model containing the accessor and buffer data.
 * @param accessorIndex The index of the accessor to build the view for.
 * @param out The AccessorView to populate with the accessor's data.
 * @return True if the AccessorView was successfully built, false otherwise.
 */
static bool BuildAccessorView(const tg3_model* model, int32_t accessorIndex, AccessorView& out)
{
    // Validate the accessor index and model pointer
    if(!model || accessorIndex < 0 || accessorIndex >= static_cast<int32_t>(model->accessors_count))
    {
        cerr << "Invalid accessor index: " << accessorIndex << endl;
        return false;
    }

    // Get the accessor from the model
    const tg3_accessor& acc = model->accessors[accessorIndex];

    // Validate the buffer view index
    if(acc.buffer_view < 0 || acc.buffer_view >= static_cast<int32_t>(model->buffer_views_count))
    {
        cerr << "Invalid buffer view index: " << acc.buffer_view << endl;
        return false;
    }

    // Get the buffer view from the model
    const tg3_buffer_view& bv = model->buffer_views[acc.buffer_view];

    // Validate the buffer index
    if(bv.buffer < 0 || bv.buffer >= static_cast<int32_t>(model->buffers_count))
    {
        cerr << "Invalid buffer index: " << bv.buffer << endl;
        return false;
    }

    // Get the buffer from the model
    const tg3_buffer& buf = model->buffers[bv.buffer];

    const int32_t compSize = tg3_component_size(acc.component_type);
    const int32_t compNum = tg3_num_components(acc.type);
    const int32_t stride = tg3_accessor_byte_stride(&acc, &bv);

    if(compSize < 0 || compNum < 0 || stride < 0)
    {
        cerr << "Invalid accessor type or component type: " << acc.type << ", " << acc.component_type << endl;
        return false;
    }

    const uint64_t elemBytes = (uint64_t)compSize * (uint64_t)compNum;
    const uint64_t baseOffset = bv.byte_offset + acc.byte_offset;
    const uint64_t neededBytes = acc.count == 0 ? 0 : ((acc.count - 1) * (uint64_t)stride + elemBytes);

    if(baseOffset + neededBytes > buf.data.count)
    {
        cerr << "Accessor byte range exceeds buffer size: " << baseOffset + neededBytes << " > " << buf.data.count << endl;
        return false;
    }

    out.base = buf.data.data + baseOffset;
    out.count = acc.count;
    out.componentType = acc.component_type;
    out.type = acc.type;
    out.stride = stride;
    return true;

}

/**
 * Helper function to read a vec3 of 32-bit floats from an AccessorView at a given index. 
 * This function calculates the byte offset for the specified index, reads the three float components, and returns them as a glm::vec3. 
 * It assumes that the AccessorView is properly set up for vec3 data (i.e., componentType is GL_FLOAT and type is VEC3).
 * @param view The AccessorView containing the buffer data and layout information.
 * @param index The index of the element to read (0-based).
 * @return A glm::vec3 containing the three float components read from the buffer.  
 */
static glm::vec3 ReadVec3F32(const AccessorView& view, uint64_t index)
{
    const uint8_t* elemPtr = view.base + index * view.stride;
    const float* floatPtr = reinterpret_cast<const float*>(elemPtr);
    return glm::vec3(floatPtr[0], floatPtr[1], floatPtr[2]);
}

/**
 * Helper function to read an index from an AccessorView at a given index. 
 * This function calculates the byte offset for the specified index, reads the index value based on the component type (e.g., unsigned byte, unsigned short, unsigned int), and returns it as a uint32_t. 
 * It assumes that the AccessorView is properly set up for index data (i.e., componentType is one of the unsigned integer types and type is SCALAR).
 * @param view The AccessorView containing the buffer data and layout information.
 * @param index The index of the element to read (0-based).
 * @return A uint32_t containing the index value read from the buffer.  
 */
static uint32_t ReadIndex(const AccessorView& view, uint64_t index)
{
    const uint8_t* ptr = view.base + index * (uint64_t)view.stride;
    switch(view.componentType)
    {
        case TG3_COMPONENT_TYPE_UNSIGNED_BYTE:
            return static_cast<uint32_t>(*reinterpret_cast<const uint8_t*>(ptr));
        case TG3_COMPONENT_TYPE_UNSIGNED_SHORT:
            return static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(ptr));
        case TG3_COMPONENT_TYPE_UNSIGNED_INT:
            return *reinterpret_cast<const uint32_t*>(ptr);
        default:
            cerr << "Unsupported index component type: " << view.componentType << endl;
            return 0; 
    }
}

/**
 * Helper function to find the index of a specific attribute in a GLTF primitive's attributes. 
 * This function iterates through the primitive's attributes and compares the attribute name with the provided name using tg3_str_equal. 
 * If a match is found, it returns the corresponding accessor index. 
 * If no match is found after iterating through all attributes, it returns -1 to indicate that the attribute was not found.
 * @param prim The GLTF primitive containing the attributes to search through.
 * @param attributeName The name of the attribute to find (e.g., "POSITION", "NORMAL", "TEXCOORD_0").
 * @return The index of the accessor corresponding to the specified attribute if found, or -1 if the attribute is not found in the primitive.   
 */
static int32_t FindPrimitiveAttribute(const tg3_primitive& prim, const char* attributeName)
{
    // Iterate through the primitive's attributes and compare the attribute name with the provided name using tg3_str_equal
    for(int32_t i = 0; i < prim.attributes_count; ++i)
    {
        if(tg3_str_equals_cstr(prim.attributes[i].key, attributeName))
        {
            return prim.attributes[i].value; 
        }
    }
    return -1; 
}



/**
 * Helper function to compute the local transformation matrix for a GLTF node. 
 * This function checks if the node has a matrix defined. If it does, it constructs a glm::mat4 from the node's matrix data. 
 * If the node does not have a matrix, it constructs the local transformation matrix by combining the 
 * translation, rotation (as a quaternion), and scale components of the node. 
 * The resulting matrix is returned as a glm::mat4. 
 * @param node The GLTF node for which to compute the local transformation matrix.
 * @return A glm::mat4 representing the local transformation of the node, either from the node's matrix or from its translation, rotation, and scale components.    
 */
static glm::mat4 NodeLocalMatrix(const tg3_node& node)
{
    if (node.has_matrix)
    {
        glm::mat4 m(1.0f);
        for (int c = 0; c < 4; ++c)
        {
            for (int r = 0; r < 4; ++r)
            {
                m[c][r] = (float)node.matrix[c * 4 + r];
            }
        }
        return m;
    }

    glm::vec3 t((float)node.translation[0], (float)node.translation[1], (float)node.translation[2]);
    glm::vec3 s((float)node.scale[0], (float)node.scale[1], (float)node.scale[2]);
    glm::quat q((float)node.rotation[3], (float)node.rotation[0], (float)node.rotation[1], (float)node.rotation[2]);

    return glm::translate(glm::mat4(1.0f), t) * glm::mat4_cast(q) * glm::scale(glm::mat4(1.0f), s);
}

static std::vector<int32_t> CollectGLTFRootNodes(const tg3_model& model)
{
    std::vector<int32_t> rootNodes;

    if (model.scenes_count > 0)
    {
        int32_t sceneIndex = model.default_scene;
        if (sceneIndex < 0 || sceneIndex >= static_cast<int32_t>(model.scenes_count))
        {
            sceneIndex = 0;
        }

        const tg3_scene& scene = model.scenes[sceneIndex];
        rootNodes.reserve(scene.nodes_count);
        for (uint32_t i = 0; i < scene.nodes_count; ++i)
        {
            rootNodes.push_back(scene.nodes[i]);
        }
        return rootNodes;
    }

    std::vector<uint8_t> isChild(model.nodes_count, 0);
    for (uint32_t i = 0; i < model.nodes_count; ++i)
    {
        const tg3_node& node = model.nodes[i];
        for (uint32_t c = 0; c < node.children_count; ++c)
        {
            const int32_t child = node.children[c];
            if (child >= 0 && child < static_cast<int32_t>(model.nodes_count))
            {
                isChild[child] = 1;
            }
        }
    }

    for (uint32_t i = 0; i < model.nodes_count; ++i)
    {
        if (!isChild[i])
        {
            rootNodes.push_back(static_cast<int32_t>(i));
        }
    }

    if (rootNodes.empty())
    {
        rootNodes.push_back(0);
    }

    return rootNodes;
}

static std::vector<GLTFNodeTransform> CollectGLTFNodeWorldTransforms(const tg3_model& model, const glm::mat4& objectTransform)
{
    struct NodeWorkItem
    {
        int32_t nodeIndex;
        glm::mat4 parentWorld;
    };

    std::vector<GLTFNodeTransform> nodeWorlds;
    std::vector<NodeWorkItem> workStack;
    const std::vector<int32_t> rootNodes = CollectGLTFRootNodes(model);

    workStack.reserve(rootNodes.size() + 16);
    for (int i = static_cast<int>(rootNodes.size()) - 1; i >= 0; --i)
    {
        workStack.push_back(NodeWorkItem{ rootNodes[i], objectTransform });
    }

    while (!workStack.empty())
    {
        const NodeWorkItem work = workStack.back();
        workStack.pop_back();

        if (work.nodeIndex < 0 || work.nodeIndex >= static_cast<int32_t>(model.nodes_count))
        {
            continue;
        }

        const tg3_node& node = model.nodes[work.nodeIndex];
        const glm::mat4 nodeWorld = work.parentWorld * NodeLocalMatrix(node);
        nodeWorlds.push_back(GLTFNodeTransform{ work.nodeIndex, nodeWorld });

        for (int c = static_cast<int>(node.children_count) - 1; c >= 0; --c)
        {
            const int32_t child = node.children[c];
            if (child < 0 || child >= static_cast<int32_t>(model.nodes_count))
            {
                continue;
            }
            workStack.push_back(NodeWorkItem{ child, nodeWorld });
        }
    }

    return nodeWorlds;
}

static glm::ivec2 BuildResolutionFromAspect(float aspectRatio)
{
    const float clampedAspect = (aspectRatio > 1e-4f) ? aspectRatio : 1.0f;
    const int height = 800;
    const int width = std::max(1, static_cast<int>(std::round(height * clampedAspect)));
    return glm::ivec2(width, height);
}

static void InitializeRenderState(RenderState& state,
                                  const glm::ivec2& resolution,
                                  float fovyDegrees,
                                  unsigned int iterations,
                                  int traceDepth,
                                  const std::string& imageName,
                                  const glm::vec3& position,
                                  const glm::vec3& lookAt,
                                  const glm::vec3& upHint)
{
    Camera& camera = state.camera;
    camera.resolution = glm::max(resolution, glm::ivec2(1));
    camera.position = position;
    camera.lookAt = lookAt;

    glm::vec3 view = lookAt - position;
    if (glm::length(view) <= 1e-6f)
    {
        view = glm::vec3(0.0f, 0.0f, -1.0f);
    }
    camera.view = glm::normalize(view);

    glm::vec3 up = upHint;
    if (glm::length(up) <= 1e-6f)
    {
        up = glm::vec3(0.0f, 1.0f, 0.0f);
    }
    up = glm::normalize(up);

    glm::vec3 right = glm::cross(camera.view, up);
    if (glm::length(right) <= 1e-6f)
    {
        const glm::vec3 fallbackAxis =
            (glm::abs(camera.view.y) < 0.99f) ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
        right = glm::cross(camera.view, fallbackAxis);
    }
    camera.right = glm::normalize(right);
    camera.up = glm::normalize(glm::cross(camera.right, camera.view));

    const float fovy = std::max(fovyDegrees, 1.0f);
    const float yscaled = tan(fovy * (PI / 180.0f));
    const float xscaled = (yscaled * camera.resolution.x) / static_cast<float>(camera.resolution.y);
    const float fovx = (atan(xscaled) * 180.0f) / PI;
    camera.fov = glm::vec2(fovx, fovy);
    camera.pixelLength = glm::vec2(
        2.0f * xscaled / static_cast<float>(camera.resolution.x),
        2.0f * yscaled / static_cast<float>(camera.resolution.y));

    state.iterations = iterations;
    state.traceDepth = traceDepth;
    state.imageName = imageName;
    state.image.assign(camera.resolution.x * camera.resolution.y, glm::vec3(0.0f));
}

static ImportedCameraSpec ExtractGLTFCamera(const tg3_model& model,
                                            const std::vector<GLTFNodeTransform>& nodeWorlds)
{
    ImportedCameraSpec spec{};

    for (const GLTFNodeTransform& nodeTransform : nodeWorlds)
    {
        const tg3_node& node = model.nodes[nodeTransform.nodeIndex];
        if (node.camera < 0 || node.camera >= static_cast<int32_t>(model.cameras_count))
        {
            continue;
        }

        const tg3_camera& camera = model.cameras[node.camera];
        const glm::mat3 basis(nodeTransform.world);

        spec.position = glm::vec3(nodeTransform.world * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
        spec.lookAt = spec.position + glm::normalize(basis * glm::vec3(0.0f, 0.0f, -1.0f));
        spec.up = glm::normalize(basis * glm::vec3(0.0f, 1.0f, 0.0f));
        spec.fovyDegrees = 45.0f;
        spec.aspectRatio = 1.0f;
        spec.valid = true;

        if (tg3_str_equals_cstr(camera.type, "perspective"))
        {
            spec.fovyDegrees = 0.5f * glm::degrees(static_cast<float>(camera.perspective.yfov));
            if (camera.perspective.aspect_ratio > 1e-6)
            {
                spec.aspectRatio = static_cast<float>(camera.perspective.aspect_ratio);
            }
        }
        else if (tg3_str_equals_cstr(camera.type, "orthographic"))
        {
            if (camera.orthographic.ymag > 1e-6)
            {
                spec.aspectRatio = static_cast<float>(camera.orthographic.xmag / camera.orthographic.ymag);
            }
            cerr << "Approximating orthographic glTF camera as a 45-degree perspective camera." << endl;
        }

        return spec;
    }

    return spec;
}

static std::string BuildImageBaseName(const std::string& sourcePath)
{
    const std::filesystem::path path(sourcePath);
    const std::string stem = path.stem().string();
    return stem.empty() ? "render" : stem;
}

static void FinalizeGeom(Geom& geom)
{
    geom.transform = utilityCore::buildTransformationMatrix(geom.translation, geom.rotation, geom.scale);
    geom.inverseTransform = glm::inverse(geom.transform);
    geom.invTranspose = glm::inverseTranspose(geom.transform);
}

static int AddMaterial(Scene& scene, const Material& material)
{
    scene.materials.push_back(material);
    return static_cast<int>(scene.materials.size()) - 1;
}

static void AddLightSphere(Scene& scene,
                           const glm::vec3& position,
                           float radius,
                           const glm::vec3& color,
                           float emittance)
{
    Geom lightGeom{};
    lightGeom.type = SPHERE;
    lightGeom.materialid = AddMaterial(scene, MakeEmissiveMaterial(color, emittance));
    lightGeom.translation = position;
    lightGeom.rotation = glm::vec3(0.0f);
    lightGeom.scale = glm::vec3(std::max(radius * 2.0f, 0.02f));
    FinalizeGeom(lightGeom);
    scene.geoms.push_back(lightGeom);
}

static void AddLightCube(Scene& scene,
                         const glm::vec3& position,
                         const glm::vec3& scale,
                         const glm::vec3& color,
                         float emittance)
{
    Geom lightGeom{};
    lightGeom.type = CUBE;
    lightGeom.materialid = AddMaterial(scene, MakeEmissiveMaterial(color, emittance));
    lightGeom.translation = position;
    lightGeom.rotation = glm::vec3(0.0f);
    lightGeom.scale = glm::max(scale, glm::vec3(0.02f));
    FinalizeGeom(lightGeom);
    scene.geoms.push_back(lightGeom);
}

static int ImportGLTFLights(Scene& scene,
                            const tg3_model& model,
                            const std::vector<GLTFNodeTransform>& nodeWorlds,
                            const SceneBounds& sceneBounds)
{
    const glm::vec3 boundsCenter = sceneBounds.center();
    const float sceneRadius = sceneBounds.radius();
    const float pointProxyRadius = std::max(sceneRadius * 0.05f, 0.05f);
    int importedLightCount = 0;

    for (const GLTFNodeTransform& nodeTransform : nodeWorlds)
    {
        const tg3_node& node = model.nodes[nodeTransform.nodeIndex];
        if (node.light < 0)
        {
            continue;
        }

        if (node.light >= static_cast<int32_t>(model.lights_count))
        {
            cerr << "Skipping glTF light with out-of-range index " << node.light << endl;
            continue;
        }

        const tg3_light& light = model.lights[node.light];
        glm::vec3 color(
            static_cast<float>(light.color[0]),
            static_cast<float>(light.color[1]),
            static_cast<float>(light.color[2]));
        if (glm::length(color) <= 1e-6f)
        {
            color = glm::vec3(1.0f);
        }

        const float emittance = std::max(static_cast<float>(light.intensity), 1.0f);
        const glm::vec3 position = glm::vec3(nodeTransform.world * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
        glm::vec3 forward = glm::vec3(nodeTransform.world * glm::vec4(0.0f, 0.0f, -1.0f, 0.0f));
        if (glm::length(forward) <= 1e-6f)
        {
            forward = glm::vec3(0.0f, 0.0f, -1.0f);
        }
        forward = glm::normalize(forward);

        if (tg3_str_equals_cstr(light.type, "directional"))
        {
            const glm::vec3 proxyPosition = boundsCenter - forward * std::max(sceneRadius * 3.0f, 2.0f);
            AddLightSphere(scene, proxyPosition, std::max(sceneRadius * 0.75f, 0.5f), color, emittance);
        }
        else
        {
            if (tg3_str_equals_cstr(light.type, "spot"))
            {
                cerr << "Approximating glTF spot light '" << TG3ToStdString(light.name)
                     << "' as an omni-directional emissive sphere." << endl;
            }
            AddLightSphere(scene, position, pointProxyRadius, color, emittance);
        }

        ++importedLightCount;
    }

    return importedLightCount;
}

static bool SceneHasEmissiveMaterial(const Scene& scene)
{
    return std::any_of(
        scene.materials.begin(),
        scene.materials.end(),
        [](const Material& material) { return material.emittance > 0.0f; });
}

static void AddDefaultSceneLight(Scene& scene, const SceneBounds& sceneBounds)
{
    const glm::vec3 center = sceneBounds.center();
    const glm::vec3 size = glm::max(sceneBounds.size(), glm::vec3(1.0f));
    const glm::vec3 scale(
        std::max(size.x * 0.3f, 0.5f),
        std::max(size.y * 0.02f, 0.05f),
        std::max(size.z * 0.3f, 0.5f));
    const glm::vec3 position(
        center.x,
        sceneBounds.valid ? (sceneBounds.max.y - std::max(scale.y, size.y * 0.05f)) : 2.0f,
        center.z);

    AddLightCube(scene, position, scale, glm::vec3(1.0f), 15.0f);
}

static ImportedCameraSpec BuildDefaultCameraSpec(const SceneBounds& sceneBounds)
{
    ImportedCameraSpec spec{};
    const glm::vec3 center = sceneBounds.center();
    const float radius = sceneBounds.radius();
    const float fovyDegrees = 45.0f;
    const float fovyRadians = glm::radians(fovyDegrees);
    const float distance = radius / std::tan(fovyRadians * 0.5f) + radius * 0.5f;

    spec.valid = true;
    spec.position = center + glm::vec3(0.0f, radius * 0.35f, distance);
    spec.lookAt = center;
    spec.up = glm::vec3(0.0f, 1.0f, 0.0f);
    spec.fovyDegrees = fovyDegrees;
    spec.aspectRatio = 1.0f;
    return spec;
}




Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    const std::string ext = std::filesystem::path(filename).extension().string();
    if (ext == ".gltf" || ext == ".glb")
    {
        loadFromGLTFScene(filename);
        return;
    }
    else
    {
        cout << "Unsupported scene format '" << ext << "'. Please provide a .gltf or .glb file." << endl;
        exit(-1);
    }
}

static bool LoadGLTFModelGeometry(Scene& scene,
                                  const tg3_model& model,
                                  const std::string& gltfPath,
                                  const std::vector<GLTFNodeTransform>& nodeWorlds,
                                  int materialOverride,
                                  SceneBounds* importedBounds)
{
    int fallbackMaterialId = -1;
    auto getFallbackMaterialId = [&scene, &fallbackMaterialId]() -> int
    {
        if (fallbackMaterialId < 0)
        {
            fallbackMaterialId = EnsureFallbackMaterial(scene.materials);
        }
        return fallbackMaterialId;
    };

    int resolvedMaterialOverride = materialOverride;
    if (resolvedMaterialOverride >= static_cast<int>(scene.materials.size()))
    {
        cerr << "Material override out of range for GLTF object, using fallback material." << endl;
        resolvedMaterialOverride = getFallbackMaterialId();
    }

    std::vector<int> gltfMaterialToScene(model.materials_count, -1);
    for (uint32_t materialIndex = 0; materialIndex < model.materials_count; ++materialIndex)
    {
        gltfMaterialToScene[materialIndex] = static_cast<int>(scene.materials.size());
        scene.materials.push_back(ConvertGLTFMaterial(model.materials[materialIndex]));
    }

    const size_t trianglesBefore = scene.triangles.size();

    for (const GLTFNodeTransform& nodeTransform : nodeWorlds)
    {
        const tg3_node& node = model.nodes[nodeTransform.nodeIndex];
        const glm::mat4& nodeWorld = nodeTransform.world;

        if (node.mesh < 0 || node.mesh >= static_cast<int32_t>(model.meshes_count))
        {
            continue;
        }

        const tg3_mesh& mesh = model.meshes[node.mesh];

        const int triStartIndex = static_cast<int>(scene.triangles.size());
        glm::vec3 aabbMin(std::numeric_limits<float>::max());
        glm::vec3 aabbMax(-std::numeric_limits<float>::max());
        bool meshHasTriangle = false;

        const float det = glm::determinant(glm::mat3(nodeWorld));
        glm::mat3 normalMatrix(1.0f);
        if (glm::abs(det) > 1e-12f)
        {
            normalMatrix = glm::transpose(glm::inverse(glm::mat3(nodeWorld)));
        }

        for (uint32_t p = 0; p < mesh.primitives_count; ++p)
        {
            const tg3_primitive& prim = mesh.primitives[p];
            if (prim.mode != -1 && prim.mode != TG3_MODE_TRIANGLES)
            {
                continue;
            }

            const int32_t posAccessorIndex = FindPrimitiveAttribute(prim, "POSITION");
            if (posAccessorIndex < 0)
            {
                continue;
            }

            AccessorView posView;
            if (!BuildAccessorView(&model, posAccessorIndex, posView))
            {
                continue;
            }

            if (posView.type != TG3_TYPE_VEC3 || posView.componentType != TG3_COMPONENT_TYPE_FLOAT)
            {
                cerr << "Skipping primitive with unsupported POSITION format in " << gltfPath << endl;
                continue;
            }

            bool hasVertexNormals = false;
            AccessorView nrmView;
            const int32_t nrmAccessorIndex = FindPrimitiveAttribute(prim, "NORMAL");
            if (nrmAccessorIndex >= 0)
            {
                if (BuildAccessorView(&model, nrmAccessorIndex, nrmView) &&
                    nrmView.type == TG3_TYPE_VEC3 &&
                    nrmView.componentType == TG3_COMPONENT_TYPE_FLOAT &&
                    nrmView.count == posView.count)
                {
                    hasVertexNormals = true;
                }
            }

            bool hasIndices = false;
            AccessorView idxView;
            if (prim.indices >= 0)
            {
                if (!BuildAccessorView(&model, prim.indices, idxView))
                {
                    continue;
                }

                const bool indexTypeSupported =
                    (idxView.componentType == TG3_COMPONENT_TYPE_UNSIGNED_BYTE) ||
                    (idxView.componentType == TG3_COMPONENT_TYPE_UNSIGNED_SHORT) ||
                    (idxView.componentType == TG3_COMPONENT_TYPE_UNSIGNED_INT);

                if (idxView.type != TG3_TYPE_SCALAR || !indexTypeSupported)
                {
                    cerr << "Skipping primitive with unsupported index format in " << gltfPath << endl;
                    continue;
                }

                hasIndices = true;
            }

            const uint64_t indexCount = hasIndices ? idxView.count : posView.count;
            if (indexCount < 3)
            {
                continue;
            }

            const uint64_t triangleCount = indexCount / 3;
            if (triangleCount == 0)
            {
                continue;
            }

            for (uint64_t tri = 0; tri < triangleCount; ++tri)
            {
                uint32_t i0 = 0;
                uint32_t i1 = 0;
                uint32_t i2 = 0;

                if (hasIndices)
                {
                    i0 = ReadIndex(idxView, tri * 3 + 0);
                    i1 = ReadIndex(idxView, tri * 3 + 1);
                    i2 = ReadIndex(idxView, tri * 3 + 2);
                }
                else
                {
                    i0 = static_cast<uint32_t>(tri * 3 + 0);
                    i1 = static_cast<uint32_t>(tri * 3 + 1);
                    i2 = static_cast<uint32_t>(tri * 3 + 2);
                }

                if (i0 >= posView.count || i1 >= posView.count || i2 >= posView.count)
                {
                    continue;
                }

                glm::vec3 p0 = ReadVec3F32(posView, i0);
                glm::vec3 p1 = ReadVec3F32(posView, i1);
                glm::vec3 p2 = ReadVec3F32(posView, i2);

                glm::vec3 w0 = glm::vec3(nodeWorld * glm::vec4(p0, 1.0f));
                glm::vec3 w1 = glm::vec3(nodeWorld * glm::vec4(p1, 1.0f));
                glm::vec3 w2 = glm::vec3(nodeWorld * glm::vec4(p2, 1.0f));

                glm::vec3 faceNormal = glm::cross(w1 - w0, w2 - w0);
                if (glm::length(faceNormal) <= 1e-12f)
                {
                    continue;
                }
                faceNormal = glm::normalize(faceNormal);

                Triangle triOut{};
                triOut.v1 = w0;
                triOut.v2 = w1;
                triOut.v3 = w2;
                if (resolvedMaterialOverride >= 0)
                {
                    triOut.materialid = resolvedMaterialOverride;
                }
                else if (prim.material >= 0 && prim.material < static_cast<int32_t>(gltfMaterialToScene.size()))
                {
                    triOut.materialid = gltfMaterialToScene[prim.material];
                }
                else
                {
                    if (prim.material >= static_cast<int32_t>(gltfMaterialToScene.size()))
                    {
                        cerr << "Primitive material index out of range in " << gltfPath
                             << ", using fallback material." << endl;
                    }
                    triOut.materialid = getFallbackMaterialId();
                }

                if (hasVertexNormals && i0 < nrmView.count && i1 < nrmView.count && i2 < nrmView.count)
                {
                    glm::vec3 n0 = glm::normalize(normalMatrix * ReadVec3F32(nrmView, i0));
                    glm::vec3 n1 = glm::normalize(normalMatrix * ReadVec3F32(nrmView, i1));
                    glm::vec3 n2 = glm::normalize(normalMatrix * ReadVec3F32(nrmView, i2));

                    if (glm::length(n0) <= 1e-12f || glm::length(n1) <= 1e-12f || glm::length(n2) <= 1e-12f)
                    {
                        triOut.n1 = faceNormal;
                        triOut.n2 = faceNormal;
                        triOut.n3 = faceNormal;
                        triOut.hasVertexNormals = 0;
                    }
                    else
                    {
                        triOut.n1 = n0;
                        triOut.n2 = n1;
                        triOut.n3 = n2;
                        triOut.hasVertexNormals = 1;
                    }
                }
                else
                {
                    triOut.n1 = faceNormal;
                    triOut.n2 = faceNormal;
                    triOut.n3 = faceNormal;
                    triOut.hasVertexNormals = 0;
                }

                scene.triangles.push_back(triOut);
                meshHasTriangle = true;

                aabbMin = glm::min(aabbMin, w0);
                aabbMin = glm::min(aabbMin, w1);
                aabbMin = glm::min(aabbMin, w2);

                aabbMax = glm::max(aabbMax, w0);
                aabbMax = glm::max(aabbMax, w1);
                aabbMax = glm::max(aabbMax, w2);
            }
        }

        if (meshHasTriangle)
        {
            MeshRange range{};
            range.triStartIndex = triStartIndex;
            range.triCount = static_cast<int>(scene.triangles.size()) - triStartIndex;
            range.aabbMin = aabbMin;
            range.aabbMax = aabbMax;
            scene.meshRanges.push_back(range);

            if (importedBounds)
            {
                importedBounds->expand(aabbMin, aabbMax);
            }
        }
    }

    const bool loadedAnyTriangle = scene.triangles.size() > trianglesBefore;
    if (!loadedAnyTriangle)
    {
        cerr << "No valid triangles were loaded from GLTF: " << gltfPath << endl;
    }

    return loadedAnyTriangle;
}

void Scene::loadFromGLTFScene(const std::string& gltfPath)
{
    tg3_model model{};
    tg3_error_stack errors{};

    if (!ParseGLTFFile(gltfPath, model, errors))
    {
        tg3_model_free(&model);
        tg3_error_stack_free(&errors);
        FailSceneLoad("Failed to load GLTF scene '" + gltfPath + "'.");
    }

    const std::vector<GLTFNodeTransform> nodeWorlds = CollectGLTFNodeWorldTransforms(model, glm::mat4(1.0f));
    SceneBounds importedBounds;

    if (!LoadGLTFModelGeometry(*this, model, gltfPath, nodeWorlds, -1, &importedBounds))
    {
        tg3_model_free(&model);
        tg3_error_stack_free(&errors);
        FailSceneLoad("No renderable geometry found in GLTF scene '" + gltfPath + "'.");
    }

    ImportedCameraSpec cameraSpec = ExtractGLTFCamera(model, nodeWorlds);
    if (!cameraSpec.valid)
    {
        cameraSpec = BuildDefaultCameraSpec(importedBounds);
    }

    const int importedLightCount = ImportGLTFLights(*this, model, nodeWorlds, importedBounds);
    if (importedLightCount == 0 && !SceneHasEmissiveMaterial(*this))
    {
        AddDefaultSceneLight(*this, importedBounds);
    }

    InitializeRenderState(
        state,
        BuildResolutionFromAspect(cameraSpec.aspectRatio),
        cameraSpec.fovyDegrees,
        5000,
        8,
        BuildImageBaseName(gltfPath),
        cameraSpec.position,
        cameraSpec.lookAt,
        cameraSpec.up);

    tg3_model_free(&model);
    tg3_error_stack_free(&errors);
}


/**
 * Helper function to load a GLTF file and add its geometry to the scene. 
 * This function uses the tinygltf library to parse the GLTF file, extract mesh data, and convert it into the internal representation of triangles and materials used by the scene. 
 * It handles the transformation of vertices based on the provided object transform and applies a material override if specified.  
 * @param gltfPath The file path to the GLTF file to load.
 * @param objectTransform A glm::mat4 representing the transformation to apply to the geometry loaded from the GLTF file.
 * @param materialOverride An integer index into the scene's materials array to use for all geometry loaded from the GLTF file, or -1 to use the materials specified in the GLTF file. 
 * @return True if the GLTF file was successfully loaded and its geometry added to the scene, false if there was an error during loading or processing.
 */
bool Scene::loadGLTFObject(const std::string& gltfPath, const glm::mat4& objectTransform, int materialOverride)
{
    tg3_model model{};
    tg3_error_stack errors{};

    if (!ParseGLTFFile(gltfPath, model, errors))
    {
        tg3_model_free(&model);
        tg3_error_stack_free(&errors);
        return false;
    }

    const std::vector<GLTFNodeTransform> nodeWorlds =
        CollectGLTFNodeWorldTransforms(model, objectTransform);
    const bool loadedAnyTriangle =
        LoadGLTFModelGeometry(*this, model, gltfPath, nodeWorlds, materialOverride, nullptr);

    tg3_model_free(&model);
    tg3_error_stack_free(&errors);
    return loadedAnyTriangle;
}
