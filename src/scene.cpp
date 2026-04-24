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
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <filesystem> // C++17 filesystem library for path manipulation
#include <limits>
#include <cstdint> // For fixed-width integer types like uint32_t
#include <algorithm>

using namespace std;
using json = nlohmann::json;

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




Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = 0.f;
            newMaterial.hasRefractive = 0.f;
            newMaterial.emittance = 0.f;

        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
            newMaterial.hasReflective = 0.f;
            newMaterial.hasRefractive = 0.f;
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.specular.color = newMaterial.color;
            newMaterial.hasReflective = 1.0f;
            newMaterial.hasRefractive = 0.f;
            newMaterial.emittance = 0.f;
            
            if (p.contains("ROUGHNESS")) 
            {
                newMaterial.specular.exponent = p["ROUGHNESS"];
            } 
            else 
            {
                newMaterial.specular.exponent = 0.0f; 
            }
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const std::filesystem::path scenePath(jsonName);
    const std::filesystem::path sceneDir = scenePath.has_parent_path()
        ? scenePath.parent_path()
        : std::filesystem::current_path();

    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const std::string type = p.value("TYPE", "");

        if (type == "gltf")
        {
            if (!p.contains("FILE"))
            {
                cerr << "GLTF object is missing FILE field." << endl;
                continue;
            }

            const std::string gltfFile = p["FILE"].get<std::string>();
            const auto& trans = p["TRANS"];
            const auto& rotat = p["ROTAT"];
            const auto& scale = p["SCALE"];

            glm::vec3 translation(trans[0], trans[1], trans[2]);
            glm::vec3 rotation(rotat[0], rotat[1], rotat[2]);
            glm::vec3 scaling(scale[0], scale[1], scale[2]);
            glm::mat4 objectTransform = utilityCore::buildTransformationMatrix(
                translation, rotation, scaling);

            int materialOverride = -1;
            if (p.contains("MATERIAL"))
            {
                const std::string materialName = p["MATERIAL"].get<std::string>();
                auto it = MatNameToID.find(materialName);
                if (it != MatNameToID.end())
                {
                    materialOverride = static_cast<int>(it->second);
                }
                else
                {
                    cerr << "GLTF object references unknown material: " << materialName << endl;
                }
            }

            std::filesystem::path resolvedPath = std::filesystem::path(gltfFile);
            if (!resolvedPath.is_absolute())
            {
                resolvedPath = sceneDir / resolvedPath;
            }
            resolvedPath = resolvedPath.lexically_normal();

            if (!loadGLTFObject(resolvedPath.string(), objectTransform, materialOverride))
            {
                cerr << "Failed to load GLTF object: " << resolvedPath.string() << endl;
            }

            continue;
        }

        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else
        {
            newGeom.type = SPHERE;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
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
    model.default_scene = -1;

    tg3_error_stack errors{};
    tg3_error_stack_init(&errors);

    tg3_parse_options options{};
    tg3_parse_options_init(&options);

    const tg3_error_code parseCode = tg3_parse_file(
        &model,
        &errors,
        gltfPath.c_str(),
        static_cast<uint32_t>(gltfPath.size()),
        &options);

    auto dumpErrors = [&errors]() {
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
    };

    if (parseCode != TG3_OK || tg3_errors_has_error(&errors))
    {
        cerr << "Failed to parse GLTF file: " << gltfPath << " (code " << parseCode << ")" << endl;
        dumpErrors();
        tg3_model_free(&model);
        tg3_error_stack_free(&errors);
        return false;
    }

    if (model.nodes_count == 0)
    {
        cerr << "GLTF has no nodes: " << gltfPath << endl;
        tg3_model_free(&model);
        tg3_error_stack_free(&errors);
        return false;
    }

    if (materials.empty())
    {
        Material fallback{};
        fallback.color = glm::vec3(0.8f);
        fallback.specular.exponent = 0.0f;
        fallback.specular.color = glm::vec3(1.0f);
        fallback.hasReflective = 0.0f;
        fallback.hasRefractive = 0.0f;
        fallback.indexOfRefraction = 1.0f;
        fallback.emittance = 0.0f;
        materials.push_back(fallback);
    }

    int resolvedMaterialId = materialOverride;
    if (resolvedMaterialId < 0 || resolvedMaterialId >= static_cast<int>(materials.size()))
    {
        if (resolvedMaterialId >= static_cast<int>(materials.size()))
        {
            cerr << "Material override out of range for GLTF object, falling back to material 0." << endl;
        }
        resolvedMaterialId = 0;
    }

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
    }
    else
    {
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
    }

    struct NodeWorkItem
    {
        int32_t nodeIndex;
        glm::mat4 parentWorld;
    };

    std::vector<NodeWorkItem> workStack;
    workStack.reserve(rootNodes.size() + 16);

    for (int i = static_cast<int>(rootNodes.size()) - 1; i >= 0; --i)
    {
        workStack.push_back(NodeWorkItem{ rootNodes[i], objectTransform });
    }

    const size_t trianglesBefore = triangles.size();

    while (!workStack.empty())
    {
        NodeWorkItem work = workStack.back();
        workStack.pop_back();

        if (work.nodeIndex < 0 || work.nodeIndex >= static_cast<int32_t>(model.nodes_count))
        {
            continue;
        }

        const tg3_node& node = model.nodes[work.nodeIndex];
        const glm::mat4 nodeWorld = work.parentWorld * NodeLocalMatrix(node);

        if (node.mesh >= 0 && node.mesh < static_cast<int32_t>(model.meshes_count))
        {
            const tg3_mesh& mesh = model.meshes[node.mesh];

            const int triStartIndex = static_cast<int>(triangles.size());
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
                    triOut.materialid = resolvedMaterialId;

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

                    triangles.push_back(triOut);
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
                range.triCount = static_cast<int>(triangles.size()) - triStartIndex;
                range.aabbMin = aabbMin;
                range.aabbMax = aabbMax;
                meshRanges.push_back(range);
            }
        }

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

    const bool loadedAnyTriangle = triangles.size() > trianglesBefore;
    if (!loadedAnyTriangle)
    {
        cerr << "No valid triangles were loaded from GLTF: " << gltfPath << endl;
    }

    tg3_model_free(&model);
    tg3_error_stack_free(&errors);
    return loadedAnyTriangle;
}
