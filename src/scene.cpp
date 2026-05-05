#include "scene.h"
#include <tinygltf3/tiny_gltf_v3.h>


#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <cfloat>
#include <cstdint>
#include <filesystem>
#include <functional>

using namespace std;
using json = nlohmann::json;


// Forward declarations for helper functions to load glTF files and convert them to our internal representation.
namespace
{
    /**
     * @param prim: a reference to a tg3_primitive struct representing a primitive in the glTF model. 
     * @param name: a C-style string representing the name of the attribute to find (e.g., "POSITION", "NORMAL", etc.). 
     * @return: the function returns an integer index corresponding to the accessor index of the specified attribute if it is found within the primitive's attributes. 
     */
    int findAttributeAccessor(const tg3_primitive& prim, const char* name)
    {
        for(int i = 0; i < prim.attributes_count; i++)
        {
            const tg3_str_int_pair& attr = prim.attributes[i]; // get the attribute key-value pair
            if(attr.key.data && std::string(attr.key.data, attr.key.len) == name)
            { // compare the attribute name with the provided name
                return attr.value; 
            }
        }
        return -1;
    }


    /**
     * A helper function to read a vec3 value from a glTF accessor. 
     * @param model: a reference to the tg3_model struct representing the loaded glTF model. 
     * @param accessorIndex: an integer index corresponding to the accessor from which to read the vec3 value. 
     * @param vertexIndex: an integer index specifying which vertex to read from the accessor. 
     */
    uint32_t readIndexValue(const uint8_t* ptr, int componentType)
    {
        switch (componentType)
        {
            case TG3_COMPONENT_TYPE_UNSIGNED_BYTE:
                return *reinterpret_cast<const uint8_t*>(ptr);
            case TG3_COMPONENT_TYPE_UNSIGNED_SHORT:
                return *reinterpret_cast<const uint16_t*>(ptr);
            case TG3_COMPONENT_TYPE_UNSIGNED_INT:
                return *reinterpret_cast<const uint32_t*>(ptr);
            default:
                std::cerr << "Unsupported index component type: " << componentType << std::endl;
                return 0;
        }
    }

    /**
     * A helper function to read a vec3 value from a glTF accessor.
     * @param model: a reference to the tg3_model struct representing the loaded glTF model.
     * @param accessorIndex: an integer index corresponding to the accessor from which to read the vec3 value.
     * @param elementIndex: an integer index specifying which element (vertex) to read from the accessor. This is used to calculate the correct byte offset when reading the vertex data.
     * @return: the function returns a glm::vec3 containing the x, y, and z components read from the specified accessor and element index. The vertex positions are expected to be stored as VEC3 with FLOAT components in the glTF file.
     */
    glm::vec3 readVec3Accessor(const tg3_model& model, int accessorIndex, uint32_t elementIndex)
    {
        const tg3_accessor& acc = model.accessors[accessorIndex];
        const tg3_buffer_view& view = model.buffer_views[acc.buffer_view];
        const tg3_buffer& buffer = model.buffers[view.buffer];

        int stride = tg3_accessor_byte_stride(&acc, &view);
        const uint8_t* base = buffer.data.data + view.byte_offset + acc.byte_offset;
        const uint8_t* ptr = base + elementIndex * stride;

        const float* f = reinterpret_cast<const float*>(ptr);
        return glm::vec3(f[0], f[1], f[2]);
    }


    // A helper function to compute the local transformation matrix for a given glTF node.
    glm::mat4 nodeLocalTransform(const tg3_node& node)
    {
        if (node.has_matrix)
        {
            glm::mat4 m(1.0f);
            const double* a = node.matrix;
            for (int col = 0; col < 4; ++col)
            {
                for (int row = 0; row < 4; ++row)
                {
                    m[col][row] = static_cast<float>(a[col * 4 + row]);
                }
            }
            return m;
        }

        glm::vec3 t(
            static_cast<float>(node.translation[0]),
            static_cast<float>(node.translation[1]),
            static_cast<float>(node.translation[2]));

        glm::quat r(
            static_cast<float>(node.rotation[3]),
            static_cast<float>(node.rotation[0]),
            static_cast<float>(node.rotation[1]),
            static_cast<float>(node.rotation[2]));

        glm::vec3 s(
            static_cast<float>(node.scale[0]),
            static_cast<float>(node.scale[1]),
            static_cast<float>(node.scale[2]));

        return glm::translate(glm::mat4(1.0f), t)
             * glm::mat4_cast(r)
             * glm::scale(glm::mat4(1.0f), s);
    }
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


/**
 * Loads a glTF file and adds its geometry to the scene. 
 * This is a helper function called by loadFromJSON when it encounters an object of type "gltf". 
 * You can assume that the glTF file only contains triangle meshes (no lines or points) 
 * and that all vertex positions are stored in accessors with type VEC3 and component type FLOAT. 
 * You should read the vertex positions from the POSITION accessor of each primitive, 
 * apply the provided objectTransform to them, and then create Triangle objects that 
 * you push into the triangles vector. 
 * You can compute the triangle normals using the cross product of the edges. 
 * The material ID for these triangles should be set to the provided materialId.
 * @param gltfPath: the file path to the glTF file to load.
 * @param materialId: the material ID to assign to the triangles created from this glTF file. This will be an index into the materials vector.
 * @param objectTransform: a transformation matrix to apply to all vertex positions read from the glTF file. This allows you to position, rotate, and scale the imported geometry in the scene.
 */
void Scene::loadGLTFObject(const std::string& gltfPath, int materialId, const glm::mat4& objectTransform)
{
    tg3_model model{};
    tg3_error_stack errors{};
    tg3_parse_options opts{};
    tg3_parse_options_init(&opts);

    tg3_error_code code = tg3_parse_file(
        &model,
        &errors,
        gltfPath.c_str(),
        (uint32_t)gltfPath.size(),
        &opts
    );

    if (code != TG3_OK) {
        std::cerr << "Failed to load glTF file: " << gltfPath << std::endl;
        tg3_model_free(&model);
        tg3_error_stack_free(&errors);

        return;
    }

    
    auto processPrimitive = [&](const tg3_primitive& prim, const glm::mat4& transform)
    {
        // Only process triangles for now
        if (prim.mode != TG3_MODE_TRIANGLES)
        {
            return;
        }

        // find the accessor index for the POSITION attribute
        int posAccessorIndex = findAttributeAccessor(prim, "POSITION"); 
        if(posAccessorIndex < 0)
        {
            std::cerr << "Primitive does not have POSITION attribute." << std::endl;
            return;
        }

        // Get the position accessor
        const tg3_accessor& posAcc = model.accessors[posAccessorIndex];
        if (posAcc.type != TG3_TYPE_VEC3 || posAcc.component_type != TG3_COMPONENT_TYPE_FLOAT)
        {
            std::cerr << "Unsupported POSITION accessor in " << gltfPath << std::endl;
            return;
        }

    
        int startTri = static_cast<int>(triangles.size()); 
        glm::vec3 aabbMin(FLT_MAX);
        glm::vec3 aabbMax(-FLT_MAX);

        /**
         * @param i0, i1, i2: the vertex indices of the triangle to push. These should be used to read the vertex positions from the POSITION accessor.
         */
        auto pushTriangle = [&](uint32_t i0, uint32_t i1, uint32_t i2)
        {
            // Read the vertex positions and apply the object transform
            glm::vec3 p0 = glm::vec3(transform * glm::vec4(readVec3Accessor(model, posAccessorIndex, i0), 1.0f));
            glm::vec3 p1 = glm::vec3(transform * glm::vec4(readVec3Accessor(model, posAccessorIndex, i1), 1.0f));
            glm::vec3 p2 = glm::vec3(transform * glm::vec4(readVec3Accessor(model, posAccessorIndex, i2), 1.0f));

            glm::vec3 n = glm::normalize(glm::cross(p1 - p0, p2 - p0)); // compute the triangle normal

            Triangle tri{}; // create a new triangle
            tri.v0 = p0;
            tri.v1 = p1;
            tri.v2 = p2;
            tri.n0 = n;
            tri.n1 = n;
            tri.n2 = n;
            tri.materialId = materialId; // set the material ID for the triangle
            triangles.push_back(tri); // push the triangle into the triangles vector

            // Update the AABB for this triangle
            aabbMin = glm::min(aabbMin, p0);
            aabbMin = glm::min(aabbMin, p1);
            aabbMin = glm::min(aabbMin, p2);
            aabbMax = glm::max(aabbMax, p0);
            aabbMax = glm::max(aabbMax, p1);
            aabbMax = glm::max(aabbMax, p2);

        };

        
        if (prim.indices >= 0)
        {// If the primitive has an index buffer, read the indices and push triangles accordingly


            // Get the index accessor and buffer
            const tg3_accessor& idxAcc = model.accessors[prim.indices];
            const tg3_buffer_view& idxView = model.buffer_views[idxAcc.buffer_view];
            const tg3_buffer& idxBuffer = model.buffers[idxView.buffer];

            // Determine the byte stride and base pointer for the index data
            int idxStride = tg3_accessor_byte_stride(&idxAcc, &idxView);
            const uint8_t* idxBase =
                idxBuffer.data.data + idxView.byte_offset + idxAcc.byte_offset;

            // Iterate over the indices in groups of three to form triangles
            for (uint64_t i = 0; i + 2 < idxAcc.count; i += 3)
            {
                uint32_t i0 = readIndexValue(idxBase + (i + 0) * idxStride, idxAcc.component_type);
                uint32_t i1 = readIndexValue(idxBase + (i + 1) * idxStride, idxAcc.component_type);
                uint32_t i2 = readIndexValue(idxBase + (i + 2) * idxStride, idxAcc.component_type);
                pushTriangle(i0, i1, i2);
            }
        }
        else
        {// If there is no index buffer, assume the vertices are listed in order and push triangles accordingly
            for (uint64_t i = 0; i + 2 < posAcc.count; i += 3)
            {
                pushTriangle(static_cast<uint32_t>(i),
                             static_cast<uint32_t>(i + 1),
                             static_cast<uint32_t>(i + 2));
            }

        }


        // After processing all triangles in this primitive, create a MeshRange for it and push it into the meshRanges vector.
        int triCount = static_cast<int>(triangles.size()) - startTri;
        if (triCount > 0)
        {
            MeshRange range{};
            range.triStartIndex = startTri;
            range.triCount = triCount;
            range.aabbMin = aabbMin;
            range.aabbMax = aabbMax;
            meshRanges.push_back(range);
        }

    };
    

    /**
     * A recursive function to traverse the node hierarchy of the glTF model. 
     * For each node, it computes the local transform from the node's translation, rotation, and scale properties, 
     * combines it with the parent transform, 
     * and if the node has a mesh, processes each primitive in the mesh using the processPrimitive function defined above. 
     * It then recursively calls itself on the node's children, passing down the combined transform. 
     * This ensures that all geometry in the glTF file is correctly transformed and added to the scene, 
     * taking into account the hierarchical structure of the glTF model.   
     */
    std::function<void(int, const glm::mat4&)> traverseNode =
        [&](int nodeIndex, const glm::mat4& parentTransform)
    {
        const tg3_node& node = model.nodes[nodeIndex];
        glm::mat4 nodeTransform = parentTransform * nodeLocalTransform(node);

        if (node.mesh >= 0)
        {
            const tg3_mesh& mesh = model.meshes[node.mesh];
            glm::mat4 finalTransform = objectTransform * nodeTransform;

            for (uint32_t i = 0; i < mesh.primitives_count; ++i)
            {
                processPrimitive(mesh.primitives[i], finalTransform);
            }
        }

        for (uint32_t i = 0; i < node.children_count; ++i)
        {
            traverseNode(node.children[i], nodeTransform);
        }
    };


    // Start traversing from the default scene (or the first scene if no default is specified) to load all geometry into the scene.
    int sceneIndex = model.default_scene >= 0 ? model.default_scene : 0;
    if (sceneIndex >= 0 && static_cast<uint32_t>(sceneIndex) < model.scenes_count)
    {
        const tg3_scene& scene = model.scenes[sceneIndex];
        for (uint32_t i = 0; i < scene.nodes_count; ++i)
        {
            traverseNode(scene.nodes[i], glm::mat4(1.0f));
        }
    }


    std::cout << "Loaded glTF: " << gltfPath
              << ", total triangles: " << triangles.size()
              << ", mesh ranges: " << meshRanges.size()
              << std::endl;


    tg3_model_free(&model);

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
        else if (p["TYPE"] == "Refractive")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = 0.0f;
            newMaterial.hasRefractive = 1.0f;
            newMaterial.emittance = 0.0f;
            newMaterial.indexOfRefraction = p.contains("IOR") ? (float)p["IOR"] : 1.5f;
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
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
        else if (type == "sphere")
        {
            newGeom.type = SPHERE;
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
        else if (type == "gltf")
        {
            int materialId = MatNameToID[p["MATERIAL"]];

            const auto& trans = p["TRANS"];
            const auto& rotat = p["ROTAT"];
            const auto& scale = p["SCALE"];

            glm::vec3 translation(trans[0], trans[1], trans[2]);
            glm::vec3 rotation(rotat[0], rotat[1], rotat[2]);
            glm::vec3 scaleVec(scale[0], scale[1], scale[2]);

            glm::mat4 objectTransform =
                utilityCore::buildTransformationMatrix(translation, rotation, scaleVec);

            std::filesystem::path scenePath(jsonName);
            std::filesystem::path resolvedPath =
                scenePath.parent_path() / std::string(p["PATH"]);
            loadGLTFObject(resolvedPath.string(), materialId, objectTransform);
        }
        
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
    camera.apertureRadius = cameraData.contains("APERTURE") ? (float)cameraData["APERTURE"] : 0.0f;
    camera.focalDistance = cameraData.contains("FOCALDIST") ? (float)cameraData["FOCALDIST"] : 0.0f;

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
