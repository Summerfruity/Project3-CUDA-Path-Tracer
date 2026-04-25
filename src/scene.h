#pragma once

#include "sceneStructs.h"
#include <vector>

class Scene
{
private:
    void loadFromGLTFScene(const std::string& gltfPath);

    // Loads a GLTF file and adds its geometry to the scene. Returns true on success, false on failure.
    bool loadGLTFObject(const std::string& gltfPath, const glm::mat4& objectTransform, int materialOverride);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Triangle> triangles;
    std::vector<MeshRange> meshRanges;
    std::vector<Material> materials;
    RenderState state;
};
