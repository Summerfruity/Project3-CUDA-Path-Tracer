#pragma once

#include "sceneStructs.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadGLTFObject(const std::string& gltfPath, int materialId, const glm::mat4& objectTransform);
public:
    Scene(std::string filename);

    std::vector<Triangle> triangles;
    std::vector<MeshRange> meshRanges;

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
};
