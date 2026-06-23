#pragma once
#include "sceneStructs.h"
#include <vector>


std::vector<BVHNode> buildBVHForRange(
    std::vector<Triangle>& triangles,
    int start,
    int count,
    int maxLeafSize = 4,
    int maxDepth = 32,
    int nodeBaseOffset = 0); // offset to add to node indices (used for recursive calls)

std::vector<BVHNode> buildBVHForMeshRanges(
    std::vector<MeshRange>& meshRanges,
    int maxLeafSize = 4,
    int maxDepth = 32);
