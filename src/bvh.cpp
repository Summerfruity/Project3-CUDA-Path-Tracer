#include "bvh.h"
#include <algorithm>
#include <vector>
#include <cfloat>

struct BVHBuildPrimitive
{
    int originalIndex;      // offset relative to MeshRange.triStartIndex 
    glm::vec3 centroid;
    glm::vec3 aabbMin;
    glm::vec3 aabbMax;
};

// Choose the axis to split the primitives along, based on the largest extent of their centroids.
static int chooseSplitAxis(const std::vector<BVHBuildPrimitive>& prims, int begin, int end)
{
    glm::vec3 cmin(FLT_MAX), cmax(-FLT_MAX);
    for (int i = begin; i < end; ++i)
    {
        cmin = glm::min(cmin, prims[i].centroid);
        cmax = glm::max(cmax, prims[i].centroid);
    }
    glm::vec3 extent = cmax - cmin;
    int axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (extent.z > extent[axis]) axis = 2;
    return axis;
}

static int recursiveBuild(std::vector<BVHNode>& nodes, std::vector<BVHBuildPrimitive>& prims,
                          int begin, int end,
                          int maxLeafSize, int maxDepth, 
                          int depth)
{
    //step1. create current node
    int nodeIndex = static_cast<int>(nodes.size());
    nodes.emplace_back();
    BVHNode& node = nodes[nodeIndex];
    node.isLeaf = 0;
    node.left = node.right = -1;

    //step2. compute the AABB of the current node
    glm::vec3 bmin(FLT_MAX), bmax(-FLT_MAX);
    for(int i = begin; i < end; i++)
    {
        bmin = glm::min(bmin, prims[i].aabbMin);
        bmax = glm::max(bmax, prims[i].aabbMax);
    }

    node.aabbMin = bmin;
    node.aabbMax = bmax;

    //step3. If leaf
    int count = end - begin;
    if(count <= maxLeafSize || depth >= maxDepth)
    {
        node.isLeaf = 1;
        node.left = begin; // Notice that begin is offset to prims, not Scene::triangles
        node.right = count;
        return nodeIndex;
    }

    //step4. If not leaf, choose the split axis
    int axis = chooseSplitAxis(prims, begin, end);
    float extend = node.aabbMax[axis] - node.aabbMin[axis];

    // The centroids of the triangles almost coincide on this axis, 
    // so there's no point in dividing them further; just make them into leaves.
    if (extend < 1e-6f)
    {
        node.isLeaf = 1;
        node.left = begin;
        node.right = count;
        return nodeIndex;
    } 

    //step5. Midpoint segmentation, divide the triangle into two groups by centroid[axis].
    int mid = begin + count / 2;
    std::nth_element(
        prims.begin() + begin,
        prims.begin() + mid,
        prims.begin() + end,
        [axis](const BVHBuildPrimitive& a, const BVHBuildPrimitive& b)
    {
        return a.centroid[axis] < b.centroid[axis];
    });

    //step6. Recursively build left and right subtrees
    int leftChild  = recursiveBuild(nodes, prims, begin, mid, maxLeafSize, maxDepth, depth + 1);
    int rightChild = recursiveBuild(nodes, prims, mid,   end, maxLeafSize, maxDepth, depth + 1);

    node.left = leftChild;
    node.right = rightChild;
    return nodeIndex;

}

std::vector<BVHNode> buildBVHForRange(std::vector<Triangle>& triangles,
                                      int start, int count,
                                      int maxLeafSize, int maxDepth,
                                      int nodeBaseOffset)
{
    std::vector<BVHBuildPrimitive> prims;
    prims.reserve(count);

    //step1. Convert Triangle to BVHBuildPrimitive
    for(int i = 0; i < count; i++)
    {
        Triangle& t = triangles[start + i];
        BVHBuildPrimitive p;
        p.originalIndex = i;
        p.centroid = (t.v0 + t.v1 + t.v2) * (1.0f / 3.0f);
        p.aabbMin = glm::min(glm::min(t.v0, t.v1), t.v2);
        p.aabbMax = glm::max(glm::max(t.v0, t.v1), t.v2);
        prims.push_back(p);

    }


    //step2. Create an array of nodes and build recursively.
    std::vector<BVHNode> nodes; 
    nodes.reserve(2 * count); 
    recursiveBuild(nodes, prims, 0, count, maxLeafSize, maxDepth, 0);

    //step3. Rearrange the actual Triangles according to BVH order.
    std::vector<Triangle> reordered;
    reordered.reserve(count);
    for(int i = 0; i < count; i++)
    {
        reordered.push_back(triangles[start + prims[i].originalIndex]);
    }
    for (int i = 0; i < count; ++i)
    {
        triangles[start + i] = reordered[i];
    }

    //step4. Change local node indexes to global indexes
    if (nodeBaseOffset != 0)
    {
        for (auto& node : nodes)
        {
            if (!node.isLeaf)
            {
                node.left  += nodeBaseOffset;
                node.right += nodeBaseOffset;
            }
        }
    }

    return nodes;

}                                      

std::vector<BVHNode> buildBVHForMeshRanges(std::vector<MeshRange>& meshRanges,
                                           int maxLeafSize,
                                           int maxDepth)
{
    std::vector<BVHBuildPrimitive> prims;
    prims.reserve(meshRanges.size());

    for (int i = 0; i < static_cast<int>(meshRanges.size()); ++i)
    {
        const MeshRange& range = meshRanges[i];
        BVHBuildPrimitive p;
        p.originalIndex = i;
        p.aabbMin = range.aabbMin;
        p.aabbMax = range.aabbMax;
        p.centroid = (range.aabbMin + range.aabbMax) * 0.5f;
        prims.push_back(p);
    }

    std::vector<BVHNode> nodes;
    if (prims.empty())
    {
        return nodes;
    }

    nodes.reserve(2 * prims.size());
    recursiveBuild(nodes, prims, 0, static_cast<int>(prims.size()), maxLeafSize, maxDepth, 0);

    std::vector<MeshRange> reordered;
    reordered.reserve(meshRanges.size());
    for (int i = 0; i < static_cast<int>(prims.size()); ++i)
    {
        reordered.push_back(meshRanges[prims[i].originalIndex]);
    }
    meshRanges = std::move(reordered);

    return nodes;
}
