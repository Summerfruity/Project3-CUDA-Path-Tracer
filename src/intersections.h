#pragma once

#include "sceneStructs.h"

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>


/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ inline glm::vec3 getPointOnRay(Ray r, float t)
{
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ inline glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v)
{
    return glm::vec3(m * v);
}

/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param surfaceNormal      Output parameter for face-forward shading normal.
 * @param geometricNormal    Output parameter for outward geometric normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& surfaceNormal,
    glm::vec3& geometricNormal,
    bool& outside);

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param surfaceNormal      Output parameter for face-forward shading normal.
 * @param geometricNormal    Output parameter for outward geometric normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& surfaceNormal,
    glm::vec3& geometricNormal,
    bool& outside);



/**
 * Test intersection between a ray and a triangle.
 * @param tri                The triangle to test against.
 * @param r                  The ray to test.
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param surfaceNormal      Output parameter for face-forward shading normal.
 * @param geometricNormal    Output parameter for outward geometric normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float triangleIntersectionTest(
    const Triangle& tri,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& surfaceNormal,
    glm::vec3& geometricNormal,
    bool& outside,
    glm::vec2& uv);


/**
 * Test intersection between a ray and an axis-aligned bounding box.
 * @param range              The AABB to test against.
 * @param r                  The ray to test.
 * @param maxT               The maximum `t` value to consider for intersections (e.g. the ray may have already hit something closer).
 * @return                   Ray parameter `t` value. -1 if no intersection or if the intersection is farther than maxT.
 */
__host__ __device__ float aabbIntersectionTest(
    const MeshRange& range,
    Ray r,
    float maxT);


__host__ __device__ float aabbIntersectionTest(
    glm::vec3 aabbMin,
    glm::vec3 aabbMax,
    Ray r,
    float maxT);

__device__ bool bvhIntersectionTest(
    const BVHNode* nodes,           // A global BVH node array (containing all nodes of the mesh).
    int rootIndex,                  // The index of the current mesh's BVH root node in the nodes list.
    const MeshRange& range,         // Triangle range information of the current mesh
    const Triangle* triangles,      // Global triangle array
    Ray r,                          // Current light
    float& bestT,                   // The smallest t value found so far (input/output)
    glm::vec3& bestPoint,           // Closest intersection point (output)
    glm::vec3& bestNormal,          // The closest intersection point normal (output)
    glm::vec3& bestGeomNormal,      // The geometric normal at the closest intersection point (output)
    bool& bestOutside,              // Was it hit from the outside (output)?
    int& bestMaterialId,            // material ID (output)
    glm::vec2& bestUv);             // Nearest intersection point UV (output)