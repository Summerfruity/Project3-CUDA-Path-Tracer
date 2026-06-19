#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &surfaceNormal,
    glm::vec3 &geometricNormal,
    bool &outside)
{
    const float PARALLEL_EPSILON = 1e-6f;
    Ray q; // ray in the coordinate system of the box

    // transform the ray into the space of the box
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n(0.0f); // outward normal of the plane that gives us tmin
    glm::vec3 tmax_n(0.0f); // outward normal of the plane that gives us tmax
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz]; // component of the ray direction in the current axis, which is the normal of the planes we are intersecting with
        if (glm::abs(qdxyz) < PARALLEL_EPSILON)
        {
            if (q.origin[xyz] < -0.5f || q.origin[xyz] > 0.5f)
            {
                return -1;
            }
            continue;
        }

        float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
        float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
        glm::vec3 nearNormal(0.0f);
        glm::vec3 farNormal(0.0f);
        nearNormal[xyz] = qdxyz > 0.0f ? -1.0f : 1.0f;
        farNormal[xyz] = -nearNormal[xyz];

        float tNear = glm::min(t1, t2);
        float tFar = glm::max(t1, t2);
        if (tNear > tmin)
        {
            tmin = tNear;
            tmin_n = nearNormal;
        }
        if (tFar < tmax)
        {
            tmax = tFar;
            tmax_n = farNormal;
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(q.origin + tmin * q.direction, 1.0f));
        geometricNormal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        surfaceNormal = outside ? geometricNormal : -geometricNormal;
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &surfaceNormal,
    glm::vec3 &geometricNormal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = rt.origin + t * rt.direction;

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    geometricNormal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    surfaceNormal = outside ? geometricNormal : -geometricNormal;

    return glm::length(r.origin - intersectionPoint);
}


__host__ __device__ float triangleIntersectionTest(
    const Triangle& tri,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& surfaceNormal,
    glm::vec3& geometricNormal,
    bool& outside,
    glm::vec2& uv)
{
    // Moller-Trumbore ray/triangle intersection in triangle space.
    const float TRIANGLE_EPSILON = 1e-6f;
    glm::vec3 dir = glm::normalize(r.direction);
    glm::vec3 edge1 = tri.v1 - tri.v0;
    glm::vec3 edge2 = tri.v2 - tri.v0;
    glm::vec3 pvec = glm::cross(dir, edge2);
    float det = glm::dot(edge1, pvec);

    if (glm::abs(det) < TRIANGLE_EPSILON)
    {
        return -1.0f;
    }

    float invDet = 1.0f / det;
    // Compute barycentric u and test bounds.
    glm::vec3 tvec = r.origin - tri.v0;
    float u = glm::dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f)
    {
        return -1.0f;
    }

    // Compute barycentric v and test bounds.
    glm::vec3 qvec = glm::cross(tvec, edge1);
    float v = glm::dot(dir, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f)
    {
        return -1.0f;
    }

    // Ray parameter along direction.
    float t = glm::dot(edge2, qvec) * invDet;
    if (t <= TRIANGLE_EPSILON)
    {
        return -1.0f;
    }

    intersectionPoint = r.origin + t * dir;

    glm::vec3 faceNormal = glm::cross(edge1, edge2);
    float normalLengthSquared = glm::dot(faceNormal, faceNormal);
    if (normalLengthSquared < TRIANGLE_EPSILON * TRIANGLE_EPSILON)
    {
        return -1.0f;
    }

    geometricNormal = glm::normalize(faceNormal);
    outside = glm::dot(dir, geometricNormal) < 0.0f;

    float w = 1.0f - u - v; // barycentric coordinate
    glm::vec3 smoothN = glm::normalize(w * tri.n0 + u * tri.n1 + v * tri.n2);
    surfaceNormal = outside ? smoothN : -smoothN;

    // uv interpolation
    uv = w * tri.uv0 + u * tri.uv1 + v * tri.uv2;

    // Return Euclidean distance to match box/sphere intersection semantics.
    return glm::length(r.origin - intersectionPoint);
}


__host__ __device__ float aabbIntersectionTest(
    glm::vec3 aabbMin,
    glm::vec3 aabbMax,
    Ray r,
    float maxT)
{

    const float PARALLEL_EPSILON = 1e-6f;

    glm::vec3 dir = glm::normalize(r.direction);

    float tEnter = -FLT_MAX;
    float tExit = FLT_MAX;

    for(int axis = 0; axis < 3; axis++)
    {
        float origin = r.origin[axis];
        float direction = dir[axis];
        float minBox = aabbMin[axis];
        float maxBox = aabbMax[axis];

        if(glm::abs(direction) < PARALLEL_EPSILON)
        {
            // need to make sure the origin is outside the range
            if(origin < minBox || origin > maxBox)
            {
                return -1.0f;
            }
            continue;
        }

        float t0 = (minBox - origin) / direction;
        float t1 = (maxBox - origin) / direction;

        if(t0 > t1)
        {
            float temp = t0;
            t0 = t1;
            t1 = temp;
        }

        tEnter = glm::max(t0, tEnter);
        tExit = glm::min(tExit, t1);

        if(tEnter > tExit)
        {
            return -1.0f;
        }

    }
    
    if(tExit <= 0.0f)
    {
        return -1.0f;
    }


    
    float tNear = glm::max(tEnter, 0.0f);
    // if tNear is greater than maxT, it means the intersection is farther than the closest intersection we have found so far, so we should ignore this intersection
    if (tNear > maxT) 
    {
        return -1.0f;
    }

    // Always return the nearest valid intersection distance.
    // If tEnter is negative (ray starts inside the box), tNear == 0, which is a valid hit.
    return tNear;

}

__host__ __device__ float aabbIntersectionTest(
    const MeshRange& range,
    Ray r,
    float maxT)
{
    return aabbIntersectionTest(range.aabbMin, range.aabbMax, r, maxT);
}

__device__ bool bvhIntersectionTest(
    const BVHNode* nodes,
    int rootIndex,
    const MeshRange& range,
    const Triangle* triangles,
    Ray r,
    float& bestT,
    glm::vec3& bestPoint,
    glm::vec3& bestNormal,
    glm::vec3& bestGeomNormal,
    bool& bestOutside,
    int& bestMaterialId,
    glm::vec2& bestUv)
{
    if(rootIndex < 0) return false;

    // Stack initialization
    const int STACK_SIZE = 128;
    int stack[STACK_SIZE];
    int stack_ptr = 0;
    stack[stack_ptr++] = rootIndex;

    // Main loop: while stack is not empty
    bool hit = false;

    while (stack_ptr > 0)
    {
        // pop the node
        int nodeIdx = stack[--stack_ptr];
        const BVHNode& node = nodes[nodeIdx];

        // AABB Quick Removal
        float aabbT = aabbIntersectionTest(node.aabbMin, node.aabbMax, r, bestT);
        if(aabbT < 0.0f) continue;

        // if leaf
        if(node.isLeaf)
        {
            for (int i = 0; i < node.right; ++i)
            {
                int triIdx = range.triStartIndex + node.left + i;
                const Triangle& tri = triangles[triIdx];

                glm::vec3 tmpPoint, tmpNormal, tmpGeomNormal;
                glm::vec2 tmpUv;
                bool tmpOutside;
                float t = triangleIntersectionTest(
                    tri, r,
                    tmpPoint, tmpNormal, tmpGeomNormal,
                    tmpOutside, tmpUv);

                if (t > 0.0f && t < bestT)
                {
                    bestT = t;
                    bestPoint = tmpPoint;
                    bestNormal = tmpNormal;
                    bestGeomNormal = tmpGeomNormal;
                    bestOutside = tmpOutside;
                    bestMaterialId = tri.materialId;
                    bestUv = tmpUv;
                    hit = true;
                }
            }
        }
        else
        {
            const BVHNode& leftChild = nodes[node.left];
            const BVHNode& rightChild = nodes[node.right];

            // Test the AABB of the two child nodes to obtain the nearest entry distance.
            float tLeft = aabbIntersectionTest(leftChild.aabbMin, leftChild.aabbMax, r, bestT);
            float tRight = aabbIntersectionTest(rightChild.aabbMin, rightChild.aabbMax, r, bestT);

            bool hitLeft  = (tLeft  >= 0.0f);
            bool hitRight = (tRight >= 0.0f);

            if (hitLeft && hitRight)
            {
                // If both child nodes intersect, push the farther one first, then the closer one.
                if (stack_ptr + 2 <= STACK_SIZE)
                {
                    if (tLeft < tRight)
                    {
                        stack[stack_ptr++] = node.right;
                        stack[stack_ptr++] = node.left;
                    }
                    else
                    {
                        stack[stack_ptr++] = node.left;
                        stack[stack_ptr++] = node.right;
                    }
                }
            }
            else if (hitLeft)
            {
                if (stack_ptr + 1 <= STACK_SIZE)
                {
                    stack[stack_ptr++] = node.left;
                }
            }
            else if (hitRight)
            {
                if (stack_ptr + 1 <= STACK_SIZE)
                {
                    stack[stack_ptr++] = node.right;
                }
            }

        }
    }
    

    return hit;


}