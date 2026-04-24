#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q; // ray in the coordinate system of the box

    // transform the ray into the space of the box
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n; // normal of the plane that gives us tmin
    glm::vec3 tmax_n; // normal of the plane that gives us tmax
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz]; // component of the ray direction in the current axis, which is the normal of the planes we are intersecting with
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n; // normal of the plane we are intersecting with, in the space of the box. We will transform it back to world space at the end.
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
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
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
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

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}


__host__ __device__ bool aabbIntersectionTest(
    const glm::vec3& bmin, // (xmin, ymin, zmin)
    const glm::vec3& bmax, // (xmax, ymax, zmax)
    const Ray& r,
    float& tEnter,
    float& tExit
)
{

    const float kInf = 1e38f;
    const float kEps = 1e-8f;

    float tmin = kEps;
    float tmax = kInf;

    for(int axis = 0; axis < 3; ++axis)
    {
        float origin = r.origin[axis];
        float dir = r.direction[axis];

        float minv = bmin[axis];
        float maxv = bmax[axis];

        // if the ray is parallel to axis && origin is not inside the box, return false
        if(fabsf(dir) < kEps)
        {
            if(origin < minv || origin > maxv)
            {
                return false;
            }
            continue;
        }

        float t1 = (minv - origin) / dir;
        float t2 = (maxv - origin) / dir;
        float ta = fminf(t1, t2);
        float tb = fmaxf(t1, t2);

        // update tmin and tmax
        tmin = fmaxf(tmin, ta);
        tmax = fminf(tmax, tb);

        // if the ray misses the box, return false
        if (tmax < tmin)
        {
            return false;
        }

    }

    tEnter = tmin;
    tExit  = tmax;

    return tExit >= 0.0f;

}

__host__ __device__ float triangleIntersectionTest(
    const Triangle& triangle,
    const Ray& r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside
)
{
    // Ensure we use a unit direction so returned t is a world-space distance.
    Ray ray = r;
    ray.direction = glm::normalize(ray.direction);

    glm::vec3 baryPosition;
    bool hit = glm::intersectRayTriangle(
        ray.origin,
        ray.direction,
        triangle.v1,
        triangle.v2,
        triangle.v3,
        baryPosition
    );

    if (!hit)
    {
        return -1.0f;
    }

    // GLM writes (u, v, t) into baryPosition.
    float t = baryPosition.z;
    if (t <= 0.0f)
    {
        return -1.0f;
    }

    intersectionPoint = getPointOnRay(ray, t);

    // Compute geometric normal as a fallback.
    glm::vec3 edge1 = triangle.v2 - triangle.v1;
    glm::vec3 edge2 = triangle.v3 - triangle.v1;
    glm::vec3 faceNormal = glm::normalize(glm::cross(edge1, edge2));

    if (triangle.hasVertexNormals != 0)
    {
        const float u = baryPosition.x;
        const float v = baryPosition.y;
        const float w = 1.0f - u - v;

        glm::vec3 interpNormal = w * triangle.n1 + u * triangle.n2 + v * triangle.n3;
        if (glm::length2(interpNormal) > 0.0f)
        {
            // If the interpolated normal is non-zero, use it. Otherwise, fall back to the face normal.
            normal = glm::normalize(interpNormal);
        }
        else
        {
            normal = faceNormal;
        }
    }
    else
    {
        normal = faceNormal;
    }

    outside = glm::dot(ray.direction, normal) < 0.0f;

    return glm::length(ray.origin - intersectionPoint);

}