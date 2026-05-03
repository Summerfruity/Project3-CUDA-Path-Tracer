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
