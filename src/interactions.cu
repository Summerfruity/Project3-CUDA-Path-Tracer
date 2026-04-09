#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

// Scatter a ray off the surface, depending on the material properties.
__host__ __device__ void scatterRay(
    PathSegment & pathSegment, // pass by ref, modify it in place
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // TODO: implement this.
    // Now just purely diffuse shading. 

    // flip the normal if the ray is inside the surface
    if(glm::dot(normal, pathSegment.ray.direction) > 0)
    {
        normal = -normal;
    }

    // if the material is emissive, don't scatter the ray, just return
    if(m.emittance > 0.0f)
    {
        pathSegment.color *= m.color * m.emittance;
        pathSegment.remainingBounces = 0;
        return; 
    }

    // 1. get the new direction
    glm::vec3 newDirection = calculateRandomDirectionInHemisphere(normal, rng);

    // 2. update the ray's dir
    pathSegment.ray.direction = glm::normalize(newDirection);

    // 3. update the ray's origin as intersect, need to add an offset in normal direction
    pathSegment.ray.origin = intersect + EPSILON * normal;

    // 4. color change(Energy decay)
    pathSegment.color *= m.color;

    // 5. count a bounce
    pathSegment.remainingBounces--;
}
