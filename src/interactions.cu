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

    // flip the normal if the ray is inside the surface
    if(glm::dot(normal, pathSegment.ray.direction) > 0)
    {
        normal = -normal;
    }

    thrust::uniform_real_distribution<float> u01(0, 1);
    float xi = u01(rng);
    float pSpec = glm::clamp(m.hasReflective, 0.0f, 1.0f);
    glm::vec3 outDir;

    if(xi < pSpec)
    {
        // Specular branch: perfect reflection + roughness blur
        glm::vec3 mirrorDir = glm::normalize(glm::reflect(pathSegment.ray.direction, normal));

        float roughness = glm::clamp(m.specular.exponent, 0.0f, 1.0f);
        if(roughness > SPECULAR_ROUGHNESS_EPS)
        {
            // Glossy reflection: mix the perfect reflection direction with a random direction in the hemisphere
            glm::vec3 glossyDir = calculateRandomDirectionInHemisphere(mirrorDir, rng);
            outDir = glm::normalize(glm::mix(mirrorDir, glossyDir, roughness * roughness));
        }
        else
        {
            outDir = mirrorDir;
        }

        // Update the path throughput by multiplying with the specular color and the probability of choosing the specular branch
        float invProb = (pSpec > 0.0f) ? (1.0 / pSpec) : 0.0f;
        pathSegment.throughput *= m.specular.color * invProb;

    }
    else
    {
        // Diffuse branch: sample a random direction in the hemisphere
        outDir = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));

        float pDiff = 1.0f - pSpec;
        float invProb = (pDiff > 0.0f) ? (1.0 / pDiff) : 0.0f;
        pathSegment.throughput *= m.color * invProb;
    }
    

    // update the ray's dir
    pathSegment.ray.direction = outDir;

    // update the ray's origin as intersect, need to add an offset in normal direction
    float side = glm::dot(outDir, normal) > 0.0f ? 1.0f : -1.0f;
    pathSegment.ray.origin = intersect + side * EPSILON * normal;

    // record a bounce count
    pathSegment.remainingBounces--;
}
