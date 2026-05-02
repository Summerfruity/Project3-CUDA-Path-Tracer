#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

// Geometry primitive types supported by the renderer.
enum GeomType
{
    SPHERE,
    CUBE
};

struct Ray
{
    // World-space ray origin.
    glm::vec3 origin;

    // World-space ray direction (typically normalized).
    glm::vec3 direction;
};

struct Geom
{
    // Primitive type.
    enum GeomType type;

    // Index into the global materials array (Material*).
    int materialid;

    // SRT components (typically read from the scene file).
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;

    // Object-to-world transform matrix.
    glm::mat4 transform;

    // World-to-object transform matrix (inverse of transform).
    glm::mat4 inverseTransform;

    // Inverse-transpose of transform, used to transform normals correctly.
    glm::mat4 invTranspose;
};

struct Material
{
    // Base color / albedo used for diffuse shading.
    glm::vec3 color;

    // Specular (glossy) parameters. How you interpret these is up to your BSDF.
    struct
    {
        // Specular exponent / shininess (often used as a roughness proxy).
        float exponent;

        // Specular color / tint.
        glm::vec3 color;
    } specular;

    // Flags (often treated as booleans in [0,1]) indicating supported lobes.
    float hasReflective;
    float hasRefractive;

    // Index of refraction (IOR) for refractive materials (e.g., glass ~ 1.5).
    float indexOfRefraction;

    // Emission strength. If > 0, the material acts like a light source.
    float emittance;
};

struct Camera
{
    // Output image resolution in pixels.
    glm::ivec2 resolution;

    // World-space camera position.
    glm::vec3 position;

    // World-space point the camera is looking at.
    glm::vec3 lookAt;

    // Forward/view direction (typically normalized).
    glm::vec3 view;

    // Camera up direction (typically normalized).
    glm::vec3 up;

    // Camera right direction (typically normalized).
    glm::vec3 right;

    // Field-of-view in degrees: (fovx, fovy).
    glm::vec2 fov;

    // Size of a pixel on the image plane in world/viewport units.
    glm::vec2 pixelLength;

    /**
     * apertureRadius = 0: pinhole camera (no depth of field)
     * apertureRadius > 0: depth of field
     */
    float apertureRadius; 

    // distance from the camera position to the focal plane
    float focalDistance;
};

struct RenderState
{
    // Camera parameters for the current render.
    Camera camera;

    // Number of iterations/samples per pixel to accumulate.
    unsigned int iterations;

    // Maximum path length (number of bounces).
    int traceDepth;

    // Host-side accumulation buffer (one vec3 per pixel).
    std::vector<glm::vec3> image;

    // Output filename for saved renders.
    std::string imageName;
};

struct PathSegment
{
    // The ray to trace for this path.
    Ray ray;

    // final Radiance accumulated along this path so far.
    glm::vec3 color;

    // Throughput is the cumulative product of BSDFs and cosines along the path, used to weight the contribution of this path to the final image.
    glm::vec3 throughput;

    // Index into the output image buffer.
    int pixelIndex;

    // How many more bounces this path is allowed to take.
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
    // Parametric distance along the ray. Convention: t < 0 means "no hit".
  float t;

    // World-space surface normal at the intersection.
  glm::vec3 surfaceNormal;

    // Index into the global materials array (Material*).
  int materialId;
};
