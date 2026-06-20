CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* (TODO) YOUR NAME HERE
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

![Cover Image](img/COVER_IMAGE_TODO.png)

## Overview

This project is a CUDA-based GPU path tracer capable of rendering globally-illuminated scenes with both analytic primitives (spheres and cubes) and arbitrary glTF 2.0 triangle meshes. It implements the core rendering pipeline on the GPU, including ray generation, intersection testing, BSDF evaluation, and progressive accumulation. Beyond the base requirements, the renderer features **adaptive stream compaction**, **material-type sorting**, **per-mesh AABB culling**, **per-mesh BVH acceleration**, **refraction**, **depth of field**, **Russian roulette path termination**, **texture mapping**, and full **glTF PBR material** support.

The implementation prioritizes interactive feedback through CUDA-OpenGL interop: the image is progressively refined while the user can toggle effects and move the camera in real time.

## Features

### Part 1 — Core Requirements

| Feature | Status | Implementation |
|---------|--------|----------------|
| Diffuse BSDF | ✅ | `interactions.cu::scatterRay` — cosine-weighted hemisphere sampling |
| Stream Compaction | ✅ | `pathtrace.cu` + `stream_compaction/efficient.cu` — work-efficient prefix-sum scan |
| Material Type Sort | ✅ | `pathtrace.cu::buildMaterialSortKeys` — buckets rays by material behavior before shading |
| Stochastic Antialiasing | ✅ | `pathtrace.cu::generateRayFromCamera` — per-pixel uniform jitter |

### Part 2 — Extra Features

| Feature | Points | Status | Implementation |
|---------|--------|--------|----------------|
| **glTF 2.0 Mesh Loading** | :four: | ✅ | `scene.cpp::loadGLTFObject` — node hierarchy, TRS/matrix transforms, multi-primitive, multi-material |
| **BVH Acceleration** | :six: | ✅ | `bvh.cpp` + `intersections.cu::bvhIntersectionTest` — per-`MeshRange` midpoint-split BVH, iterative GPU traversal |
| **AABB Intersection Culling** | — | ✅ | `intersections.cu::aabbIntersectionTest` — per-`MeshRange` slab test |
| **Refraction (Glass/Water)** | :two: | ✅ | `interactions.cu::scatterRay` — Snell's law + Schlick Fresnel + TIR handling |
| **Depth of Field** | :two: | ✅ | `pathtrace.cu::generateRayFromCamera` — thin-lens model with polar disk sampling |
| **Russian Roulette** | :one: | ✅ | `pathtrace.cu::russianRouletteTerminate` — Veach-style stochastic termination with 1/q throughput re-weighting |
| **Specular / Glossy Reflection** | — | ✅ | `interactions.cu::scatterRay` — perfect mirror + roughness hemisphere perturbation |
| **Base Color Texture** | — | ✅ | `pathtrace.cu::shadeFakeMaterial` — packed GPU texture atlas via `stbi_load` |
| **Emissive Texture** | — | ✅ | `pathtrace.cu::shadeFakeMaterial` — overrides material color for emitters |
| **glTF Material Features** | — | ✅ | `scene.cpp::convertGltfMaterial` — `alphaMode` (OPAQUE/MASK/BLEND), `doubleSided`, `alphaCutoff` |

**Total Part 2 score: 15+ points** (minimum required: 10).

## Build Instructions

This project uses **CMake** and requires a CUDA-capable GPU with the CUDA Toolkit installed.

```bash
# Clone or navigate to the repository
cd Project3-CUDA-Path-Tracer

# Generate build files
cmake -B build -S .

# Build
cmake --build build --config Release
```

On Windows with Visual Studio, open `build/cis565_path_tracer.sln` and build the `cis565_path_tracer` project in Release mode.

### Dependencies

- CUDA Toolkit
- OpenGL
- GLFW / GLEW / GLM (provided under `external/`)
- tinygltf v3 (provided)
- stb_image (provided)

## Running the Program

```bash
# Cornell Box (JSON scene with primitives)
./build/bin/Release/cis565_path_tracer.exe ./scenes/cornell.json

# glTF scene
./build/bin/Release/cis565_path_tracer.exe ./scenes/render_gltf.json
```

The renderer opens an interactive preview window and refines the image progressively. Press **S** to save the current image, or **Esc** to save and exit.

### Scene File Format

Scenes are defined in JSON. The format supports materials, camera settings, and objects. For glTF objects, use `"TYPE": "gltf"` and provide a `"PATH"` relative to the scene file:

```json
{
    "TYPE": "gltf",
    "MATERIAL": "car_material",
    "TRANS": [0.0, 0.0, 0.0],
    "ROTAT": [0.0, 0.0, 0.0],
    "SCALE": [1.0, 1.0, 1.0],
    "PATH": "ToyCar/glTF/ToyCar.gltf"
}
```

## Controls

| Input | Action |
|-------|--------|
| `Esc` | Save image and exit |
| `S` | Save current image |
| `Space` | Re-center camera at scene lookAt |
| Left Mouse | Rotate camera |
| Right Mouse (vertical) | Zoom in/out |
| Middle Mouse | Move lookAt point in X/Z plane |

### Runtime Toggles (ImGui)

The analytics window exposes several rendering options:

- **Enable Antialiasing (Jitter)** — stochastic sub-pixel sampling
- **Enable Stream Compaction** — remove terminated paths between bounces
- **Enable Adaptive Compaction** — compact only when the active ray ratio drops below a threshold
- **Enable Material Type Sort** — group rays by material behavior before shading
- **Enable Mesh AABB Culling** — skip whole meshes whose AABB is behind the current hit
- **Enable Mesh BVH** — use per-mesh BVH instead of brute-force triangle iteration
- **Enable Russian Roulette** — stochastically terminate low-contribution paths to save GPU work (unbiased)

## Feature Deep Dive

### glTF 2.0 Mesh Loading

The loader (`scene.cpp::loadGLTFObject`) uses tinygltf v3 to parse glTF files. It recursively traverses the scene graph, computes hierarchical transforms, and processes each primitive:

- Vertex positions are transformed to world space on the CPU.
- Normals are transformed with the inverse-transpose of the linear transform to support non-uniform scale.
- UV coordinates and per-primitive materials are preserved.
- Degenerate triangles are skipped to avoid NaN propagation.

![glTF Render TODO](img/GLTF_RENDER_TODO.png)

### BVH Acceleration

For each `MeshRange` with at least 16 triangles, a binary BVH is built on the CPU after scene loading:

1. Compute triangle centroids and AABBs.
2. Choose the split axis with the largest centroid extent.
3. Partition triangles by centroid using `std::nth_element` (midpoint split).
4. Recurse until a leaf contains at most 4 triangles or the maximum depth (32) is reached.
5. Reorder the global triangle array so that each leaf stores a contiguous block.

On the GPU, `bvhIntersectionTest` performs iterative stack-based traversal:

- Test each visited node's AABB against the ray, using the current closest `t` to prune.
- At internal nodes, test both children and push the farther one first so the nearer one is popped first (near-first traversal).
- At leaves, test all triangles and update the closest hit.

The BVH can be toggled at runtime to compare performance against brute-force iteration.

![BVH Comparison TODO](img/BVH_COMPARISON_TODO.png)

### Stream Compaction

After each bounce, terminated paths are gathered into the accumulation buffer. Active paths are then compacted using a work-efficient exclusive prefix-sum scan (`stream_compaction/efficient.cu`). **Adaptive compaction** avoids paying the scan cost early in a path when most rays are still alive; it only compacts when the active ratio falls below a user-defined threshold (default 0.70) and enough paths remain (default 4096).

Stream compaction helps most after several bounces, especially in closed scenes where many rays terminate on light sources.

![Stream Compaction TODO](img/STREAM_COMPACTION_TODO.png)

### Material Type Sort

Before shading, active paths are sorted by material bucket using `thrust::sort_by_key`:

1. Emissive
2. Specular
3. Refractive
4. Diffuse
5. Miss
6. Dead

Grouping similar materials reduces warp divergence in the shading mega-kernel, because threads in the same warp execute similar code paths.

### Refraction

Refractive materials use `glm::refract` for Snell's law. The Schlick approximation estimates Fresnel reflectance:

```
r0 = ((1 - ior) / (1 + ior))^2
reflectProb = r0 + (1 - r0) * (1 - cosθ)^5
```

If total internal reflection occurs or a random sample is below `reflectProb`, the ray reflects; otherwise it refracts. The ray origin is offset along the surface normal to avoid self-intersection.

![Refraction TODO](img/REFRACTION_TODO.png)

### Depth of Field

When the camera has a non-zero aperture radius, rays are sampled on a thin lens. The pinhole ray is first cast through the pixel, then its intersection with the focal plane is computed. The actual ray originates from a sampled lens position and is directed toward that focal point, producing physically-based defocus blur.

![Depth of Field TODO](img/DOF_TODO.png)

### Texture Mapping

Base color and emissive textures are loaded with `stbi_load`, packed into a single GPU texture atlas, and sampled with bilinear-ish nearest-neighbor lookup during shading. UV coordinates are interpolated at triangle intersections and wrapped to `[0, 1]`.

![Texture TODO](img/TEXTURE_TODO.png)

### Russian Roulette

After each scatter, `russianRouletteTerminate` offers the path to the roulette. The survival probability is `q = clamp(max(throughput.rgb), 0.05, 0.95)`:

1. **Re-weight** `throughput /= q` so the estimator stays unbiased.
2. Sample `ξ ~ U(0,1)`. If `ξ < 1 - q`, terminate the path.

Expected contribution `q · (throughput / q) + (1 - q) · 0 = throughput`, so the estimator is preserved.

The helper is invoked in `shadeFakeMaterial` immediately **after** `scatterRay`, which guarantees that paths that already terminated cleanly (light hit, miss, alpha discard) are not re-rolled. When disabled, the helper short-circuits and the renderer behaves exactly as it did with fixed-depth termination.

Because each thread already carries its own `PathSegment`, RR on the GPU costs almost nothing — one clamp, one divide, one RNG draw per scatter — whereas a CPU implementation would have to maintain per-pixel state on the host. The biggest performance win is in closed scenes where most paths terminate on diffuse bounces; RR prunes the long tail of low-energy paths early, letting stream compaction (which also benefits) kick in sooner.

![Russian Roulette TODO](img/RUSSIAN_ROULETTE_TODO.png)

## Performance Analysis

All timings below were collected with Nsight on the test machine listed at the top of this README. Replace the placeholders with your actual measurements.

### BVH vs. Brute Force

| Scene | Triangles | Brute Force (ms) | BVH (ms) | Speedup |
|-------|-----------|------------------|----------|---------|
| ToyCar | 108,936 | TODO | TODO | TODOx |
| Cornell (no glTF) | 0 | TODO | N/A | N/A |

Key observation: BVH benefits increase with triangle count and scene depth complexity. For very small meshes, the overhead of tree traversal can outweigh the savings.

### Stream Compaction Effect

| Depth | Active Rays (Open Scene) | Active Rays (Closed Scene) |
|-------|--------------------------|----------------------------|
| 0 | TODO | TODO |
| 1 | TODO | TODO |
| 2 | TODO | TODO |
| 3 | TODO | TODO |
| 4 | TODO | TODO |

Key observation: stream compaction provides larger savings in closed scenes where rays terminate quickly on emissive surfaces. In open scenes, most rays escape to the background and stay active longer.

### Material Sort Impact

Material sorting reduces warp divergence in the shading kernel. The benefit is most visible in scenes with many different materials interleaved across the image.

| Scene | No Sort (ms) | Sort (ms) | Improvement |
|-------|--------------|-----------|-------------|
| Cornell | TODO | TODO | TODO% |

## Known Issues & Future Work

- **No global BVH**: each `MeshRange` has its own BVH, but there is no top-level BVH across mesh ranges. Scenes with many small primitives still pay per-mesh overhead.
- **No direct lighting / NEE**: convergence in scenes with small bright lights can be slow.
- **Glossy energy conservation**: the roughness blur does not currently divide by the sampling PDF, leading to a slight brightness bias.
- **No tone mapping / gamma correction**: linear radiance is written directly to the PBO.

## References

- [PBRTv3 / PBRTv4](https://www.pbr-book.org/)
- [GPU Gems 3, Chapter 39: Parallel Prefix Sum (Scan) with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)
- [GPU Gems 3, Chapter 20: GPU-Based Importance Sampling](https://developer.nvidia.com/gpugems/gpugems3/part-iii-rendering/chapter-20-gpu-based-importance-sampling)
- [Paul Bourke — Raytracing Notes](https://paulbourke.net/miscellaneous/raytracing/)
- [tinygltf](https://github.com/syoyo/tinygltf/)

## Credits

- Base code: CIS 5650 Project 3 CUDA Path Tracer
- glTF loader: tinygltf v3
- Texture loading: stb_image
- (TODO: add any third-party models or assets used)
