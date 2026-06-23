CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* (TODO) YOUR NAME HERE
* Tested on: (TODO) Windows, CPU/GPU details

![Cover Image](img/COVER_IMAGE_TODO.png)

## Overview

This project is a CUDA-based GPU path tracer that renders globally illuminated scenes with analytic primitives (spheres and cubes) and arbitrary glTF 2.0 triangle meshes. The renderer implements ray generation, scene intersection, BSDF scattering, progressive accumulation, and interactive CUDA-OpenGL preview.

Beyond the core requirements, the renderer includes adaptive stream compaction, material-type sorting, top-level and per-mesh BVH acceleration, JSON-defined refraction, depth of field, Russian roulette path termination, texture mapping, and a practical subset of glTF material support.

## Features

### Core Requirements

| Feature | Status | Implementation |
|---------|--------|----------------|
| Diffuse BSDF | Done | `interactions.cu::scatterRay`, cosine-weighted hemisphere sampling |
| Stream Compaction | Done | `pathtrace.cu` + `stream_compaction/efficient.cu`, work-efficient prefix-sum scan |
| Material Type Sort | Done | `pathtrace.cu::buildMaterialSortKeys`, buckets rays by material behavior |
| Stochastic Antialiasing | Done | `pathtrace.cu::generateRayFromCamera`, per-pixel jitter |

### Extra Features

| Feature | Points | Status | Implementation |
|---------|--------|--------|----------------|
| glTF 2.0 Mesh Loading | 4 | Done | `scene.cpp::loadGLTFObject`, node hierarchy, TRS/matrix transforms, multi-primitive, multi-material |
| BVH Acceleration | 6 | Done | Top-level `MeshRange` BVH plus per-mesh triangle BVHs |
| AABB Culling | Included | Done | `intersections.cu::aabbIntersectionTest`, mesh-range slab tests |
| Refraction | 2 | Done | JSON `Refractive` materials use Snell's law, Schlick Fresnel, and TIR handling |
| Depth of Field | 2 | Done | Thin-lens camera model in `generateRayFromCamera` |
| Russian Roulette | 1 | Done | Veach-style stochastic termination with throughput re-weighting |
| Texture Mapping | Included | Done | glTF base-color and emissive textures via a packed GPU texture atlas |
| glTF Material Subset | Included | Done | Base color, metallic/roughness approximation, emissive/base-color textures, `alphaMode`, `doubleSided`, `alphaCutoff` |

**Important limitation:** glTF `KHR_materials_transmission` is not currently mapped to the internal refractive material path. ToyCar-style glTF glass may appear tinted, but it is not physically refractive glass in the current implementation.

## Build Instructions

This project uses CMake and requires a CUDA-capable GPU with the CUDA Toolkit installed.

```bash
cmake -B build -S .
cmake --build build --config Release
```

On Windows with Visual Studio, open `build/cis565_path_tracer.sln` and build `cis565_path_tracer` in Release mode.

## Running

```bash
# Cornell Box / primitive scene
.\build\bin\Release\cis565_path_tracer.exe .\scenes\cornell.json

# glTF scene
.\build\bin\Release\cis565_path_tracer.exe .\scenes\render_gltf.json
```

The renderer opens an interactive preview window and progressively refines the image. Press `S` to save the current image, or `Esc` to save and exit.

## Scene Format

Scenes are JSON files. Analytic primitives use `"TYPE": "cube"` or `"TYPE": "sphere"`. glTF meshes use `"TYPE": "gltf"` with a `PATH` relative to the scene file:

```json
{
    "TYPE": "gltf",
    "MATERIAL": "diffuse_white",
    "PATH": "ToyCar/glTF/ToyCar.gltf",
    "TRANS": [0.0, 0.0, 0.0],
    "ROTAT": [0.0, 0.0, 0.0],
    "SCALE": [60.0, 60.0, 60.0]
}
```

## Runtime Toggles

The ImGui panel exposes:

- Enable Antialiasing
- Enable Stream Compaction
- Enable Adaptive Compaction
- Enable Material Type Sort
- Enable Mesh AABB Culling
- Enable Mesh BVH
- Enable Russian Roulette

When Mesh BVH is enabled, mesh rays first traverse a top-level BVH over `MeshRange` AABBs, then enter the per-mesh triangle BVH for candidate mesh ranges. When disabled, the renderer falls back to brute-force mesh-range iteration.

## Implementation Notes

### glTF Loading

`Scene::loadGLTFObject` uses tinygltf v3 to parse glTF files. It recursively traverses the glTF scene graph, applies node transforms plus the JSON object transform, extracts triangle primitives, and stores world-space triangles in the renderer's native `Triangle` buffer.

Supported mesh data:

- `POSITION` as `VEC3 FLOAT`
- Optional `NORMAL` as `VEC3 FLOAT`
- Optional `TEXCOORD_0` as `VEC2 FLOAT`
- Indexed and non-indexed triangle primitives
- Per-primitive material remapping

Unsupported glTF material features include normal maps, roughness/metallic textures, clearcoat, sheen, and `KHR_materials_transmission`.

### BVH Acceleration

The mesh acceleration structure has two levels:

1. **Per-mesh BVH:** each sufficiently large `MeshRange` builds a midpoint-split triangle BVH. The global triangle buffer is reordered so each leaf references a contiguous triangle block.
2. **Top-level BVH:** all `MeshRange` AABBs are grouped into a second BVH. GPU traversal first prunes whole mesh ranges, then tests the relevant per-mesh BVH.

Both levels use iterative stack traversal on the GPU.

### Refraction

Refraction is implemented for JSON `Refractive` materials. The scatter path uses `glm::refract`, Schlick Fresnel probability, and total internal reflection handling. This code path is not currently connected to glTF transmission materials.

### Texture Mapping

Base-color and emissive textures are loaded with `stbi_load`, packed into a GPU texture atlas, and sampled from interpolated triangle UVs. The current sampler is nearest-neighbor with wrapping.

### Russian Roulette

Russian roulette computes:

```text
q = clamp(max(throughput.r, throughput.g, throughput.b), 0.05, 0.95)
```

Surviving paths are re-weighted by `1 / q` so the estimator remains unbiased.

## Known Issues & Future Work

- glTF `KHR_materials_transmission` is not implemented; glTF glass is not physically refractive.
- glTF normal maps, roughness/metallic textures, clearcoat, and sheen are ignored.
- Texture alpha is not loaded because `stbi_load` currently requests RGB data.
- No direct lighting / next-event estimation, so small lights converge slowly.
- No tone mapping or gamma correction; linear radiance is written directly to the preview buffer.
- BVH construction uses midpoint splits rather than SAH.

## References

- [PBRT](https://www.pbr-book.org/)
- [GPU Gems 3, Chapter 39: Parallel Prefix Sum (Scan) with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)
- [tinygltf](https://github.com/syoyo/tinygltf/)
- [stb_image](https://github.com/nothings/stb)

## Credits

- Base code: CIS 5650 Project 3 CUDA Path Tracer
- glTF loader: tinygltf v3
- Texture loading: stb_image
- (TODO: add third-party model or asset credits)
