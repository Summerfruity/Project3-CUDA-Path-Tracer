# Project 3 CUDA Path Tracer - Implementation Status

> **Course**: CIS 5650 - GPU Programming & Architecture  
> **Project**: CUDA Path Tracer  
> **Scene Formats**: JSON primitives + glTF 2.0 triangle meshes  
> **Build System**: CMake, CUDA/C++17, CUDA-OpenGL interop preview

---

## 1. Executive Summary

The current renderer supports classic analytic primitives (spheres and cubes) plus arbitrary glTF 2.0 triangle meshes. It implements GPU path tracing with diffuse/specular/refractive BSDFs, stochastic antialiasing, stream compaction, material sorting, depth of field, Russian roulette, texture mapping, and mesh acceleration.

Mesh acceleration now has two levels:

- A top-level BVH over `MeshRange` AABBs.
- Per-mesh triangle BVHs for large `MeshRange`s.

glTF loading supports geometry, transforms, UVs, base-color textures, emissive textures, and a practical subset of material fields. **glTF transmission/glass is not currently implemented**: `KHR_materials_transmission` is not mapped to the internal refractive material path.

---

## 2. Feature Matrix

### Part 1 - Core Features

| Feature | Status | File / Kernel | Notes |
|---------|--------|---------------|-------|
| Diffuse BSDF | Done | `interactions.cu::scatterRay` | Cosine-weighted hemisphere sampling |
| Stream Compaction | Done | `pathtrace.cu` + `stream_compaction/efficient.cu` | Work-efficient scan plus adaptive gating |
| Material Type Sort | Done | `pathtrace.cu::buildMaterialSortKeys` | Buckets rays by material behavior |
| Stochastic Antialiasing | Done | `pathtrace.cu::generateRayFromCamera` | Per-pixel jitter |

### Part 2 - Extra Features

| Feature | Points | Status | File / Kernel | Notes |
|---------|--------|--------|---------------|-------|
| glTF 2.0 Mesh Loading | 4 | Done | `scene.cpp::loadGLTFObject` | Node hierarchy, TRS/matrix transforms, multi-primitive, multi-material |
| AABB Intersection Culling | Included | Done | `intersections.cu::aabbIntersectionTest` | Mesh-range slab tests |
| BVH Acceleration | 6 | Done | `bvh.cpp`, `pathtrace.cu`, `intersections.cu` | Top-level `MeshRange` BVH plus per-mesh triangle BVHs |
| Refraction | 2 | Done | `interactions.cu::scatterRay` | JSON `Refractive` materials only |
| Depth of Field | 2 | Done | `pathtrace.cu::generateRayFromCamera` | Thin-lens model |
| Russian Roulette | 1 | Done | `pathtrace.cu::russianRouletteTerminate` | Unbiased throughput re-weighting |
| Base Color Texture | Included | Done | `scene.cpp`, `pathtrace.cu::shadeFakeMaterial` | glTF base-color texture loaded with `stbi_load` and sampled from UVs |
| Emissive Texture | Included | Done | `scene.cpp`, `pathtrace.cu::shadeFakeMaterial` | glTF emissive texture support |
| glTF Material Subset | Included | Done | `scene.cpp::convertGltfMaterial` | Base color, metallic/roughness approximation, `alphaMode`, `doubleSided`, `alphaCutoff` |

### Not Implemented / Future Work

| Feature | Notes |
|---------|-------|
| glTF transmission / glass | `KHR_materials_transmission` is not converted to `hasRefractive`; glTF glass may appear tinted but is not physically refractive |
| glTF normal mapping | Normals are loaded, but normal textures and TBN are not implemented |
| glTF metallic/roughness texture | Scalar metallic/roughness factors are approximated; packed ORM textures are ignored |
| glTF clearcoat / sheen | Extension fields are ignored |
| Direct lighting / NEE | Standard path tracing only |
| Tone mapping / gamma correction | Linear radiance is written directly to the preview/output buffer |
| Denoising | No CPU/GPU denoiser integration |
| Motion blur | No time sampling |

---

## 3. Architecture & Code Tour

### 3.1 glTF Pipeline (`scene.cpp`)

The scene loader handles JSON scenes and recognizes `"TYPE": "gltf"` objects. The glTF path is resolved relative to the scene JSON and then passed to `Scene::loadGLTFObject`.

#### 3.1.1 Parsing & Node Hierarchy

- Parser: tinygltf v3 (`tg3_*` API).
- Node traversal: recursive traversal of the glTF scene graph.
- Node transforms: supports explicit matrix and TRS composition.
- Final transform: JSON object transform is multiplied with the internal glTF node hierarchy transform.

#### 3.1.2 Triangle Extraction

For each triangle primitive:

1. Locate `POSITION`, optional `NORMAL`, and optional `TEXCOORD_0`.
2. Read indexed or non-indexed triangle data.
3. Transform positions into world space.
4. Transform normals by the inverse-transpose of the linear transform.
5. Compute face-normal fallback when vertex normals are absent.
6. Skip degenerate triangles.
7. Store vertices, normals, UVs, and material ID in the global `Triangle` array.

#### 3.1.3 MeshRange and AABB Construction

Each glTF primitive becomes a `MeshRange`:

```cpp
struct MeshRange {
    int triStartIndex;
    int triCount;
    glm::vec3 aabbMin;
    glm::vec3 aabbMax;
    int bvhRootIndex;
    int bvhNodeCount;
};
```

The AABB is accumulated from transformed triangle vertices. These ranges are used both for direct AABB culling and for top-level BVH construction.

---

## 3.2 GPU Ray Intersection

`computeIntersections` operates in two phases:

1. Analytic primitive pass over JSON `Geom`s.
2. Mesh pass over glTF triangles.

When **Enable Mesh BVH** is on and TLAS nodes exist, mesh traversal uses:

1. Top-level BVH over `MeshRange` AABBs.
2. Per-mesh triangle BVH for each selected range.

When Mesh BVH is off, the renderer falls back to the older path that scans every `MeshRange`, optionally using per-range AABB culling before brute-force triangle tests.

### Triangle Intersection

`triangleIntersectionTest` uses Moller-Trumbore in world space. Barycentric coordinates interpolate smooth normals and UVs. The `ShadeableIntersection` stores both:

- `surfaceNormal`: face-forward shading normal.
- `geometricNormal`: outward geometric normal.

This split keeps diffuse/specular shading and refractive enter/exit logic separate.

---

## 3.3 BVH Acceleration

### Per-Mesh BVH

For each `MeshRange` with at least 16 triangles:

1. Build triangle primitives with centroids and AABBs.
2. Split on the largest centroid extent.
3. Partition with `std::nth_element`.
4. Stop at max leaf size 4 or max depth 32.
5. Reorder the global triangle array so leaves point to contiguous triangle blocks.

### Top-Level BVH

`buildBVHForMeshRanges` builds a second BVH over all `MeshRange` AABBs. It also reorders `meshRanges` to match TLAS leaf order. The GPU traversal in `pathtrace.cu` first prunes whole mesh ranges before entering per-mesh BVHs.

This removes the previous limitation where every ray had to linearly test every mesh-range AABB before finding relevant triangle BVHs.

---

## 3.4 BSDF & Shading

### Diffuse

Diffuse surfaces sample a cosine-weighted hemisphere around the shading normal.

### Specular / Glossy

Specular surfaces use perfect reflection, with optional roughness blur by mixing the mirror direction with a sampled hemisphere direction.

### Refraction

JSON `Refractive` materials use:

- Snell refraction via `glm::refract`.
- Schlick Fresnel probability.
- Total internal reflection handling.
- A single offset in `scatterRay` to avoid self-intersection.

glTF `KHR_materials_transmission` is currently **not** connected to this path.

### Texture Sampling

Base-color and emissive textures are loaded with `stbi_load`, packed into GPU buffers, and sampled in `shadeFakeMaterial` from interpolated UVs. The current sampler is nearest-neighbor with wrapping.

---

## 4. Known Issues & Limitations

| Issue | Severity | Details |
|-------|----------|---------|
| glTF transmission/glass unsupported | Medium | `KHR_materials_transmission` is ignored; glTF glass is not physically refractive |
| No glTF normal maps | Medium | Normal attributes are used, but normal textures and TBN are not implemented |
| No glTF roughness/metallic textures | Medium | Scalar material factors are approximated; packed ORM texture maps are ignored |
| Alpha from texture unavailable | Low | `stbi_load` requests RGB, so texture alpha is unavailable for alpha masking |
| No direct lighting | Medium | Small lights converge slowly without next-event estimation |
| No tone mapping / gamma correction | Low | Linear radiance is written directly to the output |
| BVH build quality | Low | Midpoint split is simple; SAH/binned SAH could improve traversal |

---

## 5. How to Run

```powershell
# JSON scene
.\build\bin\Release\cis565_path_tracer.exe .\scenes\cornell.json

# glTF scene
.\build\bin\Release\cis565_path_tracer.exe .\scenes\render_gltf.json
```

Runtime toggles:

- Enable Antialiasing
- Enable Stream Compaction
- Enable Adaptive Compaction
- Enable Material Type Sort
- Enable Mesh AABB Culling
- Enable Mesh BVH
- Enable Russian Roulette

---

*Last updated: 2026-06-23*
