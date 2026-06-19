# Project 3 CUDA Path Tracer — Implementation Status

> **Course**: CIS 5650 — GPU Programming & Architecture  
> **Project**: CUDA Path Tracer  
> **Scene Formats**: JSON ( Cornell Box / primitives ) + glTF 2.0 ( arbitrary mesh )  
> **Build System**: CMake (CUDA 17 + C++17), CUDA–OpenGL interop for live preview

---

## 1. Executive Summary

This project implements a fully functional GPU path tracer using CUDA. The renderer supports classic primitives (spheres & cubes) as well as arbitrary glTF 2.0 triangle meshes with hierarchical transforms, PBR materials, and texture mapping. Beyond the core Part 1 requirements, the implementation includes **refraction**, **depth of field**, **adaptive stream compaction**, **material type sorting**, **per-mesh AABB culling**, **per-mesh BVH acceleration**, and **glTF alpha/double-sided material extensions**. The total feature score exceeds the Part 2 minimum of **10 points**.

---

## 2. Feature Matrix

### Part 1 — Core Features (Required)

| Feature | Status | File / Kernel | Notes |
|---------|--------|---------------|-------|
| Diffuse BSDF | ✅ | `interactions.cu::scatterRay` | Cosine-weighted hemisphere sampling |
| Stream Compaction | ✅ | `pathtrace.cu` + `stream_compaction/efficient.cu` | Work-efficient prefix-sum scan; adaptive threshold toggle |
| Material Type Sort | ✅ | `pathtrace.cu::buildMaterialSortKeys` | Buckets: Emissive, Specular, Refractive, Diffuse, Miss, Dead |
| Stochastic Antialiasing | ✅ | `pathtrace.cu::generateRayFromCamera` | Per-pixel uniform jitter; fallback to center sampling when disabled |

### Part 2 — Extra Features

| Feature | Points | Status | File / Kernel | Notes |
|---------|--------|--------|---------------|-------|
| **glTF 2.0 Mesh Loading** | :four: | ✅ | `scene.cpp::loadGLTFObject` | Node hierarchy, TRS/matrix transforms, multi-primitive, multi-material |
| **AABB Intersection Culling** | — | ✅ | `intersections.cu::aabbIntersectionTest` | Per-`MeshRange` slab method; togglable via ImGui |
| **Refraction (Glass/Water)** | :two: | ✅ | `interactions.cu::scatterRay` | Snell's law + Schlick Fresnel + TIR handling |
| **Depth of Field** | :two: | ✅ | `pathtrace.cu::generateRayFromCamera` | Thin-lens model, polar disk sampling |
| **Specular / Glossy Reflection** | — | ✅ | `interactions.cu::scatterRay` | Perfect mirror + roughness hemisphere perturbation |
| **BVH Acceleration (per-mesh)** | :six: | ✅ | `bvh.cpp` + `intersections.cu::bvhIntersectionTest` | Midpoint-split BVH built on CPU; iterative GPU traversal with near-first stack |
| **Base Color Texture** | — | ✅ | `pathtrace.cu::shadeFakeMaterial` | `stbi_load` → packed GPU texture atlas |
| **Emissive Texture** | — | ✅ | `pathtrace.cu::shadeFakeMaterial` | Overrides material color for light emitters |
| **glTF Material Features** | — | ✅ | `scene.cpp::convertGltfMaterial` | `alphaMode` (OPAQUE/MASK/BLEND), `doubleSided`, `alphaCutoff` |

### Not Implemented (Future Work)

| Feature | Points | Why Not |
|---------|--------|---------|
| Bump / Normal Mapping | :five:/:six: | Infrastructure ready (normals + UVs exist), but TBN matrix not computed |
| Russian Roulette | :one: | Fixed `traceDepth` termination only |
| Direct Lighting (NEE) | :two: | Standard path tracing only; no explicit light sampling per bounce |
| Wavefront Path Tracing | :six: | Single mega-kernel (`shadeFakeMaterial`) instead of material-specific kernels |
| OIDN Denoiser | :three: | No CPU denoiser integration |
| Procedural Shapes / Textures | :four: | All geometry loaded from files |
| Motion Blur | :three: | No time-dimension sampling |
| Post-processing Shaders | :three: | Linear color output directly to PBO; no tone mapping / gamma correction |

---

## 3. Architecture & Code Tour

### 3.1 glTF Pipeline (`scene.cpp`)

The scene loader supports both JSON (for simple primitives) and glTF 2.0 (for arbitrary meshes). When a JSON object has `"TYPE": "gltf"`, the path is resolved relative to the scene file and passed to `loadGLTFObject`.

#### 3.1.1 Parsing & Node Hierarchy
- **Parser**: `tinygltf v3` (`tg3_*` API) is used to load the binary/JSON glTF file.
- **Node traversal**: a recursive lambda `traverseNode(nodeIndex, parentTransform)` walks the scene graph:
  ```cpp
  glm::mat4 nodeTransform = parentTransform * nodeLocalTransform(node);
  glm::mat4 finalTransform = objectTransform * nodeTransform;
  ```
  `objectTransform` is the scene-level transform from the JSON file (translation, rotation, scale); `nodeTransform` is the internal glTF node hierarchy transform.
- **Local transform**: `nodeLocalTransform` supports both:
  - Explicit `4×4` matrix (`node.has_matrix`), read in **column-major** order to match glm.
  - TRS decomposition: `translate * rotate * scale`.

#### 3.1.2 Triangle Extraction (`processPrimitive`)
For each primitive in a mesh:
1. **Attribute accessors** are located by name (`POSITION`, `NORMAL`, `TEXCOORD_0`).
2. **Index buffer** is read if present (`prim.indices >= 0`); otherwise vertices are assumed to be a flat triangle list.
3. **Vertex transform**: positions are transformed by `finalTransform` (object × node hierarchy). Normals use the **inverse-transpose** of the linear part (`glm::mat3(transpose(inverse(transform)))`) to handle non-uniform scale correctly.
4. **Normal fallback**: if no vertex normals exist, face normals are computed via `cross(v1-v0, v2-v0)`.
5. **Degenerate triangles** (zero-area) are skipped to avoid NaN propagation.
6. **Material remapping**: glTF material indices are mapped to the internal `materials` vector; if a primitive has no material, it falls back to the material specified in the JSON scene file.

#### 3.1.3 AABB Construction
During `pushTriangle`, per-primitive axis-aligned bounding boxes are accumulated:
```cpp
aabbMin = min(aabbMin, p0/p1/p2);
aabbMax = max(aabbMax, p0/p1/p2);
```
After all triangles of a primitive are processed, a `MeshRange` is pushed:
```cpp
struct MeshRange {
    int triStartIndex;   // offset into global Triangle[]
    int triCount;        // number of triangles
    glm::vec3 aabbMin;
    glm::vec3 aabbMax;
};
```
This allows the GPU to cull entire primitives before testing individual triangles.

---

### 3.2 GPU Ray Intersection (`intersections.cu`)

#### 3.2.1 Primitives
- **Box**: Ray is transformed into object space (`inverseTransform`), then a slab test is performed on the unit cube `[-0.5, 0.5]³`. The face normal is transformed back to world space via `invTranspose`.
- **Sphere**: Ray is transformed into object space, then a standard quadratic discriminant test. The hit point and normal are transformed back to world space.

#### 3.2.2 Triangle — Möller-Trumbore
`triangleIntersectionTest` implements the Möller-Trumbore algorithm in **world space** (vertices are pre-transformed on the CPU):
```
det = dot(edge1, cross(dir, edge2))
u = dot(tvec, cross(dir, edge2)) / det
v = dot(dir, cross(tvec, edge1)) / det
t = dot(edge2, cross(tvec, edge1)) / det
```
- Barycentric coordinates `(u, v, w=1-u-v)` are used to interpolate vertex normals (`n0, n1, n2`) and UVs (`uv0, uv1, uv2`).
- `outside` is determined by `dot(rayDir, geometricNormal) < 0`.
- Returns Euclidean distance (consistent with box/sphere tests).

#### 3.2.3 AABB Culling — Slab Method
`aabbIntersectionTest` tests a ray against a `MeshRange` AABB:
1. For each axis, compute `t0 = (min - origin) / dir`, `t1 = (max - origin) / dir`, swap if necessary.
2. Track the **enter** (`max(t0)`) and **exit** (`min(t1)`) distances across all axes.
3. Early-out if `tEnter > tExit` or `tExit <= 0`.
4. Return `-1` if the nearest hit `tNear > maxT` (the current closest intersection from geoms or previous meshes), allowing early culling of entire meshes.

#### 3.2.4 Integration in `computeIntersections`
The kernel operates in two phases per ray:
1. **Primitive pass**: iterate all `Geom` (boxes/spheres), track `t_min`.
2. **Mesh pass**: iterate all `MeshRange`; if AABB culling is enabled, skip meshes whose AABB is farther than `t_min`. Otherwise, test every triangle in the range and update `t_min` if closer.

A critical state-tracking bug was fixed in a recent revision: when a geom is hit closer than a prior triangle, `hit_from_triangle` is reset to `false` so the final material selection uses the geom's material.

#### 3.2.5 Per-Mesh BVH (`bvh.cpp` + `intersections.cu::bvhIntersectionTest`)
For each `MeshRange` with at least `BVH_MIN_TRIANGLES` triangles (default 16), a binary BVH is built on the CPU after glTF loading:
1. **Build primitive**: each triangle is wrapped with its centroid and AABB.
2. **Split axis**: choose the axis with the largest centroid extent.
3. **Midpoint split**: use `std::nth_element` to partition triangles by centroid along the chosen axis.
4. **Leaf**: when `count <= maxLeafSize` (4), `depth >= maxDepth` (32), or the extent is negligible.
5. **Reorder**: the global `Triangle` array is reordered in-place so that a leaf's `left` offset directly indexes the contiguous triangle block.

On the GPU, `bvhIntersectionTest` traverses the tree iteratively with a fixed-size stack:
- Test the current node's AABB against the ray (using the current best `t` to prune).
- If a leaf is reached, test all triangles in the leaf and update the closest hit.
- If an internal node is reached, test both children and push the farther one first so the nearer one is popped first (near-first traversal).

The BVH path is toggled at runtime via the ImGui **Enable Mesh BVH** checkbox and falls back to brute-force triangle iteration when disabled or when the `MeshRange` is too small to build a BVH.

---

### 3.3 BSDF & Shading (`interactions.cu` + `pathtrace.cu`)

#### 3.3.1 Diffuse — Cosine-Weighted Hemisphere
`calculateRandomDirectionInHemisphere` generates a cosine-weighted direction:
- `cosθ = sqrt(ξ₁)`, `sinθ = sqrt(1 - cos²θ)`, `φ = 2π ξ₂`
- A temporary basis `(perpendicular1, perpendicular2, normal)` is constructed to avoid numerical instability when the normal is close to `(1,1,1)/√3`.

#### 3.3.2 Specular / Glossy
- **Perfect mirror**: `reflect(incident, normal)`.
- **Glossy blur**: if `roughness > SPECULAR_ROUGHNESS_EPS`, a random direction in the hemisphere around the mirror direction is sampled, then mixed with the perfect reflection via `mix(mirrorDir, glossyDir, roughness²)`.
- **Probability weighting**: the path uses Russian-roulette-style branching — `pSpec = hasReflective`, `pDiff = 1 - pSpec`. The throughput is divided by the branch probability to keep the estimator unbiased.

#### 3.3.3 Refraction
When `hasRefractive > 0`:
1. Determine incident side: `entering = outside`.
2. Compute `eta = n₁/n₂` (air → material or material → air).
3. Refract via `glm::refract(inDir, normal, eta)`.
4. Compute **Schlick Fresnel** reflectance:
   ```
   r0 = ((1 - ior) / (1 + ior))²
   reflectProb = r0 + (1 - r0)(1 - cosθ)⁵
   ```
5. If **total internal reflection** occurs (`|refracted|² ≈ 0`) or a random sample is below `reflectProb`, choose reflection; otherwise refraction.
6. Throughput is multiplied by `color` on refraction, or `1.0` on reflection (Fresnel weighting is implicitly handled by the probability split).
7. Origin is offset along the surface normal to avoid self-intersection.

#### 3.3.4 `shadeFakeMaterial` Kernel
Despite the name, this kernel performs full BSDF shading. Key logic:
- **Emissive**: if `emittance > 0`, accumulate `throughput * materialColor * emittance` and terminate the path.
- **Base Color Texture**: if present, sample the packed texture atlas and multiply `materialColor`.
- **Emissive Texture**: if present, overrides `materialColor` for light sources.
- **Double-sided**: if `doubleSided == 0` and ray hits back-face, terminate (or treat as miss).
- **Alpha Mask**: if `alphaMode == MASK` and `baseAlpha < alphaCutoff`, the ray passes through the surface without consuming a bounce (origin advanced by `t + EPSILON`).
- **Alpha Blend**: if `alphaMode == BLEND`, a stochastic test decides whether the ray passes through or is shaded.

---

### 3.4 Performance Optimizations

#### 3.4.1 Stream Compaction
After each bounce, terminated paths (`remainingBounces <= 0`) are gathered into the accumulation image, then active paths are compacted:
1. `mapActivePaths`: write `1` if `remainingBounces > 0`, else `0`.
2. `StreamCompaction::Efficient::scanDevice`: exclusive prefix sum over the flags.
3. `scatterActivePaths`: copy surviving paths to `dev_paths_compact`.
4. Pointer swap: `dev_paths ↔ dev_paths_compact`.

**Adaptive compaction**: if enabled, compaction only runs when `activeRatio <= threshold` and `num_paths >= min_paths`. This avoids paying the scan cost early in the path when most rays are still alive.

#### 3.4.2 Material Type Sort
Before shading, paths are sorted by material bucket (via `thrust::sort_by_key` on a zip iterator of `(dev_paths, dev_intersections)`). This groups threads executing similar code, reducing warp divergence.

#### 3.4.3 Depth of Field
In `generateRayFromCamera`:
1. Compute the pinhole ray direction through the pixel center + jitter.
2. If `apertureRadius > 0`:
   - Sample a point on the lens disk: `r = aperture * sqrt(ξ₁)`, `θ = 2π ξ₂`.
   - Compute focal plane intersection: `focalT = focalDistance / dot(pinholeDir, view)`.
   - New ray origin = lens sample; new direction = `normalize(focalPoint - lensOrigin)`.

---

## 4. Known Issues & Limitations

| Issue | Severity | Details |
|-------|----------|---------|
| No global BVH | Medium | Each MeshRange has its own BVH, but there is no top-level BVH across MeshRanges. Scenes with many small primitives still pay per-mesh overhead. |
| Energy conservation (glossy) | Low | Roughness blur uses hemisphere sampling without dividing by the sample PDF. Slightly biased brightness. |
| No gamma correction | Low | `sendImageToPBO` writes linear radiance directly; output may look dark on standard displays without post-process tone mapping. |
| Alpha from texture | Low | `stbi_load` requests 3 channels, so texture alpha is unavailable for `alphaMode == MASK`. |
| No Russian Roulette | Low | Fixed-depth termination only; some paths could terminate earlier without bias. |
| No Direct Lighting | Medium | Standard uni-directional path tracing; scenes with small bright lights converge slowly. |

---

## 5. How to Run

```bash
# JSON scene (Cornell Box)
.\build\bin\release\cis565_path_tracer.exe .\scenes\cornell.json    

# glTF scene
.\build\bin\release\cis565_path_tracer.exe .\scenes\render_gltf.json
```

**ImGui Toggles** (available at runtime):
- Enable Antialiasing
- Enable Stream Compaction
- Enable Adaptive Compaction (with ratio threshold & min paths sliders)
- Enable Material Type Sort
- Enable Mesh AABB Culling
- Enable Mesh BVH

---

*Last updated: 2026-06-19*
