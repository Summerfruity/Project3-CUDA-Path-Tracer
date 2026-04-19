#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "efficient.h"

#if defined(NDEBUG)
#define ERRORCHECK 0
#else
#define ERRORCHECK 1
#endif

void pathtraceCheckCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

static inline const char* pathtraceFilename()
{
    return strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__;
}

static inline void pathtraceCheckCUDA(const char* msg, int line)
{
    pathtraceCheckCUDAErrorFn(msg, pathtraceFilename(), line);
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
static PathSegment* dev_paths_compact = NULL; 
static int* dev_activeFlags = NULL;
static int* dev_scanIndices = NULL;
static int* dev_newNumPaths = NULL;

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need
    cudaMalloc(&dev_paths_compact, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_activeFlags, pixelcount * sizeof(int));
    cudaMemset(dev_activeFlags, 0, pixelcount * sizeof(int));

    cudaMalloc(&dev_scanIndices, pixelcount * sizeof(int));
    cudaMemset(dev_scanIndices, 0, pixelcount * sizeof(int));

    cudaMalloc(&dev_newNumPaths, sizeof(int));

    StreamCompaction::Efficient::initScanDeviceBuffer(pixelcount);

    pathtraceCheckCUDA("pathtraceInit", __LINE__);
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
    cudaFree(dev_paths_compact);
    cudaFree(dev_activeFlags);
    cudaFree(dev_scanIndices);
    cudaFree(dev_newNumPaths);
    StreamCompaction::Efficient::freeScanDeviceBuffer();
    pathtraceCheckCUDA("pathtraceFree", __LINE__);
}

/**
 * computeActivePathCount is a helper kernel to compute the number of active paths (remainingBounces > 0)
 */
__global__ void computeActivePathCount(int n, const int* scanIndices, const int* activeFlags, int* outCount)
{
    // compute the number of active paths by looking at the last element of scanIndices and activeFlags

    if(threadIdx.x == 0 && blockIdx.x == 0) // only need one thread to do this
    {
        if(n <= 0) // no paths, so count is 0 
        {
            outCount[0] = 0;
            return;
        }
        int lastScan = scanIndices[n - 1];
        int lastFlag = activeFlags[n - 1];
        outCount[0] = lastScan + lastFlag; // number of active paths is the last scan index + the last flag (if the last path is active)
    }
}


/**
 * mapActivePaths is a helper kernel to identify which paths are still active (remainingBounces > 0) 
 * and set a flag for them in an array. 
 * This can be used for stream compaction 
 * to remove terminated paths from the pathSegments array.
 */
__global__ void mapActivePaths(int num_paths, PathSegment* pathSegments, int* activeFlags)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < num_paths)
    {
        activeFlags[idx] = (pathSegments[idx].remainingBounces > 0) ? 1 : 0;
    }
}

/**
 * scatterActivePaths is a helper kernel to compact the pathSegments array by scattering only the active paths (remainingBounces > 0) 
 * to a new array based on the scan indices computed from the activeFlags array.
 */
__global__ void scatterActivePaths(int num_paths, PathSegment* pathSegments, int* activeFlags, int* scanIndices, PathSegment* compactedPaths)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < num_paths)
    {
        if(activeFlags[idx] == 1) 
        {
            int compactedIndex = scanIndices[idx];
            compactedPaths[compactedIndex] = pathSegments[idx];
        }
    }
}


/**
 * gatherTerminatedToImage is a helper kernel to add the color contributions of terminated paths (remainingBounces <= 0) 
 * to the final image and reset their color to black. 
 * This should be called before stream compaction to ensure that we don't lose the contributions of terminated paths.
 */
__global__ void gatherTerminatedToImage(int nPaths, glm::vec3* image, PathSegment* Paths)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < nPaths)
    {
        if(Paths[index].remainingBounces <= 0) 
        {
            image[Paths[index].pixelIndex] += Paths[index].color;
            Paths[index].color = glm::vec3(0.0f, 0.0f, 0.0f); // reset color after gathering to image
        }

    }
}





/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, bool enableAA)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(0.0f, 0.0f, 0.0f); // initial color is black, no contribution to the final image until we start shading
        segment.throughput = glm::vec3(1.0f, 1.0f, 1.0f); // initial throughput is white, full contribution to the final image until we start shading and updating it based on the materials we interact with

        // TODO: implement antialiasing by jittering the ray
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);

        float jitterX = 0.0f;
        float jitterY = 0.0f;

        if (enableAA) {
            jitterX = u01(rng) - 0.5f;
            jitterY = u01(rng) - 0.5f;
        }


        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + jitterX - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + jitterY - (float)cam.resolution.y * 0.5f)
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        // get the path segment for this thread
        PathSegment pathSegment = pathSegments[path_index];

        if(pathSegment.remainingBounces <= 0) {
            return;
        }

        float t; // distance along ray to intersection
        glm::vec3 intersect_point; // point of intersection
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true; // used to determine whether the intersection was from outside the surface or inside the surface, should be passed to the shader to determine how to shade the intersection

        glm::vec3 tmp_intersect; // used if the ray intersects with the current geometry
        glm::vec3 tmp_normal; 

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
        }
    }
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if(pathSegments[idx].remainingBounces <= 0) 
        {
            return;
        }
        if (intersection.t > 0.0f ) // if the intersection exists...
        {
          // Set up the RNG
          // LOOK: this is how you use thrust's RNG! Please look at
          // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);
            //thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color += pathSegments[idx].throughput * (materialColor * material.emittance);
                pathSegments[idx].remainingBounces = 0; 
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                scatterRay(
                    pathSegments[idx], // pass by ref, modify it in place
                    getPointOnRay(pathSegments[idx].ray, intersection.t), // intersect point
                    intersection.surfaceNormal,
                    material,
                    rng);
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].color += pathSegments[idx].throughput * BACKGROUND_COLOR;
            pathSegments[idx].remainingBounces = 0; // This is also a good time to terminate the ray if you don't have any more bounces left!
        }
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    const bool ENABLE_STREAM_COMPACTION = (guiData != NULL) ? guiData->enableStreamCompaction : true;
    const bool ENABLE_ADAPTIVE_COMPACTION = (guiData != NULL) ? guiData->enableAdaptiveCompaction : true;
    const float COMPACTION_ACTIVE_RATIO_THRESHOLD = (guiData != NULL) ? guiData->compactionActiveRatioThreshold : 0.70f;
    const int COMPACTION_MIN_PATHS = (guiData != NULL) ? guiData->compactionMinPaths : 4096;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths, guiData->enableAntialiasing);
    pathtraceCheckCUDA("generate camera ray", __LINE__);

    int depth = 0; // depth is how many times the ray has bounced, not to be confused with iter, which is how many paths have been traced
    PathSegment* dev_path_end = dev_paths + pixelcount; // end of the path segments array, used for stream compaction
    int num_paths = dev_path_end - dev_paths; // the number of active paths, used for iteration and stream compaction

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, num_paths * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d; // number of blocks for tracing path segments, depends on the number of active paths
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections
        );
        pathtraceCheckCUDA("trace one bounce", __LINE__);
        depth++;

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials
        );

        // gather terminated paths to add their contribution to the image and reset their color to black before compaction
        gatherTerminatedToImage<<<numblocksPathSegmentTracing, blockSize1d>>>(num_paths, dev_image, dev_paths);
        pathtraceCheckCUDA("shade one bounce", __LINE__);

        if (ENABLE_STREAM_COMPACTION)
        {
            mapActivePaths<<<numblocksPathSegmentTracing, blockSize1d>>>(num_paths, dev_paths, dev_activeFlags);
            pathtraceCheckCUDA("map active paths", __LINE__);

            StreamCompaction::Efficient::scanDevice(num_paths, dev_scanIndices, dev_activeFlags);

            computeActivePathCount<<<1, 1>>>(num_paths, dev_scanIndices, dev_activeFlags, dev_newNumPaths);

            int newNumPaths = 0;
            cudaMemcpy(&newNumPaths, dev_newNumPaths, sizeof(int), cudaMemcpyDeviceToHost);

            if (newNumPaths == 0)
            {
                num_paths = 0;
            }
            else
            {
                bool shouldCompact = true;
                if (ENABLE_ADAPTIVE_COMPACTION)
                {
                    float activeRatio = (float)newNumPaths / (float)num_paths;
                    shouldCompact = ((activeRatio <= COMPACTION_ACTIVE_RATIO_THRESHOLD) && num_paths >= COMPACTION_MIN_PATHS);
                }

                if (shouldCompact)
                {
                    // scatter active paths to compact the path segments array based on the scan indices computed from the active flags
                    scatterActivePaths<<<numblocksPathSegmentTracing, blockSize1d>>>(num_paths, dev_paths, dev_activeFlags, dev_scanIndices, dev_paths_compact);
                    pathtraceCheckCUDA("scatter active paths", __LINE__);

                    // swap pointers for compacted paths and path segments
                    PathSegment* temp = dev_paths;
                    dev_paths = dev_paths_compact;
                    dev_paths_compact = temp;
                    num_paths = newNumPaths; // update number of active paths for next iteration
                }
            }
        }


        //std::cout << "Depth: " << depth << ", Paths Remaining: " << num_paths << std::endl;

        if(depth >= traceDepth || num_paths <= 0)
            iterationComplete = true; 

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    if (num_paths > 0) 
    {
        dim3 numBlocksGather = (num_paths + blockSize1d - 1) / blockSize1d;
        finalGather<<<numBlocksGather, blockSize1d>>>(num_paths, dev_image, dev_paths);
    }
    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    pathtraceCheckCUDA("pathtrace", __LINE__);
}
