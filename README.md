CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Daniel Krupka
* Tested on: Debian testing (stretch), Intel(R) Core(TM) i7-4710HQ CPU @ 2.50GHz 8GB, GTX 850M

# Project 3 - GPU Path Tracing
My goal for this project was to extend my existing path tracer to use CUDA. All code is hand-written or
adapted from given CIS 565 code -- no external libraries used.

# Features
The primary performance features implemented are
1. Stream Compaction - Rays that are no longer alive (e.g. have hit lights, escaped the scene)
are culled.
2. Bounce Caching - Rays' first intersection are constant between iterations, and thus can be cached for some time savings.
3. Sort-by-material - As objects with similar materials can be expected to take similar code paths during shading, it may be
advantageous to sort rays by the material they've hit prior to shading.

Additionally, I implemented some extra features and improvements.
1. Fresnel scattering - The exact Fresnel equations for reflection/refraction are solved.
2. Depth of field - Physically accurate depth of field.
3. Arbitrary meshes - .obj files can be loaded, with optional AABB culling.

# Results

![Cornell Box](images/cornell.png "Cornell Box")
## Base features
The CUDA-accelerated path tracer performed much faster than the CPU version, as expected. Though I did not run CPU tests,
similar scenes generally took minutes where the GPU version takes seconds.


![First-Bounce caching comparison](images/cache.png "First-Bounce caching comparison")
As expected, caching the first ray intersection did produce a performance gain, but this diminished as the maximum
number of bounces increased. These tests were conducted on the Cornell box seen above.

## Extra features
![DOF](images/dof.png "Depth of Field")
Depth of field is physically realistic, and achieved by a "DENSITY N" and "APERTURE R" parameters in the CAMERA section, which
causes the ray origins to be jittered N^2 times within an aperture of radius R. This leads to a slow-down equivalent to
having N^2 times as many rays.

![Cow](images/cow.png "Arbitrary meshes")
![Teapot](images/teapot.png "Arbitrary meshes")
The tracer supports loading arbitrary meshes from a standard .obj file, by setting "SHAPE mesh"
and FILE file.obj" in an OBJECT section. AABB culling can be enabled/disabled by "BBCULL 1/0".
For the teapot, the non-culled render took 118 seconds, the culled 102 seconds The cow, a more complex model,
took 635 seconds non-culled, but only 593 seconds culled.

