# Credits and Sources

All code in this repository is an original implementation released under the MIT license (see LICENSE): the
`i2s` tools and library, the Blender script, the templates and the tests. Methods are cited where they
are used.

| Technique, model or tool | Source | Used for |
|---|---|---|
| Normal integration with position constraints | D. Nehab, S. Rusinkiewicz, J. Davis, R. Ramamoorthi, "Efficiently combining positions and normals for precise 3D geometry", SIGGRAPH 2005 | the normal term of every fit pass |
| Visual hull / shape from silhouette | A. Laurentini, "The visual hull concept for silhouette-based image understanding", IEEE PAMI 16 (1994) | the first volume |
| Taubin smoothing | G. Taubin, "A signal processing approach to fair surface design", SIGGRAPH 1995 | facet-free solve copy, final smoothing |
| Marching cubes | W. Lorensen, H. Cline, SIGGRAPH 1987 (scikit-image implementation) | meshing the distance field |
| Closest point on a triangle | C. Ericson, *Real-Time Collision Detection* (2004) | exact distances |
| Euclidean distance transform | SciPy `ndimage.distance_transform_edt` | the field away from the surface |
| Kabsch alignment | W. Kabsch, Acta Cryst. A32 (1976) | per-view normal tilt correction |
| Silhouette inflation from the medial axis | in the spirit of T. Igarashi et al., "Teddy", SIGGRAPH 1999 | the front-only proxy |
| MoGe (monocular geometry: metric points, focal length) | Microsoft Research; `fal-ai/moge-3` on fal | front depth and camera |
| Marigold normals | ETH Zürich PRS; `fal-ai/marigold-v2` on fal | normal maps of every view |
| BiRefNet | ZhengPeng7 et al.; `fal-ai/birefnet/v2` on fal | subject masks |
| SAM 3 | Meta AI; `fal-ai/sam-3-1/image` on fal | text-prompted masks (select, parts) |
| Nano Banana (Gemini image editing / generation) | Google; `google/nano-banana-2.1/edit`, `fal-ai/nano-banana-pro` on fal | generated views, restyle, idea-only inputs |
| Blender, Workbench, glTF/OBJ/STL exporters | Blender Foundation, Blender 5.2 Python API | replay, renders, exports |
| Repository format | the sibling skills Video 2 GLSL (gokayfem/shader-skills), Video 2 Blender Nodes (gokayfem/video-to-blender), Video 2 Physics (gokayfem/video-to-physics), Video to CSS (gokayfem/video-to-css) | layout, adapters, plugin manifests |
