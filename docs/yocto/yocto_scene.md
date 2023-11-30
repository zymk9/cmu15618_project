# Yocto/Scene: Scene representation

Yocto/Scene define a simple scene representation, and related utilities,
used in the Yocto/GL path tracer and for scene IO.
Yocto/Scene is implemented in `yocto_scene.h` and `yocto_scene.cpp`.

## Scene representation

Scenes are stored in `scene_data` structs and are comprised of arrays of
cameras, instances, shapes, materials, textures and environments.
The various objects are stored as values in arrays named like the object type.
Animation is not currently supported.
The scene representation is geared toward modeling physically-based environments.
In Yocto/Scene, lights are not explicitly defined, but implicitly comprised of
instances with emissive materials and environment maps.
All scenes and objects properties are accessible directly.

All scenes objects may have names that are used in IO. If names are defined,
that have to be unique. If not, names are automatically generated. Names are
stored separately from objects, for performance reasons. So for each object
array, Yocto/Scene stores a corresponding names array. For examples,
cameras as stored as `cameras` and their names are stored as `camera_names`.

Cameras, instances and environments have coordinate frames to define the local
to world transformation. Frames are presented as affine 3x4 matrices and are
intended to be rigid transforms, although most scene processing support frames
with scaling.

Objects are added to the scene by directly adding elements to the corresponding
arrays. References to elements are expressed as indices to the
corresponding arrays. For each element type, properties can be set directly.
Also, all scene objects are values, so you can work freely with them without
concerning yourself with memory management. The mantra we followed here is
that "if you know how to use `std::vector`, you know how to use scenes".

Here is an sketch of how to create a shape instance in a scene.

```cpp
auto scene = scene_data{};          // create a scene
auto shape = shape_data{};          // create a shape and add it
set_shape_properties(shape, ...);
scene.shapes.push_back(shape);
scene.materials.push_back({});      // create a black material directly
auto instance = instance_data{};    // create an instance of last added shape
instance.shape = (int)scene.shapes.size()-1;
instance.material = (int)scene.materials.size()-1;
```

Yocto/Scene defines several function to evaluate scene properties.
Use `compute_bounds(scene)` to compute the scene bounding boxes,  
`scene_stats(scene)` to get scene stats and
`scene_validation(scene)` to validate scene objects.

```cpp
auto scene = scene_scene{...};              // create a complete scene
auto bbox = compute_bounds(scene);          // get bounds
auto stats = scene_stats(scene);            // get stats
for(auto stat : stats) print_info(stat);    // print stats
auto errors = validate_stats(scene);        // get validation errors
for(auto error : errors) print_error(error);// print error
```

## Cameras

Cameras, represented by `camera_data`, are based on a simple lens model.
Cameras coordinate systems are defined by their frame.
Cameras projections are described in photographic terms. In particular,
we specify film size (35mm by default), film aspect ration,
the lens' focal length, the focus distance and the lens aperture.
All values are in meters. We support both perspective and orthographic cameras,
but prefer the former.

Common aspect ratios used in video and still photography are
3:2 on 35 mm (0.036 x 0.024),
16:9 on 35 mm (0.036 x 0.02025 or 0.04267 x 0.024),
2.35:1 on 35 mm (0.036 x 0.01532 or 0.05640 x 0.024),
2.39:1 on 35 mm (0.036 x 0.01506 or 0.05736 x 0.024),
2.40:1 on 35 mm (0.036 x 0.015 or 0.05760 x 0.024).
To compute good apertures, one can use the F-stop number from photography
and set the aperture to focal length over f-stop.

To create cameras, you should set the camera frame, the camera view,
via lens, aspect and film, and optionally the camera aperture and focus.

```cpp
auto camera = camera_data{};     // create a camera
camera.frame = identity3x4f;     // set frame to identity
camera.lens = 0.050;             // set as 50mm lens
camera.aspect = 1.5;             // set 3:2 aspect ratio
camera.film = 0.036;             // set the film as 35mm
camera.aperture = 0.01;          // set 10mm aperture
camera.focus = 10;               // set the focus at 10m
```

Use `get_camera(scene, name)` to get a camera by name or the default camera
is the name is not given. Use `eval_camera(camera, image_uv, lens_uv)`
to get a camera ray from the normalized image coordinates `image_uv` and
lens coordinates `lens_uv`.

```cpp
auto scene = scene_data{...};                  // create a complete scene
auto& camera = get_camera(scene);              // get default camera
auto ray = eval_camera(camera,{0.5,0.5},{0,0});// get ray though image center
```

## Instances

Instances, represented as `instance_data`, place shapes in the scene by
defining their coordinate frame, a shape index and a material index.
Through the use of instancing, Yocto/Scene scales well to large environments
without introducing more complex mechanisms.

For instances, you should set the instance frame, shape and material.

```cpp
auto instance = instance_data{};     // create an instance
instance.frame = identity3x4f;       // set frame to identity
instance.shape = shape_index;        // set shape index
instance.material = material_index;  // set material index
```

Several functions are defined to evaluate the geometric and material
properties of points on shapes and instances, indicated by the shape element id
and, when needed, the shape element barycentric coordinates. The difference
between the shape and instance methods is that the former returns quantities
in object space, while the latter in world space.
Use `eval_position(...)` to evaluate the point position,
`eval_normal(...)` to evaluate the interpolate point normal,
`eval_texcoord(...)` to evaluate the point texture coordinates,
`eval_element_normal(...)` to evaluate the point geometric normal, and
`eval_color(...)` to evaluate the interpolate point color.
Use `eval_material(...)` as a convenience function to evaluate material
properties of instance points.

```cpp
auto eid = 0; auto euv = vec3f{0.5,0.5};       // element id and uvs
auto pos  = eval_position(instance, eid, euv); // eval point position
auto norm = eval_normal(instance, eid, euv);   // eval point normal
auto st   = eval_texcoord(instance, eid, euv); // eval point texture coords
auto col  = eval_color(instance, eid, euv);    // eval point color
auto gn   = eval_element_normal(instance, eid, euv); // eval geometric normal
auto mat  = eval_material(instance, eid, euv); // eval point material
```

## Environments

Environments, represented as `environment_data`, store the background
illumination as a scene. Environments have a frame, to rotate illumination,
an emission term and an optional emission texture.
The emission texture is an HDR environment map stored in a LatLon
parametrization.

For environments, set the frame, emission and optionally the emission texture.

```cpp
auto& environment = environment_data{};  // create an environment
environment.frame = identity3x4f;         // set identity transform
environment.emission = {1,1,1};           // set emission scale
environment.emission_tex = texture_index; // add emission texture
```

Use `eval_environment(environment, direction)` to evaluate an environment
map emission along a specific direction `direction`. Use
`eval_environment(scene, direction)` to accumulate the lighting for all
environment maps.

```cpp
auto scene = new trace_scene{...};               // create a complete scene
auto enva = eval_environment(scene, dir);        // eval all environments
auto environment = scene.environments.front();  // get first environment
auto envi = eval_environment(environment, dir);  // eval environment
```

## Shapes

Shapes, represented by `shape_data`, are indexed meshes of elements.
Shapes are defined in [Yocto/Shape](yocto_shape.md), and briefly described 
here for convenience. Shapes can contain only one type of element, either
points, lines, triangles or quads. Shape elements are parametrized as in
[Yocto/Geometry](yocto_geometry.md).
Vertex properties are defined as separate arrays and include
positions, normals, texture coords, colors, radius and tangent spaces.
Additionally, Yocto/Scene supports face-varying primitives, as `fvshape_data`,
where each vertex data has its own topology.

Shapes also work as a standalone mesh representation throughout the
library and can be used even without a scene.

For shapes, you should set the shape elements, i.e. point, limes, triangles
or quads, and the vertex properties, i.e. positions, normals, texture
coordinates, colors and radia. Shapes support only one element type.

```cpp
auto shape = shape_data{};             // create a shape
shape.triangles = vector<vec3i>{...};  // set triangle indices
shape.positions = vector<vec3f>{...};  // set positions
shape.normals = vector<vec3f>{...};    // set normals
shape.texcoords = vector<vec2f>{...};  // set texture coordinates
```

Several functions are defined to evaluate the geometric properties of points
of shapes, indicated by the shape element id and, when needed, the shape element
barycentric coordinates.
Use `eval_position(...)` to evaluate the point position,
`eval_normal(...)` to evaluate the interpolate point normal,
`eval_texcoord(...)` to evaluate the point texture coordinates,
`eval_element_normal(...)` to evaluate the point geometric normal, and
`eval_color(...)` to evaluate the interpolate point color.

```cpp
auto eid = 0; auto euv = vec3f{0.5,0.5};    // element id and uvs
auto pos  = eval_position(shape, eid, euv); // eval point position
auto norm = eval_normal(shape, eid, euv);   // eval point normal
auto st   = eval_texcoord(shape, eid, euv); // eval point texture coords
auto col  = eval_color(shape, eid, euv);    // eval point color
auto gn   = eval_element_normal(shape, eid, euv); // eval geometric normal
```

Shape support random sampling with a uniform distribution using
`sample_shape(...)` and `sample_shape_cdf(shape)`. Sampling works for lines and
triangles in all cases, while for quad it requires that the elements
are rectangular.

```cpp
auto cdf = sample_shape_cdfd(shape);         // compute the shape CDF
auto points = sample_shape(shape, cdf, num); // sample many points
auto point = sample_shape(shape, cdf,        // sample a single point
  rand1f(rng), rand2f(rng));
```

For shapes, we also support the computation of smooth vertex normals with
`compute_normals(shape)` and converting to and from face-varying representations
with `shape_to_fvshape(shape)` and `fvshape_to_shape(fvshape)`.

## Materials

Materials, represented as `material_data`, are defined by a material type
and a few parameters, common to all materials. In particular, we support the
following materials:

- `matte`, for materials like concrete or stucco, implemented as a lambertian bsdf;
- `glossy`, for materials like plastic or painted wood, implemented as the sum
  of a lambertian and a microfacet dielectric lobe;
- `reflective`, for materials like metals, implemented as either a delta or
  microfacet brdf lobe;
- `transparent`, for materials for thin glass, implemented as a delta or
  microfacet transmission bsdf;
- `refractive`, for materials for glass or water, implemented as a delta or
  microfacet refraction bsdf; also support homogenous volume scattering;
- `subsurface`, for materials for skin, implemented as a microfacet refraction
  bsdf with homogenous volume scattering - for no this is like `refractive`;
- `volumetric`, for materials like homogeneous smoke or fog, implemented as the lack
  of a surface interface but with volumetric scattering.
- `gltfpbr`, for materials that range from glossy to metallic, implemented as
  the sum of a lambertian and a microfacet dielectric lobe;
  this is a compatibility material for loading and saving Khronos glTF data.

All materials can specify a diffuse surface emission `emission` with HDR values
that represent emitted radiance.

Surface scattering is controlled by specifying the main surface color `color`,
that represent the surface albedo, the surface roughness `roughness` and
the index of refraction `ior`. The physical meaning of each parameter depends
on the material type. By default surfaces are fully opaque, but
can defined a `opacity` parameter and texture to define the surface coverage.

Materials like `refractive`, `subsurface` and `volumetric` may also specify
volumetric properties. In these cases, the `color` parameter controls the volume density,
while the `scattering` also define volumetric scattering properties by setting a
`transmission` parameter controls the homogenous volume scattering.

All parameters can modulated by a corresponding textures, if present.

For materials, we need to specify the material type and color at the minimum.
We can further control the appearance by changing surface roughness, index of
refraction and volumetric properties, when appropriate. Here are some examples.

```cpp
auto matte = material_data{};            // create a matte material
matte.type = material_type::matte;
matte.color = {1,0.5,0.5};               // with base color and
matte.color_tex = texture_id;            // textured albedo
auto glossy =  material_data{};          // create a glossy material
glossy.type = material_type::glossy;
glossy.color = {0.5,1,0.5};              // with constant color
glossyv.roughness = 0.1;                 // base roughness and a
glossy.roughness_tex = texture_id;       // roughness texture
auto reflective =  material_data{};      // create a reflective material
glossy.type = material_type::reflective
reflective.color = {0.5,0.5,1};          // constant color
reflective.roughness = 0.1;              // constant roughness
auto tglass = material_data{};           // create a transparent material
tglass.type = material_type::transparent;
tglass.color = {1,1,1};                  // with constant color
auto glass = material_data{};            // create a refractive material
glass.type = material_type::transparent;
glass.color = {1,0.9,0.9};               // constant color
auto subsurf = material_data{};          // create a refractive material
subsurf.type = material_type::subsurface;
subsurf.color = {1,1,1};                 // that transmits all light
subsurf.scattering = {0.5,1,0.5};        // and has volumetric scattering
```

Lights are not explicit in Yocto/Scene but are specified by assigning emissive
materials.

```cpp
auto light = material_data{};    // create a material
light.color = {0,0,0};           // that does not reflect light
light.emission = {10,10,10};     // but emits it instead
```

Use `eval_material(material, texcoord)` to evaluate material textures and
combine them with parameter values. The function returns a
`material_point` that has the same parameters of a material but no
textures defined.

```cpp
auto mat = eval_material(scene,material,{0.5,0.5}) // eval material
```

## Textures

Textures, represented as `texture_data`, contains either 8-bit LDR or
32-bit float HDR images with four channels. Textures can be encoded in either
a linear color space or as sRGBs, depending on an internal flag. The use of
float versus byte is just a memory saving feature.

For textures, set the size, the color space, and _either_ the hdr or ldr pixels.

```cpp
auto hdr_texture = texture_data{};   // create a texture
hdr_texture.width = 512;             // set size
hdr_texture.height = 512;
hdr_texture.linear = true;           // set color space and pixels for an HDR
hdr_texture.pixelsf = vector<vec4f>{...};
auto ldr_texture = texture_data{};   // create a texture
ldr_texture.width = 512;             // set size
ldr_texture.height = 512;
ldr_texture.linear = false;          // set color space and pixels for an LDR
ldr_texture.pixelsb = vector<vec4b>{...};
```

Use `eval_texture(texture, uv)` to evaluate the texture at specific uvs.
Textures evaluation returns a color in linear color space, regardless of
the texture representation.

```cpp
auto col = eval_texture(texture,{0.5,0.5});   // eval texture
```

## Subdivs

Subdivs, represented as `subdiv_data`, support tesselation and displacement
mapping. Subdivs are represented as facee-varying shapes.
Subdivs specify a level of subdivision and can be subdivide elements
either linearly or using Catmull-Clark subdivision. Subdivs also support
displacement by specifying both a displacement texture and a displacement amount.
Differently from most systems, in Yocto/Scene displacement is specified
in the shape and not the material. Subdivs only support tesselation to shapes,
but do not directly support additional evaluation of properties.
Subdivs specified to the shape index to which they are subdivided into.

In this case, set the quads for positions, normals and texture coordinates.
Also set the subdivision level, and whether to use Catmull-Clark or linear
subdivision. Finally, displacement can also be applied by setting a displacement
scale and texture.

```cpp
auto subdiv = scene_sundiv{};             // create a subdiv
subdiv.quadspos = vector<vec4i>{...};     // set face-varying indices
subdiv.quadstexcoord = vector<vec4i>{...};// for positions and textures
subdiv.positions = vector<vec3f>{...};    // set positions
subdiv.texcoords = vector<vec2f>{...};    // set texture coordinates
subdiv.subdivisions = 2;                  // set subdivision level
subdiv.catmullclark = true;               // set Catmull-Clark subdivision
subdiv.displacement = 1;                  // set displacement scale
subdiv.displacement_tex = texture_id;     // and displacement map
```

Most properties on subdivs cannot be directly evaluated, nor they are
supported directly in scene processing. Instead, subdivs are converted to
indexed shapes using `tesselate_subdiv(subdiv, shape)` for a specific subdiv,
or `tesselate_subdivs(scene)` for the whole scene.

```cpp
tesselate_subdivs(scene);     // tesselate all subdivs in the scene
```

## Example scenes

Yocto/Scene has a function to create a simple Cornell Box scene for testing.
There are plans to increase support for more test scenes in the future.

```cpp
auto scene = make_cornellbox();             // make cornell box
```
