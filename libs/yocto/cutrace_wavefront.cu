//
// cutrace_wavefront kernel and device function implementation.
//

#include <cuda.h>

#include "yocto_color.h"
#include "yocto_geometry.h"
#include "yocto_math.h"
#include "yocto_sampling.h"
#include "yocto_shading.h"

// HACK TO ALLOW CUT&PASTING FROM YOCTO'S CODE
#define inline __forceinline__ __device__
#define static static __forceinline__ __device__
#define optix_shader extern "C" __global__
#define optix_constant extern "C" __constant__

// whether to use builtin compound types or yocto's ones
#define CUTRACE_BUILTIN_VECS 0

// -----------------------------------------------------------------------------
// SUBSTITUTES FOR STD TYPES
// -----------------------------------------------------------------------------
namespace yocto {

// pair
template <typename T1, typename T2>
struct pair_ {
  T1 first;
  T2 second;
};

template <typename T>
struct span {
  inline bool   empty() const { return _size == 0; }
  inline size_t size() const { return _size; }

  inline T&       operator[](int idx) { return _data[idx]; }
  inline const T& operator[](int idx) const { return _data[idx]; }
  inline T&       at(int idx) { return _data[idx]; }
  inline const T& at(int idx) const { return _data[idx]; }

  inline T*       begin() { return _data; }
  inline T*       end() { return _data + _size; }
  inline const T* begin() const { return _data; }
  inline const T* end() const { return _data + _size; }

  inline T&       front() { return *_data; }
  inline T&       back() { return *(_data + _size - 1); }
  inline const T& front() const { return *_data; }
  inline const T& back() const { return *(_data + _size - 1); }

  inline T*       data() { return _data; }
  inline const T* data() const { return _data; }

  T*     _data = nullptr;
  size_t _size = 0;
};

}  // namespace yocto

// -----------------------------------------------------------------------------
// SAMPLING FUNCTIONS
// -----------------------------------------------------------------------------
namespace yocto {

// simplified version of possible implementation from cpprenference.com
template <class T>
static const T* _upper_bound(const T* first, const T* last, const T& value) {
  const T*  it;
  ptrdiff_t count, step;
  count = last - first;

  while (count > 0) {
    it   = first;
    step = count / 2;
    it += step;
    if (!(value < *it)) {
      first = ++it;
      count -= step + 1;
    } else
      count = step;
  }
  return first;
}

// Sample a discrete distribution represented by its cdf.
inline int sample_discrete(const span<float>& cdf, float r) {
  r = clamp(r * cdf.back(), (float)0, cdf.back() - (float)0.00001);
  auto idx =
      (int)(_upper_bound(cdf.data(), cdf.data() + cdf.size(), r) - cdf.data());
  return clamp(idx, 0, (int)cdf.size() - 1);
}
// Pdf for uniform discrete distribution sampling.
inline float sample_discrete_pdf(const span<float>& cdf, int idx) {
  if (idx == 0) return cdf.at(0);
  return cdf.at(idx) - cdf.at(idx - 1);
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// CUDA HELPERS
// -----------------------------------------------------------------------------
namespace yocto {

template <typename T>
struct cuspan {
  inline bool     empty() const { return _size == 0; }
  inline size_t   size() const { return _size; }
  inline T&       operator[](int idx) { return _data[idx]; }
  inline const T& operator[](int idx) const { return _data[idx]; }

  inline T*       begin() { return _data; }
  inline T*       end() { return _data + _size; }
  inline const T* begin() const { return _data; }
  inline const T* end() const { return _data + _size; }

  inline T&       front() { return *_data; }
  inline T&       back() { return *(_data + _size - 1); }
  inline const T& front() const { return *_data; }
  inline const T& back() const { return *(_data + _size - 1); }

  inline operator span<T>() const { return {_data, _size}; }

  T*     _data = nullptr;
  size_t _size = 0;
};

template <typename T, size_t Size = 16>
struct svector {
  inline bool     empty() const { return _size == 0; }
  inline size_t   size() const { return _size; }
  inline T&       operator[](int idx) { return _data[idx]; }
  inline const T& operator[](int idx) const { return _data[idx]; }

  inline T*       begin() { return _data; }
  inline T*       end() { return _data + _size; }
  inline const T* begin() const { return _data; }
  inline const T* end() const { return _data + _size; }

  inline T&       front() { return *_data; }
  inline T&       back() { return *(_data + _size - 1); }
  inline const T& front() const { return *_data; }
  inline const T& back() const { return *(_data + _size - 1); }

  inline void push_back(const T& value) { _data[_size++] = value; }
  inline void pop_back() { _size--; }

  T      _data[Size] = {};
  size_t _size       = 0;
};

inline void* unpackPointer(uint32_t i0, uint32_t i1) {
  const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
  void*          ptr  = reinterpret_cast<void*>(uptr);
  return ptr;
}

inline void packPointer(void* ptr, uint32_t& i0, uint32_t& i1) {
  const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
  i0                  = uptr >> 32;
  i1                  = uptr & 0x00000000ffffffff;
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// CUTRACE TYPES
// -----------------------------------------------------------------------------
namespace yocto {

constexpr int invalidid = -1;
struct material_point;

enum struct material_type {
  // clang-format off
  matte, glossy, reflective, transparent, refractive, subsurface, volumetric, 
  gltfpbr
  // clang-format on
};

struct cutrace_path {
  cuspan<int>            indices       = {};
  cuspan<vec3f>          radiance      = {};
  cuspan<vec3f>          weights       = {};
  cuspan<ray3f>          rays          = {};
  cuspan<material_point> volume_back   = {};
  cuspan<bool>           volume_empty  = {};
  cuspan<float>          max_roughness = {};
  cuspan<bool>           hit           = {};
  cuspan<vec3f>          hit_albedo    = {};
  cuspan<vec3f>          hit_normal    = {};
  cuspan<int>            opbounces     = {};
  cuspan<int>            bounces       = {};
};

struct cutrace_intersection {
  cuspan<int>   instance = {};
  cuspan<int>   element  = {};
  cuspan<vec2f> uv       = {};
  cuspan<float> distance = {};
  cuspan<bool>  hit      = {};
};

struct cutrace_sample {
  int idx   = 0;
  int iters = 0;
};

struct cutrace_material_queue {
  // material queue
  cuspan<int> indices = {};  // path indices

  // material point info
  cuspan<vec3f> emission     = {};
  cuspan<vec3f> color        = {};
  cuspan<float> opacity      = {};
  cuspan<float> roughness    = {};
  cuspan<float> metallic     = {};
  cuspan<float> ior          = {};
  cuspan<vec3f> density      = {};
  cuspan<vec3f> scattering   = {};
  cuspan<float> scanisotropy = {};
  cuspan<float> trdepth      = {};

  cuspan<vec3f> outgoing = {};
  cuspan<vec3f> normal   = {};
};

struct cutrace_state {
  int               width            = 0;
  int               height           = 0;
  int               samples          = 0;
  cuspan<vec4f>     image            = {};
  cuspan<vec3f>     albedo           = {};
  cuspan<vec3f>     normal           = {};
  cuspan<int>       pixel_samples    = {};
  cuspan<rng_state> rngs             = {};
  cuspan<vec4f>     denoised         = {};
  cuspan<byte>      denoiser_state   = {};
  cuspan<byte>      denoiser_scratch = {};

  cuspan<cutrace_sample> sample_queue = {};

  cutrace_material_queue material_queue = {};

  cutrace_path         path         = {};
  cutrace_intersection intersection = {};
};

struct cucamera_data {
  frame3f frame        = {};
  float   lens         = {};
  float   film         = {};
  float   aspect       = {};
  float   focus        = {};
  float   aperture     = {};
  bool    orthographic = {};
};

struct cutexture_data {
  int                 width   = 0;
  int                 height  = 0;
  bool                linear  = false;
  bool                nearest = false;
  bool                clamp   = false;
  cudaTextureObject_t texture = 0;
  cudaArray_t         array   = nullptr;
};

struct cumaterial_data {
  material_type type         = material_type::matte;
  vec3f         emission     = {0, 0, 0};
  vec3f         color        = {0, 0, 0};
  float         roughness    = 0;
  float         metallic     = 0;
  float         ior          = 1.5f;
  vec3f         scattering   = {0, 0, 0};
  float         scanisotropy = 0;
  float         trdepth      = 0.01f;
  float         opacity      = 1;

  int emission_tex   = invalidid;
  int color_tex      = invalidid;
  int roughness_tex  = invalidid;
  int scattering_tex = invalidid;
  int normal_tex     = invalidid;
};

struct cuinstance_data {
  frame3f frame    = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 0}};
  int     shape    = -1;
  int     material = -1;
};

struct cushape_data {
  cuspan<vec3f> positions = {};
  cuspan<vec3f> normals   = {};
  cuspan<vec2f> texcoords = {};
  cuspan<vec4f> colors    = {};
  cuspan<vec3i> triangles = {};
};

struct cuenvironment_data {
  frame3f frame        = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 0}};
  vec3f   emission     = {0, 0, 0};
  int     emission_tex = invalidid;
};

struct cuscene_data {
  cuspan<cucamera_data>      cameras      = {};
  cuspan<cutexture_data>     textures     = {};
  cuspan<cumaterial_data>    materials    = {};
  cuspan<cushape_data>       shapes       = {};
  cuspan<cuinstance_data>    instances    = {};
  cuspan<cuenvironment_data> environments = {};
};

struct bvh_node {
  bbox3f  bbox        = invalidb3f;
  int32_t start       = 0;
  int16_t num         = 0;
  int8_t  axis        = 0;
  bool    internal    = false;
  int8_t  slot_map[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  int8_t  slot_pos    = 0;
};

struct cubvh_tree {
  cuspan<bvh_node> nodes      = {};
  cuspan<int>      primitives = {};
};

struct cushape_bvh {
  cubvh_tree bvh = {};
};

struct cuscene_bvh {
  cubvh_tree          bvh    = {};
  cuspan<cushape_bvh> shapes = {};
};

// Type of tracing algorithm
enum struct trace_sampler_type {
  path,        // path tracing
  pathdirect,  // path tracing with direct
  pathmis,     // path tracing with mis
  pathtest,    // path tracing for testing
  naive,       // naive path tracing
  eyelight,    // eyelight rendering
  diagram,     // diagram rendering
  furnace,     // furnace test
  falsecolor,  // false color rendering
};
// Type of false color visualization
enum struct trace_falsecolor_type {
  // clang-format off
  position, normal, frontfacing, gnormal, gfrontfacing, texcoord, mtype, color,
  emission, roughness, opacity, metallic, delta, instance, shape, material, 
  element, highlight
  // clang-format on
};

// Default trace seed
constexpr auto trace_default_seed = 961748941ull;

// params
struct trace_params {
  int                   camera         = 0;
  int                   resolution     = 1280;
  trace_sampler_type    sampler        = trace_sampler_type::path;
  trace_falsecolor_type falsecolor     = trace_falsecolor_type::color;
  int                   samples        = 512;
  int                   bounces        = 8;
  float                 clamp          = 10;
  bool                  nocaustics     = false;
  bool                  envhidden      = false;
  bool                  tentfilter     = false;
  uint64_t              seed           = trace_default_seed;
  bool                  embreebvh      = false;
  bool                  highqualitybvh = false;
  bool                  noparallel     = false;
  int                   pratio         = 8;
  bool                  denoise        = false;
  int                   batch          = 1;
  bool                  wbvh           = false;
};

using cutrace_bvh = cuscene_bvh;

// light
struct cutrace_light {
  int           instance     = invalidid;
  int           environment  = invalidid;
  cuspan<float> elements_cdf = {};
};

// lights
struct cutrace_lights {
  cuspan<cutrace_light> lights = {};
};

struct cutrace_globals {
  cutrace_state  state  = {};
  cuscene_data   scene  = {};
  cuscene_bvh    bvh    = {};
  cutrace_lights lights = {};
  trace_params   params = {};
};

// global data
__constant__ cutrace_globals globals;

// compatibility aliases
using trace_bvh    = cutrace_bvh;
using trace_lights = cutrace_lights;

}  // namespace yocto

// -----------------------------------------------------------------------------
// SCENE FUNCTIONS
// -----------------------------------------------------------------------------
namespace yocto {

// compatibility aliases
using scene_data       = cuscene_data;
using camera_data      = cucamera_data;
using material_data    = cumaterial_data;
using texture_data     = cutexture_data;
using instance_data    = cuinstance_data;
using shape_data       = cushape_data;
using environment_data = cuenvironment_data;

// constant values
constexpr auto min_roughness = 0.03f * 0.03f;

// Evaluates an image at a point `uv`.
static vec4f eval_texture(const texture_data& texture, const vec2f& texcoord,
    bool as_linear = false, bool no_interpolation = false,
    bool clamp_to_edge = false) {
  auto fromTexture = tex2D<float4>(texture.texture, texcoord.x, texcoord.y);
  auto color       = vec4f{
      fromTexture.x, fromTexture.y, fromTexture.z, fromTexture.w};
  if (as_linear && !texture.linear) {
    return srgb_to_rgb(color);
  } else {
    return color;
  }
}

// Helpers
static vec4f eval_texture(const scene_data& scene, int texture, const vec2f& uv,
    bool ldr_as_linear = false, bool no_interpolation = false,
    bool clamp_to_edge = false) {
  if (texture == invalidid) return {1, 1, 1, 1};
  return eval_texture(
      scene.textures[texture], uv, ldr_as_linear, no_interpolation);
}

// Material parameters evaluated at a point on the surface
struct material_point {
  material_type type         = material_type::gltfpbr;
  vec3f         emission     = {0, 0, 0};
  vec3f         color        = {0, 0, 0};
  float         opacity      = 1;
  float         roughness    = 0;
  float         metallic     = 0;
  float         ior          = 1;
  vec3f         density      = {0, 0, 0};
  vec3f         scattering   = {0, 0, 0};
  float         scanisotropy = 0;
  float         trdepth      = 0.01f;
};

// Eval position
static vec3f eval_position(const scene_data& scene,
    const instance_data& instance, int element, const vec2f& uv) {
  auto& shape = scene.shapes[instance.shape];
  if (!shape.triangles.empty()) {
    auto t = shape.triangles[element];
    return transform_point(
        instance.frame, interpolate_triangle(shape.positions[t.x],
                            shape.positions[t.y], shape.positions[t.z], uv));
  } else {
    return {0, 0, 0};
  }
}

// Shape element normal.
static vec3f eval_element_normal(
    const scene_data& scene, const instance_data& instance, int element) {
  auto& shape = scene.shapes[instance.shape];
  if (!shape.triangles.empty()) {
    auto t = shape.triangles[element];
    return transform_normal(
        instance.frame, triangle_normal(shape.positions[t.x],
                            shape.positions[t.y], shape.positions[t.z]));
  } else {
    return {0, 0, 0};
  }
}

// Eval normal
static vec3f eval_normal(const scene_data& scene, const instance_data& instance,
    int element, const vec2f& uv) {
  auto& shape = scene.shapes[instance.shape];
  if (shape.normals.empty())
    return eval_element_normal(scene, instance, element);
  if (!shape.triangles.empty()) {
    auto t = shape.triangles[element];
    return transform_normal(
        instance.frame, normalize(interpolate_triangle(shape.normals[t.x],
                            shape.normals[t.y], shape.normals[t.z], uv)));
  } else {
    return {0, 0, 0};
  }
}

// Eval texcoord
static vec2f eval_texcoord(const scene_data& scene,
    const instance_data& instance, int element, const vec2f& uv) {
  auto& shape = scene.shapes[instance.shape];
  if (shape.texcoords.empty()) return uv;
  if (!shape.triangles.empty()) {
    auto t = shape.triangles[element];
    return interpolate_triangle(
        shape.texcoords[t.x], shape.texcoords[t.y], shape.texcoords[t.z], uv);
  } else {
    return {0, 0};
  }
}

// Shape element normal.
static pair_<vec3f, vec3f> eval_element_tangents(
    const scene_data& scene, const instance_data& instance, int element) {
  auto& shape = scene.shapes[instance.shape];
  if (!shape.triangles.empty() && !shape.texcoords.empty()) {
    auto t   = shape.triangles[element];
    auto tuv = triangle_tangents_fromuv(shape.positions[t.x],
        shape.positions[t.y], shape.positions[t.z], shape.texcoords[t.x],
        shape.texcoords[t.y], shape.texcoords[t.z]);
    return {transform_direction(instance.frame, tuv.first),
        transform_direction(instance.frame, tuv.second)};
  } else {
    return {};
  }
}

static vec3f eval_normalmap(const scene_data& scene,
    const instance_data& instance, int element, const vec2f& uv) {
  auto& shape    = scene.shapes[instance.shape];
  auto& material = scene.materials[instance.material];
  // apply normal mapping
  auto normal   = eval_normal(scene, instance, element, uv);
  auto texcoord = eval_texcoord(scene, instance, element, uv);
  if (material.normal_tex != invalidid && (!shape.triangles.empty())) {
    auto& normal_tex = scene.textures[material.normal_tex];
    auto  normalmap  = -1 + 2 * xyz(eval_texture(normal_tex, texcoord, false));
    auto  tuv        = eval_element_tangents(scene, instance, element);
    auto  frame      = frame3f{tuv.first, tuv.second, normal, {0, 0, 0}};
    frame.x          = orthonormalize(frame.x, frame.z);
    frame.y          = normalize(cross(frame.z, frame.x));
    auto flip_v      = dot(frame.y, tuv.second) < 0;
    normalmap.y *= flip_v ? 1 : -1;  // flip vertical axis
    normal = transform_normal(frame, normalmap);
  }
  return normal;
}

// Eval shading position
static vec3f eval_shading_position(const scene_data& scene,
    const instance_data& instance, int element, const vec2f& uv,
    const vec3f& outgoing) {
  auto& shape = scene.shapes[instance.shape];
  if (!shape.triangles.empty()) {
    return eval_position(scene, instance, element, uv);
  } else {
    return {0, 0, 0};
  }
}

// Eval shading normal
static vec3f eval_shading_normal(const scene_data& scene,
    const instance_data& instance, int element, const vec2f& uv,
    const vec3f& outgoing) {
  auto& shape    = scene.shapes[instance.shape];
  auto& material = scene.materials[instance.material];
  if (!shape.triangles.empty()) {
    auto normal = eval_normal(scene, instance, element, uv);
    if (material.normal_tex != invalidid) {
      normal = eval_normalmap(scene, instance, element, uv);
    }
    if (material.type == material_type::refractive) return normal;
    return dot(normal, outgoing) >= 0 ? normal : -normal;
  } else {
    return {0, 0, 0};
  }
}

// Eval color
static vec4f eval_color(const scene_data& scene, const instance_data& instance,
    int element, const vec2f& uv) {
  auto& shape = scene.shapes[instance.shape];
  if (shape.colors.empty()) return {1, 1, 1, 1};
  if (!shape.triangles.empty()) {
    auto t = shape.triangles[element];
    return interpolate_triangle(
        shape.colors[t.x], shape.colors[t.y], shape.colors[t.z], uv);
  } else {
    return {0, 0, 0, 0};
  }
}

// Evaluate material
static material_point eval_material(const scene_data& scene,
    const instance_data& instance, int element, const vec2f& uv) {
  auto& material = scene.materials[instance.material];
  auto  texcoord = eval_texcoord(scene, instance, element, uv);

  // evaluate textures
  auto emission_tex = eval_texture(
      scene, material.emission_tex, texcoord, true);
  auto color_shp     = eval_color(scene, instance, element, uv);
  auto color_tex     = eval_texture(scene, material.color_tex, texcoord, true);
  auto roughness_tex = eval_texture(
      scene, material.roughness_tex, texcoord, false);
  auto scattering_tex = eval_texture(
      scene, material.scattering_tex, texcoord, true);

  // material point
  auto point         = material_point{};
  point.type         = material.type;
  point.emission     = material.emission * xyz(emission_tex);
  point.color        = material.color * xyz(color_tex) * xyz(color_shp);
  point.opacity      = material.opacity * color_tex.w * color_shp.w;
  point.metallic     = material.metallic * roughness_tex.z;
  point.roughness    = material.roughness * roughness_tex.y;
  point.roughness    = point.roughness * point.roughness;
  point.ior          = material.ior;
  point.scattering   = material.scattering * xyz(scattering_tex);
  point.scanisotropy = material.scanisotropy;
  point.trdepth      = material.trdepth;

  // volume density
  if (material.type == material_type::refractive ||
      material.type == material_type::volumetric ||
      material.type == material_type::subsurface) {
    point.density = -log(clamp(point.color, 0.0001f, 1.0f)) / point.trdepth;
  } else {
    point.density = {0, 0, 0};
  }

  // fix roughness
  if (point.type == material_type::matte ||
      point.type == material_type::gltfpbr ||
      point.type == material_type::glossy) {
    point.roughness = clamp(point.roughness, min_roughness, 1.0f);
  } else if (material.type == material_type::volumetric) {
    point.roughness = 0;
  } else {
    if (point.roughness < min_roughness) point.roughness = 0;
  }

  return point;
}

static bool is_volumetric(const material_data& material) {
  return material.type == material_type::refractive ||
         material.type == material_type::volumetric ||
         material.type == material_type::subsurface;
}

// check if an instance is volumetric
static bool is_volumetric(
    const scene_data& scene, const instance_data& instance) {
  return is_volumetric(scene.materials[instance.material]);
}

// check if a brdf is a delta
static bool is_delta(const material_point& material) {
  return (material.type == material_type::reflective &&
             material.roughness == 0) ||
         (material.type == material_type::refractive &&
             material.roughness == 0) ||
         (material.type == material_type::transparent &&
             material.roughness == 0) ||
         (material.type == material_type::volumetric);
}

static ray3f eval_camera(
    const cucamera_data& camera, const vec2f& image_uv, const vec2f& lens_uv) {
  auto film = camera.aspect >= 1
                  ? vec2f{camera.film, camera.film / camera.aspect}
                  : vec2f{camera.film * camera.aspect, camera.film};
  auto q    = vec3f{
      film.x * (0.5f - image_uv.x), film.y * (image_uv.y - 0.5f), camera.lens};
  // ray direction through the lens center
  auto dc = -normalize(q);
  // point on the lens
  auto e = vec3f{
      lens_uv.x * camera.aperture / 2, lens_uv.y * camera.aperture / 2, 0};
  // point on the focus plane
  auto p = dc * camera.focus / abs(dc.z);
  // correct ray direction to account for camera focusing
  auto d = normalize(p - e);
  // done
  return ray3f{
      transform_point(camera.frame, e), transform_direction(camera.frame, d)};
}

// Evaluate environment color.
static vec3f eval_environment(const scene_data& scene,
    const environment_data& environment, const vec3f& direction) {
  auto wl       = transform_direction_inverse(environment.frame, direction);
  auto texcoord = vec2f{
      atan2(wl.z, wl.x) / (2 * pif), acos(clamp(wl.y, -1.0f, 1.0f)) / pif};
  if (texcoord.x < 0) texcoord.x += 1;
  return environment.emission *
         xyz(eval_texture(scene, environment.emission_tex, texcoord));
}

// Evaluate all environment color.
static vec3f eval_environment(const scene_data& scene, const vec3f& direction) {
  auto emission = vec3f{0, 0, 0};
  for (auto& environment : scene.environments) {
    emission += eval_environment(scene, environment, direction);
  }
  return emission;
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// RAY-SCENE INTERSECTION
// -----------------------------------------------------------------------------
namespace yocto {

// intersection result
struct scene_intersection {
  int   instance = -1;
  int   element  = -1;
  vec2f uv       = {0, 0};
  float distance = 0;
  bool  hit      = false;
  float _pad     = 0;
};

struct shape_intersection {
  int   element  = -1;
  vec2f uv       = {0, 0};
  float distance = 0;
  bool  hit      = false;
};

static shape_intersection intersect_shape_wbvh(const cushape_bvh& sbvh,
    const shape_data& shape, const ray3f& ray_, bool find_any) {
  // get bvh tree
  auto& bvh = sbvh.bvh;

  // check empty
  if (bvh.nodes.empty()) return {};

  // node stack
  int  node_stack[512];
  auto node_cur          = 0;
  node_stack[node_cur++] = 0;

  // shared variables
  auto intersection = shape_intersection{};

  // copy ray to modify it
  auto ray = ray_;

  // prepare ray for fast queries
  auto ray_dinv  = vec3f{1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z};
  auto ray_dsign = vec3i{(ray_dinv.x < 0) ? 1 : 0, (ray_dinv.y < 0) ? 1 : 0,
      (ray_dinv.z < 0) ? 1 : 0};

  // walking stack
  while (node_cur != 0) {
    // grab node
    auto& node = bvh.nodes[node_stack[--node_cur]];

    // intersect bbox
    // if (!intersect_bbox(ray, ray_dinv, ray_dsign, node.bbox)) continue;
    if (!intersect_bbox(ray, ray_dinv, node.bbox)) continue;

    // intersect node, switching based on node type
    // for each type, iterate over the the primitive list
    if (node.internal) {
      int oct    = ray_dsign[0] + (ray_dsign[1] << 1) + (ray_dsign[2] << 2);
      int offset = node.slot_map[oct];

      for (int i = node.num - 1; i >= 0; i--) {
        node_stack[node_cur + i] = node.start + offset;
        offset                   = (offset + 1) % node.num;
      }

      node_cur += node.num;
    } else if (!shape.triangles.empty()) {
      for (auto idx = node.start; idx < node.start + node.num; idx++) {
        auto& t             = shape.triangles[bvh.primitives[idx]];
        auto  pintersection = intersect_triangle(ray, shape.positions[t.x],
            shape.positions[t.y], shape.positions[t.z]);
        if (!pintersection.hit) continue;
        intersection = {bvh.primitives[idx], pintersection.uv,
            pintersection.distance, true};
        ray.tmax     = pintersection.distance;
      }
    }

    // check for early exit
    if (find_any && intersection.hit) return intersection;
  }

  return intersection;
}

static shape_intersection intersect_shape_bvh(const cushape_bvh& sbvh,
    const shape_data& shape, const ray3f& ray_, bool find_any) {
  // get bvh tree
  auto& bvh = sbvh.bvh;

  // check empty
  if (bvh.nodes.empty()) return {};

  // node stack
  int  node_stack[128];
  auto node_cur          = 0;
  node_stack[node_cur++] = 0;

  // shared variables
  auto intersection = shape_intersection{};

  // copy ray to modify it
  auto ray = ray_;

  // prepare ray for fast queries
  auto ray_dinv  = vec3f{1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z};
  auto ray_dsign = vec3i{(ray_dinv.x < 0) ? 1 : 0, (ray_dinv.y < 0) ? 1 : 0,
      (ray_dinv.z < 0) ? 1 : 0};

  // walking stack
  while (node_cur != 0) {
    // grab node
    auto& node = bvh.nodes[node_stack[--node_cur]];

    // intersect bbox
    // if (!intersect_bbox(ray, ray_dinv, ray_dsign, node.bbox)) continue;
    if (!intersect_bbox(ray, ray_dinv, node.bbox)) continue;

    // intersect node, switching based on node type
    // for each type, iterate over the the primitive list
    if (node.internal) {
      // for internal nodes, attempts to proceed along the
      // split axis from smallest to largest nodes
      if (ray_dsign[node.axis] != 0) {
        node_stack[node_cur++] = node.start + 0;
        node_stack[node_cur++] = node.start + 1;
      } else {
        node_stack[node_cur++] = node.start + 1;
        node_stack[node_cur++] = node.start + 0;
      }
    } else if (!shape.triangles.empty()) {
      for (auto idx = node.start; idx < node.start + node.num; idx++) {
        auto& t             = shape.triangles[bvh.primitives[idx]];
        auto  pintersection = intersect_triangle(ray, shape.positions[t.x],
            shape.positions[t.y], shape.positions[t.z]);
        if (!pintersection.hit) continue;
        intersection = {bvh.primitives[idx], pintersection.uv,
            pintersection.distance, true};
        ray.tmax     = pintersection.distance;
      }
    }

    // check for early exit
    if (find_any && intersection.hit) return intersection;
  }

  return intersection;
}

static scene_intersection intersect_scene_wbvh(const cuscene_bvh& sbvh,
    const scene_data& scene, const ray3f& ray_, bool find_any) {
  // get instances bvh
  auto& bvh = sbvh.bvh;

  // check empty
  if (bvh.nodes.empty()) return {};

  // node stack
  int  node_stack[512];
  auto node_cur          = 0;
  node_stack[node_cur++] = 0;

  // intersection
  auto intersection = scene_intersection{};

  // copy ray to modify it
  auto ray = ray_;

  // prepare ray for fast queries
  auto ray_dinv  = vec3f{1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z};
  auto ray_dsign = vec3i{(ray_dinv.x < 0) ? 1 : 0, (ray_dinv.y < 0) ? 1 : 0,
      (ray_dinv.z < 0) ? 1 : 0};

  // walking stack
  while (node_cur != 0) {
    // grab node
    auto& node = bvh.nodes[node_stack[--node_cur]];

    // intersect bbox
    // if (!intersect_bbox(ray, ray_dinv, ray_dsign, node.bbox)) continue;
    if (!intersect_bbox(ray, ray_dinv, node.bbox)) continue;

    // intersect node, switching based on node type
    // for each type, iterate over the the primitive list
    if (node.internal) {
      int oct    = ray_dsign[0] + (ray_dsign[1] << 1) + (ray_dsign[2] << 2);
      int offset = node.slot_map[oct];

      for (int i = node.num - 1; i >= 0; i--) {
        node_stack[node_cur + i] = node.start + offset;
        offset                   = (offset + 1) % node.num;
      }

      node_cur += node.num;
    } else {
      for (auto idx = node.start; idx < node.start + node.num; idx++) {
        auto& instance_ = scene.instances[bvh.primitives[idx]];
        auto  inv_ray   = transform_ray(inverse(instance_.frame, true), ray);
        auto  sintersection = intersect_shape_wbvh(sbvh.shapes[instance_.shape],
            scene.shapes[instance_.shape], inv_ray, find_any);
        if (!sintersection.hit) continue;
        intersection = {bvh.primitives[idx], sintersection.element,
            sintersection.uv, sintersection.distance, true};
        ray.tmax     = sintersection.distance;
      }
    }

    // check for early exit
    if (find_any && intersection.hit) return intersection;
  }

  return intersection;
}

static scene_intersection intersect_scene_bvh(const cuscene_bvh& sbvh,
    const scene_data& scene, const ray3f& ray_, bool find_any) {
  // get instances bvh
  auto& bvh = sbvh.bvh;

  // check empty
  if (bvh.nodes.empty()) return {};

  // node stack
  int  node_stack[128];
  auto node_cur          = 0;
  node_stack[node_cur++] = 0;

  // intersection
  auto intersection = scene_intersection{};

  // copy ray to modify it
  auto ray = ray_;

  // prepare ray for fast queries
  auto ray_dinv  = vec3f{1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z};
  auto ray_dsign = vec3i{(ray_dinv.x < 0) ? 1 : 0, (ray_dinv.y < 0) ? 1 : 0,
      (ray_dinv.z < 0) ? 1 : 0};

  // walking stack
  while (node_cur != 0) {
    // grab node
    auto& node = bvh.nodes[node_stack[--node_cur]];

    // intersect bbox
    // if (!intersect_bbox(ray, ray_dinv, ray_dsign, node.bbox)) continue;
    if (!intersect_bbox(ray, ray_dinv, node.bbox)) continue;

    // intersect node, switching based on node type
    // for each type, iterate over the the primitive list
    if (node.internal) {
      // for internal nodes, attempts to proceed along the
      // split axis from smallest to largest nodes
      if (ray_dsign[node.axis] != 0) {
        node_stack[node_cur++] = node.start + 0;
        node_stack[node_cur++] = node.start + 1;
      } else {
        node_stack[node_cur++] = node.start + 1;
        node_stack[node_cur++] = node.start + 0;
      }
    } else {
      for (auto idx = node.start; idx < node.start + node.num; idx++) {
        auto& instance_ = scene.instances[bvh.primitives[idx]];
        auto  inv_ray   = transform_ray(inverse(instance_.frame, true), ray);
        auto  sintersection = intersect_shape_bvh(sbvh.shapes[instance_.shape],
            scene.shapes[instance_.shape], inv_ray, find_any);
        if (!sintersection.hit) continue;
        intersection = {bvh.primitives[idx], sintersection.element,
            sintersection.uv, sintersection.distance, true};
        ray.tmax     = sintersection.distance;
      }
    }

    // check for early exit
    if (find_any && intersection.hit) return intersection;
  }

  return intersection;
}

static scene_intersection intersect_scene(const cuscene_bvh& sbvh,
    const scene_data& scene, const ray3f& ray_, bool find_any) {
  if (globals.params.wbvh) {
    return intersect_scene_wbvh(sbvh, scene, ray_, find_any);
  } else {
    return intersect_scene_bvh(sbvh, scene, ray_, find_any);
  }
}

// instance intersection, for now manual
static scene_intersection intersect_instance(const trace_bvh& bvh,
    const cuscene_data& scene, int instance_id, const ray3f& ray) {
  auto& instance     = scene.instances[instance_id];
  auto& shape        = scene.shapes[instance.shape];
  auto  intersection = scene_intersection{};
  auto  tray         = ray3f{transform_point_inverse(instance.frame, ray.o),
      transform_vector_inverse(instance.frame, ray.d)};
  for (auto element = 0; element < shape.triangles.size(); element++) {
    auto& triangle = shape.triangles[element];
    auto  isec     = intersect_triangle(tray, shape.positions[triangle.x],
        shape.positions[triangle.y], shape.positions[triangle.z]);
    if (!isec.hit) continue;
    intersection.hit      = true;
    intersection.instance = instance_id;
    intersection.element  = element;
    intersection.uv       = isec.uv;
    intersection.distance = isec.distance;
    tray.tmax             = isec.distance;
  }
  return intersection;
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// TRACE FUNCTIONS
// -----------------------------------------------------------------------------
namespace yocto {

// Convenience functions
[[maybe_unused]] static vec3f eval_position(
    const scene_data& scene, const scene_intersection& intersection) {
  return eval_position(scene, scene.instances[intersection.instance],
      intersection.element, intersection.uv);
}
[[maybe_unused]] static vec3f eval_normal(
    const scene_data& scene, const scene_intersection& intersection) {
  return eval_normal(scene, scene.instances[intersection.instance],
      intersection.element, intersection.uv);
}
[[maybe_unused]] static vec3f eval_element_normal(
    const scene_data& scene, const scene_intersection& intersection) {
  return eval_element_normal(
      scene, scene.instances[intersection.instance], intersection.element);
}
[[maybe_unused]] static vec3f eval_shading_position(const scene_data& scene,
    const scene_intersection& intersection, const vec3f& outgoing) {
  return eval_shading_position(scene, scene.instances[intersection.instance],
      intersection.element, intersection.uv, outgoing);
}
[[maybe_unused]] static vec3f eval_shading_normal(const scene_data& scene,
    const scene_intersection& intersection, const vec3f& outgoing) {
  return eval_shading_normal(scene, scene.instances[intersection.instance],
      intersection.element, intersection.uv, outgoing);
}
[[maybe_unused]] static vec2f eval_texcoord(
    const scene_data& scene, const scene_intersection& intersection) {
  return eval_texcoord(scene, scene.instances[intersection.instance],
      intersection.element, intersection.uv);
}
[[maybe_unused]] static material_point eval_material(
    const scene_data& scene, const scene_intersection& intersection) {
  return eval_material(scene, scene.instances[intersection.instance],
      intersection.element, intersection.uv);
}
[[maybe_unused]] static bool is_volumetric(
    const scene_data& scene, const scene_intersection& intersection) {
  return is_volumetric(scene, scene.instances[intersection.instance]);
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// TRACE FUNCTIONS
// -----------------------------------------------------------------------------
namespace yocto {

// Evaluates/sample the BRDF scaled by the cosine of the incoming direction.
static vec3f eval_emission(const material_point& material, const vec3f& normal,
    const vec3f& outgoing) {
  return dot(normal, outgoing) >= 0 ? material.emission : vec3f{0, 0, 0};
}

// Evaluates/sample the BRDF scaled by the cosine of the incoming direction.
static vec3f eval_bsdfcos(const material_point& material, const vec3f& normal,
    const vec3f& outgoing, const vec3f& incoming) {
  if (material.roughness == 0) return {0, 0, 0};

  if (material.type == material_type::matte) {
    return eval_matte(material.color, normal, outgoing, incoming);
  } else if (material.type == material_type::glossy) {
    return eval_glossy(material.color, material.ior, material.roughness, normal,
        outgoing, incoming);
  } else if (material.type == material_type::reflective) {
    return eval_reflective(
        material.color, material.roughness, normal, outgoing, incoming);
  } else if (material.type == material_type::transparent) {
    return eval_transparent(material.color, material.ior, material.roughness,
        normal, outgoing, incoming);
  } else if (material.type == material_type::refractive) {
    return eval_refractive(material.color, material.ior, material.roughness,
        normal, outgoing, incoming);
  } else if (material.type == material_type::subsurface) {
    return eval_refractive(material.color, material.ior, material.roughness,
        normal, outgoing, incoming);
  } else if (material.type == material_type::gltfpbr) {
    return eval_gltfpbr(material.color, material.ior, material.roughness,
        material.metallic, normal, outgoing, incoming);
  } else {
    return {0, 0, 0};
  }
}

static vec3f eval_delta(const material_point& material, const vec3f& normal,
    const vec3f& outgoing, const vec3f& incoming) {
  if (material.roughness != 0) return {0, 0, 0};

  if (material.type == material_type::reflective) {
    return eval_reflective(material.color, normal, outgoing, incoming);
  } else if (material.type == material_type::transparent) {
    return eval_transparent(
        material.color, material.ior, normal, outgoing, incoming);
  } else if (material.type == material_type::refractive) {
    return eval_refractive(
        material.color, material.ior, normal, outgoing, incoming);
  } else if (material.type == material_type::volumetric) {
    return eval_passthrough(material.color, normal, outgoing, incoming);
  } else {
    return {0, 0, 0};
  }
}

// Picks a direction based on the BRDF
static vec3f sample_bsdfcos(const material_point& material, const vec3f& normal,
    const vec3f& outgoing, float rnl, const vec2f& rn) {
  if (material.roughness == 0) return {0, 0, 0};

  if (material.type == material_type::matte) {
    return sample_matte(material.color, normal, outgoing, rn);
  } else if (material.type == material_type::glossy) {
    return sample_glossy(material.color, material.ior, material.roughness,
        normal, outgoing, rnl, rn);
  } else if (material.type == material_type::reflective) {
    return sample_reflective(
        material.color, material.roughness, normal, outgoing, rn);
  } else if (material.type == material_type::transparent) {
    return sample_transparent(material.color, material.ior, material.roughness,
        normal, outgoing, rnl, rn);
  } else if (material.type == material_type::refractive) {
    return sample_refractive(material.color, material.ior, material.roughness,
        normal, outgoing, rnl, rn);
  } else if (material.type == material_type::subsurface) {
    return sample_refractive(material.color, material.ior, material.roughness,
        normal, outgoing, rnl, rn);
  } else if (material.type == material_type::gltfpbr) {
    return sample_gltfpbr(material.color, material.ior, material.roughness,
        material.metallic, normal, outgoing, rnl, rn);
  } else {
    return {0, 0, 0};
  }
}

static vec3f sample_delta(const material_point& material, const vec3f& normal,
    const vec3f& outgoing, float rnl) {
  if (material.roughness != 0) return {0, 0, 0};

  if (material.type == material_type::reflective) {
    return sample_reflective(material.color, normal, outgoing);
  } else if (material.type == material_type::transparent) {
    return sample_transparent(
        material.color, material.ior, normal, outgoing, rnl);
  } else if (material.type == material_type::refractive) {
    return sample_refractive(
        material.color, material.ior, normal, outgoing, rnl);
  } else if (material.type == material_type::volumetric) {
    return sample_passthrough(material.color, normal, outgoing);
  } else {
    return {0, 0, 0};
  }
}

// Compute the weight for sampling the BRDF
static float sample_bsdfcos_pdf(const material_point& material,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming) {
  if (material.roughness == 0) return 0;

  if (material.type == material_type::matte) {
    return sample_matte_pdf(material.color, normal, outgoing, incoming);
  } else if (material.type == material_type::glossy) {
    return sample_glossy_pdf(material.color, material.ior, material.roughness,
        normal, outgoing, incoming);
  } else if (material.type == material_type::reflective) {
    return sample_reflective_pdf(
        material.color, material.roughness, normal, outgoing, incoming);
  } else if (material.type == material_type::transparent) {
    return sample_tranparent_pdf(material.color, material.ior,
        material.roughness, normal, outgoing, incoming);
  } else if (material.type == material_type::refractive) {
    return sample_refractive_pdf(material.color, material.ior,
        material.roughness, normal, outgoing, incoming);
  } else if (material.type == material_type::subsurface) {
    return sample_refractive_pdf(material.color, material.ior,
        material.roughness, normal, outgoing, incoming);
  } else if (material.type == material_type::gltfpbr) {
    return sample_gltfpbr_pdf(material.color, material.ior, material.roughness,
        material.metallic, normal, outgoing, incoming);
  } else {
    return 0;
  }
}

static float sample_delta_pdf(const material_point& material,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming) {
  if (material.roughness != 0) return 0;

  if (material.type == material_type::reflective) {
    return sample_reflective_pdf(material.color, normal, outgoing, incoming);
  } else if (material.type == material_type::transparent) {
    return sample_tranparent_pdf(
        material.color, material.ior, normal, outgoing, incoming);
  } else if (material.type == material_type::refractive) {
    return sample_refractive_pdf(
        material.color, material.ior, normal, outgoing, incoming);
  } else if (material.type == material_type::volumetric) {
    return sample_passthrough_pdf(material.color, normal, outgoing, incoming);
  } else {
    return 0;
  }
}

static vec3f eval_scattering(const material_point& material,
    const vec3f& outgoing, const vec3f& incoming) {
  if (material.density == vec3f{0, 0, 0}) return {0, 0, 0};
  return material.scattering * material.density *
         eval_phasefunction(material.scanisotropy, outgoing, incoming);
}

static vec3f sample_scattering(const material_point& material,
    const vec3f& outgoing, float rnl, const vec2f& rn) {
  if (material.density == vec3f{0, 0, 0}) return {0, 0, 0};
  return sample_phasefunction(material.scanisotropy, outgoing, rn);
}

static float sample_scattering_pdf(const material_point& material,
    const vec3f& outgoing, const vec3f& incoming) {
  if (material.density == vec3f{0, 0, 0}) return 0;
  return sample_phasefunction_pdf(material.scanisotropy, outgoing, incoming);
}

// Sample camera
static ray3f sample_camera(const camera_data& camera, const vec2i& ij,
    const vec2i& image_size, const vec2f& puv, const vec2f& luv, bool tent) {
  if (!tent) {
    auto uv = vec2f{
        (ij.x + puv.x) / image_size.x, (ij.y + puv.y) / image_size.y};
    return eval_camera(camera, uv, sample_disk(luv));
  } else {
    const auto width  = 2.0f;
    const auto offset = 0.5f;
    auto       fuv =
        width *
            vec2f{
                puv.x < 0.5f ? sqrt(2 * puv.x) - 1 : 1 - sqrt(2 - 2 * puv.x),
                puv.y < 0.5f ? sqrt(2 * puv.y) - 1 : 1 - sqrt(2 - 2 * puv.y),
            } +
        offset;
    auto uv = vec2f{
        (ij.x + fuv.x) / image_size.x, (ij.y + fuv.y) / image_size.y};
    return eval_camera(camera, uv, sample_disk(luv));
  }
}

// Sample lights wrt solid angle
static vec3f sample_lights(const scene_data& scene, const trace_lights& lights,
    const vec3f& position, float rl, float rel, const vec2f& ruv) {
  auto  light_id = sample_uniform((int)lights.lights.size(), rl);
  auto& light    = lights.lights[light_id];
  if (light.instance != invalidid) {
    auto& instance  = scene.instances[light.instance];
    auto& shape     = scene.shapes[instance.shape];
    auto  element   = sample_discrete(light.elements_cdf, rel);
    auto  uv        = (!shape.triangles.empty()) ? sample_triangle(ruv) : ruv;
    auto  lposition = eval_position(scene, instance, element, uv);
    return normalize(lposition - position);
  } else if (light.environment != invalidid) {
    auto& environment = scene.environments[light.environment];
    if (environment.emission_tex != invalidid) {
      auto& emission_tex = scene.textures[environment.emission_tex];
      auto  idx          = sample_discrete(light.elements_cdf, rel);
      auto  uv = vec2f{((idx % emission_tex.width) + 0.5f) / emission_tex.width,
          ((idx / emission_tex.width) + 0.5f) / emission_tex.height};
      return transform_direction(environment.frame,
          {cos(uv.x * 2 * pif) * sin(uv.y * pif), cos(uv.y * pif),
              sin(uv.x * 2 * pif) * sin(uv.y * pif)});
    } else {
      return sample_sphere(ruv);
    }
  } else {
    return {0, 0, 0};
  }
}

// Sample lights pdf
static float sample_lights_pdf(const scene_data& scene, const trace_bvh& bvh,
    const trace_lights& lights, const vec3f& position, const vec3f& direction) {
  auto pdf = 0.0f;
  for (auto& light : lights.lights) {
    if (light.instance != invalidid) {
      auto& instance = scene.instances[light.instance];
      // check all intersection
      auto lpdf          = 0.0f;
      auto next_position = position;
      for (auto bounce = 0; bounce < 100; bounce++) {
        auto intersection = intersect_instance(
            bvh, scene, light.instance, {next_position, direction});
        if (!intersection.hit) break;
        // accumulate pdf
        auto lposition = eval_position(
            scene, instance, intersection.element, intersection.uv);
        auto lnormal = eval_element_normal(
            scene, instance, intersection.element);
        // prob triangle * area triangle = area triangle mesh
        auto area = light.elements_cdf.back();
        lpdf += distance_squared(lposition, position) /
                (abs(dot(lnormal, direction)) * area);
        // continue
        next_position = lposition + direction * 1e-3f;
      }
      pdf += lpdf;
    } else if (light.environment != invalidid) {
      auto& environment = scene.environments[light.environment];
      if (environment.emission_tex != invalidid) {
        auto& emission_tex = scene.textures[environment.emission_tex];
        auto  wl = transform_direction_inverse(environment.frame, direction);
        auto  texcoord = vec2f{atan2(wl.z, wl.x) / (2 * pif),
            acos(clamp(wl.y, -1.0f, 1.0f)) / pif};
        if (texcoord.x < 0) texcoord.x += 1;
        auto i = clamp(
            (int)(texcoord.x * emission_tex.width), 0, emission_tex.width - 1);
        auto j    = clamp((int)(texcoord.y * emission_tex.height), 0,
            emission_tex.height - 1);
        auto prob = sample_discrete_pdf(
                        light.elements_cdf, j * emission_tex.width + i) /
                    light.elements_cdf.back();
        auto angle = (2 * pif / emission_tex.width) *
                     (pif / emission_tex.height) *
                     sin(pif * (j + 0.5f) / emission_tex.height);
        pdf += prob / angle;
      } else {
        pdf += 1 / (4 * pif);
      }
    }
  }
  pdf *= sample_uniform_pdf((int)lights.lights.size());
  return pdf;
}

struct trace_result {
  vec3f radiance = {0, 0, 0};
  bool  hit      = false;
  vec3f albedo   = {0, 0, 0};
  vec3f normal   = {0, 0, 0};
};

// Recursive path tracing.
static trace_result trace_path(const scene_data& scene, const trace_bvh& bvh,
    const trace_lights& lights, const ray3f& ray_, rng_state& rng,
    const trace_params& params) {
  // initialize
  auto radiance      = vec3f{0, 0, 0};
  auto weight        = vec3f{1, 1, 1};
  auto ray           = ray_;
  auto volume_stack  = svector<material_point>{};
  auto max_roughness = 0.0f;
  auto hit           = false;
  auto hit_albedo    = vec3f{0, 0, 0};
  auto hit_normal    = vec3f{0, 0, 0};
  auto opbounce      = 0;

  // trace  path
  for (auto bounce = 0; bounce < params.bounces; bounce++) {
    // intersect next point
    auto intersection = intersect_scene(bvh, scene, ray, false);
    if (!intersection.hit) {
      if (bounce > 0 || !params.envhidden)
        radiance += weight * eval_environment(scene, ray.d);
      break;
    }

    // handle transmission if inside a volume
    auto in_volume = false;
    if (!volume_stack.empty()) {
      auto& vsdf     = volume_stack.back();
      auto  distance = sample_transmittance(
          vsdf.density, intersection.distance, rand1f(rng), rand1f(rng));
      weight *= eval_transmittance(vsdf.density, distance) /
                sample_transmittance_pdf(
                    vsdf.density, distance, intersection.distance);
      in_volume             = distance < intersection.distance;
      intersection.distance = distance;
    }

    // switch between surface and volume
    if (!in_volume) {
      // prepare shading point
      auto outgoing = -ray.d;
      auto position = eval_shading_position(scene, intersection, outgoing);
      auto normal   = eval_shading_normal(scene, intersection, outgoing);
      auto material = eval_material(scene, intersection);

      // correct roughness
      if (params.nocaustics) {
        max_roughness      = max(material.roughness, max_roughness);
        material.roughness = max_roughness;
      }

      // handle opacity
      if (material.opacity < 1 && rand1f(rng) >= material.opacity) {
        if (opbounce++ > 128) break;
        ray = {position + ray.d * 1e-2f, ray.d};
        bounce -= 1;
        continue;
      }

      // set hit variables
      if (bounce == 0) {
        hit        = true;
        hit_albedo = material.color;
        hit_normal = normal;
      }

      // accumulate emission
      radiance += weight * eval_emission(material, normal, outgoing);

      // next direction
      auto incoming = vec3f{0, 0, 0};
      if (!is_delta(material)) {
        if (rand1f(rng) < 0.5f) {
          incoming = sample_bsdfcos(
              material, normal, outgoing, rand1f(rng), rand2f(rng));
        } else {
          incoming = sample_lights(
              scene, lights, position, rand1f(rng), rand1f(rng), rand2f(rng));
        }
        if (incoming == vec3f{0, 0, 0}) break;
        weight *=
            eval_bsdfcos(material, normal, outgoing, incoming) /
            (0.5f * sample_bsdfcos_pdf(material, normal, outgoing, incoming) +
                0.5f *
                    sample_lights_pdf(scene, bvh, lights, position, incoming));
      } else {
        incoming = sample_delta(material, normal, outgoing, rand1f(rng));
        weight *= eval_delta(material, normal, outgoing, incoming) /
                  sample_delta_pdf(material, normal, outgoing, incoming);
      }

      // update volume stack
      if (is_volumetric(scene, intersection) &&
          dot(normal, outgoing) * dot(normal, incoming) < 0) {
        if (volume_stack.empty()) {
          auto material = eval_material(scene, intersection);
          volume_stack.push_back(material);
        } else {
          volume_stack.pop_back();
        }
      }

      // setup next iteration
      ray = {position, incoming};
    } else {
      // prepare shading point
      auto  outgoing = -ray.d;
      auto  position = ray.o + ray.d * intersection.distance;
      auto& vsdf     = volume_stack.back();

      // accumulate emission
      // radiance += weight * eval_volemission(emission, outgoing);

      // next direction
      auto incoming = vec3f{0, 0, 0};
      if (rand1f(rng) < 0.5f) {
        incoming = sample_scattering(vsdf, outgoing, rand1f(rng), rand2f(rng));
      } else {
        incoming = sample_lights(
            scene, lights, position, rand1f(rng), rand1f(rng), rand2f(rng));
      }
      if (incoming == vec3f{0, 0, 0}) break;
      weight *=
          eval_scattering(vsdf, outgoing, incoming) /
          (0.5f * sample_scattering_pdf(vsdf, outgoing, incoming) +
              0.5f * sample_lights_pdf(scene, bvh, lights, position, incoming));

      // setup next iteration
      ray = {position, incoming};
    }

    // check weight
    if (weight == vec3f{0, 0, 0} || !isfinite(weight)) break;

    // russian roulette
    if (bounce > 3) {
      auto rr_prob = min((float)0.99, max(weight));
      if (rand1f(rng) >= rr_prob) break;
      weight *= 1 / rr_prob;
    }
  }

  return {radiance, hit, hit_albedo, hit_normal};
}

static void submit_material(const material_point& material, const vec3f& normal,
    const vec3f& outgoing, int idx, int* queue_fronts) {
  int material_map[8] = {0, 1, 2, 3, 4, 4, 4, 5};
  int material_id     = material_map[(int)(material.type)];

  int   queue_size     = globals.state.width * globals.state.height;
  auto& material_queue = globals.state.material_queue;

  int queue_idx = atomicAdd(&queue_fronts[material_id], 1);
  queue_idx += material_id * queue_size;

  // write material
  material_queue.indices[queue_idx] = idx;
  material_queue.emission[idx]      = material.emission;
  material_queue.color[idx]         = material.color;
  material_queue.opacity[idx]       = material.opacity;
  material_queue.roughness[idx]     = material.roughness;
  material_queue.metallic[idx]      = material.metallic;
  material_queue.ior[idx]           = material.ior;
  material_queue.density[idx]       = material.density;
  material_queue.scattering[idx]    = material.scattering;
  material_queue.scanisotropy[idx]  = material.scanisotropy;
  material_queue.trdepth[idx]       = material.trdepth;

  material_queue.outgoing[idx] = outgoing;
  material_queue.normal[idx]   = normal;
}

// eval one segment of the path, returns true if the ray terminated
static bool eval_path(const scene_data& scene, const trace_bvh& bvh,
    const trace_lights& lights, cutrace_path& paths,
    const cutrace_intersection& intersections, int idx, rng_state& rng,
    const trace_params& params) {
  // read from globals
  // we need to write back to globals at the end if the ray is not terminated
  auto radiance      = paths.radiance[idx];
  auto weight        = paths.weights[idx];
  auto ray           = paths.rays[idx];
  auto volume_back   = paths.volume_back[idx];
  auto volume_empty  = paths.volume_empty[idx];
  auto max_roughness = paths.max_roughness[idx];
  auto opbounce      = paths.opbounces[idx];
  auto bounce        = paths.bounces[idx];

  // read intersection from globals
  auto intersection = scene_intersection{intersections.instance[idx],
      intersections.element[idx], intersections.uv[idx],
      intersections.distance[idx], intersections.hit[idx]};

  if (!intersection.hit) {
    if (bounce > 0 || !params.envhidden) {
      paths.radiance[idx] = radiance + weight * eval_environment(scene, ray.d);
    }
    return true;
  }

  // handle transmission if inside a volume
  auto in_volume = false;
  if (!volume_empty) {
    auto distance = sample_transmittance(
        volume_back.density, intersection.distance, rand1f(rng), rand1f(rng));
    weight *= eval_transmittance(volume_back.density, distance) /
              sample_transmittance_pdf(
                  volume_back.density, distance, intersection.distance);
    paths.weights[idx]    = weight;
    in_volume             = distance < intersection.distance;
    intersection.distance = distance;
  }

  // switch between surface and volume
  if (!in_volume) {
    // prepare shading point
    auto outgoing = -ray.d;
    auto position = eval_shading_position(scene, intersection, outgoing);
    auto normal   = eval_shading_normal(scene, intersection, outgoing);
    auto material = eval_material(scene, intersection);

    // correct roughness
    if (params.nocaustics) {
      max_roughness            = max(material.roughness, max_roughness);
      paths.max_roughness[idx] = max_roughness;
      material.roughness       = max_roughness;
    }

    // handle opacity
    if (material.opacity < 1 && rand1f(rng) >= material.opacity) {
      if (opbounce > 128) {
        return true;
      }
      paths.opbounces[idx] = opbounce + 1;
      paths.rays[idx]      = {position + ray.d * 1e-2f, ray.d};
      return false;
    }

    // set hit variables
    if (bounce == 0) {
      paths.hit[idx]        = true;
      paths.hit_albedo[idx] = material.color;
      paths.hit_normal[idx] = normal;
    }

    // accumulate emission
    radiance += weight * eval_emission(material, normal, outgoing);
    paths.radiance[idx] = radiance;

    // next direction
    auto incoming = vec3f{0, 0, 0};
    if (!is_delta(material)) {
      if (rand1f(rng) < 0.5f) {
        incoming = sample_bsdfcos(
            material, normal, outgoing, rand1f(rng), rand2f(rng));
      } else {
        incoming = sample_lights(
            scene, lights, position, rand1f(rng), rand1f(rng), rand2f(rng));
      }
      if (incoming == vec3f{0, 0, 0}) return true;
      weight *=
          eval_bsdfcos(material, normal, outgoing, incoming) /
          (0.5f * sample_bsdfcos_pdf(material, normal, outgoing, incoming) +
              0.5f * sample_lights_pdf(scene, bvh, lights, position, incoming));
    } else {
      incoming = sample_delta(material, normal, outgoing, rand1f(rng));
      weight *= eval_delta(material, normal, outgoing, incoming) /
                sample_delta_pdf(material, normal, outgoing, incoming);
    }

    // update volume stack
    if (is_volumetric(scene, intersection) &&
        dot(normal, outgoing) * dot(normal, incoming) < 0) {
      if (volume_empty) {
        paths.volume_back[idx] = eval_material(scene, intersection);
      } else {
        paths.volume_empty[idx] = true;
      }
    }

    // setup next iteration
    ray = {position, incoming};
  } else {
    // prepare shading point
    auto outgoing = -ray.d;
    auto position = ray.o + ray.d * intersection.distance;

    // accumulate emission
    // radiance += weight * eval_volemission(emission, outgoing);

    // next direction
    auto incoming = vec3f{0, 0, 0};
    if (rand1f(rng) < 0.5f) {
      incoming = sample_scattering(
          volume_back, outgoing, rand1f(rng), rand2f(rng));
    } else {
      incoming = sample_lights(
          scene, lights, position, rand1f(rng), rand1f(rng), rand2f(rng));
    }
    if (incoming == vec3f{0, 0, 0}) return true;
    weight *=
        eval_scattering(volume_back, outgoing, incoming) /
        (0.5f * sample_scattering_pdf(volume_back, outgoing, incoming) +
            0.5f * sample_lights_pdf(scene, bvh, lights, position, incoming));

    // setup next iteration
    ray = {position, incoming};
  }

  // check weight
  if (weight == vec3f{0, 0, 0} || !isfinite(weight)) return true;

  // russian roulette
  if (bounce > 3) {
    auto rr_prob = min((float)0.99, max(weight));
    if (rand1f(rng) >= rr_prob) return true;
    weight *= 1 / rr_prob;
  }

  // finish eval one segment
  bounce++;
  if (bounce >= params.bounces) return true;

  // write to globals
  paths.rays[idx]    = ray;
  paths.weights[idx] = weight;
  paths.bounces[idx] = bounce;

  return false;
}

// do not eval material
static bool eval_path_partial(const scene_data& scene, const trace_bvh& bvh,
    const trace_lights& lights, cutrace_path& paths,
    const cutrace_intersection& intersections, int idx, rng_state& rng,
    const trace_params& params, int* queue_fronts) {
  // read from globals
  // we need to write back to globals at the end if the ray is not terminated
  auto radiance      = paths.radiance[idx];
  auto weight        = paths.weights[idx];
  auto ray           = paths.rays[idx];
  auto volume_back   = paths.volume_back[idx];
  auto volume_empty  = paths.volume_empty[idx];
  auto max_roughness = paths.max_roughness[idx];
  auto opbounce      = paths.opbounces[idx];
  auto bounce        = paths.bounces[idx];

  // material eval terminates the ray
  if (weight == vec3f{0, 0, 0} || !isfinite(weight)) return true;

  // read intersection from globals
  auto intersection = scene_intersection{intersections.instance[idx],
      intersections.element[idx], intersections.uv[idx],
      intersections.distance[idx], intersections.hit[idx]};

  if (!intersection.hit) {
    if (bounce > 0 || !params.envhidden) {
      paths.radiance[idx] = radiance + weight * eval_environment(scene, ray.d);
    }
    return true;
  }

  // handle transmission if inside a volume
  auto in_volume = false;
  if (!volume_empty) {
    auto distance = sample_transmittance(
        volume_back.density, intersection.distance, rand1f(rng), rand1f(rng));
    weight *= eval_transmittance(volume_back.density, distance) /
              sample_transmittance_pdf(
                  volume_back.density, distance, intersection.distance);
    paths.weights[idx]    = weight;
    in_volume             = distance < intersection.distance;
    intersection.distance = distance;
  }

  // switch between surface and volume
  bool sample_mat = false;
  bool weight_unk = false;
  if (!in_volume) {
    // prepare shading point
    auto outgoing = -ray.d;
    auto position = eval_shading_position(scene, intersection, outgoing);
    auto normal   = eval_shading_normal(scene, intersection, outgoing);
    auto material = eval_material(scene, intersection);

    // correct roughness
    if (params.nocaustics) {
      max_roughness            = max(material.roughness, max_roughness);
      paths.max_roughness[idx] = max_roughness;
      material.roughness       = max_roughness;
    }

    // handle opacity
    if (material.opacity < 1 && rand1f(rng) >= material.opacity) {
      if (opbounce > 128) {
        return true;
      }
      paths.opbounces[idx] = opbounce + 1;
      paths.rays[idx]      = {position + ray.d * 1e-2f, ray.d};
      return false;
    }

    // set hit variables
    if (bounce == 0) {
      paths.hit[idx]        = true;
      paths.hit_albedo[idx] = material.color;
      paths.hit_normal[idx] = normal;
    }

    // accumulate emission
    radiance += weight * eval_emission(material, normal, outgoing);
    paths.radiance[idx] = radiance;

    // next direction
    auto incoming = vec3f{0, 0, 0};
    if (!is_delta(material)) {
      if (rand1f(rng) < 0.5f) {
        sample_mat = true;
        // incoming = sample_bsdfcos(
        //     material, normal, outgoing, rand1f(rng), rand2f(rng));
      } else {
        incoming = sample_lights(
            scene, lights, position, rand1f(rng), rand1f(rng), rand2f(rng));
        if (incoming == vec3f{0, 0, 0}) return true;
      }

      if (material.roughness == 0) {
        return true;
      }

      weight_unk = true;
      // if (incoming == vec3f{0, 0, 0}) return true;
      // weight *=
      //     eval_bsdfcos(material, normal, outgoing, incoming) /
      //     (0.5f * sample_bsdfcos_pdf(material, normal, outgoing, incoming) +
      //         0.5f * sample_lights_pdf(scene, bvh, lights, position,
      //         incoming));
    } else {
      incoming = sample_delta(material, normal, outgoing, rand1f(rng));
      weight *= eval_delta(material, normal, outgoing, incoming) /
                sample_delta_pdf(material, normal, outgoing, incoming);
    }

    // update volume stack
    if (!sample_mat && is_volumetric(scene, intersection) &&
        dot(normal, outgoing) * dot(normal, incoming) < 0) {
      if (volume_empty) {
        paths.volume_back[idx]  = eval_material(scene, intersection);
        paths.volume_empty[idx] = false;
      } else {
        paths.volume_empty[idx] = true;
      }
    }

    if (weight_unk && bounce + 1 < params.bounces) {
      // do not eval material if last bounce
      submit_material(material, normal, outgoing, idx, queue_fronts);
    }

    // setup next iteration
    ray = {position, incoming};
  } else {
    // prepare shading point
    auto outgoing = -ray.d;
    auto position = ray.o + ray.d * intersection.distance;

    // accumulate emission
    // radiance += weight * eval_volemission(emission, outgoing);

    // next direction
    auto incoming = vec3f{0, 0, 0};
    if (rand1f(rng) < 0.5f) {
      incoming = sample_scattering(
          volume_back, outgoing, rand1f(rng), rand2f(rng));
    } else {
      incoming = sample_lights(
          scene, lights, position, rand1f(rng), rand1f(rng), rand2f(rng));
    }
    if (incoming == vec3f{0, 0, 0}) return true;
    weight *=
        eval_scattering(volume_back, outgoing, incoming) /
        (0.5f * sample_scattering_pdf(volume_back, outgoing, incoming) +
            0.5f * sample_lights_pdf(scene, bvh, lights, position, incoming));

    // setup next iteration
    ray = {position, incoming};
  }

  if (!weight_unk) {
    // check weight
    if (weight == vec3f{0, 0, 0} || !isfinite(weight)) return true;

    // russian roulette
    if (bounce > 3) {
      auto rr_prob = min((float)0.99, max(weight));
      if (rand1f(rng) >= rr_prob) return true;
      weight *= 1 / rr_prob;
    }
  }

  // finish eval one segment
  bounce++;
  if (bounce >= params.bounces) return true;

  // write to globals
  paths.rays[idx]    = ray;
  paths.weights[idx] = weight;
  paths.bounces[idx] = bounce;

  return false;
}

static void fetch_sample(cutrace_state& state, const trace_params& params,
    int idx, int* sample_queue_front) {
  int queue_size = state.width * state.height * params.batch;
  if (*sample_queue_front >= queue_size) {
    state.path.indices[idx] = -1;
    return;
  }

  int sample_idx = atomicAdd(sample_queue_front, 1);
  if (sample_idx >= queue_size) {
    state.path.indices[idx] = -1;
    return;
  }

  int pixel_idx = sample_idx / params.batch;
  int pixel_x   = pixel_idx % state.width;
  int pixel_y   = pixel_idx / state.width;

  auto& camera = globals.scene.cameras[globals.params.camera];
  auto& rng    = globals.state.rngs[idx];

  auto ray = sample_camera(camera, {pixel_x, pixel_y},
      {state.width, state.height}, rand2f(rng), rand2f(rng),
      globals.params.tentfilter);

  auto& path = state.path;

  path.indices[idx]       = pixel_idx;
  path.radiance[idx]      = vec3f{0, 0, 0};
  path.weights[idx]       = vec3f{1, 1, 1};
  path.rays[idx]          = ray;
  path.volume_back[idx]   = {};
  path.volume_empty[idx]  = true;
  path.max_roughness[idx] = 0.0f;
  path.hit[idx]           = false;
  path.hit_albedo[idx]    = vec3f{0, 0, 0};
  path.hit_normal[idx]    = vec3f{0, 0, 0};
  path.opbounces[idx]     = 0;
  path.bounces[idx]       = 0;
}

static void trace_sample(cutrace_state& state, const cuscene_data& scene,
    const cutrace_bvh& bvh, const cutrace_lights& lights, int i, int j,
    const trace_params& params, int* num_samples_done, int* sample_queue_front,
    int* mat_queue_fronts) {
  auto thread_idx = state.width * j + i;
  auto pixel_idx  = state.path.indices[thread_idx];
  if (pixel_idx < 0) {
    return;
  }

  // auto result = eval_path(scene, bvh, lights, state.path, state.intersection,
  //     thread_idx, state.rngs[thread_idx], params);

  auto result = eval_path_partial(scene, bvh, lights, state.path,
      state.intersection, thread_idx, state.rngs[thread_idx], params,
      mat_queue_fronts);

  if (result) {
    // ray is terminated, update image and generate a new ray
    auto& path = state.path;

    auto radiance = path.radiance[thread_idx];
    auto hit      = path.hit[thread_idx];
    auto albedo   = path.hit_albedo[thread_idx];
    auto normal   = path.hit_normal[thread_idx];
    auto ray      = path.rays[thread_idx];

    if (!isfinite(radiance)) radiance = {0, 0, 0};
    if (max(radiance) > params.clamp)
      radiance = radiance * (params.clamp / max(radiance));

    float alpha = 1.0f;
    if (!hit) {
      if (!params.envhidden && !scene.environments.empty()) {
        albedo = {1, 1, 1};
      } else {
        alpha    = 0.0f;
        radiance = {0, 0, 0};
        albedo   = {0, 0, 0};
      }
      normal = -ray.d;
    }

    // atomic update image
    float* image_ptr  = (float*)&state.image[pixel_idx];
    float* albedo_ptr = (float*)&state.albedo[pixel_idx];
    float* normal_ptr = (float*)&state.normal[pixel_idx];

    atomicAdd(&image_ptr[0], radiance.x);
    atomicAdd(&image_ptr[1], radiance.y);
    atomicAdd(&image_ptr[2], radiance.z);
    atomicAdd(&image_ptr[3], alpha);

    atomicAdd(&albedo_ptr[0], albedo.x);
    atomicAdd(&albedo_ptr[1], albedo.y);
    atomicAdd(&albedo_ptr[2], albedo.z);

    atomicAdd(&normal_ptr[0], normal.x);
    atomicAdd(&normal_ptr[1], normal.y);
    atomicAdd(&normal_ptr[2], normal.z);

    atomicAdd(num_samples_done, 1);

    fetch_sample(state, params, thread_idx, sample_queue_front);
  }
}

static material_point get_material_from_queue(material_type type, int idx) {
  material_point material;
  material.type         = type;
  material.emission     = globals.state.material_queue.emission[idx];
  material.color        = globals.state.material_queue.color[idx];
  material.opacity      = globals.state.material_queue.opacity[idx];
  material.roughness    = globals.state.material_queue.roughness[idx];
  material.metallic     = globals.state.material_queue.metallic[idx];
  material.ior          = globals.state.material_queue.ior[idx];
  material.density      = globals.state.material_queue.density[idx];
  material.scattering   = globals.state.material_queue.scattering[idx];
  material.scanisotropy = globals.state.material_queue.scanisotropy[idx];
  material.trdepth      = globals.state.material_queue.trdepth[idx];

  return material;
}

static void eval_material_epilogue(material_type type, int idx,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming,
    const vec3f& weight) {
  bool is_volumetric = type == material_type::refractive ||
                       type == material_type::volumetric ||
                       type == material_type::subsurface;

  auto& paths = globals.state.path;

  // update volume stack
  if (is_volumetric && dot(normal, outgoing) * dot(normal, incoming) < 0) {
    if (paths.volume_empty[idx]) {
      paths.volume_back[idx]  = get_material_from_queue(type, idx);
      paths.volume_empty[idx] = false;
    } else {
      paths.volume_empty[idx] = true;
    }
  }

  // russian roulette
  if (paths.bounces[idx] > 4) {
    auto rr_prob = min((float)0.99, max(weight));
    if (rand1f(globals.state.rngs[idx]) >= rr_prob) {
      paths.weights[idx] = {0, 0, 0};
      return;
    }
    paths.weights[idx] *= 1 / rr_prob;
  }
}

extern "C" __global__ void eval_matte_kernel(int num_materials) {
  int mat_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (mat_idx >= num_materials) {
    return;
  }

  auto& material_queue = globals.state.material_queue;
  int   queue_size     = globals.state.width * globals.state.height;
  int   offset         = 0;
  int   queue_idx      = offset * queue_size + mat_idx;
  int   idx            = material_queue.indices[queue_idx];

  auto  color    = material_queue.color[idx];
  auto  normal   = material_queue.normal[idx];
  auto  outgoing = material_queue.outgoing[idx];
  auto& rng      = globals.state.rngs[idx];
  auto  rn       = rand2f(rng);

  auto incoming = globals.state.path.rays[idx].d;
  auto position = globals.state.path.rays[idx].o;
  if (incoming == vec3f{0, 0, 0}) {  // otherwise use ray direction
    incoming = sample_matte(color, normal, outgoing, rn);
  }

  if (incoming == vec3f{0, 0, 0}) {
    globals.state.path.weights[idx] = {0, 0, 0};  // terminate
    return;
  }

  auto bsdfcos     = eval_matte(color, normal, outgoing, incoming);
  auto bsdfcos_pdf = sample_matte_pdf(color, normal, outgoing, incoming);
  auto lights_pdf  = sample_lights_pdf(
      globals.scene, globals.bvh, globals.lights, position, incoming);

  globals.state.path.weights[idx] *= bsdfcos /
                                     (0.5f * bsdfcos_pdf + 0.5f * lights_pdf);

  eval_material_epilogue(material_type::matte, idx, normal, outgoing, incoming,
      globals.state.path.weights[idx]);

  globals.state.path.rays[idx] = {position, incoming};
}

extern "C" __global__ void eval_glossy_kernel(int num_materials) {
  int mat_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (mat_idx >= num_materials) {
    return;
  }

  auto& material_queue = globals.state.material_queue;
  int   queue_size     = globals.state.width * globals.state.height;
  int   offset         = 1;
  int   queue_idx      = offset * queue_size + mat_idx;
  int   idx            = material_queue.indices[queue_idx];

  auto  color     = material_queue.color[idx];
  auto  ior       = material_queue.ior[idx];
  auto  roughness = material_queue.roughness[idx];
  auto  normal    = material_queue.normal[idx];
  auto  outgoing  = material_queue.outgoing[idx];
  auto& rng       = globals.state.rngs[idx];
  auto  rn        = rand2f(rng);
  auto  rnl       = rand1f(rng);

  auto incoming = globals.state.path.rays[idx].d;
  auto position = globals.state.path.rays[idx].o;
  if (incoming == vec3f{0, 0, 0}) {  // otherwise use ray direction
    incoming = sample_glossy(color, ior, roughness, normal, outgoing, rnl, rn);
  }

  if (incoming == vec3f{0, 0, 0}) {
    globals.state.path.weights[idx] = {0, 0, 0};  // terminate
    return;
  }

  auto bsdfcos = eval_glossy(color, ior, roughness, normal, outgoing, incoming);
  auto bsdfcos_pdf = sample_glossy_pdf(
      color, ior, roughness, normal, outgoing, incoming);
  auto lights_pdf = sample_lights_pdf(
      globals.scene, globals.bvh, globals.lights, position, incoming);

  globals.state.path.weights[idx] *= bsdfcos /
                                     (0.5f * bsdfcos_pdf + 0.5f * lights_pdf);

  eval_material_epilogue(material_type::glossy, idx, normal, outgoing, incoming,
      globals.state.path.weights[idx]);

  globals.state.path.rays[idx] = {position, incoming};
}

extern "C" __global__ void eval_reflective_kernel(int num_materials) {
  int mat_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (mat_idx >= num_materials) {
    return;
  }

  auto& material_queue = globals.state.material_queue;
  int   queue_size     = globals.state.width * globals.state.height;
  int   offset         = 2;
  int   queue_idx      = offset * queue_size + mat_idx;
  int   idx            = material_queue.indices[queue_idx];

  auto  color     = material_queue.color[idx];
  auto  roughness = material_queue.roughness[idx];
  auto  normal    = material_queue.normal[idx];
  auto  outgoing  = material_queue.outgoing[idx];
  auto& rng       = globals.state.rngs[idx];
  auto  rn        = rand2f(rng);

  auto incoming = globals.state.path.rays[idx].d;
  auto position = globals.state.path.rays[idx].o;
  if (incoming == vec3f{0, 0, 0}) {  // otherwise use ray direction
    incoming = sample_reflective(color, roughness, normal, outgoing, rn);
  }

  if (incoming == vec3f{0, 0, 0}) {
    globals.state.path.weights[idx] = {0, 0, 0};  // terminate
    return;
  }

  auto bsdfcos = eval_reflective(color, roughness, normal, outgoing, incoming);
  auto bsdfcos_pdf = sample_reflective_pdf(
      color, roughness, normal, outgoing, incoming);
  auto lights_pdf = sample_lights_pdf(
      globals.scene, globals.bvh, globals.lights, position, incoming);

  globals.state.path.weights[idx] *= bsdfcos /
                                     (0.5f * bsdfcos_pdf + 0.5f * lights_pdf);

  eval_material_epilogue(material_type::reflective, idx, normal, outgoing,
      incoming, globals.state.path.weights[idx]);

  globals.state.path.rays[idx] = {position, incoming};
}

extern "C" __global__ void eval_transparent_kernel(int num_materials) {
  int mat_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (mat_idx >= num_materials) {
    return;
  }

  auto& material_queue = globals.state.material_queue;
  int   queue_size     = globals.state.width * globals.state.height;
  int   offset         = 3;
  int   queue_idx      = offset * queue_size + mat_idx;
  int   idx            = material_queue.indices[queue_idx];

  auto  color     = material_queue.color[idx];
  auto  ior       = material_queue.ior[idx];
  auto  roughness = material_queue.roughness[idx];
  auto  normal    = material_queue.normal[idx];
  auto  outgoing  = material_queue.outgoing[idx];
  auto& rng       = globals.state.rngs[idx];
  auto  rn        = rand2f(rng);
  auto  rnl       = rand1f(rng);

  auto incoming = globals.state.path.rays[idx].d;
  auto position = globals.state.path.rays[idx].o;
  if (incoming == vec3f{0, 0, 0}) {  // otherwise use ray direction
    incoming = sample_transparent(
        color, ior, roughness, normal, outgoing, rnl, rn);
  }

  if (incoming == vec3f{0, 0, 0}) {
    globals.state.path.weights[idx] = {0, 0, 0};  // terminate
    return;
  }

  auto bsdfcos = eval_transparent(
      color, ior, roughness, normal, outgoing, incoming);
  auto bsdfcos_pdf = sample_tranparent_pdf(
      color, ior, roughness, normal, outgoing, incoming);
  auto lights_pdf = sample_lights_pdf(
      globals.scene, globals.bvh, globals.lights, position, incoming);

  globals.state.path.weights[idx] *= bsdfcos /
                                     (0.5f * bsdfcos_pdf + 0.5f * lights_pdf);

  eval_material_epilogue(material_type::transparent, idx, normal, outgoing,
      incoming, globals.state.path.weights[idx]);

  globals.state.path.rays[idx] = {position, incoming};
}

extern "C" __global__ void eval_refractive_kernel(int num_materials) {
  int mat_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (mat_idx >= num_materials) {
    return;
  }

  auto& material_queue = globals.state.material_queue;
  int   queue_size     = globals.state.width * globals.state.height;
  int   offset         = 4;
  int   queue_idx      = offset * queue_size + mat_idx;
  int   idx            = material_queue.indices[queue_idx];

  auto  color     = material_queue.color[idx];
  auto  ior       = material_queue.ior[idx];
  auto  roughness = material_queue.roughness[idx];
  auto  normal    = material_queue.normal[idx];
  auto  outgoing  = material_queue.outgoing[idx];
  auto& rng       = globals.state.rngs[idx];
  auto  rn        = rand2f(rng);
  auto  rnl       = rand1f(rng);

  auto incoming = globals.state.path.rays[idx].d;
  auto position = globals.state.path.rays[idx].o;
  if (incoming == vec3f{0, 0, 0}) {  // otherwise use ray direction
    incoming = sample_refractive(
        color, ior, roughness, normal, outgoing, rnl, rn);
  }

  if (incoming == vec3f{0, 0, 0}) {
    globals.state.path.weights[idx] = {0, 0, 0};  // terminate
    return;
  }

  auto bsdfcos = eval_refractive(
      color, ior, roughness, normal, outgoing, incoming);
  auto bsdfcos_pdf = sample_refractive_pdf(
      color, ior, roughness, normal, outgoing, incoming);
  auto lights_pdf = sample_lights_pdf(
      globals.scene, globals.bvh, globals.lights, position, incoming);

  globals.state.path.weights[idx] *= bsdfcos /
                                     (0.5f * bsdfcos_pdf + 0.5f * lights_pdf);

  eval_material_epilogue(material_type::refractive, idx, normal, outgoing,
      incoming, globals.state.path.weights[idx]);

  globals.state.path.rays[idx] = {position, incoming};
}

extern "C" __global__ void eval_gltfpbr_kernel(int num_materials) {
  int mat_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (mat_idx >= num_materials) {
    return;
  }

  auto& material_queue = globals.state.material_queue;
  int   queue_size     = globals.state.width * globals.state.height;
  int   offset         = 5;
  int   queue_idx      = offset * queue_size + mat_idx;
  int   idx            = material_queue.indices[queue_idx];

  auto  color     = material_queue.color[idx];
  auto  ior       = material_queue.ior[idx];
  auto  roughness = material_queue.roughness[idx];
  auto  metallic  = material_queue.metallic[idx];
  auto  normal    = material_queue.normal[idx];
  auto  outgoing  = material_queue.outgoing[idx];
  auto& rng       = globals.state.rngs[idx];
  auto  rn        = rand2f(rng);
  auto  rnl       = rand1f(rng);

  auto incoming = globals.state.path.rays[idx].d;
  auto position = globals.state.path.rays[idx].o;
  if (incoming == vec3f{0, 0, 0}) {  // otherwise use ray direction
    incoming = sample_gltfpbr(
        color, ior, roughness, metallic, normal, outgoing, rnl, rn);
  }

  if (incoming == vec3f{0, 0, 0}) {
    globals.state.path.weights[idx] = {0, 0, 0};  // terminate
    return;
  }

  auto bsdfcos = eval_gltfpbr(
      color, ior, roughness, metallic, normal, outgoing, incoming);
  auto bsdfcos_pdf = sample_gltfpbr_pdf(
      color, ior, roughness, metallic, normal, outgoing, incoming);
  auto lights_pdf = sample_lights_pdf(
      globals.scene, globals.bvh, globals.lights, position, incoming);

  globals.state.path.weights[idx] *= bsdfcos /
                                     (0.5f * bsdfcos_pdf + 0.5f * lights_pdf);

  eval_material_epilogue(material_type::gltfpbr, idx, normal, outgoing,
      incoming, globals.state.path.weights[idx]);

  globals.state.path.rays[idx] = {position, incoming};
}

// logic phase of the wavefront algorithm
// we generate a new ray if its the first invocation, or if the
// last ray terminated due to a miss/max depth reached/opacity/rr
extern "C" __global__ void trace_pixel_logic(
    int* num_samples_done, int* sample_queue_front, int* mat_queue_fronts) {
  // pixel index
  uint2 ij;
  ij.x = blockIdx.x * blockDim.x + threadIdx.x;
  ij.y = blockIdx.y * blockDim.y + threadIdx.y;

  if (ij.x >= globals.state.width || ij.y >= globals.state.height) {
    return;
  }

  trace_sample(globals.state, globals.scene, globals.bvh, globals.lights, ij.x,
      ij.y, globals.params, num_samples_done, sample_queue_front,
      mat_queue_fronts);
}

// extend phase of the wavefront algorithm
// we fetch the next ray to trace from the queue, trace it,
// and write the result back to the queue
extern "C" __global__ void trace_pixel_extend() {
  // pixel index
  uint2 ij;
  ij.x = blockIdx.x * blockDim.x + threadIdx.x;
  ij.y = blockIdx.y * blockDim.y + threadIdx.y;

  if (ij.x >= globals.state.width || ij.y >= globals.state.height) {
    return;
  }

  auto idx = ij.y * globals.state.width + ij.x;

  auto pixel_idx = globals.state.path.indices[idx];
  if (pixel_idx < 0) {
    return;
  }

  auto ray = globals.state.path.rays[idx];
  if (ray.d == vec3f{0, 0, 0}) {
    globals.state.intersection.hit[idx] = false;
    return;
  }

  auto intersection = intersect_scene(globals.bvh, globals.scene, ray, false);

  auto& intersections         = globals.state.intersection;
  intersections.instance[idx] = intersection.instance;
  intersections.element[idx]  = intersection.element;
  intersections.uv[idx]       = intersection.uv;
  intersections.distance[idx] = intersection.distance;
  intersections.hit[idx]      = intersection.hit;
}

// fill in the sample queue by generating #batch rays for each pixel
extern "C" __global__ void trace_pixel_raygen(int* sample_queue_front) {
  // pixel index
  uint2 ij;
  ij.x = blockIdx.x * blockDim.x + threadIdx.x;
  ij.y = blockIdx.y * blockDim.y + threadIdx.y;

  int width  = globals.state.width;
  int height = globals.state.height;

  if (ij.x >= width || ij.y >= height) {
    return;
  }

  auto idx = ij.y * globals.state.width + ij.x;

  if (globals.state.samples == 0) {
    globals.state.image[idx] = {0, 0, 0, 0};
    globals.state.rngs[idx]  = make_rng(98273987, idx * 2 + 1);
  }

  // scale the image by the number of samples
  globals.state.image[idx] *= globals.state.samples;
  globals.state.albedo[idx] *= globals.state.samples;
  globals.state.normal[idx] *= globals.state.samples;

  // globals.state.sample_queue[idx] = {(int)idx, 0};

  // fetch the first sample
  fetch_sample(globals.state, globals.params, idx, sample_queue_front);
}

// postprocess the image to convert to final pixel values
extern "C" __global__ void trace_pixel_epilogue() {
  // pixel index
  uint2 ij;
  ij.x = blockIdx.x * blockDim.x + threadIdx.x;
  ij.y = blockIdx.y * blockDim.y + threadIdx.y;

  int width  = globals.state.width;
  int height = globals.state.height;

  if (ij.x >= width || ij.y >= height) {
    return;
  }

  float sample = globals.state.samples + globals.params.batch;
  int   idx    = ij.y * globals.state.width + ij.x;

  // get the mean pixel value
  globals.state.image[idx] /= sample;
  globals.state.albedo[idx] /= sample;
  globals.state.normal[idx] /= sample;
}

// dispatch trace_pixel for each pixel
extern "C" void cutrace_samples(CUdeviceptr trace_globals) {
  auto globals_cpu = cutrace_globals{};
  auto result      = cuMemcpyDtoH(
      &globals_cpu, trace_globals, sizeof(cutrace_globals));
  if (result != CUDA_SUCCESS) {
    const char* error_name;
    cuGetErrorName(result, &error_name);
    printf("cutrace_samples: cuMemcpyDtoH error %s\n", error_name);
  }

  auto cpyResult = cudaMemcpyToSymbol(
      globals, &globals_cpu, sizeof(cutrace_globals));
  if (cpyResult != cudaSuccess) {
    printf("cutrace_samples: cudaMemcpyToSymbol error %s\n",
        cudaGetErrorName(cpyResult));
  }

  int* num_samples_done;
  int* sample_queue_front;
  int* mat_queue_fronts_gpu;

  cudaMalloc(&num_samples_done, sizeof(int));
  cudaMalloc(&sample_queue_front, sizeof(int));
  cudaMalloc(&mat_queue_fronts_gpu, sizeof(int) * 6);

  cudaMemset(num_samples_done, 0, sizeof(int));
  cudaMemset(sample_queue_front, 0, sizeof(int));

  int num_samples_done_cpu = 0;
  int mat_queue_fronts[6]  = {0, 0, 0, 0, 0, 0};

  int width  = globals_cpu.state.width;
  int height = globals_cpu.state.height;

  dim3 blockSize = {16, 16, 1};
  dim3 gridSize  = {(width + blockSize.x - 1) / blockSize.x,
      (height + blockSize.y - 1) / blockSize.y, 1};

  trace_pixel_raygen<<<gridSize, blockSize>>>(
      sample_queue_front);  // generate rays

  int cur            = 0;
  int target_samples = width * height * globals_cpu.params.batch;

  if ((long)width * height * globals_cpu.params.batch != target_samples) {
    throw std::runtime_error("too many samples, overflow");
  }

  while (num_samples_done_cpu < target_samples) {
    cudaMemset(mat_queue_fronts_gpu, 0, sizeof(int) * 6);

    trace_pixel_extend<<<gridSize, blockSize>>>();

    trace_pixel_logic<<<gridSize, blockSize>>>(
        num_samples_done, sample_queue_front, mat_queue_fronts_gpu);

    cudaMemcpy(mat_queue_fronts, mat_queue_fronts_gpu, sizeof(int) * 6,
        cudaMemcpyDeviceToHost);

    auto& material_queue = globals_cpu.state.material_queue;
    int   mat_block_size = 256;
    if (mat_queue_fronts[0] > 0) {
      int num_materials = mat_queue_fronts[0];
      int num_blocks    = (num_materials + mat_block_size - 1) / mat_block_size;
      eval_matte_kernel<<<num_blocks, mat_block_size>>>(num_materials);
    }

    if (mat_queue_fronts[1] > 0) {
      int num_materials = mat_queue_fronts[1];
      int num_blocks    = (num_materials + mat_block_size - 1) / mat_block_size;
      eval_glossy_kernel<<<num_blocks, mat_block_size>>>(num_materials);
    }

    if (mat_queue_fronts[2] > 0) {
      int num_materials = mat_queue_fronts[2];
      int num_blocks    = (num_materials + mat_block_size - 1) / mat_block_size;
      eval_reflective_kernel<<<num_blocks, mat_block_size>>>(num_materials);
    }

    if (mat_queue_fronts[3] > 0) {
      int num_materials = mat_queue_fronts[3];
      int num_blocks    = (num_materials + mat_block_size - 1) / mat_block_size;
      eval_transparent_kernel<<<num_blocks, mat_block_size>>>(num_materials);
    }

    if (mat_queue_fronts[4] > 0) {
      int num_materials = mat_queue_fronts[4];
      int num_blocks    = (num_materials + mat_block_size - 1) / mat_block_size;
      eval_refractive_kernel<<<num_blocks, mat_block_size>>>(num_materials);
    }

    if (mat_queue_fronts[5] > 0) {
      int num_materials = mat_queue_fronts[5];
      int num_blocks    = (num_materials + mat_block_size - 1) / mat_block_size;
      eval_gltfpbr_kernel<<<num_blocks, mat_block_size>>>(num_materials);
    }

    cudaMemcpy(&num_samples_done_cpu, num_samples_done, sizeof(int),
        cudaMemcpyDeviceToHost);

    printf("iteration %d, num samples done: %d/%d, %f\n", cur++,
        num_samples_done_cpu, target_samples,
        (float)num_samples_done_cpu / target_samples);

    printf("mat queue: %d %d %d %d %d %d\n", mat_queue_fronts[0],
        mat_queue_fronts[1], mat_queue_fronts[2], mat_queue_fronts[3],
        mat_queue_fronts[4], mat_queue_fronts[5]);
  }

  trace_pixel_epilogue<<<gridSize, blockSize>>>();

  cudaFree(num_samples_done);
  cudaFree(sample_queue_front);
  cudaFree(mat_queue_fronts_gpu);
}

}  // namespace yocto