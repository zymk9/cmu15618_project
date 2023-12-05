//
// Implementation for cuda_trace.
//

#include "cuda_trace.h"

#if defined(YOCTO_CUDA) && defined(CUSTOM_CUDA)

#include "yocto_sampling.h"

// do not reorder
#include <cuda.h>
// do not reorder
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#ifdef YOCTO_DENOISE
#include <OpenImageDenoise/oidn.hpp>
#endif

// -----------------------------------------------------------------------------
// CUDA HELPERS
// -----------------------------------------------------------------------------
namespace yocto {

static void check_result(CUresult result) {
  if (result != CUDA_SUCCESS) {
    const char* error_name;
    cuGetErrorName(result, &error_name);
    throw std::runtime_error{"Cuda error: " + string{error_name}};
  }
}

static void sync_gpu(CUstream stream) {
  check_result(cuStreamSynchronize(stream));
}

// static void check_result(OptixResult result) {
//   if (result != OPTIX_SUCCESS) {
//     throw std::runtime_error{"Optix error"};
//   }
// }

// make a buffer
template <typename T>
static cuspan<T> make_buffer(CUstream stream, size_t size, const T* data) {
  auto buffer  = cuspan<T>{};
  buffer._size = size;
  check_result(cuMemAlloc(&buffer._data, buffer.size_in_bytes()));
  if (data) {
    check_result(cuMemcpyHtoDAsync(
        buffer.device_ptr(), data, buffer.size_in_bytes(), stream));
  }
  return buffer;
}
template <typename T>
static cuspan<T> make_buffer(CUstream stream, const vector<T>& data) {
  if (data.empty()) return {};
  return make_buffer(stream, data.size(), data.data());
}
template <typename T>
static cuspan<T> make_buffer(CUstream stream, const T& data) {
  return make_buffer(stream, 1, &data);
}

// resize a buffer
template <typename T>
static void resize_buffer(
    CUstream stream, cuspan<T>& buffer, size_t size, const T* data) {
  if (buffer._size != size) {
    if (buffer._size != 0) check_result(cuMemFree(buffer._data));
    buffer._size = size;
    check_result(cuMemAlloc(&buffer._data, buffer.size_in_bytes()));
  }
  if (data) {
    check_result(cuMemcpyHtoDAsync(
        buffer.device_ptr(), data, buffer.size_in_bytes(), stream));
  }
}

// update a buffer
template <typename T>
static void update_buffer(
    CUstream stream, cuspan<T>& buffer, size_t size, const T* data) {
  if (buffer.size() != size) throw std::runtime_error{"Cuda buffer error"};
  check_result(cuMemcpyHtoDAsync(
      buffer.device_ptr(), data, buffer.size_in_bytes(), stream));
}
template <typename T>
static void update_buffer(
    CUstream stream, cuspan<T>& buffer, const vector<T>& data) {
  return update_buffer(stream, buffer, data.size(), data.data());
}
template <typename T>
static void update_buffer(CUstream stream, cuspan<T>& buffer, const T& data) {
  return update_buffer(stream, buffer, 1, &data);
}

// update a buffer
template <typename T, typename T1>
static void update_buffer_value(CUstream stream, cuspan<T>& buffer,
    size_t offset, size_t size, const T1* data) {
  check_result(cuMemcpyHtoDAsync(
      buffer.device_ptr() + offset, data, size * sizeof(T1), stream));
}
template <typename T, typename T1>
static void update_buffer_value(
    CUstream stream, cuspan<T>& buffer, size_t offset, const T1& data) {
  return update_buffer_value(stream, buffer, offset, 1, &data);
}

// download buffer --- these are synched to avoid errors
template <typename T>
static void download_buffer(const cuspan<T>& buffer, size_t size, void* data) {
  if (buffer.size() != size) throw std::runtime_error{"Cuda download error"};
  check_result(cuMemcpyDtoH(data, buffer.device_ptr(), buffer.size_in_bytes()));
}
template <typename T>
static void download_buffer(const cuspan<T>& buffer, vector<T>& data) {
  return download_buffer(buffer, data.size(), data.data());
}
template <typename T>
static void download_buffer(const cuspan<T>& buffer, T& data) {
  return download_buffer(buffer, 1, &data);
}
template <typename T>
static vector<T> download_buffer_vector(const cuspan<T>& buffer) {
  auto data = vector<T>(buffer.size());
  download_buffer(buffer, data.size(), data.data());
  return data;
}
template <typename T>
static T download_buffer_value(const cuspan<T>& buffer) {
  if (buffer.size() != 1) throw std::runtime_error{"Cuda download error"};
  auto data = T{};
  download_buffer(buffer, 1, &data);
  return data;
}

// free buffer
template <typename T>
static void clear_buffer(cuspan<T>& buffer) {
  if (buffer.device_ptr() == 0) return;
  check_result(cuMemFree(buffer.device_ptr()));
  buffer._data = 0;
  buffer._size = 0;
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// HACKS
// -----------------------------------------------------------------------------
namespace yocto {

extern "C" char yocto_cutrace_ptx[];
extern "C" void cutrace_samples(CUdeviceptr trace_globals);

cuscene_data::cuscene_data(cuscene_data&& other) {
  cameras.swap(other.cameras);
  textures.swap(other.textures);
  materials.swap(other.materials);
  shapes.swap(other.shapes);
  instances.swap(other.instances);
  environments.swap(other.environments);
}
cuscene_data& cuscene_data::operator=(cuscene_data&& other) {
  cameras.swap(other.cameras);
  textures.swap(other.textures);
  materials.swap(other.materials);
  shapes.swap(other.shapes);
  instances.swap(other.instances);
  environments.swap(other.environments);
  return *this;
}
cuscene_data::~cuscene_data() {
  if (!textures.empty()) {
    auto textures_ = download_buffer_vector(textures);
    for (auto& texture : textures_) {
      cuArrayDestroy(texture.array);
      // TODO: texture
    }
  }
  if (!shapes.empty()) {
    auto shapes_ = download_buffer_vector(shapes);
    for (auto& shape : shapes_) {
      clear_buffer(shape.positions);
      clear_buffer(shape.normals);
      clear_buffer(shape.texcoords);
      clear_buffer(shape.colors);
      clear_buffer(shape.triangles);
    }
  }
  clear_buffer(cameras);
  clear_buffer(textures);
  clear_buffer(materials);
  clear_buffer(shapes);
  clear_buffer(instances);
  clear_buffer(environments);
};

cubvh_tree::cubvh_tree(cubvh_tree&& other) {
  nodes.swap(other.nodes);
  primitives.swap(other.primitives);
}
cubvh_tree& cubvh_tree::operator=(cubvh_tree&& other) {
  nodes.swap(other.nodes);
  primitives.swap(other.primitives);
  return *this;
}
cubvh_tree::~cubvh_tree() {
  clear_buffer(nodes);
  clear_buffer(primitives);
}

cushape_bvh::cushape_bvh(cushape_bvh&& other) {
  bvh.nodes.swap(other.bvh.nodes);
  bvh.primitives.swap(other.bvh.primitives);
}
cushape_bvh& cushape_bvh::operator=(cushape_bvh&& other) {
  bvh.nodes.swap(other.bvh.nodes);
  bvh.primitives.swap(other.bvh.primitives);
  return *this;
}
cushape_bvh::~cushape_bvh() {}

cuscene_bvh::cuscene_bvh(cuscene_bvh&& other) {
  shapes.swap(other.shapes);
  bvh.nodes.swap(other.bvh.nodes);
  bvh.primitives.swap(other.bvh.primitives);
}
cuscene_bvh& cuscene_bvh::operator=(cuscene_bvh&& other) {
  shapes.swap(other.shapes);
  bvh.nodes.swap(other.bvh.nodes);
  bvh.primitives.swap(other.bvh.primitives);
  return *this;
}
cuscene_bvh::~cuscene_bvh() { clear_buffer(shapes); }

cutrace_context::cutrace_context(cutrace_context&& other) {
  globals_buffer.swap(other.globals_buffer);
  std::swap(cuda_stream, other.cuda_stream);
  std::swap(cuda_context, other.cuda_context);
}
cutrace_context& cutrace_context::operator=(cutrace_context&& other) {
  globals_buffer.swap(other.globals_buffer);
  std::swap(cuda_stream, other.cuda_stream);
  std::swap(cuda_context, other.cuda_context);
  return *this;
}

cutrace_state::cutrace_state(cutrace_state&& other) {
  std::swap(width, other.width);
  std::swap(height, other.height);
  std::swap(samples, other.samples);
  image.swap(other.image);
  albedo.swap(other.albedo);
  normal.swap(other.normal);
  hits.swap(other.hits);
  rngs.swap(other.rngs);
  denoised.swap(other.denoised);
  denoiser_state.swap(other.denoiser_state);
  denoiser_scratch.swap(other.denoiser_scratch);
}
cutrace_state& cutrace_state::operator=(cutrace_state&& other) {
  std::swap(width, other.width);
  std::swap(height, other.height);
  std::swap(samples, other.samples);
  image.swap(other.image);
  albedo.swap(other.albedo);
  normal.swap(other.normal);
  hits.swap(other.hits);
  rngs.swap(other.rngs);
  denoised.swap(other.denoised);
  denoiser_state.swap(other.denoiser_state);
  denoiser_scratch.swap(other.denoiser_scratch);
  return *this;
}
cutrace_state::~cutrace_state() {
  clear_buffer(image);
  clear_buffer(albedo);
  clear_buffer(normal);
  clear_buffer(hits);
  clear_buffer(rngs);
  clear_buffer(denoised);
  clear_buffer(denoiser_state);
  clear_buffer(denoiser_scratch);
}

cutrace_lights::cutrace_lights(cutrace_lights&& other) {
  lights.swap(other.lights);
}
cutrace_lights& cutrace_lights::operator=(cutrace_lights&& other) {
  lights.swap(other.lights);
  return *this;
}
cutrace_lights::~cutrace_lights() {
  if (!lights.empty()) {
    auto lights_ = download_buffer_vector(lights);
    for (auto& light : lights_) {
      clear_buffer(light.elements_cdf);
    }
  }
  clear_buffer(lights);
}

cutrace_context::~cutrace_context() {
  // global buffer
  clear_buffer(globals_buffer);

  // context
  cuStreamDestroy(cuda_stream);
  cuCtxDestroy(cuda_context);
}

static void optix_log_callback(
    unsigned int level, const char* tag, const char* message, void* cbdata) {
  printf("[%s] %s\n", tag, message);
}

// init cuda and optix context
cutrace_context make_cutrace_context(const trace_params& params) {
  // context
  auto context = cutrace_context{};

  // init cuda
  check_result(cuInit(0));
  auto device = CUdevice{0};
  check_result(cuCtxCreate(&context.cuda_context, CU_CTX_SCHED_SPIN, device));

  // init optix
  // check_result(optixInit());

  // init cuda device
  check_result(cuStreamCreate(&context.cuda_stream, CU_STREAM_DEFAULT));

  // globals
  context.globals_buffer = make_buffer(context.cuda_stream, cutrace_globals{});

  // sync gpu
  sync_gpu(context.cuda_stream);

  return context;
}

// start a new render
void trace_start(cutrace_context& context, cutrace_state& state,
    const cuscene_data& cuscene, const cuscene_bvh& bvh,
    const cutrace_lights& lights, const scene_data& scene,
    const trace_params& params) {
  auto globals = cutrace_globals{};
  update_buffer_value(context.cuda_stream, context.globals_buffer,
      offsetof(cutrace_globals, state), state);
  update_buffer_value(context.cuda_stream, context.globals_buffer,
      offsetof(cutrace_globals, scene), cuscene);
  update_buffer_value(context.cuda_stream, context.globals_buffer,
      offsetof(cutrace_globals, bvh), bvh);
  update_buffer_value(context.cuda_stream, context.globals_buffer,
      offsetof(cutrace_globals, lights), lights);
  update_buffer_value(context.cuda_stream, context.globals_buffer,
      offsetof(cutrace_globals, params), params);
  // sync to avoid errors
  sync_gpu(context.cuda_stream);
}

// render a batch of samples
void trace_samples(cutrace_context& context, cutrace_state& state,
    const cuscene_data& cuscene, const cuscene_bvh& bvh,
    const cutrace_lights& lights, const scene_data& scene,
    const trace_params& params) {
  if (state.samples >= params.samples) return;
  auto nsamples = params.batch;
  update_buffer_value(context.cuda_stream, context.globals_buffer,
      offsetof(cutrace_globals, state) + offsetof(cutrace_state, samples),
      state.samples);

  sync_gpu(context.cuda_stream);
  cutrace_samples(context.globals_buffer.device_ptr());

  state.samples += nsamples;
  if (params.denoise) {
    denoise_image(context, state);
  }
  // sync so we can get the image
  sync_gpu(context.cuda_stream);
}

cuscene_data make_cutrace_scene(cutrace_context& context,
    const scene_data& scene, const trace_params& params) {
  auto cuscene = cuscene_data{};

  auto cucameras = vector<cucamera_data>{};
  for (auto& camera : scene.cameras) {
    auto& cucamera        = cucameras.emplace_back();
    cucamera.frame        = camera.frame;
    cucamera.lens         = camera.lens;
    cucamera.aspect       = camera.aspect;
    cucamera.film         = camera.film;
    cucamera.aperture     = camera.aperture;
    cucamera.focus        = camera.focus;
    cucamera.orthographic = camera.orthographic;
  }
  cuscene.cameras = make_buffer(context.cuda_stream, cucameras);

  // shapes
  auto cushapes = vector<cushape_data>{};
  for (auto& shape : scene.shapes) {
    auto& cushape     = cushapes.emplace_back();
    cushape.positions = make_buffer(context.cuda_stream, shape.positions);
    cushape.triangles = make_buffer(context.cuda_stream, shape.triangles);
    if (!shape.normals.empty())
      cushape.normals = make_buffer(context.cuda_stream, shape.normals);
    if (!shape.texcoords.empty())
      cushape.texcoords = make_buffer(context.cuda_stream, shape.texcoords);
    if (!shape.colors.empty())
      cushape.colors = make_buffer(context.cuda_stream, shape.colors);
  }
  cuscene.shapes = make_buffer(context.cuda_stream, cushapes);

  // textures
  auto cutextures = vector<cutexture_data>{};
  for (auto& texture : scene.textures) {
    auto& cutexture   = cutextures.emplace_back();
    cutexture.width   = texture.width;
    cutexture.height  = texture.height;
    cutexture.linear  = texture.linear;
    cutexture.nearest = texture.nearest;
    cutexture.clamp   = texture.clamp;

    auto as_byte = !texture.pixelsb.empty();

    auto array_descriptor        = CUDA_ARRAY_DESCRIPTOR{};
    array_descriptor.Width       = texture.width;
    array_descriptor.Height      = texture.height;
    array_descriptor.NumChannels = 4;
    array_descriptor.Format      = as_byte ? CU_AD_FORMAT_UNSIGNED_INT8
                                           : CU_AD_FORMAT_FLOAT;
    check_result(cuArrayCreate(&cutexture.array, &array_descriptor));

    auto memcpy_descriptor          = CUDA_MEMCPY2D{};
    memcpy_descriptor.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    memcpy_descriptor.dstArray      = cutexture.array;
    memcpy_descriptor.dstXInBytes   = 0;
    memcpy_descriptor.dstY          = 0;
    memcpy_descriptor.srcMemoryType = CU_MEMORYTYPE_HOST;
    memcpy_descriptor.srcHost       = nullptr;
    memcpy_descriptor.srcXInBytes   = 0;
    memcpy_descriptor.srcY          = 0;
    memcpy_descriptor.srcPitch      = texture.width * (as_byte ? 4 : 16);
    memcpy_descriptor.WidthInBytes  = texture.width * (as_byte ? 4 : 16);
    memcpy_descriptor.Height        = texture.height;
    if (!texture.pixelsb.empty()) {
      memcpy_descriptor.srcHost = texture.pixelsb.data();
      check_result(cuMemcpy2D(&memcpy_descriptor));
    }
    if (!texture.pixelsf.empty()) {
      memcpy_descriptor.srcHost = texture.pixelsf.data();
      check_result(cuMemcpy2D(&memcpy_descriptor));
    }

    auto resource_descriptor               = CUDA_RESOURCE_DESC{};
    resource_descriptor.resType            = CU_RESOURCE_TYPE_ARRAY;
    resource_descriptor.res.array.hArray   = cutexture.array;
    auto texture_descriptor                = CUDA_TEXTURE_DESC{};
    texture_descriptor.addressMode[0]      = CU_TR_ADDRESS_MODE_WRAP;
    texture_descriptor.addressMode[1]      = CU_TR_ADDRESS_MODE_WRAP;
    texture_descriptor.addressMode[2]      = CU_TR_ADDRESS_MODE_WRAP;
    texture_descriptor.filterMode          = CU_TR_FILTER_MODE_LINEAR;
    texture_descriptor.flags               = CU_TRSF_NORMALIZED_COORDINATES;
    texture_descriptor.maxAnisotropy       = 1;
    texture_descriptor.maxMipmapLevelClamp = 99;
    texture_descriptor.minMipmapLevelClamp = 0;
    texture_descriptor.mipmapFilterMode    = CU_TR_FILTER_MODE_POINT;
    texture_descriptor.borderColor[0]      = 1.0f;
    texture_descriptor.borderColor[1]      = 1.0f;
    texture_descriptor.borderColor[2]      = 1.0f;
    texture_descriptor.borderColor[3]      = 1.0f;
    check_result(cuTexObjectCreate(&cutexture.texture, &resource_descriptor,
        &texture_descriptor, nullptr));
  }
  cuscene.textures = make_buffer(context.cuda_stream, cutextures);

  auto materials = vector<cumaterial_data>{};
  for (auto& material : scene.materials) {
    auto& cumaterial      = materials.emplace_back();
    cumaterial.type       = material.type;
    cumaterial.emission   = material.emission;
    cumaterial.color      = material.color;
    cumaterial.roughness  = material.roughness;
    cumaterial.metallic   = material.metallic;
    cumaterial.ior        = material.ior;
    cumaterial.scattering = material.scattering;
    cumaterial.trdepth    = material.trdepth;
    cumaterial.opacity    = material.opacity;

    cumaterial.emission_tex   = material.emission_tex;
    cumaterial.color_tex      = material.color_tex;
    cumaterial.roughness_tex  = material.roughness_tex;
    cumaterial.scattering_tex = material.scattering_tex;
    cumaterial.normal_tex     = material.normal_tex;
  }
  cuscene.materials = make_buffer(context.cuda_stream, materials);

  auto instances = vector<cuinstance_data>{};
  for (auto& instance : scene.instances) {
    auto& cuinstance    = instances.emplace_back();
    cuinstance.frame    = instance.frame;
    cuinstance.shape    = instance.shape;
    cuinstance.material = instance.material;
  }
  cuscene.instances = make_buffer(context.cuda_stream, instances);

  auto environments = vector<cuenvironment_data>{};
  for (auto& environment : scene.environments) {
    auto& cuenvironment        = environments.emplace_back();
    cuenvironment.frame        = environment.frame;
    cuenvironment.emission     = environment.emission;
    cuenvironment.emission_tex = environment.emission_tex;
  }
  cuscene.environments = make_buffer(context.cuda_stream, environments);

  // sync gpu
  sync_gpu(context.cuda_stream);

  return cuscene;
}

void update_cutrace_cameras(cutrace_context& context, cuscene_data& cuscene,
    const scene_data& scene, const trace_params& params) {
  auto cucameras = vector<cucamera_data>{};
  for (auto& camera : scene.cameras) {
    auto& cucamera        = cucameras.emplace_back();
    cucamera.frame        = camera.frame;
    cucamera.lens         = camera.lens;
    cucamera.aspect       = camera.aspect;
    cucamera.film         = camera.film;
    cucamera.aperture     = camera.aperture;
    cucamera.focus        = camera.focus;
    cucamera.orthographic = camera.orthographic;
  }
  update_buffer(context.cuda_stream, cuscene.cameras, cucameras);
  sync_gpu(context.cuda_stream);
}

cubvh_tree make_cubvh_tree(cutrace_context& context, const bvh_tree& bvh_cpu) {
  auto bvh = cubvh_tree{};

  // nodes
  bvh.nodes = make_buffer(context.cuda_stream, bvh_cpu.nodes);

  // primitives
  bvh.primitives = make_buffer(context.cuda_stream, bvh_cpu.primitives);

  // sync
  sync_gpu(context.cuda_stream);

  return bvh;
}

cutrace_bvh make_cutrace_bvh(cutrace_context& context,
    const cuscene_data& scene, const trace_params& params,
    const scene_bvh& bvh_cpu) {
  auto bvh = cutrace_bvh{};

  bvh.bvh = make_cubvh_tree(context, bvh_cpu.bvh);
  context.shape_bvhs.resize(bvh_cpu.shapes.size());

  for (size_t i = 0; i < bvh_cpu.shapes.size(); i++) {
    context.shape_bvhs[i].bvh = make_cubvh_tree(context, bvh_cpu.shapes[i].bvh);
  }

  bvh.shapes = make_buffer(context.cuda_stream, context.shape_bvhs);

  // sync gpu
  sync_gpu(context.cuda_stream);

  // done
  return bvh;
}

// Initialize state.
cutrace_state make_cutrace_state(cutrace_context& context,
    const scene_data& scene, const trace_params& params) {
  auto& camera = scene.cameras[params.camera];
  auto  state  = cutrace_state{};
  if (camera.aspect >= 1) {
    state.width  = params.resolution;
    state.height = (int)round(params.resolution / camera.aspect);
  } else {
    state.height = params.resolution;
    state.width  = (int)round(params.resolution * camera.aspect);
  }
  state.samples = 0;
  state.image   = make_buffer(
      context.cuda_stream, state.width * state.height, (vec4f*)nullptr);
  state.albedo = make_buffer(
      context.cuda_stream, state.width * state.height, (vec3f*)nullptr);
  state.normal = make_buffer(
      context.cuda_stream, state.width * state.height, (vec3f*)nullptr);
  state.hits = make_buffer(
      context.cuda_stream, state.width * state.height, (int*)nullptr);
  state.rngs = make_buffer(
      context.cuda_stream, state.width * state.height, (rng_state*)nullptr);
  // if (params.denoise) {
  //   auto denoiser_sizes = OptixDenoiserSizes{};
  //   check_result(optixDenoiserComputeMemoryResources(
  //       context.denoiser, state.width, state.height, &denoiser_sizes));
  //   state.denoised = make_buffer(
  //       context.cuda_stream, state.width * state.height, (vec4f*)nullptr);
  //   state.denoiser_state = make_buffer(
  //       context.cuda_stream, denoiser_sizes.stateSizeInBytes,
  //       (byte*)nullptr);
  //   state.denoiser_scratch = make_buffer(context.cuda_stream,
  //       denoiser_sizes.withoutOverlapScratchSizeInBytes, (byte*)nullptr);
  // }
  sync_gpu(context.cuda_stream);
  return state;
};

void reset_cutrace_state(cutrace_context& context, cutrace_state& state,
    const scene_data& scene, const trace_params& params) {
  auto& camera = scene.cameras[params.camera];
  if (camera.aspect >= 1) {
    state.width  = params.resolution;
    state.height = (int)round(params.resolution / camera.aspect);
  } else {
    state.height = params.resolution;
    state.width  = (int)round(params.resolution * camera.aspect);
  }
  state.samples = 0;
  resize_buffer(context.cuda_stream, state.image, state.width * state.height,
      (vec4f*)nullptr);
  resize_buffer(context.cuda_stream, state.albedo, state.width * state.height,
      (vec3f*)nullptr);
  resize_buffer(context.cuda_stream, state.normal, state.width * state.height,
      (vec3f*)nullptr);
  resize_buffer(context.cuda_stream, state.hits, state.width * state.height,
      (int*)nullptr);
  resize_buffer(context.cuda_stream, state.rngs, state.width * state.height,
      (rng_state*)nullptr);
  // if (params.denoise) {
  //   auto denoiser_sizes = OptixDenoiserSizes{};
  //   check_result(optixDenoiserComputeMemoryResources(
  //       context.denoiser, state.width, state.height, &denoiser_sizes));
  //   resize_buffer(context.cuda_stream, state.denoised,
  //       state.width * state.height, (vec4f*)nullptr);
  //   resize_buffer(context.cuda_stream, state.denoiser_state,
  //       denoiser_sizes.stateSizeInBytes, (byte*)nullptr);
  //   resize_buffer(context.cuda_stream, state.denoiser_scratch,
  //       denoiser_sizes.withoutOverlapScratchSizeInBytes, (byte*)nullptr);
  // } else {
  //   clear_buffer(state.denoised);
  //   clear_buffer(state.denoiser_state);
  //   clear_buffer(state.denoiser_scratch);
  // }
  sync_gpu(context.cuda_stream);
}

// Init trace lights
cutrace_lights make_cutrace_lights(cutrace_context& context,
    const scene_data& scene, const trace_params& params) {
  auto lights    = make_trace_lights(scene, (const trace_params&)params);
  auto culights_ = vector<cutrace_light>{};
  for (auto& light : lights.lights) {
    auto& culight        = culights_.emplace_back();
    culight.instance     = light.instance;
    culight.environment  = light.environment;
    culight.elements_cdf = make_buffer(context.cuda_stream, light.elements_cdf);
  }
  auto culights   = cutrace_lights{};
  culights.lights = make_buffer(context.cuda_stream, culights_);
  sync_gpu(context.cuda_stream);
  return culights;
}

// Copmutes an image
image_data cutrace_image(const scene_data& scene, const trace_params& params,
    const scene_bvh& bvh_cpu) {
  // initialization
  auto context = make_cutrace_context(params);
  auto cuscene = make_cutrace_scene(context, scene, params);
  auto bvh     = make_cutrace_bvh(context, cuscene, params, bvh_cpu);
  auto state   = make_cutrace_state(context, scene, params);
  auto lights  = make_cutrace_lights(context, scene, params);

  // rendering
  trace_start(context, state, cuscene, bvh, lights, scene, params);
  for (auto sample = 0; sample < params.samples; sample++) {
    trace_samples(context, state, cuscene, bvh, lights, scene, params);
  }

  // copy back image and return
  return get_image(state);
}

// render preview
void trace_preview(image_data& image, cutrace_context& context,
    cutrace_state& pstate, const cuscene_data& cuscene, const cutrace_bvh& bvh,
    const cutrace_lights& lights, const scene_data& scene,
    const trace_params& params) {
  auto pparams = params;
  pparams.resolution /= params.pratio;
  pparams.samples = 1;
  reset_cutrace_state(context, pstate, scene, pparams);
  trace_start(context, pstate, cuscene, bvh, lights, scene, pparams);
  trace_samples(context, pstate, cuscene, bvh, lights, scene, pparams);
  auto preview = get_image(pstate);
  for (auto idx = 0; idx < image.width * image.height; idx++) {
    auto i = idx % image.width, j = idx / image.width;
    auto pi           = clamp(i / params.pratio, 0, preview.width - 1),
         pj           = clamp(j / params.pratio, 0, preview.height - 1);
    image.pixels[idx] = preview.pixels[pj * preview.width + pi];
  }
}

// Get resulting render
image_data get_image(const cutrace_state& state) {
  auto image = make_image(state.width, state.height, true);
  get_image(image, state);
  return image;
}
void get_image(image_data& image, const cutrace_state& state) {
  if (state.denoised.empty()) {
    download_buffer(state.image, image.pixels);
  } else {
    download_buffer(state.denoised, image.pixels);
  }
}

// Get resulting render
image_data get_rendered_image(const cutrace_state& state) {
  auto image = make_image(state.width, state.height, true);
  get_rendered_image(image, state);
  return image;
}
void get_rendered_image(image_data& image, const cutrace_state& state) {
  download_buffer(state.image, image.pixels);
}

// Get denoised result
image_data get_denoised_image(const cutrace_state& state) {
  auto image = make_image(state.width, state.height, true);
  get_denoised_image(image, state);
  return image;
}
void get_denoised_image(image_data& image, const cutrace_state& state) {
#if YOCTO_DENOISE
  // Create an Intel Open Image Denoise device
  oidn::DeviceRef device = oidn::newDevice();
  device.commit();

  // get image
  get_rendered_image(image, state);

  // get albedo and normal
  auto albedo = download_buffer_vector(state.albedo);
  auto normal = download_buffer_vector(state.normal);

  // Create a denoising filter
  oidn::FilterRef filter = device.newFilter("RT");  // ray tracing filter
  filter.setImage("color", (void*)image.pixels.data(), oidn::Format::Float3,
      state.width, state.height, 0, sizeof(vec4f), sizeof(vec4f) * state.width);
  filter.setImage("albedo", (void*)albedo.data(), oidn::Format::Float3,
      state.width, state.height);
  filter.setImage("normal", (void*)normal.data(), oidn::Format::Float3,
      state.width, state.height);
  filter.setImage("output", image.pixels.data(), oidn::Format::Float3,
      state.width, state.height, 0, sizeof(vec4f), sizeof(vec4f) * state.width);
  filter.set("inputScale", 1.0f);  // set scale as fixed
  filter.set("hdr", true);         // image is HDR
  filter.commit();

  // Filter the image
  filter.execute();
#else
  get_rendered_image(image, state);
#endif
}

// Get denoising buffers
image_data get_albedo_image(const cutrace_state& state) {
  auto albedo = make_image(state.width, state.height, true);
  get_albedo_image(albedo, state);
  return albedo;
}
void get_albedo_image(image_data& image, const cutrace_state& state) {
  auto albedo = vector<vec3f>(state.width * state.height);
  download_buffer(state.albedo, albedo);
  for (auto idx = 0; idx < state.width * state.height; idx++) {
    image.pixels[idx] = {albedo[idx].x, albedo[idx].y, albedo[idx].z, 1.0f};
  }
}
image_data get_normal_image(const cutrace_state& state) {
  auto normal = make_image(state.width, state.height, true);
  get_normal_image(normal, state);
  return normal;
}
void get_normal_image(image_data& image, const cutrace_state& state) {
  auto normal = vector<vec3f>(state.width * state.height);
  download_buffer(state.normal, normal);
  for (auto idx = 0; idx < state.width * state.height; idx++) {
    image.pixels[idx] = {normal[idx].x, normal[idx].y, normal[idx].z, 1.0f};
  }
}

// denoise image
void denoise_image(cutrace_context& context, cutrace_state& state) {}

bool is_display(const cutrace_context& context) {
  auto device = 0, is_display = 0;
  // check_result(cuDevice(&current_device));
  check_result(cuDeviceGetAttribute(
      &is_display, CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, device));
  return (bool)is_display;
}

}  // namespace yocto

#endif
