// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <string>
#include <utility>
#include <vector>

#include "core/providers/webgpu/webgpu_utils.h"
#include "core/providers/webgpu/nn/im2col_matmul.h"
#include "core/providers/webgpu/nn/conv.h"
#include "core/providers/webgpu/nn/activation_util.h"

namespace onnxruntime {
namespace webgpu {
namespace {
// Width of the major (workgroup/thread) tile dimension.
constexpr uint32_t kMajorTile = 64u;

// Minimum number of dispatched workgroups (per batch) needed to keep the GPU
// busy. Below this, prefer smaller register tiles to spawn more workgroups.
constexpr uint32_t kMinDispatch = 128u;

// Result of tile-size selection: the (M, N) tile shape and whether the kernel
// runs in M-major or N-major orientation.
struct Im2ColTileConfig {
  uint32_t tile_m;
  uint32_t tile_n;
  bool is_m_major;
};

// Chooses the optimal tile shape and orientation for the im2col operation.
// N-major is preferred: the 64-wide "workgroup/thread" dimension spans N and the
// smaller M dimension is tiled by 32 or 16 (the per-thread register tile).
// When im2col_n is smaller than the 64-wide tile (tile_n), N-major would waste
// threads, so the kernel switches to M-major (each thread owns one M row).
// The tile sizes are performance-tuned and vary depending on the target device.
Im2ColTileConfig ChooseTileSize(uint32_t im2col_m, uint32_t im2col_n) {
  const bool is_m_major = im2col_n < kMajorTile;

  // Candidate sizes for the minor (register-tiled) dimension, in descending
  // order of preference. Larger tiles do more work per thread, so they are
  // tried first and used as long as there are enough workgroups to dispatch.
  const std::vector<uint32_t> small_tiles = {32, 16};

  for (const uint32_t tile_small : small_tiles) {
    const uint32_t tile_m = is_m_major ? kMajorTile : tile_small;
    const uint32_t tile_n = is_m_major ? tile_small : kMajorTile;

    const uint32_t dispatch = CeilDiv(im2col_m, tile_m) * CeilDiv(im2col_n, tile_n);
    if (dispatch >= kMinDispatch) {
      return {tile_m, tile_n, is_m_major};
    }
  }

  // None of the tile sizes met the dispatch requirement; fall back to the
  // smallest tile to maximize the number of dispatched workgroups.
  const uint32_t tile_small = small_tiles.back();
  return {is_m_major ? kMajorTile : tile_small, is_m_major ? tile_small : kMajorTile, is_m_major};
}

// Add support for more devices.
bool IsDeviceSupported(const ComputeContextBase& context) {
  const wgpu::AdapterInfo& adapter_info = context.AdapterInfo();

  if (adapter_info.vendor == std::string_view("intel")) {
    if (adapter_info.architecture == std::string_view("xe-2lpg") ||
        adapter_info.architecture == std::string_view("xe-2hpg") ||
        adapter_info.architecture == std::string_view("xe-3lpg") ||
        adapter_info.architecture == std::string_view("xe-3lpg-xs")) {
      return true;
    }
  }

  return false;
}

}  // namespace

Status Im2ColMatMulProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& src = shader.AddInput("src", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  const auto& weight = shader.AddInput("weight", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  if (has_bias_) {
    shader.AddInput("bias", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  }
  const auto& output = shader.AddOutput("output", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);

  ORT_ENFORCE(vec_size_ == 1 || vec_size_ == 2 || vec_size_ == 4, "vec_size must be 1, 2 or 4.");
  if (is_m_major_) {
    ORT_ENFORCE(tile_m_ == 64, "tile_m must be 64 for m-major.");
    ORT_ENFORCE(tile_n_ == 16 || tile_n_ == 32, "tile_n must be 16 or 32 for m-major.");
  } else {
    ORT_ENFORCE(tile_m_ == 16 || tile_m_ == 32, "tile_m must be 16 or 32 for n-major.");
    ORT_ENFORCE(tile_n_ == 64, "tile_n must be 64 for n-major.");
  }

  return WGSL_TEMPLATE_APPLY(shader, "nn/im2col_matmul.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(has_bias, has_bias_),
                             WGSL_TEMPLATE_PARAMETER(is_m_major, is_m_major_),
                             WGSL_TEMPLATE_PARAMETER(tile_m, tile_m_),
                             WGSL_TEMPLATE_PARAMETER(tile_n, tile_n_),
                             WGSL_TEMPLATE_PARAMETER(use_subgroup, use_subgroup_),
                             WGSL_TEMPLATE_PARAMETER(vec_size, vec_size_),
                             WGSL_TEMPLATE_VARIABLE(output, output),
                             WGSL_TEMPLATE_VARIABLE(src, src),
                             WGSL_TEMPLATE_VARIABLE(weight, weight));
}

Status ApplyIm2ColMatMulProgram(ComputeContext& context,
                                bool is_channels_last,
                                const std::vector<uint32_t>& dilations,
                                const std::vector<uint32_t>& pads,
                                const std::vector<uint32_t>& strides,
                                Tensor* output) {
  const auto* src = context.Input<Tensor>(0);
  const auto* weight = context.Input<Tensor>(1);
  const bool has_bias = context.InputCount() > 2;
  const auto* bias = has_bias ? context.Input<Tensor>(2) : nullptr;

  TensorShape weight_shape = weight->Shape();
  const uint32_t channel_output = onnxruntime::narrow<uint32_t>(weight_shape[0]);
  const uint32_t channel_input = onnxruntime::narrow<uint32_t>(weight_shape[1]);
  const uint32_t kernel_height = onnxruntime::narrow<uint32_t>(weight_shape[2]);
  const uint32_t kernel_width = onnxruntime::narrow<uint32_t>(weight_shape[3]);

  // Transpose OIHW Weight to OHWI
  // TODO: Use prepack
  Tensor ohwi_weight;
  ORT_RETURN_IF_ERROR(TransposeKernel(context, weight, weight->Shape(), &ohwi_weight, {0, 2, 3, 1}));

  // im2col-matmul
  const TensorShape src_shape = src->Shape();
  const TensorShape output_shape = output->Shape();

  const uint32_t batch = onnxruntime::narrow<uint32_t>(src_shape[0]);
  const uint32_t src_height = onnxruntime::narrow<uint32_t>(src_shape[is_channels_last ? 1 : 2]);
  const uint32_t src_width = onnxruntime::narrow<uint32_t>(src_shape[is_channels_last ? 2 : 3]);
  const uint32_t output_height = onnxruntime::narrow<uint32_t>(output_shape[is_channels_last ? 1 : 2]);
  const uint32_t output_width = onnxruntime::narrow<uint32_t>(output_shape[is_channels_last ? 2 : 3]);

  const uint32_t im2col_m = output_height * output_width;
  const uint32_t im2col_k = kernel_height * kernel_width * channel_input;
  const uint32_t im2col_n = channel_output;

  const auto [tile_m, tile_n, is_m_major] = ChooseTileSize(im2col_m, im2col_n);
  // The workgroup always spans 64 threads over the major (64-wide) dimension.
  const uint32_t workgroup_size = is_m_major ? tile_m : tile_n;

  // Check the device's subgroup size before shader compilation to avoid potential performance penalties
  // associated with conditional checks in the shader runtime.
  //
  // Ensure the subgroup size must be greater than or equal to `tile_m` to safely enable `use_subgroup`.
  // If the status of this condition is uncertain, the feature must be disabled.
  const bool use_subgroup = false;
  const uint32_t vec_size = channel_input % 4 == 0 ? 4 : (channel_input % 2 == 0 ? 2 : 1);
  Im2ColMatMulProgram im2col_mm_program{has_bias, tile_m, tile_n, vec_size, use_subgroup, is_m_major};
  im2col_mm_program.SetWorkgroupSize(workgroup_size);

  const uint32_t M_tiles = CeilDiv(im2col_m, tile_m);
  const uint32_t N_tiles = CeilDiv(im2col_n, tile_n);
  im2col_mm_program.SetDispatchGroupSize(M_tiles, N_tiles, batch);

  im2col_mm_program.AddInput({src,
                              ProgramTensorMetadataDependency::TypeAndRank,
                              static_cast<int>(vec_size)});
  im2col_mm_program.AddInput({&ohwi_weight,
                              ProgramTensorMetadataDependency::TypeAndRank,
                              static_cast<int>(vec_size)});
  if (has_bias) {
    im2col_mm_program.AddInput({bias,
                                ProgramTensorMetadataDependency::TypeAndRank});
  }
  im2col_mm_program.AddOutput({output,
                               ProgramTensorMetadataDependency::TypeAndRank});
  im2col_mm_program.AddUniformVariables({{batch},
                                         {src_height},
                                         {src_width},
                                         {channel_input},
                                         {kernel_height},
                                         {kernel_width},
                                         {output_height},
                                         {output_width},
                                         {im2col_m},
                                         {im2col_k},
                                         {im2col_n},
                                         {M_tiles},
                                         {N_tiles},
                                         {CeilDiv(CeilDiv(im2col_k, 4u), 4u)},
                                         {dilations},
                                         {pads},
                                         {strides}});
  im2col_mm_program.CacheHint(has_bias, tile_m, tile_n, vec_size, use_subgroup, is_m_major);

  return context.RunProgram(im2col_mm_program);
}

bool CanApplyIm2ColMatMulProgram(ComputeContextBase& context,
                                 const bool is_channels_last,
                                 const bool is_fused,
                                 const TensorShape weight_shape,
                                 const uint32_t group,
                                 const MLDataType data_type) {
  if (!IsDeviceSupported(context)) {
    return false;
  }

  // The im2col-matmul kernel is performance-tuned for fp16. Use the default
  // conv path for fp32.
  if (data_type != DataTypeImpl::GetType<MLFloat16>()) {
    return false;
  }

  // TODO: Support !is_channels_last
  // TODO: Support fuse
  // TODO: Support group conv
  if (!is_channels_last || is_fused || group != 1) {
    return false;
  }

  // TODO: Support conv1d
  // TODO: Support conv2d_1x1
  const uint32_t kernel_height = onnxruntime::narrow<uint32_t>(weight_shape[2]);
  const uint32_t kernel_width = onnxruntime::narrow<uint32_t>(weight_shape[3]);
  if (kernel_height == 1 || kernel_width == 1) {
    return false;
  }

  return true;
}

}  // namespace webgpu
}  // namespace onnxruntime
