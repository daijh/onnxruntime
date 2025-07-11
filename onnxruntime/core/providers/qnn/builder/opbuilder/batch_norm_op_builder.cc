// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>
#include <cmath>
#include <utility>

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/op_builder_factory.h"

namespace onnxruntime {
namespace qnn {
class BatchNormOpBuilder : public BaseOpBuilder {
 public:
  BatchNormOpBuilder() : BaseOpBuilder("BatchNormOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(BatchNormOpBuilder);

  Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger,
                       std::vector<std::string>& input_names,
                       bool do_op_validation) const override ORT_MUST_USE_RESULT;

  Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger) const override final ORT_MUST_USE_RESULT;

  inline Status GetValueOnQnnDataType(const Qnn_DataType_t qnn_data_type,
                                      const uint8_t* raw_ptr,
                                      double& value,
                                      int& offset) const {
    switch (qnn_data_type) {
      case QNN_DATATYPE_INT_8:
      case QNN_DATATYPE_SFIXED_POINT_8: {
        value = static_cast<double>(*reinterpret_cast<const int8_t*>(raw_ptr));
        offset += sizeof(int8_t);
        break;
      }
      case QNN_DATATYPE_INT_16:
      case QNN_DATATYPE_SFIXED_POINT_16: {
        value = static_cast<double>(*reinterpret_cast<const int16_t*>(raw_ptr));
        offset += sizeof(int16_t);
        break;
      }
      case QNN_DATATYPE_INT_32:
      case QNN_DATATYPE_SFIXED_POINT_32: {
        value = static_cast<double>(*reinterpret_cast<const int32_t*>(raw_ptr));
        offset += sizeof(int32_t);
        break;
      }
      case QNN_DATATYPE_INT_64: {
        value = static_cast<double>(*reinterpret_cast<const int64_t*>(raw_ptr));
        offset += sizeof(int64_t);
        break;
      }
      case QNN_DATATYPE_UINT_8:
      case QNN_DATATYPE_UFIXED_POINT_8: {
        value = static_cast<double>(*reinterpret_cast<const uint8_t*>(raw_ptr));
        offset += sizeof(uint8_t);
        break;
      }
      case QNN_DATATYPE_UINT_16:
      case QNN_DATATYPE_UFIXED_POINT_16: {
        value = static_cast<double>(*reinterpret_cast<const uint16_t*>(raw_ptr));
        offset += sizeof(uint16_t);
        break;
      }
      case QNN_DATATYPE_UINT_32:
      case QNN_DATATYPE_UFIXED_POINT_32: {
        value = static_cast<double>(*reinterpret_cast<const uint32_t*>(raw_ptr));
        offset += sizeof(uint32_t);
        break;
      }
      case QNN_DATATYPE_UINT_64: {
        value = static_cast<double>(*reinterpret_cast<const uint64_t*>(raw_ptr));
        offset += sizeof(uint64_t);
        break;
      }
      case QNN_DATATYPE_FLOAT_32: {
        value = static_cast<double>(*reinterpret_cast<const float*>(raw_ptr));
        offset += sizeof(float);
        break;
      }
      case QNN_DATATYPE_FLOAT_16: {
        value = static_cast<double>(reinterpret_cast<const MLFloat16*>(raw_ptr)->ToFloat());
        offset += sizeof(MLFloat16);
        break;
      }
      case QNN_DATATYPE_BOOL_8:
      case QNN_DATATYPE_STRING:
      default:
        ORT_RETURN_IF(true, "Qnn Data Type: %d not supported yet.", qnn_data_type);
    }
    return Status::OK();
  }

  inline Status AssertUnpackedTensorSize(const Qnn_DataType_t qnn_data_type,
                                         const uint32_t channel,
                                         const size_t raw_ptr_length) const {
    switch (qnn_data_type) {
      case QNN_DATATYPE_INT_8:
      case QNN_DATATYPE_SFIXED_POINT_8: {
        ORT_RETURN_IF_NOT(channel == static_cast<uint32_t>(raw_ptr_length / sizeof(int8_t)),
                          "initializer size not match Qnn data type.");
        break;
      }
      case QNN_DATATYPE_INT_16:
      case QNN_DATATYPE_SFIXED_POINT_16: {
        ORT_RETURN_IF_NOT(channel == static_cast<uint32_t>(raw_ptr_length / sizeof(int16_t)),
                          "initializer size not match Qnn data type.");
        break;
      }
      case QNN_DATATYPE_INT_32:
      case QNN_DATATYPE_SFIXED_POINT_32: {
        ORT_RETURN_IF_NOT(channel == static_cast<uint32_t>(raw_ptr_length / sizeof(int32_t)),
                          "initializer size not match Qnn data type.");
        break;
      }
      case QNN_DATATYPE_INT_64: {
        ORT_RETURN_IF_NOT(channel == static_cast<uint32_t>(raw_ptr_length / sizeof(int64_t)),
                          "initializer size not match Qnn data type.");
        break;
      }
      case QNN_DATATYPE_UINT_8:
      case QNN_DATATYPE_UFIXED_POINT_8: {
        ORT_RETURN_IF_NOT(channel == static_cast<uint32_t>(raw_ptr_length / sizeof(uint8_t)),
                          "initializer size not match Qnn data type.");
        break;
      }
      case QNN_DATATYPE_UINT_16:
      case QNN_DATATYPE_UFIXED_POINT_16: {
        ORT_RETURN_IF_NOT(channel == static_cast<uint32_t>(raw_ptr_length / sizeof(uint16_t)),
                          "initializer size not match Qnn data type.");
        break;
      }
      case QNN_DATATYPE_UINT_32:
      case QNN_DATATYPE_UFIXED_POINT_32: {
        ORT_RETURN_IF_NOT(channel == static_cast<uint32_t>(raw_ptr_length / sizeof(uint32_t)),
                          "initializer size not match Qnn data type.");
        break;
      }
      case QNN_DATATYPE_UINT_64: {
        ORT_RETURN_IF_NOT(channel == static_cast<uint32_t>(raw_ptr_length / sizeof(uint64_t)),
                          "initializer size not match Qnn data type.");
        break;
      }
      case QNN_DATATYPE_FLOAT_32: {
        ORT_RETURN_IF_NOT(channel == static_cast<uint32_t>(raw_ptr_length / sizeof(float)),
                          "initializer size not match Qnn data type.");
        break;
      }
      case QNN_DATATYPE_FLOAT_16: {
        ORT_RETURN_IF_NOT(channel == static_cast<uint32_t>(raw_ptr_length / sizeof(MLFloat16)),
                          "initializer size not match Qnn data type.");
        break;
      }
      case QNN_DATATYPE_BOOL_8:
      case QNN_DATATYPE_STRING:
      default:
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Qnn Data Type: ", qnn_data_type, " is not supported yet.");
    }
    return Status::OK();
  }

  inline Status ConvertToRawOnQnnDataType(const Qnn_DataType_t qnn_data_type,
                                          const std::vector<double>& double_tensor,
                                          std::vector<uint8_t>& raw_tensor) const {
    switch (qnn_data_type) {
      case QNN_DATATYPE_INT_8: {
        raw_tensor.resize(double_tensor.size() * sizeof(int8_t));
        int8_t* raw_ptr = reinterpret_cast<int8_t*>(raw_tensor.data());
        for (size_t i = 0; i < double_tensor.size(); ++i) {
          raw_ptr[i] = static_cast<int8_t>(double_tensor[i]);
        }
        break;
      }
      case QNN_DATATYPE_INT_16: {
        raw_tensor.resize(double_tensor.size() * sizeof(int16_t));
        int16_t* raw_ptr = reinterpret_cast<int16_t*>(raw_tensor.data());
        for (size_t i = 0; i < double_tensor.size(); ++i) {
          raw_ptr[i] = static_cast<int16_t>(double_tensor[i]);
        }
        break;
      }
      case QNN_DATATYPE_INT_32: {
        raw_tensor.resize(double_tensor.size() * sizeof(int32_t));
        int32_t* raw_ptr = reinterpret_cast<int32_t*>(raw_tensor.data());
        for (size_t i = 0; i < double_tensor.size(); ++i) {
          raw_ptr[i] = static_cast<int32_t>(double_tensor[i]);
        }
        break;
      }
      case QNN_DATATYPE_INT_64: {
        raw_tensor.resize(double_tensor.size() * sizeof(int64_t));
        int64_t* raw_ptr = reinterpret_cast<int64_t*>(raw_tensor.data());
        for (size_t i = 0; i < double_tensor.size(); ++i) {
          raw_ptr[i] = static_cast<int64_t>(double_tensor[i]);
        }
        break;
      }
      case QNN_DATATYPE_UINT_8: {
        raw_tensor.resize(double_tensor.size() * sizeof(uint8_t));
        uint8_t* raw_ptr = reinterpret_cast<uint8_t*>(raw_tensor.data());
        for (size_t i = 0; i < double_tensor.size(); ++i) {
          raw_ptr[i] = static_cast<uint8_t>(double_tensor[i]);
        }
        break;
      }
      case QNN_DATATYPE_UINT_16: {
        raw_tensor.resize(double_tensor.size() * sizeof(uint16_t));
        uint16_t* raw_ptr = reinterpret_cast<uint16_t*>(raw_tensor.data());
        for (size_t i = 0; i < double_tensor.size(); ++i) {
          raw_ptr[i] = static_cast<uint16_t>(double_tensor[i]);
        }
        break;
      }
      case QNN_DATATYPE_UINT_32: {
        raw_tensor.resize(double_tensor.size() * sizeof(uint32_t));
        uint32_t* raw_ptr = reinterpret_cast<uint32_t*>(raw_tensor.data());
        for (size_t i = 0; i < double_tensor.size(); ++i) {
          raw_ptr[i] = static_cast<uint32_t>(double_tensor[i]);
        }
        break;
      }
      case QNN_DATATYPE_UINT_64: {
        raw_tensor.resize(double_tensor.size() * sizeof(uint64_t));
        uint64_t* raw_ptr = reinterpret_cast<uint64_t*>(raw_tensor.data());
        for (size_t i = 0; i < double_tensor.size(); ++i) {
          raw_ptr[i] = static_cast<uint64_t>(double_tensor[i]);
        }
        break;
      }
      case QNN_DATATYPE_FLOAT_32: {
        raw_tensor.resize(double_tensor.size() * sizeof(float));
        float* raw_ptr = reinterpret_cast<float*>(raw_tensor.data());
        for (size_t i = 0; i < double_tensor.size(); ++i) {
          raw_ptr[i] = static_cast<float>(double_tensor[i]);
        }
        break;
      }
      case QNN_DATATYPE_FLOAT_16: {
        raw_tensor.resize(double_tensor.size() * sizeof(MLFloat16));
        MLFloat16* raw_ptr = reinterpret_cast<MLFloat16*>(raw_tensor.data());
        for (size_t i = 0; i < double_tensor.size(); ++i) {
          raw_ptr[i] = MLFloat16(static_cast<float>(double_tensor[i]));
        }
        break;
      }
      case QNN_DATATYPE_UFIXED_POINT_32:
      case QNN_DATATYPE_UFIXED_POINT_16:
      case QNN_DATATYPE_UFIXED_POINT_8:
      case QNN_DATATYPE_SFIXED_POINT_32:
      case QNN_DATATYPE_SFIXED_POINT_16:
      case QNN_DATATYPE_SFIXED_POINT_8:
      case QNN_DATATYPE_BOOL_8:
      case QNN_DATATYPE_STRING:
      default:
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Qnn Data Type: ", qnn_data_type, " is not supported yet.");
    }
    return Status::OK();
  }

  Status PreprocessMean(const TensorInfo& mean_info,
                        const uint8_t* mean_raw_ptr,
                        const size_t mean_raw_ptr_length,
                        std::vector<double>& mean_out) const {
    // tensor length (channel)
    uint32_t channel = mean_info.shape[0];
    mean_out.resize(channel);
    ORT_RETURN_IF_ERROR(AssertUnpackedTensorSize(mean_info.qnn_data_type, channel, mean_raw_ptr_length));

    const bool is_quantized = mean_info.quant_param.IsQuantized();
    ORT_RETURN_IF_NOT(!is_quantized || mean_info.quant_param.IsPerTensor(),
                      "BatchNormalization's input_mean does not support per-channel quantization");
    int i = 0;
    int offset = 0;
    const Qnn_QuantizeParams_t& quant_param = mean_info.quant_param.Get();
    for (; i < static_cast<int>(channel); ++i) {
      double mean_value = 0.0;
      ORT_RETURN_IF_ERROR(GetValueOnQnnDataType(mean_info.qnn_data_type, mean_raw_ptr + offset, mean_value, offset));
      mean_out[i] = (is_quantized) ? utils::Dequantize(quant_param.scaleOffsetEncoding.offset,
                                                       quant_param.scaleOffsetEncoding.scale,
                                                       mean_value)
                                   : mean_value;
    }
    return Status::OK();
  }

  Status PreprocessStd(const TensorInfo& var_info,
                       const uint8_t* var_raw_ptr,
                       const size_t var_raw_ptr_length,
                       const float epsilon,
                       std::vector<double>& std_out) const {
    // tensor length (channel)
    uint32_t channel = var_info.shape[0];
    std_out.resize(channel);
    ORT_RETURN_IF_ERROR(AssertUnpackedTensorSize(var_info.qnn_data_type, channel, var_raw_ptr_length));

    const bool is_quantized = var_info.quant_param.IsQuantized();
    ORT_RETURN_IF_NOT(!is_quantized || var_info.quant_param.IsPerTensor(),
                      "BatchNormalization's input_var does not support per-channel quantization");
    int i = 0;
    int offset = 0;
    const Qnn_QuantizeParams_t& quant_param = var_info.quant_param.Get();
    for (; i < static_cast<int>(channel); ++i) {
      double var_value = 0.0;
      ORT_RETURN_IF_ERROR(GetValueOnQnnDataType(var_info.qnn_data_type, var_raw_ptr + offset, var_value, offset));
      std_out[i] = (is_quantized) ? utils::Dequantize(quant_param.scaleOffsetEncoding.offset,
                                                      quant_param.scaleOffsetEncoding.scale,
                                                      var_value)
                                  : var_value;
      std_out[i] = std::sqrt(std_out[i] + static_cast<double>(epsilon));
    }
    return Status::OK();
  }

  Status PreprocessScale(const TensorInfo& scale_info,
                         const uint8_t* scale_raw_ptr,
                         const size_t scale_raw_ptr_length,
                         const std::vector<double>& std_double_tensor,
                         double& rmax,
                         double& rmin,
                         std::vector<double>& scale_out) const {
    // tensor length (channel)
    uint32_t channel = scale_info.shape[0];
    scale_out.resize(channel);
    ORT_RETURN_IF_ERROR(AssertUnpackedTensorSize(scale_info.qnn_data_type, channel, scale_raw_ptr_length));

    const bool is_quantized = scale_info.quant_param.IsQuantized();
    ORT_RETURN_IF_NOT(!is_quantized || scale_info.quant_param.IsPerTensor(),
                      "BatchNormalization's scale input does not support per-channel quantization");
    int i = 0;
    int offset = 0;
    const Qnn_QuantizeParams_t& quant_param = scale_info.quant_param.Get();
    for (; i < static_cast<int>(channel); ++i) {
      double scale_value = 0.0;
      ORT_RETURN_IF_ERROR(GetValueOnQnnDataType(scale_info.qnn_data_type, scale_raw_ptr + offset, scale_value, offset));
      scale_out[i] = (is_quantized) ? utils::Dequantize(quant_param.scaleOffsetEncoding.offset,
                                                        quant_param.scaleOffsetEncoding.scale,
                                                        scale_value)
                                    : scale_value;
      scale_out[i] = scale_out[i] / std_double_tensor[i];
      rmax = std::max(rmax, scale_out[i]);
      rmin = std::min(rmin, scale_out[i]);
    }
    return Status::OK();
  }

  Status PreprocessBias(const TensorInfo& bias_info,
                        const uint8_t* bias_raw_ptr,
                        const size_t bias_raw_ptr_length,
                        const std::vector<double>& scale_double_tensor,
                        const std::vector<double>& mean_double_tensor,
                        double& rmax,
                        double& rmin,
                        std::vector<double>& bias_out) const {
    // tensor length (channel)
    uint32_t channel = bias_info.shape[0];
    bias_out.resize(channel);
    ORT_RETURN_IF_ERROR(AssertUnpackedTensorSize(bias_info.qnn_data_type, channel, bias_raw_ptr_length));

    const bool is_quantized = bias_info.quant_param.IsQuantized();
    ORT_RETURN_IF_NOT(!is_quantized || bias_info.quant_param.IsPerTensor(),
                      "BatchNormalization's bias input does not support per-channel quantization");
    int i = 0;
    int offset = 0;
    const Qnn_QuantizeParams_t& quant_param = bias_info.quant_param.Get();
    for (; i < static_cast<int>(channel); ++i) {
      double bias_value = 0.0;
      ORT_RETURN_IF_ERROR(GetValueOnQnnDataType(bias_info.qnn_data_type, bias_raw_ptr + offset, bias_value, offset));
      bias_out[i] = (is_quantized) ? utils::Dequantize(quant_param.scaleOffsetEncoding.offset,
                                                       quant_param.scaleOffsetEncoding.scale,
                                                       bias_value)
                                   : bias_value;
      bias_out[i] = bias_out[i] - (mean_double_tensor[i] * scale_double_tensor[i]);
      rmax = std::max(rmax, bias_out[i]);
      rmin = std::min(rmin, bias_out[i]);
    }
    return Status::OK();
  }

  Status Postprocess(const TensorInfo& info,
                     const std::vector<double>& double_tensor,
                     const double rmax,
                     const double rmin,
                     QnnQuantParamsWrapper& quant_param,
                     std::vector<uint8_t>& raw_tensor) const {
    bool symmetric = false;
    if (info.quant_param.IsQuantized()) {
      size_t data_size = double_tensor.size();
      // QNN BatchNorm int32 bias requires symmetric quantizated
      if (info.qnn_data_type == QNN_DATATYPE_SFIXED_POINT_32) {
        data_size *= sizeof(int32_t);
        symmetric = true;
      }
      raw_tensor.resize(data_size);
      float scale = 0.0f;
      int32_t zero_point = 0;
      ORT_RETURN_IF_ERROR(utils::GetQuantParams(static_cast<float>(rmin),
                                                static_cast<float>(rmax),
                                                info.qnn_data_type,
                                                scale,
                                                zero_point,
                                                symmetric));
      quant_param = QnnQuantParamsWrapper(scale, zero_point);
      for (size_t i = 0; i < double_tensor.size(); ++i) {
        // onnx only supports 8 bits quantization
        int quant_value_int = 0;
        ORT_RETURN_IF_ERROR(utils::Quantize(double_tensor[i], scale, zero_point, info.qnn_data_type, quant_value_int));
        if (info.qnn_data_type == QNN_DATATYPE_UFIXED_POINT_8) {
          raw_tensor[i] = static_cast<uint8_t>(quant_value_int);
        } else if (info.qnn_data_type == QNN_DATATYPE_SFIXED_POINT_8) {
          int8_t quant_value = static_cast<int8_t>(quant_value_int);
          raw_tensor[i] = *reinterpret_cast<uint8_t*>(&quant_value);
        } else if (info.qnn_data_type == QNN_DATATYPE_SFIXED_POINT_32) {
          int32_t quant_value = static_cast<int32_t>(quant_value_int);
          size_t pos = i * sizeof(int32_t);
          std::memcpy(&raw_tensor[pos], reinterpret_cast<uint8_t*>(&quant_value), sizeof(int32_t));
        } else {
          // TODO(adrianlizarraga): Should support 16-bit quantization as well.
          ORT_RETURN_IF(true, "Qnn Data Type: %d not supported yet.", info.qnn_data_type);
        }
      }
    } else {
      ORT_RETURN_IF_ERROR(ConvertToRawOnQnnDataType(info.qnn_data_type, double_tensor, raw_tensor));
    }
    return Status::OK();
  }

 protected:
  Status CheckCpuDataTypes(const std::vector<Qnn_DataType_t> in_dtypes,
                           const std::vector<Qnn_DataType_t> out_dtypes) const override ORT_MUST_USE_RESULT;

  Status CheckHtpDataTypes(const std::vector<Qnn_DataType_t> in_dtypes,
                           const std::vector<Qnn_DataType_t> out_dtypes) const override ORT_MUST_USE_RESULT;
};

// BatchNorm is sensitive with data layout, no special validation so far
// The nodes from 1st call of GetCapability do not get layout transformer applied, it's still NCHW
// The nodes from 2nd call of GetCapability get layout transformer applied, it's NHWC
Status BatchNormOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                         const NodeUnit& node_unit,
                                         const logging::Logger& logger) const {
  if (node_unit.Domain() == kMSInternalNHWCDomain) {
    // It's useless to fallback the node after layout transformation because CPU EP can't support it anyway
    // Still do it here so hopefully QNN Op validation API can tell us some details why it's not supported
    return AddToModelBuilder(qnn_model_wrapper, node_unit, logger, true);
  } else {
    // Check input datatype. Can't use Qnn Op validation API since it's before layout transformation
    ORT_RETURN_IF_ERROR(ProcessDataTypes(qnn_model_wrapper, node_unit));

    const auto& inputs = node_unit.Inputs();
    ORT_RETURN_IF_NOT(inputs.size() == 5, "5 input expected per BatchNorm Onnx Spec.");

    std::vector<uint32_t> input_shape;
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].node_arg, input_shape), "Cannot get shape of input 0.");
    const size_t input_rank = input_shape.size();

    ORT_RETURN_IF(input_rank > 4, "QNN BatchNorm only supports input ranks of size <= 4.");

    const uint32_t num_channels = input_shape[1];

    std::vector<uint32_t> scale_shape;
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[1].node_arg, scale_shape), "Cannot get shape of input 1 (scale).");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.IsConstantInput(inputs[1].node_arg.Name()),
                      "QNN BatchNorm doesn't support dynamic scale.");
    ORT_RETURN_IF(scale_shape.size() != 1 || scale_shape[0] != num_channels,
                  "QNN BatchNorm input 1 (scale) must have 1D shape [channel].");

    std::vector<uint32_t> bias_shape;
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[2].node_arg, bias_shape), "Cannot get shape of input 2 (bias).");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.IsConstantInput(inputs[2].node_arg.Name()),
                      "QNN BatchNorm doesn't support dynamic bias.");

    ORT_RETURN_IF(bias_shape.size() != 1 || bias_shape[0] != num_channels,
                  "QNN BatchNorm input 2 (bias) must have 1D shape [channel].");

    std::vector<uint32_t> mean_shape;
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[3].node_arg, mean_shape), "Cannot get shape of input 3 (mean).");
    ORT_RETURN_IF(mean_shape.size() != 1 || mean_shape[0] != num_channels,
                  "QNN BatchNorm input 3 (mean) must have 1D shape [channel].");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.IsConstantInput(inputs[3].node_arg.Name()),
                      "QNN BatchNorm doesn't support dynamic mean.");

    std::vector<uint32_t> var_shape;
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[4].node_arg, var_shape), "Cannot get shape of input 4 (var).");
    ORT_RETURN_IF(var_shape.size() != 1 || var_shape[0] != num_channels,
                  "QNN BatchNorm input 4 (var) must have 1D shape [channel].");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.IsConstantInput(inputs[4].node_arg.Name()),
                      "QNN BatchNorm doesn't support dynamic var.");

    ORT_RETURN_IF(node_unit.Outputs().size() > 1, "QNN BatchNorm only support 1 output.");
  }

  return Status::OK();
}

Status BatchNormOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                         const NodeUnit& node_unit,
                                         const logging::Logger& logger,
                                         std::vector<std::string>& input_names,
                                         bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);
  ORT_UNUSED_PARAMETER(logger);

  const auto& inputs = node_unit.Inputs();
  //
  // Input 0
  //
  ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[0], logger, input_names));

  //
  // Input 1: scale
  // Input 2: bias
  // QNN only accept 3 input. We need to first combine mean and variance into scale and bias.
  //
  {
    const std::string& scale_name = inputs[1].node_arg.Name();
    const std::string& bias_name = inputs[2].node_arg.Name();
    TensorInfo var_info = {};
    TensorInfo mean_info = {};
    TensorInfo scale_info = {};
    TensorInfo bias_info = {};
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[1], scale_info));
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[2], bias_info));
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[3], mean_info));
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[4], var_info));

    // scale, bias, mean, and var must be initializers
    ORT_RETURN_IF_NOT(scale_info.is_initializer, "scale must be initializers");
    ORT_RETURN_IF_NOT(bias_info.is_initializer, "bias must be initializers");
    ORT_RETURN_IF_NOT(mean_info.is_initializer, "mean must be initializers");
    ORT_RETURN_IF_NOT(var_info.is_initializer, "var must be initializers");

    std::vector<uint8_t> scale_unpacked_tensor;
    std::vector<uint8_t> bias_unpacked_tensor;
    std::vector<uint8_t> var_unpacked_tensor;
    std::vector<uint8_t> mean_unpacked_tensor;
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*scale_info.initializer_tensor, scale_unpacked_tensor));
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*bias_info.initializer_tensor, bias_unpacked_tensor));
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*mean_info.initializer_tensor, mean_unpacked_tensor));
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*var_info.initializer_tensor, var_unpacked_tensor));

    std::vector<double> mean_double_tensor;
    std::vector<double> std_double_tensor;
    std::vector<double> scale_double_tensor;
    std::vector<double> bias_double_tensor;

    NodeAttrHelper node_helper(node_unit);
    const float epsilon = node_helper.Get("epsilon", 1e-05f);  // Default is 1e-05 according to ONNX spec.

    double scale_rmax = std::numeric_limits<double>::min();
    double scale_rmin = std::numeric_limits<double>::max();
    double bias_rmax = std::numeric_limits<double>::min();
    double bias_rmin = std::numeric_limits<double>::max();

    // Calculate and convert new scale, new bias, mean and std to double array (may be dequantized)
    ORT_RETURN_IF_ERROR(PreprocessMean(mean_info,
                                       mean_unpacked_tensor.data(),
                                       mean_unpacked_tensor.size(),
                                       mean_double_tensor));
    ORT_RETURN_IF_ERROR(PreprocessStd(var_info,
                                      var_unpacked_tensor.data(),
                                      var_unpacked_tensor.size(),
                                      epsilon,
                                      std_double_tensor));
    ORT_RETURN_IF_ERROR(PreprocessScale(scale_info,
                                        scale_unpacked_tensor.data(),
                                        scale_unpacked_tensor.size(),
                                        std_double_tensor,
                                        scale_rmax,
                                        scale_rmin,
                                        scale_double_tensor));
    ORT_RETURN_IF_ERROR(PreprocessBias(bias_info,
                                       bias_unpacked_tensor.data(),
                                       bias_unpacked_tensor.size(),
                                       scale_double_tensor,
                                       mean_double_tensor,
                                       bias_rmax,
                                       bias_rmin,
                                       bias_double_tensor));

    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(scale_name)) {
      std::vector<uint8_t> scale_raw_tensor;
      QnnQuantParamsWrapper scale_quant_param = scale_info.quant_param;
      ORT_RETURN_IF_ERROR(Postprocess(scale_info,
                                      scale_double_tensor,
                                      scale_rmax,
                                      scale_rmin,
                                      scale_quant_param,
                                      scale_raw_tensor));
      Qnn_TensorType_t scale_tensor_type = qnn_model_wrapper.GetTensorType(scale_name);
      QnnTensorWrapper input_tensorwrapper(scale_name, scale_tensor_type, scale_info.qnn_data_type,
                                           std::move(scale_quant_param), std::move(scale_info.shape),
                                           std::move(scale_raw_tensor));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
    }
    input_names.push_back(scale_name);

    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(bias_name)) {
      std::vector<uint8_t> bias_raw_tensor;
      QnnQuantParamsWrapper bias_quant_param = bias_info.quant_param;
      ORT_RETURN_IF_ERROR(Postprocess(bias_info,
                                      bias_double_tensor,
                                      bias_rmax,
                                      bias_rmin,
                                      bias_quant_param,
                                      bias_raw_tensor));
      Qnn_TensorType_t bias_tensor_type = qnn_model_wrapper.GetTensorType(bias_name);
      QnnTensorWrapper input_tensorwrapper(bias_name, bias_tensor_type, bias_info.qnn_data_type,
                                           std::move(bias_quant_param), std::move(bias_info.shape),
                                           std::move(bias_raw_tensor));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
    }
    input_names.push_back(bias_name);
  }

  return Status::OK();
}

void CreateBatchNormOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<BatchNormOpBuilder>());
}

Status BatchNormOpBuilder::CheckCpuDataTypes(const std::vector<Qnn_DataType_t> in_dtypes,
                                             const std::vector<Qnn_DataType_t> out_dtypes) const {
  bool is_supported_dtype = false;
  // in_dtypes: [X, scale, B, input_mean, input_var]
  std::vector<Qnn_DataType_t> all_dtypes(in_dtypes.begin(), in_dtypes.begin() + 3);
  // out_dtypes: [Y, running_mean, running_var]
  all_dtypes.insert(all_dtypes.end(), out_dtypes.begin(), out_dtypes.begin() + 1);
  // FP32
  if (
      (all_dtypes == std::vector<Qnn_DataType_t>{QNN_DATATYPE_FLOAT_32, QNN_DATATYPE_FLOAT_32, QNN_DATATYPE_FLOAT_32, QNN_DATATYPE_FLOAT_32})) {
    is_supported_dtype = true;
  }
  // INT8
  else if (
      (all_dtypes == std::vector<Qnn_DataType_t>{QNN_DATATYPE_UFIXED_POINT_8, QNN_DATATYPE_UFIXED_POINT_8, QNN_DATATYPE_UFIXED_POINT_8, QNN_DATATYPE_UFIXED_POINT_8}) ||
      (all_dtypes == std::vector<Qnn_DataType_t>{QNN_DATATYPE_UFIXED_POINT_8, QNN_DATATYPE_UFIXED_POINT_8, QNN_DATATYPE_SFIXED_POINT_32, QNN_DATATYPE_UFIXED_POINT_8})) {
    is_supported_dtype = true;
  }
  ORT_RETURN_IF_NOT(is_supported_dtype, "QNN Batchnorm unsupported datatype on CPU.");
  return Status::OK();
}

Status BatchNormOpBuilder::CheckHtpDataTypes(const std::vector<Qnn_DataType_t> in_dtypes,
                                             const std::vector<Qnn_DataType_t> out_dtypes) const {
  bool is_supported_dtype = false;
  // in_dtypes: [X, scale, B, input_mean, input_var]
  std::vector<Qnn_DataType_t> all_dtypes(in_dtypes.begin(), in_dtypes.begin() + 3);
  // out_dtypes: [Y, running_mean, running_var]
  all_dtypes.insert(all_dtypes.end(), out_dtypes.begin(), out_dtypes.begin() + 1);
  // FP16
  if (
      (all_dtypes == std::vector<Qnn_DataType_t>{QNN_DATATYPE_FLOAT_16, QNN_DATATYPE_FLOAT_16, QNN_DATATYPE_FLOAT_16, QNN_DATATYPE_FLOAT_16}) ||
      (all_dtypes == std::vector<Qnn_DataType_t>{QNN_DATATYPE_FLOAT_32, QNN_DATATYPE_FLOAT_32, QNN_DATATYPE_FLOAT_32, QNN_DATATYPE_FLOAT_32})) {
    is_supported_dtype = true;
  }
  // INT16
  else if (
      (all_dtypes == std::vector<Qnn_DataType_t>{QNN_DATATYPE_UFIXED_POINT_16, QNN_DATATYPE_UFIXED_POINT_8, QNN_DATATYPE_UFIXED_POINT_8, QNN_DATATYPE_UFIXED_POINT_16}) ||
      (all_dtypes == std::vector<Qnn_DataType_t>{QNN_DATATYPE_UFIXED_POINT_16, QNN_DATATYPE_UFIXED_POINT_8, QNN_DATATYPE_SFIXED_POINT_32, QNN_DATATYPE_UFIXED_POINT_16}) ||
      (all_dtypes == std::vector<Qnn_DataType_t>{QNN_DATATYPE_UFIXED_POINT_16, QNN_DATATYPE_UFIXED_POINT_16, QNN_DATATYPE_UFIXED_POINT_8, QNN_DATATYPE_UFIXED_POINT_16}) ||
      (all_dtypes == std::vector<Qnn_DataType_t>{QNN_DATATYPE_UFIXED_POINT_16, QNN_DATATYPE_UFIXED_POINT_16, QNN_DATATYPE_SFIXED_POINT_32, QNN_DATATYPE_UFIXED_POINT_16}) ||
      (all_dtypes == std::vector<Qnn_DataType_t>{QNN_DATATYPE_UFIXED_POINT_16, QNN_DATATYPE_SFIXED_POINT_16, QNN_DATATYPE_UFIXED_POINT_8, QNN_DATATYPE_UFIXED_POINT_16}) ||
      (all_dtypes == std::vector<Qnn_DataType_t>{QNN_DATATYPE_UFIXED_POINT_16, QNN_DATATYPE_SFIXED_POINT_16, QNN_DATATYPE_SFIXED_POINT_32, QNN_DATATYPE_UFIXED_POINT_16})) {
    is_supported_dtype = true;
  }
  // INT8
  else if (
      (all_dtypes == std::vector<Qnn_DataType_t>{QNN_DATATYPE_UFIXED_POINT_8, QNN_DATATYPE_UFIXED_POINT_8, QNN_DATATYPE_UFIXED_POINT_8, QNN_DATATYPE_UFIXED_POINT_8}) ||
      (all_dtypes == std::vector<Qnn_DataType_t>{QNN_DATATYPE_UFIXED_POINT_8, QNN_DATATYPE_UFIXED_POINT_8, QNN_DATATYPE_SFIXED_POINT_32, QNN_DATATYPE_UFIXED_POINT_8}) ||
      (all_dtypes == std::vector<Qnn_DataType_t>{QNN_DATATYPE_SFIXED_POINT_8, QNN_DATATYPE_SFIXED_POINT_8, QNN_DATATYPE_SFIXED_POINT_8, QNN_DATATYPE_SFIXED_POINT_8})) {
    is_supported_dtype = true;
  }
  ORT_RETURN_IF_NOT(is_supported_dtype, "QNN Batchnorm unsupported datatype on HTP.");
  return Status::OK();
};

}  // namespace qnn
}  // namespace onnxruntime
