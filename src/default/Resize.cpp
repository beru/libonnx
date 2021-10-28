#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct Resize_operator : public operator_t {

	enum coordinate_transformation_mode_t {
		half_pixel,
		asymmetric,
		pytorch_half_pixel,
		tf_half_pixel_for_nn,
		align_corners,
		tf_crop_and_resize,
	} coordinate_transformation_mode;

	float cubic_coeff_a;
	int exclude_outside;
	float extrapolation_value;

	enum interpolation_mode_t {
		nearest,
		linear,
		cubic,
	} mode;

	enum nearest_mode_t {
		round_prefer_floor,
		round_prefer_ceil,
		floor,
		ceil,
	} nearest_mode;

	bool init() override {

		if (inputs.size() < 1 || inputs.size() > 4 || outputs.size() != 1) {
			return false;
		}

		coordinate_transformation_mode = attribute<coordinate_transformation_mode_t>("coordinate_transformation_mode", "half_pixel");
		cubic_coeff_a = attribute("cubic_coeff_a", -0.75f);
		exclude_outside = attribute("exclude_outside", 0);
		extrapolation_value = attribute("extrapolation_value", 0.0f);
		mode = attribute<interpolation_mode_t>("mode", "nearest");
		nearest_mode = attribute<nearest_mode_t>("nearest_mode", "round_prefer_floor");

		return true;
	}

	bool reshape() override {
		return true;
	}

	template <typename T>
	void exec() {
		tensor_t* y = outputs[0];
		const tensor_t* x = inputs[0];
		const tensor_t* roi = inputs.size() >= 2 ? inputs[1] : nullptr;
		const tensor_t* scales = inputs.size() >= 3 ? inputs[2] : nullptr;
		const tensor_t* sizes = inputs.size() == 4 ? inputs[3] : nullptr;

	}

	void exec() override {
		tensor_type_t type = inputs[0]->type;
		if (opset >= 13) {
			typed_exec<Resize_operator,
				uint8_t, uint16_t, uint32_t, uint64_t,
				int8_t, int16_t, int32_t, int64_t,
				float16_t, float, double, bfloat16_t,
				std::complex<float>, std::complex<double>,
				std::string
			>(this, type);
		}else if (opset >= 11) {
			typed_exec<Resize_operator,
				uint8_t, uint16_t, uint32_t, uint64_t,
				int8_t, int16_t, int32_t, int64_t,
				float16_t, float, double,
				std::complex<float>, std::complex<double>,
				std::string
			>(this, type);
		}else if (opset >= 10) {
			typed_exec<Resize_operator,
				uint8_t, uint16_t, uint32_t, uint64_t,
				int8_t, int16_t, int32_t, int64_t,
				float16_t, float, double,
				std::complex<float>, std::complex<double>,
				std::string
			>(this, type);
		}
	}
};


} // namespace

operator_t* resolver_default_op_Resize(int opset)
{
	return new (std::nothrow) Resize_operator;
}

} // namespace onnx
