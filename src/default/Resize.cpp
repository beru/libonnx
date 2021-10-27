#include "onnx.h"
#include "util.h"

namespace onnx {

namespace {

struct Resize_operator : public operator_t {

	enum class coordinate_transformation_mode_t {
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

	enum class interpolation_mode_t {
		nearest,
		linear,
		cubic,
	} mode;

	enum class nearest_mode_t {
		round_prefer_floor,
		round_prefer_ceil,
		floor,
		ceil,
	} nearest_mode;

	bool init() override {

		static const std::unordered_map<std::string_view, coordinate_transformation_mode_t> m0 {
			#define X(a) { #a, coordinate_transformation_mode_t:: a }
			X(half_pixel),
			X(asymmetric),
			X(pytorch_half_pixel),
			X(tf_half_pixel_for_nn),
			X(align_corners),
			X(tf_crop_and_resize),
			#undef X
		};
		coordinate_transformation_mode = m0.at(attribute("coordinate_transformation_mode", "half_pixel"));
		cubic_coeff_a = attribute("cubic_coeff_a", -0.75f);
		exclude_outside = attribute("exclude_outside", 0);
		extrapolation_value = attribute("extrapolation_value", 0.0f);

		static const std::unordered_map<std::string_view, interpolation_mode_t> m1 {
			#define X(a) { #a, interpolation_mode_t:: a }
			X(nearest),
			X(linear),
			X(cubic),
			#undef X
		};
		mode = m1.at(attribute("mode", "nearest"));

		static const std::unordered_map<std::string_view, nearest_mode_t> m2 {
			#define X(a) { #a, nearest_mode_t:: a }
			X(round_prefer_floor),
			X(round_prefer_ceil),
			X(floor),
			X(ceil),
			#undef X
		};
		nearest_mode = m2.at(attribute("nearest_mode", "round_prefer_floor"));

		return true;
	}

	bool reshape() override {
		return true;
	}

	void exec() override {
		//if (opset >= 13) {
		//}else if (opset >= 11) {
		//}else if (opset >= 10) {
		//}
	}
};


} // namespace

operator_t* resolver_default_op_Resize(int opset)
{
	return new (std::nothrow) Resize_operator;
}

} // namespace onnx
