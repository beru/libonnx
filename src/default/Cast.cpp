#include <onnx.h>
#include "util.h"

namespace onnx {

namespace {

void Cast_from_string(
	const std::string* from_data,
	tensor_type_t to_type, void* to_data,
	size_t ndata)
{
	size_t i;

	switch (to_type) {
	case ONNX_TENSOR_TYPE_BOOL:
		{
			bool_t* py = (bool_t*)to_data;
			for (i = 0; i < ndata; i++)
				py[i] = strtoul(from_data[i].c_str(), 0, 0);
		}
		break;
	case ONNX_TENSOR_TYPE_INT8:
		{
			int8_t* py = (int8_t*)to_data;
			for (i = 0; i < ndata; i++)
				py[i] = (int8_t)strtol(from_data[i].c_str(), 0, 0);
		}
		break;
	case ONNX_TENSOR_TYPE_INT16:
		{
			int16_t* py = (int16_t*)to_data;
			for (i = 0; i < ndata; i++)
				py[i] = (int16_t)strtol(from_data[i].c_str(), 0, 0);
		}
		break;
	case ONNX_TENSOR_TYPE_INT32:
		{
			int32_t* py = (int32_t*)to_data;
			for (i = 0; i < ndata; i++)
				py[i] = (int32_t)strtol(from_data[i].c_str(), 0, 0);
		}
		break;
	case ONNX_TENSOR_TYPE_INT64:
		{
			int64_t* py = (int64_t*)to_data;
			for (i = 0; i < ndata; i++)
				py[i] = (int64_t)strtoll(from_data[i].c_str(), 0, 0);
		}
		break;
	case ONNX_TENSOR_TYPE_UINT8:
		{
			uint8_t* py = (uint8_t*)to_data;
			for (i = 0; i < ndata; i++)
				py[i] = (uint8_t)strtoul(from_data[i].c_str(), 0, 0);
		}
		break;
	case ONNX_TENSOR_TYPE_UINT16:
		{
			uint16_t* py = (uint16_t*)to_data;
			for (i = 0; i < ndata; i++)
				py[i] = (uint16_t)strtoul(from_data[i].c_str(), 0, 0);
		}
		break;
	case ONNX_TENSOR_TYPE_UINT32:
		{
			uint32_t* py = (uint32_t*)to_data;
			for (i = 0; i < ndata; i++)
				py[i] = (uint32_t)strtoul(from_data[i].c_str(), 0, 0);
		}
		break;
	case ONNX_TENSOR_TYPE_UINT64:
		{
			uint64_t* py = (uint64_t*)to_data;
			for (i = 0; i < ndata; i++)
				py[i] = (uint64_t)strtoull(from_data[i].c_str(), 0, 0);
		}
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
		{
			bfloat16_t* py = (bfloat16_t*)to_data;
			for (i = 0; i < ndata; i++)
				py[i] = (float)strtod(from_data[i].c_str(), nullptr);
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		{
			float16_t* py = (float16_t*)to_data;
			for (i = 0; i < ndata; i++)
				py[i] = (float)strtod(from_data[i].c_str(), nullptr);
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		{
			float* py = (float*)to_data;
			for (i = 0; i < ndata; i++)
				py[i] = (float)strtod(from_data[i].c_str(), nullptr);
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		{
			double* py = (double*)to_data;
			for (i = 0; i < ndata; i++)
				py[i] = (double)strtod(from_data[i].c_str(), nullptr);
		}
		break;
	default:
		break;
	}
}

template <typename TreatT, typename DataT>
void Cast_to_string(const DataT* from_data, std::string* to_data, size_t ndata)
{
	for (size_t i=0; i<ndata; ++i) {
		to_data[i] = std::to_string((TreatT)from_data[i]);
	}
}

void Cast_to_string(
	tensor_type_t from_type, const void* from_data,
	std::string* to_data,
	size_t ndata)
{
	switch (from_type) {
#define X(enum_type, data_type, treat_type) case enum_type: Cast_to_string<treat_type>((const data_type*)from_data, to_data, ndata); return;
	X(ONNX_TENSOR_TYPE_BOOL, bool_t, bool)
	X(ONNX_TENSOR_TYPE_INT8, int8_t, int32_t)
	X(ONNX_TENSOR_TYPE_INT16, int16_t, int32_t)
	X(ONNX_TENSOR_TYPE_INT32, int32_t, int32_t)
	X(ONNX_TENSOR_TYPE_INT64, int64_t, int64_t)
	X(ONNX_TENSOR_TYPE_UINT8, uint8_t, uint32_t)
	X(ONNX_TENSOR_TYPE_UINT16, uint16_t, uint32_t)
	X(ONNX_TENSOR_TYPE_UINT32, uint32_t, uint32_t)
	X(ONNX_TENSOR_TYPE_UINT64, uint64_t, uint64_t)
	X(ONNX_TENSOR_TYPE_BFLOAT16, bfloat16_t, float)
	X(ONNX_TENSOR_TYPE_FLOAT16, float16_t, float)
	X(ONNX_TENSOR_TYPE_FLOAT32, float, float)
	X(ONNX_TENSOR_TYPE_FLOAT64, double, double)
#undef X
	}
}

void Copy_array(
	tensor_type_t enum_type,
	const void* from_data, void* to_data,
	size_t ndata)
{
	switch (enum_type) {
#define X(enum_type, data_type) case enum_type: std::copy_n((const data_type*)from_data, ndata, (data_type*)to_data); return;
	X(ONNX_TENSOR_TYPE_BOOL, bool_t)
	X(ONNX_TENSOR_TYPE_INT8, int8_t)
	X(ONNX_TENSOR_TYPE_INT16, int16_t)
	X(ONNX_TENSOR_TYPE_INT32, int32_t)
	X(ONNX_TENSOR_TYPE_INT64, int64_t)
	X(ONNX_TENSOR_TYPE_UINT8, uint8_t)
	X(ONNX_TENSOR_TYPE_UINT16, uint16_t)
	X(ONNX_TENSOR_TYPE_UINT32, uint32_t)
	X(ONNX_TENSOR_TYPE_UINT64, uint64_t)
	X(ONNX_TENSOR_TYPE_BFLOAT16, bfloat16_t)
	X(ONNX_TENSOR_TYPE_FLOAT16, float16_t)
	X(ONNX_TENSOR_TYPE_FLOAT32, float)
	X(ONNX_TENSOR_TYPE_FLOAT64, double)
	X(ONNX_TENSOR_TYPE_STRING, std::string)
#undef X
	}
}

template <typename FromT, typename ToT>
void Cast_array(
	const FromT* from_data,
	ToT* to_data,
	size_t ndata)
{
	for (size_t i=0; i<ndata; ++i) {
		to_data[i] = (ToT)from_data[i];
	}
}

template <typename FromT>
void Cast_array(
	const FromT* from_data,
	tensor_type_t to_type, void* to_data,
	size_t ndata)
{
	switch (to_type) {
#define X(enum_type, data_type) case enum_type: Cast_array(from_data, (data_type*)to_data, ndata); return;
	X(ONNX_TENSOR_TYPE_BOOL, bool_t)
	X(ONNX_TENSOR_TYPE_INT8, int8_t)
	X(ONNX_TENSOR_TYPE_INT16, int16_t)
	X(ONNX_TENSOR_TYPE_INT32, int32_t)
	X(ONNX_TENSOR_TYPE_INT64, int64_t)
	X(ONNX_TENSOR_TYPE_UINT8, uint8_t)
	X(ONNX_TENSOR_TYPE_UINT16, uint16_t)
	X(ONNX_TENSOR_TYPE_UINT32, uint32_t)
	X(ONNX_TENSOR_TYPE_UINT64, uint64_t)
	X(ONNX_TENSOR_TYPE_BFLOAT16, bfloat16_t)
	X(ONNX_TENSOR_TYPE_FLOAT16, float16_t)
	X(ONNX_TENSOR_TYPE_FLOAT32, float)
	X(ONNX_TENSOR_TYPE_FLOAT64, double)
#undef X
	}
}

void Cast_array(
	tensor_type_t from_type, const void* from_data,
	tensor_type_t to_type, void* to_data,
	size_t ndata)
{
	if (from_type == to_type) {
		Copy_array(from_type, from_data, to_data, ndata);
	}else if (to_type == ONNX_TENSOR_TYPE_STRING) {
		Cast_to_string(from_type, from_data, (std::string*)to_data, ndata);
	}else if (from_type == ONNX_TENSOR_TYPE_STRING) {
		Cast_from_string((const std::string*)from_data, to_type, to_data, ndata);
	}else {
		switch (from_type) {
#define X(enum_type, data_type) case enum_type: Cast_array((const data_type*)from_data, to_type, to_data, ndata); return;
		X(ONNX_TENSOR_TYPE_BOOL, bool_t)
		X(ONNX_TENSOR_TYPE_INT8, int8_t)
		X(ONNX_TENSOR_TYPE_INT16, int16_t)
		X(ONNX_TENSOR_TYPE_INT32, int32_t)
		X(ONNX_TENSOR_TYPE_INT64, int64_t)
		X(ONNX_TENSOR_TYPE_UINT8, uint8_t)
		X(ONNX_TENSOR_TYPE_UINT16, uint16_t)
		X(ONNX_TENSOR_TYPE_UINT32, uint32_t)
		X(ONNX_TENSOR_TYPE_UINT64, uint64_t)
		X(ONNX_TENSOR_TYPE_BFLOAT16, bfloat16_t)
		X(ONNX_TENSOR_TYPE_FLOAT16, float16_t)
		X(ONNX_TENSOR_TYPE_FLOAT32, float)
		X(ONNX_TENSOR_TYPE_FLOAT64, double)
#undef X
		}
	}
}

struct Cast_operator : public operator_t {
	tensor_type_t to;

	bool init() override {
		if (!is_inout_size(1, 1)) {
			return false;
		}
		to = (tensor_type_t)attribute("to", inputs[0]->type);
		return true;
	}

	bool reshape() override {
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];
		return y->reshape_identity(x, to);
	}

	template <typename T>
	void exec() {
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];
		Cast_array(x->type, x->data, y->type, y->data, y->ndata);
	}

	void exec() override {
		if (opset >= 13) {
			TYPED_EXEC(inputs[0]->type,
				bool_t,
				uint8_t, uint16_t, uint32_t, uint64_t,
				int8_t, int16_t, int32_t, int64_t,
				float16_t, float, double, bfloat16_t,
				std::string
			)
		}else if (opset >= 9) {
			TYPED_EXEC(inputs[0]->type,
				bool_t,
				uint8_t, uint16_t, uint32_t, uint64_t,
				int8_t, int16_t, int32_t, int64_t,
				float16_t, float, double,
				std::string
			)
		}else if (opset >= 6) {
			TYPED_EXEC(inputs[0]->type,
				bool_t,
				uint8_t, uint16_t, uint32_t, uint64_t,
				int8_t, int16_t, int32_t, int64_t,
				float16_t, float, double
			)
		}else if (opset >= 1) {
		}
	}

};

} // namespace {

operator_t* resolver_default_op_Cast()
{
	return new Cast_operator;
}

} // namespace onnx
