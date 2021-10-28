#pragma once

#include "onnxconf.h"

#include "onnx.proto3.pb-c.h"

#define LIBONNX_MAJOR			(1)
#define LIBONNX_MINIOR			(0)
#define LIBONNX_PATCH			(0)
#define LIBONNX_VERSION			((LIBONNX_MAJOR * 10000) + (LIBONNX_MINIOR * 100) + LIBONNX_PATCH)

namespace onnx {

struct tensor_t;
struct graph_t;
struct context_t;

enum tensor_type_t {
	ONNX_TENSOR_TYPE_UNDEFINED	= 0,
	ONNX_TENSOR_TYPE_BOOL		= 9,
	ONNX_TENSOR_TYPE_INT8		= 3,
	ONNX_TENSOR_TYPE_INT16		= 5,
	ONNX_TENSOR_TYPE_INT32		= 6,
	ONNX_TENSOR_TYPE_INT64		= 7,
	ONNX_TENSOR_TYPE_UINT8		= 2,
	ONNX_TENSOR_TYPE_UINT16		= 4,
	ONNX_TENSOR_TYPE_UINT32		= 12,
	ONNX_TENSOR_TYPE_UINT64		= 13,
	ONNX_TENSOR_TYPE_BFLOAT16	= 16,
	ONNX_TENSOR_TYPE_FLOAT16	= 10,
	ONNX_TENSOR_TYPE_FLOAT32	= 1,
	ONNX_TENSOR_TYPE_FLOAT64	= 11,
	ONNX_TENSOR_TYPE_COMPLEX64	= 14,
	ONNX_TENSOR_TYPE_COMPLEX128	= 15,
	ONNX_TENSOR_TYPE_STRING		= 8,
};

std::string_view tensor_type_tostring(tensor_type_t type);
size_t tensor_type_sizeof(tensor_type_t type);
size_t tensor_type_sizeof(const tensor_t* tensor);
bool tensor_equal(const tensor_t* a, const tensor_t* b);

struct tensor_t {
	tensor_t(std::string_view name, tensor_type_t type, int* dims, int ndim);
	~tensor_t();

	static tensor_t* alloc_from_file(std::string_view filename);

	void reinit(tensor_type_t type, const int* dims, int ndim);
	void apply(const void* buf, size_t len);
	void apply(const tensor_t& t);
	
	void dump(bool detail) const;

	int indices_to_offset(const int* indices) const
	{
		int offset = 0;
		for (int i = 0; i < ndim; ++i) {
			offset += indices[i] * strides[i];
		}
		return offset;
	}

	void offset_to_indices(int offset, int* indices) const
	{
		for (int i = ndim - 1; i >= 0; i--) {
			indices[i] = offset % dims[i];
			offset /= dims[i];
		}
	}

	bool reshape(const int* dims, int ndim, tensor_type_t type);

	bool reshape_identity(const tensor_t* x, tensor_type_t type);

	bool reshape_identity(const tensor_t* x)
	{
		return reshape_identity(x, x->type);
	}

	bool reshape_multi_broadcast(const tensor_t* a, const tensor_t* b, tensor_type_t type);

	bool is_scalar() const
	{
		return ((ndim == 0) && (ndata == 1));
	}

	bool broadcast_is_valid(const int* dims, int ndim) const
	{
		if (this->ndim > ndim) {
			return false;
		}
		for (int i = 1; i <= this->ndim; ++i) {
			if ((this->dims[this->ndim - i] != 1) && (this->dims[this->ndim - i] != dims[ndim - i])) {
				return false;
			}
		}
		return true;
	}

	void* broadcast_map_address(const tensor_t* y, int offset);

	const void* broadcast_map_address(const tensor_t* y, int offset) const
	{
		return (const void*) const_cast<tensor_t*>(this)->broadcast_map_address(y, offset);
	}

	std::string name;
	tensor_type_t type = ONNX_TENSOR_TYPE_UNDEFINED;
	std::vector<int> strides;
	std::vector<int> dims;
	int ndim = 0;
	void* data = nullptr;
	size_t ndata = 0;
};

void copy_data(tensor_t* y, const tensor_t* x);

struct operator_t {
	virtual ~operator_t() = default;
	void dump(bool detail) const;
	Onnx__AttributeProto* find_attribute(std::string_view name);
	float attribute(std::string_view name, float def);
	int32_t attribute(std::string_view name, int32_t def);
	int64_t attribute(std::string_view name, int64_t def);
	std::string_view attribute(std::string_view name, std::string_view def);
	int attribute(std::string_view name, int64_t*& ints);
	int attribute(std::string_view name, float*& floats);
	int attribute(std::string_view name, tensor_t* t);

	template <typename T>
	T attribute(std::string_view name, std::string_view def) {
		static_assert(std::is_enum_v<T>);
		auto v0 = magic_enum::enum_cast<T>(attribute(name, def));
		if (v0.has_value()) {
			return v0.value();
		}else {
			return magic_enum::enum_cast<T>(def).value();
		}
	}

	Onnx__GraphProto* attribute(std::string_view name, Onnx__GraphProto* def);
	Onnx__SparseTensorProto* attribute(std::string_view name, Onnx__SparseTensorProto* def);

	context_t* ctx = nullptr;
	int opset = 0;
	std::vector<tensor_t*> inputs;
	std::vector<tensor_t*> outputs;
	Onnx__NodeProto* proto = nullptr;

	virtual bool init() { return true; }
	virtual bool reshape()
	{
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];
		if (x && y) {
			return y->reshape_identity(x);
		}else {
			return false;
		}
	}
	virtual void exec() = 0;

	bool is_inout_size(size_t in_size, size_t out_size) const
	{
		return (inputs.size() == in_size) && (outputs.size() == out_size);
	}

	template <typename T, typename FuncT>
	void foreach_tensor(FuncT func)
	{
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];
		const T* px = (const T*)x->data;
		T* py = (T*)y->data;

		for (size_t i = 0, l = y->ndata; i < l; ++i) {
			py[i] = (T)func(px[i]);
		}
	}

};

struct graph_t {
	graph_t() = default;
	graph_t(const graph_t&) = delete;
	graph_t& operator=(const graph_t&) = delete;
	~graph_t() = default;

	bool init(context_t* ctx, Onnx__GraphProto* graph);

	void dump(bool detail) const;

	std::vector<operator_t*> nodes;
};

struct context_t {
	context_t() = default;
	context_t(const context_t&) = delete;
	context_t& operator=(const context_t&) = delete;
	~context_t();

	bool alloc(const void* buf, size_t len);
	bool alloc_from_file(std::string_view filename);
	void dump(bool detail) const;
	void run();
	tensor_t* search_tensor(std::string_view name);

	Onnx__ModelProto* model = nullptr;
	std::map<std::string_view, tensor_t*> map;
	std::unique_ptr<graph_t> graph;
};

static inline int dim_next(int ndim, int* dims, const int* dim_max)
{
	if (ndim == 0) {
		return 0;
	}
	while (1) {
		ndim = ndim - 1;
		dims[ndim] += 1;
		if (dims[ndim] < dim_max[ndim]) {
			return 1;
		}else {
			if (ndim == 0) {
				return 0;
			}
			dims[ndim] = 0;
		}
	}
}

static inline int dim_offset(int ndim, const int* dims, const int* dim_max)
{
	int i, o, s;

	for (i = ndim - 1, o = 0, s = 1; i >= 0; i--) {
		o += dims[i] * s;
		s *= dim_max[i];
	}
	return o;
}

} // namespace onnx
