#pragma once

#include <onnxconf.h>

#include <onnx.proto3.pb-c.h>

#define LIBONNX_MAJOR			(1)
#define LIBONNX_MINIOR			(0)
#define LIBONNX_PATCH			(0)
#define LIBONNX_VERSION			((LIBONNX_MAJOR * 10000) + (LIBONNX_MINIOR * 100) + LIBONNX_PATCH)

namespace onnx {

struct tensor_t;
struct graph_t;
struct context_t;
struct resolver_t;

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
int tensor_type_sizeof(tensor_type_t type);
int tensor_type_sizeof(const tensor_t* tensor);
bool tensor_equal(const tensor_t* a, const tensor_t* b);

struct tensor_t {
	tensor_t(std::string_view name, tensor_type_t type, int* dims, int ndim);
	~tensor_t();

	void reinit(tensor_type_t type, const int* dims, int ndim);
	void apply(const void* buf, size_t len);
	
	void dump(int detail) const;

	int indices_to_offset(const int* indices) const
	{
		int offset = 0;
		for (int i = 0; i < ndim; i++)
			offset += indices[i] * strides[i];
		return offset;
	}

	void offset_to_indices(int offset, int* indices) const
	{
		for (int i = ndim - 1; i >= 0; i--) {
			indices[i] = offset % dims[i];
			offset /= dims[i];
		}
	}

	bool reshape(const int* dims, int ndim, tensor_type_t type)
	{
		if ((this->ndim != ndim) || (dims && (memcmp(&this->dims[0], dims, sizeof(int) * ndim) != 0)) || (this->type != type))
			reinit(type, dims, ndim);
		return true;
	}

	bool reshape_identity(const tensor_t* x, tensor_type_t type)
	{
		if ((this->ndim != x->ndim) || (memcmp(&this->dims[0], &x->dims[0], sizeof(int) * this->ndim) != 0) || (this->type != type))
			reinit(type, &x->dims[0], x->ndim);
		return true;
	}

	bool reshape_identity(const tensor_t* x)
	{
		return reshape_identity(x, x->type);
	}

	bool reshape_multi_broadcast(const tensor_t* a, const tensor_t* b, tensor_type_t type)
	{
		int ndim = max(a->ndim, b->ndim);
		std::vector<int> dims(ndim);
		if (ndim > 0)
		{
			int i, j, k;
			for (i = a->ndim - 1, j = b->ndim - 1, k = ndim - 1; k >= 0; k--) {
				if (i < 0)
					dims[k] = b->dims[j--];
				else if (j < 0)
					dims[k] = a->dims[i--];
				else {
					if (a->dims[i] == b->dims[j])
						dims[k] = a->dims[i];
					else if ((a->dims[i] == 1) || (b->dims[j] == 1))
						dims[k] = (a->dims[i] > b->dims[j]) ? a->dims[i] : b->dims[j];
					else
						return false;
					i--;
					j--;
				}
			}
		}
		if ((this->type != type) || (this->ndim != ndim) || (memcmp(&this->dims[0], &dims[0], sizeof(int) * ndim) != 0))
			reinit(type, &dims[0], ndim);
		return true;
	}

	bool is_scalar() const
	{
		return ((ndim == 0) && (ndata == 1));
	}

	bool broadcast_is_valid(const int* dims, int ndim) const
	{
		if (this->ndim > ndim)
			return false;
		for (int i = 1; i <= this->ndim; i++) {
			if ((this->dims[this->ndim - i] != 1) && (this->dims[this->ndim - i] != dims[ndim - i]))
				return false;
		}
		return true;
	}

	void* broadcast_map_address(const tensor_t* y, int offset)
	{
		int xndim = this->ndim;
		int yndim = y->ndim;

		if ((xndim > 0) && (yndim > 0)) {
			int dndim = yndim - xndim;
			std::vector<int> ix(xndim);
			std::vector<int> iy(yndim);
			int i;

			y->offset_to_indices(offset, &iy[0]);
			for (i = 0; i < xndim; i++)
				ix[i] = iy[dndim + i] % this->dims[i];
			return (char*)this->data + this->indices_to_offset(&ix[0]) * tensor_type_sizeof(this);
		}
		return this->data;
	}

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

struct operator_t {
	virtual ~operator_t() {
	}
	void dump(int detail) const;
	Onnx__AttributeProto* find_attribute(std::string_view name);
	float attribute(std::string_view name, float def);
	int32_t attribute(std::string_view name, int32_t def);
	int64_t attribute(std::string_view name, int64_t def);
	std::string_view attribute(std::string_view name, std::string_view def);
	int attribute(std::string_view name, int64_t** ints);
	int attribute(std::string_view name, float** floats);
	int attribute(std::string_view name, tensor_t* t);
	Onnx__GraphProto* attribute(std::string_view name, Onnx__GraphProto* def);
	Onnx__SparseTensorProto* attribute(std::string_view name, Onnx__SparseTensorProto* def);

	context_t* ctx = nullptr;
	resolver_t* r = nullptr;
	void* rctx = nullptr;
	int opset = 0;
	std::vector<tensor_t*> inputs;
	std::vector<tensor_t*> outputs;
	Onnx__NodeProto* proto = nullptr;

	virtual bool init() { return true; }
	virtual bool reshape() {
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];
		return y->reshape_identity(x);
	}
	virtual void exec() = 0;

	bool is_inout_size(size_t in_size, size_t out_size) const {
		return (inputs.size() == in_size) && (outputs.size() == out_size);
	}

	template <typename T, typename FuncT>
	void foreach_tensor(FuncT func)
	{
		const tensor_t* x = inputs[0];
		tensor_t* y = outputs[0];
		const T* px = (const T*)x->data;
		T* py = (T*)y->data;

		for (size_t i = 0, l = y->ndata; i < l; i++) {
			py[i] = func(px[i]);
		}
	}

};

struct graph_t {
	graph_t(context_t* ctx, Onnx__GraphProto* graph);
	graph_t(const graph_t&) = delete;
	graph_t& operator=(const graph_t&) = delete;
	~graph_t();

	void dump(int detail) const;

	std::vector<operator_t*> nodes;
};

struct context_t {
	context_t(const void* buf, size_t len, resolver_t** r, int rlen);
	context_t(std::string_view filename, resolver_t** r, int rlen);
	context_t(const context_t&) = delete;
	context_t& operator=(const context_t&) = delete;
	~context_t();

	void dump(int detail) const;
	void run();
	tensor_t* search_tensor(std::string_view name);

	Onnx__ModelProto* model;
	std::map<std::string_view, tensor_t*> map;
	std::vector<resolver_t*> resolvers;
	std::vector<void*> rctx;
	std::unique_ptr<graph_t> graph;
};

struct resolver_t {
	const char* name;

	virtual void* create(void) = 0;
	virtual void destroy(void* rctx) = 0;
	virtual operator_t* solve_operator(std::string_view op_type) = 0;

};

static inline int dim_next(int ndim, int* dims, const int* dim_max)
{
	if (ndim == 0)
		return 0;
	while (1) {
		ndim = ndim - 1;
		dims[ndim] += 1;
		if (dims[ndim] < dim_max[ndim])
			return 1;
		else {
			if (ndim == 0)
				return 0;
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
