#pragma once

#include <onnxconf.h>

#include <onnx.proto3.pb-c.h>

#define LIBONNX_MAJOR			(1)
#define LIBONNX_MINIOR			(0)
#define LIBONNX_PATCH			(0)
#define LIBONNX_VERSION			((LIBONNX_MAJOR * 10000) + (LIBONNX_MINIOR * 100) + LIBONNX_PATCH)

struct onnx_tensor_t;
struct onnx_node_t;
struct onnx_graph_t;
struct onnx_context_t;
struct onnx_resolver_t;

enum onnx_tensor_type_t {
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

const char* onnx_tensor_type_tostring(onnx_tensor_type_t type);
int onnx_tensor_type_sizeof(onnx_tensor_type_t type);
onnx_tensor_t* onnx_tensor_alloc_from_file(const char* filename);
int onnx_tensor_equal(const onnx_tensor_t* a, const onnx_tensor_t* b);

struct onnx_tensor_t {
	onnx_tensor_t(const char* name, onnx_tensor_type_t type, int* dims, int ndim);
	~onnx_tensor_t();

	void reinit(onnx_tensor_type_t type, const int* dims, int ndim);
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

	int reshape(const int* dims, int ndim, onnx_tensor_type_t type)
	{
		if ((this->ndim != ndim) || (dims && (memcmp(this->dims, dims, sizeof(int) * ndim) != 0)) || (this->type != type))
			reinit(type, dims, ndim);
		return 1;
	}

	int reshape_identity(const onnx_tensor_t* x, onnx_tensor_type_t type)
	{
		if ((this->ndim != x->ndim) || (memcmp(this->dims, x->dims, sizeof(int) * this->ndim) != 0) || (this->type != type))
			reinit(type, x->dims, x->ndim);
		return 1;
	}

	int reshape_multi_broadcast(const onnx_tensor_t* a, const onnx_tensor_t* b, onnx_tensor_type_t type)
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
						return 0;
					i--;
					j--;
				}
			}
		}
		if ((this->type != type) || (this->ndim != ndim) || (memcmp(this->dims, &dims[0], sizeof(int) * ndim) != 0))
			this->reinit(type, &dims[0], ndim);
		return 1;
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

	void* broadcast_map_address(const onnx_tensor_t* y, int offset)
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
			return (char*)this->datas + this->indices_to_offset(&ix[0]) * onnx_tensor_type_sizeof(this->type);
		}
		return this->datas;
	}

	std::string name;
	onnx_tensor_type_t type;
	int* strides;
	int* dims;
	int ndim;
	void* datas;
	size_t ndata;
};

struct onnx_node_t {
	void dump(int detail) const;
	Onnx__AttributeProto* search_attribute(const char* name);
	float attribute_read_float(const char* name, float def);
	int64_t attribute_read_int(const char* name, int64_t def);
	const char* attribute_read_string(const char* name, const char* def);
	int attribute_read_ints(const char* name, int64_t** ints);
	int attribute_read_floats(const char* name, float** floats);
	int attribute_read_tensor(const char* name, onnx_tensor_t* t);
	Onnx__GraphProto* attribute_read_graph(const char* name, Onnx__GraphProto* def);
	Onnx__SparseTensorProto* attribute_read_sparse_tensor(const char* name, Onnx__SparseTensorProto* def);

	onnx_context_t* ctx;
	onnx_resolver_t* r;
	void* rctx;
	int opset;
	std::vector<onnx_tensor_t*> inputs;
	std::vector<onnx_tensor_t*> outputs;
	Onnx__NodeProto* proto;

	int (*init)(onnx_node_t* n);
	int (*exit)(onnx_node_t* n);
	int (*reshape)(onnx_node_t* n);
	void (*ope)(onnx_node_t* n);
	void* priv;
};

struct onnx_graph_t {
	onnx_graph_t(onnx_context_t* ctx, Onnx__GraphProto* graph);
	onnx_graph_t(const onnx_graph_t&) = delete;
	onnx_graph_t& operator=(const onnx_graph_t&) = delete;
	~onnx_graph_t();

	void dump(int detail) const;

	std::vector<onnx_node_t> nodes;
};

struct onnx_context_t {
	onnx_context_t(const void* buf, size_t len, onnx_resolver_t** r, int rlen);
	onnx_context_t(const char* filename, onnx_resolver_t** r, int rlen);
	onnx_context_t(const onnx_context_t&) = delete;
	onnx_context_t& operator=(const onnx_context_t&) = delete;
	~onnx_context_t();

	void dump(int detail) const;
	void run();
	onnx_tensor_t* tensor_search(const char* name);

	Onnx__ModelProto* model;
	std::map<const char*, onnx_tensor_t*> map;
	std::vector<onnx_resolver_t*> resolvers;
	std::vector<void*> rctx;
	onnx_graph_t* graph;
};

struct onnx_resolver_t {
	const char* name;

	void* (*create)(void);
	void (*destroy)(void* rctx);

	using ope_t = void (*)(onnx_node_t* n);
	std::map<const char*, ope_t> op_map;
};

static inline int dim_next(int ndim, int* dims, int* dim_max)
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

static inline int dim_offset(int ndim, int* dims, int* dim_max)
{
	int i, o, s;

	for (i = ndim - 1, o = 0, s = 1; i >= 0; i--) {
		o += dims[i] * s;
		s *= dim_max[i];
	}
	return o;
}

struct onnx_ope_type_selector {
	using ope_t = void (*)(onnx_node_t* n);
	ope_t bool_ = nullptr;
	ope_t int8_ = nullptr;
	ope_t int16_ = nullptr;
	ope_t int32_ = nullptr;
	ope_t int64_ = nullptr;
	ope_t uint8_ = nullptr;
	ope_t uint16_ = nullptr;
	ope_t uint32_ = nullptr;
	ope_t uint64_ = nullptr;
	ope_t bfloat16_ = nullptr;
	ope_t float16_ = nullptr;
	ope_t float32_ = nullptr;
	ope_t float64_ = nullptr;
	ope_t complex64_ = nullptr;
	ope_t complex128_ = nullptr;
	ope_t string_ = nullptr;

	ope_t select(onnx_tensor_type_t type) const {
		switch (type) {
		case ONNX_TENSOR_TYPE_INT8: return int8_;
		case ONNX_TENSOR_TYPE_INT16: return int16_;
		case ONNX_TENSOR_TYPE_INT32: return int32_;
		case ONNX_TENSOR_TYPE_INT64: return int64_;
		case ONNX_TENSOR_TYPE_UINT8: return uint8_;
		case ONNX_TENSOR_TYPE_UINT16: return uint16_;
		case ONNX_TENSOR_TYPE_UINT32: return uint32_;
		case ONNX_TENSOR_TYPE_UINT64: return uint64_;
		case ONNX_TENSOR_TYPE_BFLOAT16: return bfloat16_;
		case ONNX_TENSOR_TYPE_FLOAT16: return float16_;
		case ONNX_TENSOR_TYPE_FLOAT32: return float32_;
		case ONNX_TENSOR_TYPE_FLOAT64: return float64_;
		default: return nullptr;
		}
	}
};

