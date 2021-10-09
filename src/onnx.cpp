/*
 * onnx.cpp
 *
 * Copyright(c) 2007-2020 Jianjun Jiang <8192542@qq.com>
 * Mobile phone: +86-18665388956
 * QQ: 8192542
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

#include <complex>
#include "onnx.h"
#include "default/default.h"
#include "default/util.h"

#define ONNX_LOG(...)	printf(__VA_ARGS__)

namespace onnx {

bool context_t::alloc_from_file(std::string_view filename, resolver_t** r, int rlen)
{
	FILE* fp = fopen(filename.data(), "rb");
	if (!fp) {
		return false;
	}
	bool ret = true;
	fseek(fp, 0L, SEEK_END);
	long l = ftell(fp);
	fseek(fp, 0L, SEEK_SET);
	if (l > 0) {
		std::vector<char> buf(l);
		size_t len;
		for (len = 0; len < l; len += fread(&buf[len], 1, l - len, fp))
			;
		ret = alloc(&buf[0], len, r, rlen);
	}
	fclose(fp);
	return ret;
}

context_t::~context_t()
{
	for (size_t i = 0; i < resolvers.size(); ++i) {
		if (resolvers[i]) {
			resolvers[i]->destroy(rctx[i]);
		}
	}
	for (auto it = map.begin(); it != map.end(); ++it) {
		delete it->second;
	}
	if (model) {
		onnx__model_proto__free_unpacked(model, nullptr);
	}
}

static tensor_t* tensor_alloc_from_value_info(Onnx__ValueInfoProto* v)
{
	tensor_t* t;
	tensor_type_t type;
	std::vector<int> dims;
	int ndim;

	if (!v || !v->name) {
		return nullptr;
	}

	switch (v->type->value_case) {
	case ONNX__TYPE_PROTO__VALUE_TENSOR_TYPE:
		type = (tensor_type_t)v->type->tensor_type->elem_type;
		ndim = v->type->tensor_type->shape->n_dim;
		if (ndim > 0) {
			dims.resize(ndim);
			for (int i = 0; i < ndim; ++i) {
				switch (v->type->tensor_type->shape->dim[i]->value_case) {
				case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_VALUE:
					dims[i] = v->type->tensor_type->shape->dim[i]->dim_value;
					break;
				case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_PARAM:
					if (strcmp(v->type->tensor_type->shape->dim[i]->dim_param, "batch_size") == 0) {
						dims[i] = 1;
					}else {
						dims[i] = 1;
					}
					break;
				default:
					dims[i] = 1;
					break;
				}
			}
		}
		t = new tensor_t(v->name, type, &dims[0], ndim);
		break;
	case ONNX__TYPE_PROTO__VALUE_SEQUENCE_TYPE:
		t = nullptr;
		break;
	case ONNX__TYPE_PROTO__VALUE_MAP_TYPE:
		t = nullptr;
		break;
	default:
		t = nullptr;
		break;
	}
	return t;
}

static void tensor_copy_from_tensor_proto(tensor_t* t, Onnx__TensorProto* o)
{
	if (!t || !o) {
		return;
	}

	if (t->type != o->data_type) {
		return;
	}

	int sz = tensor_type_sizeof(t);
	if (sz <= 0) {
		return;
	}

	if ((o->raw_data.len > 0) && o->raw_data.data) {
		switch (o->data_type) {
		case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
		{
			float* p = (float*)t->data;
			uint32_t* q = (uint32_t*)o->raw_data.data;
			union { uint32_t u; float f; } v;
			if (t->ndata > 0) {
				size_t n = min(t->ndata, (size_t)o->raw_data.len / sz);
				for (size_t i = 0; i < n; ++i) {
					v.u = le32_to_cpu(q[i]);
					p[i] = v.f;
				}
			}
		}
		break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8:
		{
			uint8_t* p = (uint8_t*)t->data;
			uint8_t* q = (uint8_t*)o->raw_data.data;
			if (t->ndata > 0) {
				size_t n = min(t->ndata, (size_t)o->raw_data.len);
				memcpy(p, q, n);
			}
		}
		break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__INT8:
		{
			int8_t* p = (int8_t*)t->data;
			int8_t* q = (int8_t*)o->raw_data.data;
			if (t->ndata > 0) {
				size_t n = min(t->ndata, (size_t)o->raw_data.len);
				memcpy(p, q, n);
			}
		}
		break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__UINT16:
		{
			uint16_t* p = (uint16_t*)t->data;
			uint16_t* q = (uint16_t*)o->raw_data.data;
			if (t->ndata > 0) {
				size_t n = min(t->ndata, (size_t)o->raw_data.len / sz);
				for (size_t i = 0; i < n; ++i) {
					p[i] = le16_to_cpu(q[i]);
				}
			}
		}
		break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__INT16:
		{
			int16_t* p = (int16_t*)t->data;
			int16_t* q = (int16_t*)o->raw_data.data;
			if (t->ndata > 0) {
				size_t n = min(t->ndata, (size_t)o->raw_data.len / sz);
				for (size_t i = 0; i < n; ++i) {
					p[i] = le16_to_cpu(q[i]);
				}
			}
		}
		break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
		{
			int32_t* p = (int32_t*)t->data;
			int32_t* q = (int32_t*)o->raw_data.data;
			if (t->ndata > 0) {
				size_t n = min(t->ndata, (size_t)o->raw_data.len / sz);
				for (size_t i = 0; i < n; ++i) {
					p[i] = le32_to_cpu(q[i]);
				}
			}
		}
		break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
		{
			int64_t* p = (int64_t*)t->data;
			int64_t* q = (int64_t*)o->raw_data.data;
			if (t->ndata > 0) {
				size_t n = min(t->ndata, (size_t)o->raw_data.len / sz);
				for (size_t i = 0; i < n; ++i) {
					p[i] = le64_to_cpu(q[i]);
				}
			}
		}
		break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__STRING:
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL:
		{
			uint8_t* p = (uint8_t*)t->data;
			uint8_t* q = (uint8_t*)o->raw_data.data;
			if (t->ndata > 0) {
				size_t n = min(t->ndata, (size_t)o->raw_data.len);
				memcpy(p, q, n);
			}
		}
		break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16:
		{
			uint16_t* p = (uint16_t*)t->data;
			uint16_t* q = (uint16_t*)o->raw_data.data;
			if (t->ndata > 0) {
				size_t n = min(t->ndata, (size_t)o->raw_data.len / sz);
				for (size_t i = 0; i < n; ++i) {
					p[i] = le16_to_cpu(q[i]);
				}
			}
		}
		break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
		{
			double* p = (double*)t->data;
			uint64_t* q = (uint64_t*)o->raw_data.data;
			union { uint64_t u; double f; } v;
			if (t->ndata > 0) {
				size_t n = min(t->ndata, (size_t)o->raw_data.len / sz);
				for (size_t i = 0; i < n; ++i) {
					v.u = le64_to_cpu(q[i]);
					p[i] = v.f;
				}
			}
		}
		break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__UINT32:
		{
			uint32_t* p = (uint32_t*)t->data;
			uint32_t* q = (uint32_t*)o->raw_data.data;
			if (t->ndata > 0) {
				size_t n = min(t->ndata, (size_t)o->raw_data.len / sz);
				for (size_t i = 0; i < n; ++i) {
					p[i] = le32_to_cpu(q[i]);
				}
			}
		}
		break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__UINT64:
		{
			uint64_t* p = (uint64_t*)t->data;
			uint64_t* q = (uint64_t*)o->raw_data.data;
			if (t->ndata > 0) {
				size_t n = min(t->ndata, (size_t)o->raw_data.len / sz);
				for (size_t i = 0; i < n; ++i) {
					p[i] = le64_to_cpu(q[i]);
				}
			}
		}
		break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64:
		{
			float* p = (float*)t->data;
			uint32_t* q = (uint32_t*)o->raw_data.data;
			union { uint32_t u; float f; } v;
			if (t->ndata > 0) {
				size_t n = min(t->ndata, (size_t)o->raw_data.len / sz) * 2;
				for (size_t i = 0; i < n; ++i) {
					v.u = le32_to_cpu(q[i]);
					p[i] = v.f;
				}
			}
		}
		break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128:
		{
			double* p = (double*)t->data;
			uint64_t* q = (uint64_t*)o->raw_data.data;
			union { uint64_t u; double f; } v;
			if (t->ndata > 0) {
				size_t n = min(t->ndata, (size_t)o->raw_data.len / sz) * 2;
				for (size_t i = 0; i < n; ++i) {
					v.u = le64_to_cpu(q[i]);
					p[i] = v.f;
				}
			}
		}
		break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16:
		{
			uint16_t* p = (uint16_t*)t->data;
			uint16_t* q = (uint16_t*)o->raw_data.data;
			if (t->ndata > 0) {
				size_t n = min(t->ndata, (size_t)o->raw_data.len / sz);
				for (size_t i = 0; i < n; ++i) {
					p[i] = le16_to_cpu(q[i]);
				}
			}
		}
		break;
		default:
			break;
		}
	}else {
		switch (o->data_type) {
		case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
		{
			size_t n = min(t->ndata, (size_t)o->n_float_data);
			if ((n > 0) && t->data && o->float_data) {
				memcpy(t->data, o->float_data, sizeof(float) * n);
			}
		}
		break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8:
		case ONNX__TENSOR_PROTO__DATA_TYPE__INT8:
		case ONNX__TENSOR_PROTO__DATA_TYPE__UINT16:
		case ONNX__TENSOR_PROTO__DATA_TYPE__INT16:
		case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
		case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL:
		case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16:
		case ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16:
		{
			//TODO
			size_t n = min(t->ndata, (size_t)o->n_int32_data);
			if ((n > 0) && t->data && o->int32_data) {
				memcpy(t->data, o->int32_data, sz * n);
			}
		}
		break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__STRING:
		{
			size_t n = min(t->ndata, (size_t)o->n_string_data);
			if ((n > 0) && t->data && o->string_data) {
				std::string* str = (std::string*)t->data;
				for (size_t i = 0; i < n; ++i) {
					str[i].assign((const char*)o->string_data[i].data, o->string_data[i].len);
				}
			}
		}
		break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
		{
			size_t n = min(t->ndata, (size_t)o->n_int64_data);
			if ((n > 0) && t->data && o->int64_data) {
				memcpy(t->data, o->int64_data, sizeof(int64_t) * n);
			}
		}
		break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
		{
			size_t n = min(t->ndata, (size_t)o->n_double_data);
			if ((n > 0) && t->data && o->double_data) {
				memcpy(t->data, o->double_data, sizeof(double) * n);
			}
		}
		break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__UINT32:
		case ONNX__TENSOR_PROTO__DATA_TYPE__UINT64:
		{
			//TODO
			size_t n = min(t->ndata, (size_t)o->n_uint64_data);
			if ((n > 0) && t->data && o->uint64_data) {
				memcpy(t->data, o->uint64_data, sz * n);
			}
		}
		break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64:
		{
			size_t n = min(t->ndata, (size_t)(o->n_float_data / 2));
			if ((n > 0) && t->data && o->float_data) {
				memcpy(t->data, o->float_data, sizeof(float) * 2 * n);
			}
		}
		break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128:
		{
			size_t n = min(t->ndata, (size_t)(o->n_double_data / 2));
			if ((n > 0) && t->data && o->double_data) {
				memcpy(t->data, o->double_data, sizeof(double) * 2 * n);
			}
		}
		break;
		default:
			break;
		}
	}
}

struct operator_dummy : public operator_t {
	void exec() override {
		ONNX_LOG("\033[45;37mUnsupported opset\033[0m => %s-%d (%s)\r\n", proto->op_type, opset, (strlen(proto->domain) > 0) ? proto->domain : "ai.onnx");
	}
};

graph_t::graph_t(context_t* ctx, Onnx__GraphProto* graph)
{
	assert(graph);

	nodes.resize(graph->n_node);

	for (int i = 0; i < graph->n_input; ++i) {
		Onnx__ValueInfoProto* v = graph->input[i];
		if (ctx->search_tensor(v->name)) {
			continue;
		}
		tensor_t* t = tensor_alloc_from_value_info(v);
		if (!t) {
			continue;
		}
		for (int j = 0; j < graph->n_initializer; ++j) {
			if (graph->initializer[j]->name == t->name) {
				tensor_copy_from_tensor_proto(t, graph->initializer[j]);
				break;
			}
		}
		ctx->map[t->name] = t;
	}

	for (int i = 0; i < graph->n_output; ++i) {
		Onnx__ValueInfoProto* v = graph->output[i];
		if (ctx->search_tensor(v->name)) {
			continue;
		}
		tensor_t* t = tensor_alloc_from_value_info(v);
		if (t) {
			ctx->map[t->name] = t;
		}
	}

	for (int i = 0; i < graph->n_value_info; ++i) {
		Onnx__ValueInfoProto* v = graph->value_info[i];
		if (ctx->search_tensor(v->name)) {
			continue;
		}
		tensor_t* t = tensor_alloc_from_value_info(v);
		if (t) {
			ctx->map[t->name] = t;
		}
	}

	for (int i = 0; i < graph->n_node; ++i) {
		for (int j = 0; j < graph->node[i]->n_output; ++j) {
			char* name = graph->node[i]->output[j];
			if (ctx->search_tensor(name)) {
				continue;
			}
			tensor_t* t = new tensor_t(name, ONNX_TENSOR_TYPE_UNDEFINED, nullptr, 0);
			ctx->map[name] = t;
		}
	}

	for (int i = 0; i < graph->n_node; ++i) {
		for (int j = 0; j < graph->node[i]->n_input; ++j) {
			std::string_view name = graph->node[i]->input[j];
			if (ctx->search_tensor(name)) {
				continue;
			}
			for (int k = 0; k < graph->n_initializer; ++k) {
				if (graph->initializer[k]->name != name) {
					continue;
				}
				Onnx__TensorProto* o = graph->initializer[k];
				if (!o) {
					continue;
				}
				const int ndim = o->n_dims;
				std::vector<int> dims(ndim);
				for (int l = 0; l < ndim; ++l) {
					dims[l] = o->dims[l];
				}
				tensor_t* t = new tensor_t(name, (tensor_type_t)o->data_type, &dims[0], ndim);
				tensor_copy_from_tensor_proto(t, o);
				ctx->map[name] = t;
				break;
			}
			//assert(ctx->search_tensor(name));
		}
	}

	for (int i = 0; i < nodes.size(); ++i) {
		operator_t* n = nullptr;
		Onnx__NodeProto* proto = graph->node[i];
		int opset = -1;
		const char* domain = proto->domain;
		if (!domain || (strlen(domain) == 0)) {
			domain = "ai.onnx";
		}
		for (int j = 0; j < ctx->model->n_opset_import; ++j) {
			const char* p = ctx->model->opset_import[j]->domain;
			if (!p || (strlen(p) == 0)) {
				p = "ai.onnx";
			}
			if (strcmp(domain, p) == 0) {
				opset = ctx->model->opset_import[j]->version;
				break;
			}
		}
		for (int j = 0; j < ctx->resolvers.size(); ++j) {
			auto resolver = ctx->resolvers[j];
			n = resolver->solve_operator(proto->op_type, opset);
			if (n) {
				n->r = resolver;
				n->rctx = ctx->rctx[j];
				break;
			}
		}
		if (!n) {
			n = resolver_default->solve_operator(proto->op_type, opset);
			if (n) {
				n->r = resolver_default;
				n->rctx = nullptr;
			}
		}
		if (!n) {
			n = new operator_dummy;
		}
		nodes[i] = n;
		n->ctx = ctx;
		n->proto = proto;
		n->opset = opset;
		if (n->proto->n_input > 0) {
			n->inputs.resize(n->proto->n_input);
			for (size_t j = 0; j < n->inputs.size(); ++j) {
				n->inputs[j] = ctx->search_tensor(n->proto->input[j]);
			}
		}
		if (n->proto->n_output > 0) {
			n->outputs.resize(n->proto->n_output);
			for (size_t j = 0; j < n->outputs.size(); ++j) {
				n->outputs[j] = ctx->search_tensor(n->proto->output[j]);
			}
		}
		if (!n->init()) {
			nodes.clear();
			return;
		}
		n->reshape();
	}
}

graph_t::~graph_t()
{
}

std::string_view tensor_type_tostring(tensor_type_t type)
{
	static std::string_view typestr[17] = {
		"undefined",
		"float32",
		"uint8",
		"int8",
		"uint16",
		"int16",
		"int32",
		"int64",
		"string",
		"bool",
		"float16",
		"float64",
		"uint32",
		"uint64",
		"complex64",
		"complex128",
		"bfloat16",
	};
	if ((type > 0) && (type < (sizeof(typestr) / sizeof((typestr)[0])))) {
		return typestr[type];
	}
	return typestr[0];
}

int tensor_type_sizeof(tensor_type_t type)
{
	static const int typesz[17] = {
		0,
		sizeof(float),
		sizeof(uint8_t),
		sizeof(int8_t),
		sizeof(uint16_t),
		sizeof(int16_t),
		sizeof(int32_t),
		sizeof(int64_t),
		sizeof(std::string),
		sizeof(bool_t),
		sizeof(float16_t),
		sizeof(double),
		sizeof(uint32_t),
		sizeof(uint64_t),
		sizeof(std::complex<float>),
		sizeof(std::complex<double>),
		sizeof(bfloat16_t),
	};
	if ((type > 0) && (type < (sizeof(typesz) / sizeof((typesz)[0])))) {
		return typesz[type];
	}
	return typesz[0];
}

int tensor_type_sizeof(const tensor_t* tensor)
{
	return tensor_type_sizeof(tensor->type);
}

tensor_t* context_t::search_tensor(std::string_view name)
{
	auto it = map.find(name);
	return (it == map.end()) ? nullptr : it->second;
}

tensor_t::tensor_t(std::string_view name, tensor_type_t type, int* dims, int ndim)
	:
	name(name)
{
	reinit(type, dims, ndim);
}

tensor_t* tensor_t::alloc_from_file(std::string_view filename)
{
	tensor_t* t = nullptr;
	Onnx__TensorProto* pb;
	size_t len;
	int ndim = 0;

	FILE* fp = fopen(filename.data(), "rb");
	if (!fp) {
		return nullptr;
	}
	fseek(fp, 0L, SEEK_END);
	size_t l = ftell(fp);
	fseek(fp, 0L, SEEK_SET);
	if (l > 0) {
		{
			std::vector<char> buf(l);
			for (len = 0; len < l; len += fread(&buf[len], 1, l - len, fp))
				;
			pb = onnx__tensor_proto__unpack(nullptr, len, (const uint8_t*)&buf[0]);
		}
		if (pb) {
			std::vector<int> dims;
			if (pb->n_dims > 0) {
				dims.resize(pb->n_dims);
				for (int i = 0; i < pb->n_dims; ++i) {
					dims[i] = pb->dims[i];
				}
				ndim = pb->n_dims;
			}
			t = new tensor_t(pb->name, (tensor_type_t)pb->data_type, &dims[0], ndim);
			tensor_copy_from_tensor_proto(t, pb);
			onnx__tensor_proto__free_unpacked(pb, nullptr);
		}
	}
	fclose(fp);
	return t;
}

inline
void delete_data(void* data, tensor_type_t type)
{
	if (type == ONNX_TENSOR_TYPE_STRING) {
		delete[] (std::string*)data;
	}else {
		delete[] data;
	}
}

tensor_t::~tensor_t()
{
	if ((ndata > 0) && data) {
		delete_data(data, type);
	}
}

bool tensor_equal(const tensor_t* a, const tensor_t* b)
{
	if (!a || !b) {
		return false;
	}
	if (a->type != b->type) {
		return false;
	}
	if (a->ndim != b->ndim) {
		return false;
	}
	if (a->ndata != b->ndata) {
		return false;
	}
	if (a->ndim > 0) {
		if (memcmp(&a->dims[0], &b->dims[0], sizeof(int) * a->ndim) != 0) {
			return false;
		}
	}
	switch (a->type) {
	case ONNX_TENSOR_TYPE_BOOL:
	case ONNX_TENSOR_TYPE_INT8:
	case ONNX_TENSOR_TYPE_INT16:
	case ONNX_TENSOR_TYPE_INT32:
	case ONNX_TENSOR_TYPE_INT64:
	case ONNX_TENSOR_TYPE_UINT8:
	case ONNX_TENSOR_TYPE_UINT16:
	case ONNX_TENSOR_TYPE_UINT32:
	case ONNX_TENSOR_TYPE_UINT64:
		if (memcmp(a->data, b->data, a->ndata * tensor_type_sizeof(a)) != 0) {
			return false;
		}
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
	{
		const bfloat16_t* p = (const bfloat16_t*)a->data;
		const bfloat16_t* q = (const bfloat16_t*)b->data;
		for (size_t i = 0; i < a->ndata; ++i) {
			if (fabsf(p[i] - q[i]) > 1e-3) {
				return false;
			}
		}
	}
	break;
	case ONNX_TENSOR_TYPE_FLOAT16:
	{
		const float16_t* p = (const float16_t*)a->data;
		const float16_t* q = (const float16_t*)b->data;
		for (size_t i = 0; i < a->ndata; ++i) {
			if (fabsf(p[i] - q[i]) > 1e-3) {
				return false;
			}
		}
	}
	break;
	case ONNX_TENSOR_TYPE_FLOAT32:
	{
		const float* p = (const float*)a->data;
		const float* q = (const float*)b->data;
		for (size_t i = 0; i < a->ndata; ++i) {
			if (fabsf(p[i] - q[i]) > 1e-3) {
				return false;
			}
		}
	}
	break;
	case ONNX_TENSOR_TYPE_FLOAT64:
	{
		const double* p = (const double*)a->data;
		const double* q = (const double*)b->data;
		for (size_t i = 0; i < a->ndata; ++i) {
			if (fabs(p[i] - q[i]) > 1e-3) {
				return false;
			}
		}
	}
	break;
	case ONNX_TENSOR_TYPE_COMPLEX64:
	{
		const std::complex<float>* p = (const std::complex<float>*)a->data;
		const std::complex<float>* q = (const std::complex<float>*)b->data;
		for (size_t i = 0; i < a->ndata; ++i) {
			if (std::abs(p[i] - q[i]) > 1e-3) {
				return false;
			}
		}
	}
	break;
	case ONNX_TENSOR_TYPE_COMPLEX128:
	{
		const std::complex<double>* p = (const std::complex<double>*)a->data;
		const std::complex<double>* q = (const std::complex<double>*)b->data;
		for (size_t i = 0; i < a->ndata; ++i) {
			if (std::abs(p[i] - q[i]) > 1e-3) {
				return false;
			}
		}
	}
	break;
	case ONNX_TENSOR_TYPE_STRING:
	{
		const std::string* p = (const std::string*)a->data;
		const std::string* q = (const std::string*)b->data;
		for (size_t i = 0; i < a->ndata; ++i) {
			if (!p[i].empty() && !q[i].empty() && (p[i] != q[i])) {
				return false;
			}
		}
	}
	break;
	default:
		break;
	}
	return true;
}

void tensor_t::reinit(tensor_type_t type, const int* dims, int ndim)
{
	size_t n;

	if (ndim > 0) {
		this->ndim = 0;
	}
	if ((ndata > 0) && data) {
		delete_data(data, this->type);
		data = nullptr;
		ndata = 0;
	}
	this->type = type;
	if (type == ONNX_TENSOR_TYPE_UNDEFINED) {
		return;
	}
	if ((ndim > 0) && dims) {
		for (int i = 0; i < ndim; ++i) {
			if (dims[i] <= 0) {
				return;
			}
		}
		strides.resize(ndim);
		strides[ndim - 1] = 1;
		for (int i = ndim - 2; i >= 0; i--) {
			strides[i] = dims[i + 1] * strides[i + 1];
		}
		this->dims.assign(dims, dims+ndim);
		this->ndim = ndim;
		n = multiply_accumulate(&dims[0], &dims[ndim], 1);
	}else {
		n = 1;
	}
	switch (type) {
	case ONNX_TENSOR_TYPE_UNDEFINED: break;
	case ONNX_TENSOR_TYPE_BOOL: data = new bool[n]; break;
	case ONNX_TENSOR_TYPE_INT8: data = new int8_t[n]; break;
	case ONNX_TENSOR_TYPE_INT16: data = new int16_t[n]; break;
	case ONNX_TENSOR_TYPE_INT32: data = new int32_t[n]; break;
	case ONNX_TENSOR_TYPE_INT64: data = new int64_t[n]; break;
	case ONNX_TENSOR_TYPE_UINT8: data = new uint8_t[n]; break;
	case ONNX_TENSOR_TYPE_UINT16: data = new uint16_t[n]; break;
	case ONNX_TENSOR_TYPE_UINT32: data = new uint32_t[n]; break;
	case ONNX_TENSOR_TYPE_UINT64: data = new uint64_t[n]; break;
	case ONNX_TENSOR_TYPE_BFLOAT16: data = new uint16_t[n]; break;
	case ONNX_TENSOR_TYPE_FLOAT16: data = new uint16_t[n]; break;
	case ONNX_TENSOR_TYPE_FLOAT32: data = new float[n]; break;
	case ONNX_TENSOR_TYPE_FLOAT64: data = new double[n]; break;
	case ONNX_TENSOR_TYPE_COMPLEX64: data = new std::complex<float>[n]; break;
	case ONNX_TENSOR_TYPE_COMPLEX128: data = new std::complex<double>[n]; break;
	case ONNX_TENSOR_TYPE_STRING: data = new std::string[n]; break;
	}
	ndata = n;
}

void tensor_t::apply(const void* buf, size_t len)
{
	if (!data || !buf || (len == 0)) {
		return;
	}
	int sz = tensor_type_sizeof(type);
	if (sz <= 0) {
		return;
	}
	if (type == ONNX_TENSOR_TYPE_STRING) {
		std::string* p = (std::string*)data;
		std::string* q = (std::string*)buf;
		for (size_t idx = 0; idx < ndata; ++idx) {
			p[idx].clear();
		}
		size_t l = min(ndata, (size_t)len);
		for (size_t idx = 0; idx < l; ++idx) {
			p[idx] = q[idx];
		}
	}else {
		size_t l = ndata * sz;
		if (l > 0) {
			memcpy(data, buf, min(l, len));
		}
	}
}

Onnx__AttributeProto* operator_t::find_attribute(std::string_view name)
{
	if (name.empty()) {
		return nullptr;
	}
	for (int i = 0; i < proto->n_attribute; ++i) {
		Onnx__AttributeProto* attr = proto->attribute[i];
		if (attr->name == name) {
			return attr;
		}
	}
	return nullptr;
}

float operator_t::attribute(std::string_view name, float def)
{
	Onnx__AttributeProto* attr = find_attribute(name);

	if (attr && (attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOAT)) {
		return attr->f;
	}
	return def;
}

int32_t operator_t::attribute(std::string_view name, int32_t def)
{
	return (int32_t)attribute(name, (int64_t)def);
}

int64_t operator_t::attribute(std::string_view name, int64_t def)
{
	Onnx__AttributeProto* attr = find_attribute(name);

	if (attr && (attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INT)) {
		return attr->i;
	}
	return def;
}

std::string_view operator_t::attribute(std::string_view name, std::string_view def)
{
	Onnx__AttributeProto* attr = find_attribute(name);

	if (attr && (attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__STRING)) {
		if (attr->s.len > 0) {
			attr->s.data[attr->s.len] = 0;
			return (char*)attr->s.data;
		}
	}
	return def;
}

int operator_t::attribute(std::string_view name, float*& floats)
{
	Onnx__AttributeProto* attr = find_attribute(name);

	if (attr && (attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOATS)) {
		floats = attr->floats;
		return attr->n_floats;
	}
	return 0;
}

int operator_t::attribute(std::string_view name, int64_t*& ints)
{
	Onnx__AttributeProto* attr = find_attribute(name);

	if (attr && (attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INTS)) {
		ints = attr->ints;
		return attr->n_ints;
	}
	return 0;
}

int operator_t::attribute(std::string_view name, tensor_t* t)
{
	Onnx__AttributeProto* attr = find_attribute(name);
	int ndim = 0;

	if (attr && (attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TENSOR)) {
		if (attr->t) {
			std::vector<int> dims;
			if (attr->t->n_dims > 0) {
				dims.resize(attr->t->n_dims);
				for (int i = 0; i < attr->t->n_dims; ++i) {
					dims[i] = attr->t->dims[i];
				}
				ndim = attr->t->n_dims;
			}
			if ((t->ndim != ndim) || (memcmp(&t->dims[0], &dims[0], sizeof(int) * ndim) != 0) || (t->type != (tensor_type_t)attr->t->data_type)) {
				t->reinit((tensor_type_t)attr->t->data_type, &dims[0], ndim);
			}
			tensor_copy_from_tensor_proto(t, attr->t);
			return 1;
		}
	}
	return 0;
}

Onnx__GraphProto* operator_t::attribute(std::string_view name, Onnx__GraphProto* def)
{
	Onnx__AttributeProto* attr = find_attribute(name);

	if (attr && (attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__GRAPH)) {
		if (attr->g) {
			return attr->g;
		}
	}
	return def;
}

Onnx__SparseTensorProto* operator_t::attribute(std::string_view name, Onnx__SparseTensorProto* def)
{
	Onnx__AttributeProto* attr = find_attribute(name);

	if (attr && (attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__SPARSE_TENSOR)) {
		if (attr->sparse_tensor) {
			return attr->sparse_tensor;
		}
	}
	return def;
}

void tensor_t::dump(int detail) const
{
	ONNX_LOG("%s: %s", name.c_str(), tensor_type_tostring(type).data());
	if (ndim > 0) {
		ONNX_LOG("[");
		for (int i = 0; i < ndim; ++i) {
			ONNX_LOG("%d", dims[i]);
			if (i != ndim - 1) {
				ONNX_LOG(" x ");
			}
		}
		ONNX_LOG("]");
		if (detail) {
			ONNX_LOG(" = \r\n");
			for (int i = 0; i < ndim; ++i) {
				if (dims[i] <= 0) {
					return;
				}
			}
			std::vector<int> sizes(ndim);
			std::vector<int> levels(ndim);
			sizes[ndim - 1] = dims[ndim - 1];
			levels[ndim - 1] = 0;
			std::vector<char> lbuf(ndim + 1);
			std::vector<char> rbuf(ndim + 1);
			char* lp = &lbuf[0];
			char* rp = &rbuf[0];
			for (int i = ndim - 2; i >= 0; i--) {
				sizes[i] = dims[i] * sizes[i + 1];
				levels[i] = 0;
			}
			for (size_t idx = 0; idx < ndata; ++idx) {
				for (int j = 0; j < ndim; ++j) {
					if ((idx % sizes[j]) == 0) {
						levels[j]++;
					}
					if (levels[j] == 1) {
						*lp++ = '[';
						levels[j]++;
					}
					if (levels[j] == 3) {
						*rp++ = ']';
						if ((j != 0) && (levels[j] > levels[j - 1])) {
							*lp++ = '[';
							levels[j] = 2;
						}else {
							levels[j] = 0;
						}
					}
				}
				*lp = *rp = '\0';
				ONNX_LOG("%s", &rbuf[0]);
				if (rbuf[0] != '\0') {
					ONNX_LOG("\r\n");
					for (int k = ndim - strlen(&rbuf[0]); k > 0; k--) {
						ONNX_LOG(" ");
					}
				}
				ONNX_LOG("%s", &lbuf[0]);
				if (lbuf[0] == '\0') {
					ONNX_LOG(" ");
				}
				void* p = (void*)((char*)data + tensor_type_sizeof(type) * idx);
				switch (type) {
				case ONNX_TENSOR_TYPE_BOOL:
					ONNX_LOG("%s,", *((uint8_t*)p) ? "true" : "false");
					break;
				case ONNX_TENSOR_TYPE_INT8:
					ONNX_LOG("%d,", *((int8_t*)p));
					break;
				case ONNX_TENSOR_TYPE_INT16:
					ONNX_LOG("%d,", *((int16_t*)p));
					break;
				case ONNX_TENSOR_TYPE_INT32:
					ONNX_LOG("%d,", *((int32_t*)p));
					break;
				case ONNX_TENSOR_TYPE_INT64:
					ONNX_LOG("%" PRId64 ",", *((int64_t*)p));
					break;
				case ONNX_TENSOR_TYPE_UINT8:
					ONNX_LOG("%u,", *((uint8_t*)p));
					break;
				case ONNX_TENSOR_TYPE_UINT16:
					ONNX_LOG("%u,", *((uint16_t*)p));
					break;
				case ONNX_TENSOR_TYPE_UINT32:
					ONNX_LOG("%u,", *((uint32_t*)p));
					break;
				case ONNX_TENSOR_TYPE_UINT64:
					ONNX_LOG("%" PRIu64 ",", *((uint64_t*)p));
					break;
				case ONNX_TENSOR_TYPE_BFLOAT16:
					ONNX_LOG("%g,", bfloat16_to_float32(*((uint16_t*)p)));
					break;
				case ONNX_TENSOR_TYPE_FLOAT16:
					ONNX_LOG("%g,", float16_to_float32(*((uint16_t*)p)));
					break;
				case ONNX_TENSOR_TYPE_FLOAT32:
					ONNX_LOG("%g,", *((float*)p));
					break;
				case ONNX_TENSOR_TYPE_FLOAT64:
					ONNX_LOG("%g,", *((double*)p));
					break;
				case ONNX_TENSOR_TYPE_COMPLEX64:
					ONNX_LOG("%g + %gi,", *((float*)p), *((float*)((char*)p + sizeof(float))));
					break;
				case ONNX_TENSOR_TYPE_COMPLEX128:
					ONNX_LOG("%g + %gi,", *((double*)p), *((double*)((char*)p + sizeof(double))));
					break;
				case ONNX_TENSOR_TYPE_STRING:
					ONNX_LOG("%s", (*(std::string*)p).c_str());
					break;
				default:
					ONNX_LOG("?,");
					break;
				}
				lp = &lbuf[0];
				rp = &rbuf[0];
			}
			for (int j = 0; j < ndim; ++j) {
				ONNX_LOG("]");
			}
			ONNX_LOG("\r\n");
		}else {
			ONNX_LOG(" = ");
			ONNX_LOG("[...]");
			ONNX_LOG("\r\n");
		}
	}else if (ndata == 1) {
		ONNX_LOG(" = ");
		void* p = data;
		switch (type) {
		case ONNX_TENSOR_TYPE_BOOL:
			ONNX_LOG("%s", *((uint8_t*)p) ? "true" : "false");
			break;
		case ONNX_TENSOR_TYPE_INT8:
			ONNX_LOG("%d", *((int8_t*)p));
			break;
		case ONNX_TENSOR_TYPE_INT16:
			ONNX_LOG("%d", *((int16_t*)p));
			break;
		case ONNX_TENSOR_TYPE_INT32:
			ONNX_LOG("%d", *((int32_t*)p));
			break;
		case ONNX_TENSOR_TYPE_INT64:
			ONNX_LOG("%" PRId64, *((int64_t*)p));
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			ONNX_LOG("%u", *((uint8_t*)p));
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			ONNX_LOG("%u", *((uint16_t*)p));
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			ONNX_LOG("%u", *((uint32_t*)p));
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			ONNX_LOG("%" PRIu64, *((uint64_t*)p));
			break;
		case ONNX_TENSOR_TYPE_BFLOAT16:
			ONNX_LOG("%g", bfloat16_to_float32(*((uint16_t*)p)));
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			ONNX_LOG("%g", float16_to_float32(*((uint16_t*)p)));
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			ONNX_LOG("%g", *((float*)p));
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			ONNX_LOG("%g", *((double*)p));
			break;
		case ONNX_TENSOR_TYPE_COMPLEX64:
			ONNX_LOG("%g + %gi", *((float*)p), *((float*)((char*)p + sizeof(float))));
			break;
		case ONNX_TENSOR_TYPE_COMPLEX128:
			ONNX_LOG("%g + %gi", *((double*)p), *((double*)((char*)p + sizeof(double))));
			break;
		case ONNX_TENSOR_TYPE_STRING:
			ONNX_LOG("%s", (*(std::string*)p).c_str());
			break;
		default:
			ONNX_LOG("?");
			break;
		}
		ONNX_LOG("\r\n");
	}else {
		ONNX_LOG(" = ");
		ONNX_LOG("null");
		ONNX_LOG("\r\n");
	}
}

void operator_t::dump(int detail) const
{
	ONNX_LOG("%s: %s-%d (%s)\r\n", proto->name, proto->op_type, opset, (strlen(proto->domain) > 0) ? proto->domain : "ai.onnx");
	if (inputs.size() > 0) {
		ONNX_LOG("\tInputs:\r\n");
		for (size_t i = 0; i < inputs.size(); ++i) {
			ONNX_LOG("\t\t");
			inputs[i]->dump(detail);
		}
	}
	if (outputs.size() > 0) {
		ONNX_LOG("\tOutputs:\r\n");
		for (size_t i = 0; i < outputs.size(); ++i) {
			ONNX_LOG("\t\t");
			outputs[i]->dump(detail);
		}
	}
}

void graph_t::dump(int detail) const
{
	for (int i = 0; i < nodes.size(); ++i) {
		nodes[i]->dump(detail);
	}
}

bool context_t::alloc(const void* buf, size_t len, resolver_t** r, int rlen)
{
	model = onnx__model_proto__unpack(nullptr, len, (const uint8_t*)buf);
	if (!model) {
		return false;
	}
	resolvers.resize(rlen);
	rctx.resize(rlen);

	for (int i = 0; i < rlen; ++i) {
		resolvers[i] = r[i];
		if (r[i]) {
			rctx[i] = r[i]->create();
		}
	}
	graph.reset(new graph_t(this, model->graph));
	return true;
}

void context_t::dump(int detail) const
{
	if (model) {
		ONNX_LOG("IR Version: v%" PRId64 "\r\n", model->ir_version);
		ONNX_LOG("Producer: %s %s\r\n", model->producer_name, model->producer_version);
		ONNX_LOG("Domain: %s\r\n", model->domain);
		ONNX_LOG("Imports:\r\n");
		for (int i = 0; i < model->n_opset_import; ++i) {
			ONNX_LOG("\t%s v%" PRId64 "\r\n", (strlen(model->opset_import[i]->domain) > 0) ? model->opset_import[i]->domain : "ai.onnx", model->opset_import[i]->version);
		}
	}
	if (graph) {
		graph->dump(detail);
	}
}

void context_t::run()
{
	for (size_t i = 0; i < graph->nodes.size(); ++i) {
		operator_t* n = graph->nodes[i];
		if (n->reshape()) {
			n->exec();
		}
	}
}

bool tensor_t::reshape(const int* dims, int ndim, tensor_type_t type)
{
	if ((this->ndim != ndim) || (dims && (memcmp(&this->dims[0], dims, sizeof(int) * ndim) != 0)) || (this->type != type)) {
		reinit(type, dims, ndim);
	}
	return true;
}

bool tensor_t::reshape_identity(const tensor_t* x, tensor_type_t type)
{
	if ((this->ndim != x->ndim) || (memcmp(&this->dims[0], &x->dims[0], sizeof(int) * this->ndim) != 0) || (this->type != type)) {
		reinit(type, &x->dims[0], x->ndim);
	}
	return true;
}

bool tensor_t::reshape_multi_broadcast(const tensor_t* a, const tensor_t* b, tensor_type_t type)
{
	const int ndim = max(a->ndim, b->ndim);
	std::vector<int> dims(ndim);
	if (ndim > 0) {
		for (int i = a->ndim - 1, j = b->ndim - 1, k = ndim - 1; k >= 0; k--) {
			if (i < 0) {
				dims[k] = b->dims[j--];
			}else if (j < 0) {
				dims[k] = a->dims[i--];
			}else {
				if (a->dims[i] == b->dims[j]) {
					dims[k] = a->dims[i];
				}else if ((a->dims[i] == 1) || (b->dims[j] == 1)) {
					dims[k] = (a->dims[i] > b->dims[j]) ? a->dims[i] : b->dims[j];
				}else {
					return false;
				}
				i--;
				j--;
			}
		}
	}
	if ((this->type != type) || (this->ndim != ndim) || (memcmp(&this->dims[0], &dims[0], sizeof(int) * ndim) != 0)) {
		reinit(type, &dims[0], ndim);
	}
	return true;
}

void* tensor_t::broadcast_map_address(const tensor_t* y, int offset)
{
	int xndim = this->ndim;
	int yndim = y->ndim;

	if ((xndim > 0) && (yndim > 0)) {
		int dndim = yndim - xndim;
		std::vector<int> ix(xndim);
		std::vector<int> iy(yndim);
		int i;

		y->offset_to_indices(offset, &iy[0]);
		for (i = 0; i < xndim; ++i) {
			ix[i] = iy[dndim + i] % this->dims[i];
		}
		return (char*)this->data + this->indices_to_offset(&ix[0]) * tensor_type_sizeof(this);
	}
	return this->data;
}

} // namespace onnx
