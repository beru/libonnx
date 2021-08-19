/*
 * onnx.c
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

#include <onnx.h>
#include <default/default.h>

#define ONNX_LOG(...)	printf(__VA_ARGS__)

onnx_context_t::onnx_context_t(const void* buf, size_t len, onnx_resolver_t** r, int rlen)
{
	model = onnx__model_proto__unpack(NULL, len, (const uint8_t*)buf);
	assert(model);

	resolvers.resize(rlen);
	rctx.resize(rlen);

	for (int i = 0; i < rlen; i++) {
		resolvers[i] = r[i];
		if (r[i] && r[i]->create)
			rctx[i] = r[i]->create();
	}

	graph = new onnx_graph_t(this, model->graph);
}

onnx_context_t::onnx_context_t(const char* filename, onnx_resolver_t** r, int rlen)
{
	FILE* fp;
	void* buf;
	size_t l, len;

	fp = fopen(filename, "rb");
	if (fp) {
		fseek(fp, 0L, SEEK_END);
		l = ftell(fp);
		fseek(fp, 0L, SEEK_SET);
		if (l > 0) {
			buf = malloc(l);
			if (buf) {
				for (len = 0; len < l; len += fread((char*)buf + len, 1, l - len, fp));
				onnx_context_t::onnx_context_t(buf, len, r, rlen);
				free(buf);
			}
		}
		fclose(fp);
	}
}

onnx_context_t::~onnx_context_t()
{
	delete graph;
	for (size_t i = 0; i < resolvers.size(); i++) {
		if (resolvers[i] && resolvers[i]->destroy)
			resolvers[i]->destroy(rctx[i]);
	}
	for (auto it = map.begin(); it != map.end(); ++it) {
		delete it->second;
	}
	if (model)
		onnx__model_proto__free_unpacked(model, NULL);
}

static onnx_tensor_t* onnx_tensor_alloc_from_value_info(Onnx__ValueInfoProto* v)
{
	onnx_tensor_t* t;
	onnx_tensor_type_t type;
	int* dims = NULL;
	int ndim;
	int i;

	if (!v || !v->name)
		return NULL;

	switch (v->type->value_case) {
	case ONNX__TYPE_PROTO__VALUE_TENSOR_TYPE:
		type = (onnx_tensor_type_t)v->type->tensor_type->elem_type;
		ndim = v->type->tensor_type->shape->n_dim;
		if (ndim > 0) {
			dims = (int*)malloc(sizeof(int) * ndim);
			if (dims) {
				for (i = 0; i < ndim; i++) {
					switch (v->type->tensor_type->shape->dim[i]->value_case) {
					case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_VALUE:
						dims[i] = v->type->tensor_type->shape->dim[i]->dim_value;
						break;
					case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_PARAM:
						if (strcmp(v->type->tensor_type->shape->dim[i]->dim_param, "batch_size") == 0)
							dims[i] = 1;
						else
							dims[i] = 1;
						break;
					default:
						dims[i] = 1;
						break;
					}
				}
			}
		}
		t = new onnx_tensor_t(v->name, type, dims, ndim);
		if (dims)
			free(dims);
		break;
	case ONNX__TYPE_PROTO__VALUE_SEQUENCE_TYPE:
		t = NULL;
		break;
	case ONNX__TYPE_PROTO__VALUE_MAP_TYPE:
		t = NULL;
		break;
	default:
		t = NULL;
		break;
	}
	return t;
}

static void onnx_tensor_copy_from_tensor_proto(onnx_tensor_t* t, Onnx__TensorProto* o)
{
	size_t n, i;
	int sz;

	if (t && o) {
		if (t->type == o->data_type) {
			sz = onnx_tensor_type_sizeof(t->type);
			if (sz > 0) {
				if ((o->raw_data.len > 0) && o->raw_data.data) {
					switch (o->data_type) {
					case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
					{
						float* p = (float*)t->datas;
						uint32_t* q = (uint32_t*)o->raw_data.data;
						union { uint32_t u; float f; } v;
						if (t->ndata > 0) {
							n = min(t->ndata, (size_t)o->raw_data.len / sz);
							for (i = 0; i < n; i++) {
								v.u = le32_to_cpu(q[i]);
								p[i] = v.f;
							}
						}
					}
					break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8:
					{
						uint8_t* p = (uint8_t*)t->datas;
						uint8_t* q = (uint8_t*)o->raw_data.data;
						if (t->ndata > 0) {
							n = min(t->ndata, (size_t)o->raw_data.len);
							memcpy(p, q, n);
						}
					}
					break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__INT8:
					{
						int8_t* p = (int8_t*)t->datas;
						int8_t* q = (int8_t*)o->raw_data.data;
						if (t->ndata > 0) {
							n = min(t->ndata, (size_t)o->raw_data.len);
							memcpy(p, q, n);
						}
					}
					break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__UINT16:
					{
						uint16_t* p = (uint16_t*)t->datas;
						uint16_t* q = (uint16_t*)o->raw_data.data;
						if (t->ndata > 0) {
							n = min(t->ndata, (size_t)o->raw_data.len / sz);
							for (i = 0; i < n; i++)
								p[i] = le16_to_cpu(q[i]);
						}
					}
					break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__INT16:
					{
						int16_t* p = (int16_t*)t->datas;
						int16_t* q = (int16_t*)o->raw_data.data;
						if (t->ndata > 0) {
							n = min(t->ndata, (size_t)o->raw_data.len / sz);
							for (i = 0; i < n; i++)
								p[i] = le16_to_cpu(q[i]);
						}
					}
					break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
					{
						int32_t* p = (int32_t*)t->datas;
						int32_t* q = (int32_t*)o->raw_data.data;
						if (t->ndata > 0) {
							n = min(t->ndata, (size_t)o->raw_data.len / sz);
							for (i = 0; i < n; i++)
								p[i] = le32_to_cpu(q[i]);
						}
					}
					break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
					{
						int64_t* p = (int64_t*)t->datas;
						int64_t* q = (int64_t*)o->raw_data.data;
						if (t->ndata > 0) {
							n = min(t->ndata, (size_t)o->raw_data.len / sz);
							for (i = 0; i < n; i++)
								p[i] = le64_to_cpu(q[i]);
						}
					}
					break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__STRING:
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL:
					{
						uint8_t* p = (uint8_t*)t->datas;
						uint8_t* q = (uint8_t*)o->raw_data.data;
						if (t->ndata > 0) {
							n = min(t->ndata, (size_t)o->raw_data.len);
							memcpy(p, q, n);
						}
					}
					break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16:
					{
						uint16_t* p = (uint16_t*)t->datas;
						uint16_t* q = (uint16_t*)o->raw_data.data;
						if (t->ndata > 0) {
							n = min(t->ndata, (size_t)o->raw_data.len / sz);
							for (i = 0; i < n; i++)
								p[i] = le16_to_cpu(q[i]);
						}
					}
					break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
					{
						double* p = (double*)t->datas;
						uint64_t* q = (uint64_t*)o->raw_data.data;
						union { uint64_t u; double f; } v;
						if (t->ndata > 0) {
							n = min(t->ndata, (size_t)o->raw_data.len / sz);
							for (i = 0; i < n; i++) {
								v.u = le64_to_cpu(q[i]);
								p[i] = v.f;
							}
						}
					}
					break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__UINT32:
					{
						uint32_t* p = (uint32_t*)t->datas;
						uint32_t* q = (uint32_t*)o->raw_data.data;
						if (t->ndata > 0) {
							n = min(t->ndata, (size_t)o->raw_data.len / sz);
							for (i = 0; i < n; i++)
								p[i] = le32_to_cpu(q[i]);
						}
					}
					break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__UINT64:
					{
						uint64_t* p = (uint64_t*)t->datas;
						uint64_t* q = (uint64_t*)o->raw_data.data;
						if (t->ndata > 0) {
							n = min(t->ndata, (size_t)o->raw_data.len / sz);
							for (i = 0; i < n; i++)
								p[i] = le64_to_cpu(q[i]);
						}
					}
					break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64:
					{
						float* p = (float*)t->datas;
						uint32_t* q = (uint32_t*)o->raw_data.data;
						union { uint32_t u; float f; } v;
						if (t->ndata > 0) {
							n = min(t->ndata, (size_t)o->raw_data.len / sz) * 2;
							for (i = 0; i < n; i++) {
								v.u = le32_to_cpu(q[i]);
								p[i] = v.f;
							}
						}
					}
					break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128:
					{
						double* p = (double*)t->datas;
						uint64_t* q = (uint64_t*)o->raw_data.data;
						union { uint64_t u; double f; } v;
						if (t->ndata > 0) {
							n = min(t->ndata, (size_t)o->raw_data.len / sz) * 2;
							for (i = 0; i < n; i++) {
								v.u = le64_to_cpu(q[i]);
								p[i] = v.f;
							}
						}
					}
					break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16:
					{
						uint16_t* p = (uint16_t*)t->datas;
						uint16_t* q = (uint16_t*)o->raw_data.data;
						if (t->ndata > 0) {
							n = min(t->ndata, (size_t)o->raw_data.len / sz);
							for (i = 0; i < n; i++)
								p[i] = le16_to_cpu(q[i]);
						}
					}
					break;
					default:
						break;
					}
				}else {
					switch (o->data_type) {
					case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
						n = min(t->ndata, (size_t)o->n_float_data);
						if ((n > 0) && t->datas && o->float_data)
							memcpy(t->datas, o->float_data, sizeof(float) * n);
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8:
					case ONNX__TENSOR_PROTO__DATA_TYPE__INT8:
					case ONNX__TENSOR_PROTO__DATA_TYPE__UINT16:
					case ONNX__TENSOR_PROTO__DATA_TYPE__INT16:
					case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
					case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL:
					case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16:
					case ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16:
						//TODO
						n = min(t->ndata, (size_t)o->n_int32_data);
						if ((n > 0) && t->datas && o->int32_data)
							memcpy(t->datas, o->int32_data, sz * n);
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__STRING:
						n = min(t->ndata, (size_t)o->n_string_data);
						if ((n > 0) && t->datas && o->string_data) {
							char** str = (char**)t->datas;
							for (i = 0; i < t->ndata; i++) {
								if (str[i]) {
									free(str[i]);
									str[i] = NULL;
								}
							}
							for (i = 0; i < n; i++) {
								str[i] = (char*)malloc(o->string_data[i].len + 1);
								if (str[i]) {
									str[i][o->string_data[i].len] = 0;
									memcpy(str[i], o->string_data[i].data, o->string_data[i].len);
								}
							}
						}
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
						n = min(t->ndata, (size_t)o->n_int64_data);
						if ((n > 0) && t->datas && o->int64_data)
							memcpy(t->datas, o->int64_data, sizeof(int64_t) * n);
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
						n = min(t->ndata, (size_t)o->n_double_data);
						if ((n > 0) && t->datas && o->double_data)
							memcpy(t->datas, o->double_data, sizeof(double) * n);
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__UINT32:
					case ONNX__TENSOR_PROTO__DATA_TYPE__UINT64:
						//TODO
						n = min(t->ndata, (size_t)o->n_uint64_data);
						if ((n > 0) && t->datas && o->uint64_data)
							memcpy(t->datas, o->uint64_data, sz * n);
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64:
						n = min(t->ndata, (size_t)(o->n_float_data / 2));
						if ((n > 0) && t->datas && o->float_data)
							memcpy(t->datas, o->float_data, sizeof(float) * 2 * n);
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128:
						n = min(t->ndata, (size_t)(o->n_double_data / 2));
						if ((n > 0) && t->datas && o->double_data)
							memcpy(t->datas, o->double_data, sizeof(double) * 2 * n);
						break;
					default:
						break;
					}
				}
			}
		}
	}
}

static int reshape_dummy(onnx_node_t* n)
{
	return 1;
}

static void operator_dummy(onnx_node_t* n)
{
	ONNX_LOG("\033[45;37mUnsupported opset\033[0m => %s-%d (%s)\r\n", n->proto->op_type, n->opset, (strlen(n->proto->domain) > 0) ? n->proto->domain : "ai.onnx");
}

static void resolver_solve_operator(onnx_resolver_t* r, onnx_node_t* n)
{
	if (r && n) {
		auto it = r->op_map.find(n->proto->op_type);
		if (it != r->op_map.end()) {
			it->second(n);
		}
	}
}

onnx_graph_t::onnx_graph_t(onnx_context_t* ctx, Onnx__GraphProto* graph)
{
	onnx_node_t* n;
	onnx_tensor_t* t;
	Onnx__TensorProto* o;
	Onnx__ValueInfoProto* v;
	const char* p;
	const char* domain;
	char* name;
	int i, j, k, l;

	assert(graph);

	memset(this, 0, sizeof(onnx_graph_t));

	nodes.resize(graph->n_node);

	for (i = 0; i < graph->n_input; i++) {
		v = graph->input[i];
		if (!ctx->tensor_search(v->name)) {
			t = onnx_tensor_alloc_from_value_info(v);
			if (t) {
				for (j = 0; j < graph->n_initializer; j++) {
					if (graph->initializer[j]->name == t->name) {
						onnx_tensor_copy_from_tensor_proto(t, graph->initializer[j]);
						break;
					}
				}
				ctx->map[t->name.c_str()] = t;
			}
		}
	}

	for (i = 0; i < graph->n_output; i++) {
		v = graph->output[i];
		if (!ctx->tensor_search(v->name)) {
			t = onnx_tensor_alloc_from_value_info(v);
			if (t)
				ctx->map[t->name.c_str()] = t;
		}
	}

	for (i = 0; i < graph->n_value_info; i++) {
		v = graph->value_info[i];
		if (!ctx->tensor_search(v->name)) {
			t = onnx_tensor_alloc_from_value_info(v);
			if (t)
				ctx->map[t->name.c_str()] = t;
		}
	}

	for (i = 0; i < graph->n_node; i++) {
		for (j = 0; j < graph->node[i]->n_output; j++) {
			name = graph->node[i]->output[j];
			if (!ctx->tensor_search(name)) {
				t = new onnx_tensor_t(name, ONNX_TENSOR_TYPE_UNDEFINED, NULL, 0);
				ctx->map[name] = t;
			}
		}
	}

	for (i = 0; i < graph->n_node; i++) {
		for (j = 0; j < graph->node[i]->n_input; j++) {
			name = graph->node[i]->input[j];
			if (!ctx->tensor_search(name)) {
				for (k = 0; k < graph->n_initializer; k++) {
					if (strcmp(graph->initializer[k]->name, name) == 0) {
						o = graph->initializer[k];
						if (o) {
							int ndim = o->n_dims;
							std::vector<int> dims(ndim);
							for (l = 0; l < ndim; l++)
								dims[l] = o->dims[l];
							t = new onnx_tensor_t(name, (onnx_tensor_type_t)o->data_type, &dims[0], ndim);
							onnx_tensor_copy_from_tensor_proto(t, o);
							ctx->map[name] = t;
							break;
						}
					}
				}
				assert(ctx->tensor_search(name));
			}
		}
	}

	for (i = 0; i < nodes.size(); i++) {
		n = &nodes[i];
		memset(n, 0, sizeof(onnx_node_t));

		n->ctx = ctx;
		n->proto = graph->node[i];
		domain = n->proto->domain;
		if (!domain || (strlen(domain) == 0))
			domain = "ai.onnx";
		for (j = 0; j < ctx->model->n_opset_import; j++) {
			p = ctx->model->opset_import[j]->domain;
			if (!p || (strlen(p) == 0))
				p = "ai.onnx";
			if (strcmp(domain, p) == 0) {
				n->opset = ctx->model->opset_import[j]->version;
				break;
			}
		}
		if (n->proto->n_input > 0) {
			n->inputs.resize(n->proto->n_input);
			for (j = 0; j < n->inputs.size(); j++)
				n->inputs[j] = ctx->tensor_search(n->proto->input[j]);
		}
		if (n->proto->n_output > 0) {
			n->outputs.resize(n->proto->n_output);
			for (j = 0; j < n->outputs.size(); j++)
				n->outputs[j] = ctx->tensor_search(n->proto->output[j]);
		}
		for (j = 0; j < ctx->resolvers.size(); j++) {
			resolver_solve_operator(ctx->resolvers[j], n);
			if (n->ope) {
				n->r = ctx->resolvers[j];
				n->rctx = ctx->rctx[j];
				break;
			}
		}
		if (!n->ope) {
			resolver_solve_operator(&resolver_default, n);
			if (n->ope) {
				n->r = &resolver_default;
				n->rctx = NULL;
			}
		}
		if (!n->reshape)
			n->reshape = reshape_dummy;
		if (!n->ope)
			n->ope = operator_dummy;
		if (n->init) {
			if (n->init(n) <= 0) {
				for (j = 0; j < nodes.size(); j++) {
					n = &nodes[j];
					if (n->exit)
						n->exit(n);
				}
				return;
			}
		}
		if (n->reshape)
			n->reshape(n);
	}
}

onnx_graph_t::~onnx_graph_t()
{
	for (size_t i = 0; i < nodes.size(); i++) {
		onnx_node_t* n = &nodes[i];
		if (n->exit)
			n->exit(n);
	}
}

const char* onnx_tensor_type_tostring(onnx_tensor_type_t type)
{
	static const char* typestr[17] = {
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
	if ((type > 0) && (type < (sizeof(typestr) / sizeof((typestr)[0]))))
		return typestr[type];
	return typestr[0];
}

int onnx_tensor_type_sizeof(onnx_tensor_type_t type)
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
		sizeof(char*),
		sizeof(uint8_t),
		sizeof(uint16_t),
		sizeof(double),
		sizeof(uint32_t),
		sizeof(uint64_t),
		sizeof(float) * 2,
		sizeof(double) * 2,
		sizeof(char*),
	};
	if ((type > 0) && (type < (sizeof(typesz) / sizeof((typesz)[0]))))
		return typesz[type];
	return typesz[0];
}

onnx_tensor_t* onnx_context_t::tensor_search(const char* name)
{
	auto it = map.find(name);
	return (it == map.end()) ? nullptr : it->second;
}

onnx_tensor_t::onnx_tensor_t(const char* name, onnx_tensor_type_t type, int* dims, int ndim)
{
	this->name = name;
	reinit(type, dims, ndim);
}

onnx_tensor_t* onnx_tensor_alloc_from_file(const char* filename)
{
	onnx_tensor_t* t = NULL;
	Onnx__TensorProto* pb;
	FILE* fp;
	void* buf;
	size_t l, len;
	int* dims = NULL;
	int ndim = 0;
	int i;

	fp = fopen(filename, "rb");
	if (fp) {
		fseek(fp, 0L, SEEK_END);
		l = ftell(fp);
		fseek(fp, 0L, SEEK_SET);
		if (l > 0) {
			buf = malloc(l);
			if (buf) {
				for (len = 0; len < l; len += fread((char*)buf + len, 1, l - len, fp));
				pb = onnx__tensor_proto__unpack(NULL, len, (const uint8_t*)buf);
				free(buf);
				if (pb) {
					if (pb->n_dims > 0) {
						dims = (int*)malloc(sizeof(int) * pb->n_dims);
						if (dims) {
							for (i = 0; i < pb->n_dims; i++)
								dims[i] = pb->dims[i];
							ndim = pb->n_dims;
						}
					}
					t = new onnx_tensor_t(pb->name, (onnx_tensor_type_t)pb->data_type, dims, ndim);
					if (dims)
						free(dims);
					onnx_tensor_copy_from_tensor_proto(t, pb);
					onnx__tensor_proto__free_unpacked(pb, NULL);
				}
			}
		}
		fclose(fp);
	}
	return t;
}

onnx_tensor_t::~onnx_tensor_t()
{
	char** str;
	if (ndim > 0) {
		if (strides)
			free(strides);
		if (dims)
			free(dims);
	}
	if ((ndata > 0) && datas) {
		if (type == ONNX_TENSOR_TYPE_STRING) {
			str = (char**)datas;
			for (size_t idx = 0; idx < ndata; idx++) {
				if (str[idx])
					free(str[idx]);
			}
		}
		free(datas);
	}
}

int onnx_tensor_equal(const onnx_tensor_t* a, const onnx_tensor_t* b)
{
	size_t i;

	if (!a || !b)
		return 0;
	if (a->type != b->type)
		return 0;
	if (a->ndim != b->ndim)
		return 0;
	if (a->ndata != b->ndata)
		return 0;
	if (a->ndim > 0) {
		if (memcmp(a->dims, b->dims, sizeof(int) * a->ndim) != 0)
			return 0;
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
		if (memcmp(a->datas, b->datas, a->ndata * onnx_tensor_type_sizeof(a->type)) != 0)
			return 0;
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
	{
		uint16_t* p = (uint16_t*)a->datas;
		uint16_t* q = (uint16_t*)b->datas;
		for (i = 0; i < a->ndata; i++) {
			if (fabsf(bfloat16_to_float32(p[i]) - bfloat16_to_float32(q[i])) > 1e-3)
				return 0;
		}
	}
	break;
	case ONNX_TENSOR_TYPE_FLOAT16:
	{
		uint16_t* p = (uint16_t*)a->datas;
		uint16_t* q = (uint16_t*)b->datas;
		for (i = 0; i < a->ndata; i++) {
			if (fabsf(float16_to_float32(p[i]) - float16_to_float32(q[i])) > 1e-3)
				return 0;
		}
	}
	break;
	case ONNX_TENSOR_TYPE_FLOAT32:
	{
		float* p = (float*)a->datas;
		float* q = (float*)b->datas;
		for (i = 0; i < a->ndata; i++) {
			if (fabsf(p[i] - q[i]) > 1e-3)
				return 0;
		}
	}
	break;
	case ONNX_TENSOR_TYPE_FLOAT64:
	{
		double* p = (double*)a->datas;
		double* q = (double*)b->datas;
		for (i = 0; i < a->ndata; i++) {
			if (fabs(p[i] - q[i]) > 1e-3)
				return 0;
		}
	}
	break;
	case ONNX_TENSOR_TYPE_COMPLEX64:
	{
		float* p = (float*)a->datas;
		float* q = (float*)b->datas;
		for (i = 0; i < a->ndata * 2; i++) {
			if (fabsf(p[i] - q[i]) > 1e-3)
				return 0;
		}
	}
	break;
	case ONNX_TENSOR_TYPE_COMPLEX128:
	{
		double* p = (double*)a->datas;
		double* q = (double*)b->datas;
		for (i = 0; i < a->ndata * 2; i++) {
			if (fabs(p[i] - q[i]) > 1e-3)
				return 0;
		}
	}
	break;
	case ONNX_TENSOR_TYPE_STRING:
	{
		char** p = (char**)a->datas;
		char** q = (char**)b->datas;
		for (i = 0; i < a->ndata; i++) {
			if (p[i] && q[i] && (strcmp(p[i], q[i]) != 0))
				return 0;
		}
	}
	break;
	default:
		break;
	}
	return 1;
}

void onnx_tensor_t::reinit(onnx_tensor_type_t type, const int* dims, int ndim)
{
	char** str;
	size_t n;
	int sz, i;

	if (ndim > 0) {
		if (strides) {
			free(strides);
			strides = NULL;
		}
		if (dims) {
			free(this->dims);
			this->dims = NULL;
		}
		this->ndim = 0;
	}
	if ((ndata > 0) && datas) {
		if (type == ONNX_TENSOR_TYPE_STRING) {
			str = (char**)datas;
			for (size_t idx = 0; idx < ndata; idx++) {
				if (str[idx]) {
					free(str[idx]);
					str[idx] = NULL;
				}
			}
		}
		free(datas);
		datas = NULL;
		ndata = 0;
	}
	this->type = type;
	if (type != ONNX_TENSOR_TYPE_UNDEFINED) {
		if ((ndim > 0) && dims) {
			for (i = 0; i < ndim; i++) {
				if (dims[i] <= 0)
					return;
			}
			strides = (int*)malloc(sizeof(int) * ndim);
			this->dims = (int*)malloc(sizeof(int) * ndim);
			strides[ndim - 1] = 1;
			for (i = ndim - 2; i >= 0; i--)
				strides[i] = dims[i + 1] * strides[i + 1];
			memcpy(this->dims, dims, sizeof(int) * ndim);
			this->ndim = ndim;
			for (i = 0, n = 1; i < ndim; i++)
				n *= dims[i];
			sz = onnx_tensor_type_sizeof(type);
			if (sz > 0) {
				datas = malloc(n * sz);
				if (datas) {
					memset(datas, 0, n * sz);
					ndata = n;
				}
			}
		}else {
			sz = onnx_tensor_type_sizeof(type);
			if (sz > 0) {
				datas = malloc(sz);
				if (datas) {
					memset(datas, 0, sz);
					ndata = 1;
				}
			}
		}
	}
}

void onnx_tensor_t::apply(const void* buf, size_t len)
{
	if (datas && buf && (len > 0)) {
		int sz = onnx_tensor_type_sizeof(type);
		if (sz > 0) {
			size_t l;
			if (type == ONNX_TENSOR_TYPE_STRING) {
				char** p = (char**)datas;
				char** q = (char**)buf;
				for (size_t idx = 0; idx < ndata; idx++) {
					if (p[idx]) {
						free(p[idx]);
						p[idx] = NULL;
					}
				}
				l = min(ndata, (size_t)len);
				for (size_t idx = 0; idx < l; idx++)
					p[idx] = strdup(q[idx]);
			}else {
				l = ndata * sz;
				if (l > 0)
					memcpy(datas, buf, min(l, len));
			}
		}
	}
}

Onnx__AttributeProto* onnx_node_t::search_attribute(const char* name)
{
	if (name) {
		for (int i = 0; i < proto->n_attribute; i++) {
			Onnx__AttributeProto* attr = proto->attribute[i];
			if (strcmp(attr->name, name) == 0)
				return attr;
		}
	}
	return NULL;
}

float onnx_node_t::attribute_read_float(const char* name, float def)
{
	Onnx__AttributeProto* attr = search_attribute(name);

	if (attr && (attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOAT))
		return attr->f;
	return def;
}

int64_t onnx_node_t::attribute_read_int(const char* name, int64_t def)
{
	Onnx__AttributeProto* attr = search_attribute(name);

	if (attr && (attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INT))
		return attr->i;
	return def;
}

const char* onnx_node_t::attribute_read_string(const char* name, const char* def)
{
	Onnx__AttributeProto* attr = search_attribute(name);

	if (attr && (attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__STRING)) {
		if (attr->s.len > 0) {
			attr->s.data[attr->s.len] = 0;
			return (char*)attr->s.data;
		}
	}
	return def;
}

int onnx_node_t::attribute_read_floats(const char* name, float** floats)
{
	Onnx__AttributeProto* attr = search_attribute(name);

	if (attr && (attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOATS)) {
		*floats = attr->floats;
		return attr->n_floats;
	}
	return 0;
}

int onnx_node_t::attribute_read_ints(const char* name, int64_t** ints)
{
	Onnx__AttributeProto* attr = search_attribute(name);

	if (attr && (attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INTS)) {
		*ints = attr->ints;
		return attr->n_ints;
	}
	return 0;
}

int onnx_node_t::attribute_read_tensor(const char* name, onnx_tensor_t* t)
{
	Onnx__AttributeProto* attr = search_attribute(name);
	int* dims = NULL;
	int ndim = 0;
	int i;

	if (attr && (attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TENSOR)) {
		if (attr->t) {
			if (attr->t->n_dims > 0) {
				dims = (int*)malloc(sizeof(int) * attr->t->n_dims);
				if (dims) {
					for (i = 0; i < attr->t->n_dims; i++)
						dims[i] = attr->t->dims[i];
					ndim = attr->t->n_dims;
				}
			}
			if ((t->ndim != ndim) || (memcmp(t->dims, dims, sizeof(int) * ndim) != 0) || (t->type != (onnx_tensor_type_t)attr->t->data_type))
				t->reinit((onnx_tensor_type_t)attr->t->data_type, dims, ndim);
			if ((ndim > 0) && dims)
				free(dims);
			onnx_tensor_copy_from_tensor_proto(t, attr->t);
			return 1;
		}
	}
	return 0;
}

Onnx__GraphProto* onnx_node_t::attribute_read_graph(const char* name, Onnx__GraphProto* def)
{
	Onnx__AttributeProto* attr = search_attribute(name);

	if (attr && (attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__GRAPH)) {
		if (attr->g)
			return attr->g;
	}
	return def;
}

Onnx__SparseTensorProto* onnx_node_t::attribute_read_sparse_tensor(const char* name, Onnx__SparseTensorProto* def)
{
	Onnx__AttributeProto* attr = search_attribute(name);

	if (attr && (attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__SPARSE_TENSOR)) {
		if (attr->sparse_tensor)
			return attr->sparse_tensor;
	}
	return def;
}

void onnx_tensor_t::dump(int detail) const
{
	int *sizes, *levels;
	char *lbuf, *rbuf;
	char *lp, *rp;
	void* p;
	int i, j, k;

	ONNX_LOG("%s: %s", name.c_str(), onnx_tensor_type_tostring(type));
	if (ndim > 0) {
		ONNX_LOG("[");
		for (i = 0; i < ndim; i++) {
			ONNX_LOG("%d", dims[i]);
			if (i != ndim - 1)
				ONNX_LOG(" x ");
		}
		ONNX_LOG("]");
		if (detail) {
			ONNX_LOG(" = \r\n");
			for (i = 0; i < ndim; i++) {
				if (dims[i] <= 0)
					return;
			}
			sizes = (int*)malloc(sizeof(int) * ndim);
			levels = (int*)malloc(sizeof(int) * ndim);
			sizes[ndim - 1] = dims[ndim - 1];
			levels[ndim - 1] = 0;
			lbuf = (char*)malloc(sizeof(char) * (ndim + 1));
			rbuf = (char*)malloc(sizeof(char) * (ndim + 1));
			lp = lbuf;
			rp = rbuf;
			for (i = ndim - 2; i >= 0; i--) {
				sizes[i] = dims[i] * sizes[i + 1];
				levels[i] = 0;
			}
			for (size_t idx = 0; idx < ndata; idx++) {
				for (j = 0; j < ndim; j++) {
					if ((idx % sizes[j]) == 0)
						levels[j]++;
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
				ONNX_LOG("%s", rbuf);
				if (*rbuf != '\0') {
					ONNX_LOG("\r\n");
					for (k = ndim - strlen(rbuf); k > 0; k--)
						ONNX_LOG(" ");
				}
				ONNX_LOG("%s", lbuf);
				if (*lbuf == '\0')
					ONNX_LOG(" ");
				p = (void*)((char*)datas + onnx_tensor_type_sizeof(type) * idx);
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
					ONNX_LOG("%s,", (char*)(((char**)p)[0]));
					break;
				default:
					ONNX_LOG("?,");
					break;
				}
				lp = lbuf;
				rp = rbuf;
			}
			for (j = 0; j < ndim; j++)
				ONNX_LOG("]");
			free(sizes);
			free(levels);
			free(lbuf);
			free(rbuf);
			ONNX_LOG("\r\n");
		}else {
			ONNX_LOG(" = ");
			ONNX_LOG("[...]");
			ONNX_LOG("\r\n");
		}
	}else if (ndata == 1) {
		ONNX_LOG(" = ");
		p = (void*)(datas);
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
			ONNX_LOG("%s", (char*)(((char**)p)[0]));
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

void onnx_node_t::dump(int detail) const
{
	ONNX_LOG("%s: %s-%d (%s)\r\n", proto->name, proto->op_type, opset, (strlen(proto->domain) > 0) ? proto->domain : "ai.onnx");
	if (inputs.size() > 0) {
		ONNX_LOG("\tInputs:\r\n");
		for (size_t i = 0; i < inputs.size(); i++) {
			ONNX_LOG("\t\t");
			inputs[i]->dump(detail);
		}
	}
	if (outputs.size() > 0) {
		ONNX_LOG("\tOutputs:\r\n");
		for (size_t i = 0; i < outputs.size(); i++) {
			ONNX_LOG("\t\t");
			outputs[i]->dump(detail);
		}
	}
}

void onnx_graph_t::dump(int detail) const
{
	for (int i = 0; i < nodes.size(); i++)
		nodes[i].dump(detail);
}

void onnx_context_t::dump(int detail) const
{
	if (model) {
		ONNX_LOG("IR Version: v%" PRId64 "\r\n", model->ir_version);
		ONNX_LOG("Producer: %s %s\r\n", model->producer_name, model->producer_version);
		ONNX_LOG("Domain: %s\r\n", model->domain);
		ONNX_LOG("Imports:\r\n");
		for (int i = 0; i < model->n_opset_import; i++)
			ONNX_LOG("\t%s v%" PRId64 "\r\n", (strlen(model->opset_import[i]->domain) > 0) ? model->opset_import[i]->domain : "ai.onnx", model->opset_import[i]->version);
	}
	if (graph)
		graph->dump(detail);
}

void onnx_context_t::run()
{
	for (size_t i = 0; i < graph->nodes.size(); i++) {
		onnx_node_t* n = &graph->nodes[i];
		if (n->reshape(n))
			n->ope(n);
	}
}

