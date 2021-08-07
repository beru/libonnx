#include <onnx.h>

struct operator_pdata_t {
	onnx_graph_t * else_branch;
	onnx_graph_t * then_branch;
};

static int If_init(onnx_node_t * n)
{
	if((n->inputs.size() == 1) && (n->outputs.size() >= 1))
	{
		operator_pdata_t * pdat = new operator_pdata_t;
		pdat->else_branch = new onnx_graph_t(n->ctx, n->attribute_read_graph("else_branch", NULL));
		pdat->then_branch = new onnx_graph_t(n->ctx, n->attribute_read_graph("then_branch", NULL));
		if(!pdat->else_branch || !pdat->then_branch)
		{
			if(pdat->else_branch)
				delete pdat->else_branch;
			if(pdat->then_branch)
				delete pdat->then_branch;
			delete pdat;
			return 0;
		}
		n->priv = pdat;
		return 1;
	}
	return 0;
}

static int If_exit(onnx_node_t * n)
{
	operator_pdata_t * pdat = (operator_pdata_t *)n->priv;

	if(pdat)
	{
		if(pdat->else_branch)
			delete pdat->else_branch;
		if(pdat->then_branch)
			delete pdat->then_branch;
		delete pdat;
	}
	return 1;
}

static int If_reshape(onnx_node_t * n)
{
	operator_pdata_t * pdat = (operator_pdata_t *)n->priv;
	onnx_tensor_t * x = n->inputs[0];
	uint8_t * px = (uint8_t *)x->datas;
	onnx_graph_t * g;
	onnx_node_t * t;
	int i;

	if(px[0])
		g = pdat->then_branch;
	else
		g = pdat->else_branch;
	if(g->nodes.size() > 0)
	{
		for(i = 0; i < g->nodes.size(); i++)
		{
			t = &g->nodes[i];
			t->reshape(t);
		}
		if(t)
		{
			for(i = 0; i < min(t->outputs.size(), n->outputs.size()); i++)
			{
				onnx_tensor_t * a = t->outputs[i];
				onnx_tensor_t * b = n->outputs[i];
				b->reshape_identity(a, a->type);
			}
		}
	}
	return 1;
}

static void If_operator(onnx_node_t * n)
{
	operator_pdata_t * pdat = (operator_pdata_t *)n->priv;
	onnx_tensor_t * x = n->inputs[0];
	uint8_t * px = (uint8_t *)x->datas;
	onnx_graph_t * g;
	onnx_node_t * t;
	int i;

	if(px[0])
		g = pdat->then_branch;
	else
		g = pdat->else_branch;
	if(g->nodes.size() > 0)
	{
		for(i = 0; i < g->nodes.size(); i++)
		{
			t = &g->nodes[i];
			t->ope(t);
		}
		if(t)
		{
			for(i = 0; i < min(t->outputs.size(), n->outputs.size()); i++)
			{
				onnx_tensor_t * a = t->outputs[i];
				onnx_tensor_t * b = n->outputs[i];
				if(x->type == ONNX_TENSOR_TYPE_STRING)
				{
					char ** pa = (char **)a->datas;
					char ** pb = (char **)b->datas;
					for(size_t o = 0; o < b->ndata; o++)
					{
						if(pb[o])
							free(pb[o]);
						pb[o] = strdup(pa[o]);
					}
				}
				else
				{
					memcpy(b->datas, a->datas, a->ndata * onnx_tensor_type_sizeof(a->type));
				}
			}
		}
	}
}

void resolver_default_op_If(onnx_node_t * n)
{
	if(n->opset >= 13)
	{
		n->init = If_init;
		n->exit = If_exit;
		n->reshape = If_reshape;
		n->ope = If_operator;
	}
	else if(n->opset >= 11)
	{
		n->init = If_init;
		n->exit = If_exit;
		n->reshape = If_reshape;
		n->ope = If_operator;
	}
	else if(n->opset >= 1)
	{
		n->init = If_init;
		n->exit = If_exit;
		n->reshape = If_reshape;
		n->ope = If_operator;
	}
}
