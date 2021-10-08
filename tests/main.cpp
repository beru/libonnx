#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <malloc.h>
#include <set>
#include <filesystem>
#include "onnx.h"

#define PATH_MAX (260)

static void testcase(const std::filesystem::path& path, onnx::resolver_t** r, int rlen)
{
	int data_set_index;
	int ninput, noutput;
	int okay;
	int len;

	onnx::context_t ctx((path / "model.onnx").string(), r, rlen);
	if (true) {
		data_set_index = 0;
		while (1) {
			char tmp[512];
			sprintf(tmp, "test_data_set_%d", data_set_index);
			std::filesystem::path data_set_path = path / tmp;
			if (!std::filesystem::is_directory(data_set_path)) {
				break;
			}
			ninput = 0;
			noutput = 0;
			okay = 0;
			while (1) {
				sprintf(tmp, "input_%d.pb", ninput);
				if (!std::filesystem::is_regular_file(data_set_path / tmp)) {
					break;
				}
				if (ninput > ctx.model->graph->n_input) {
					break;
				}
				onnx::tensor_t* t = ctx.search_tensor(ctx.model->graph->input[ninput]->name);
				onnx::tensor_t* o = onnx::tensor_t::alloc_from_file((data_set_path / tmp).string());
				t->apply(o->data, o->ndata * onnx::tensor_type_sizeof(o->type));
				delete o;
				okay++;
				ninput++;
			}
			ctx.run();
			while (1) {
				sprintf(tmp, "output_%d.pb", noutput);
				if (!std::filesystem::is_regular_file(data_set_path / tmp)) {
					break;
				}
				if (noutput > ctx.model->graph->n_output) {
					break;
				}
				onnx::tensor_t* t = ctx.search_tensor(ctx.model->graph->output[noutput]->name);
				onnx::tensor_t* o = onnx::tensor_t::alloc_from_file((data_set_path / tmp).string());
				if (onnx::tensor_equal(t, o)) {
					okay++;
				}
				delete o;
				noutput++;
			}

			len = printf("[%s](test_data_set_%d)", path.string().c_str(), data_set_index);
			printf("%*s\r\n", 100 + 12 - 6 - len, ((ninput + noutput == okay) && (okay > 0)) ? "\033[42;37m[OKAY]\033[0m" : "\033[41;37m[FAIL]\033[0m");
			data_set_index++;
		}
	}else {
		len = printf("[%s]", path.string().c_str());
		printf("%*s\r\n", 100 + 12 - 6 - len, "\033[41;37m[FAIL]\033[0m");
	}
}

static void usage(void)
{
	printf("usage:\r\n");
	printf("    tests <DIRECTORY>\r\n");
	printf("examples:\r\n");
	printf("    tests ./tests/model\r\n");
	printf("    tests ./tests/node\r\n");
	printf("    tests ./tests/pytorch-converted\r\n");
	printf("    tests ./tests/pytorch-operator\r\n");
	printf("    tests ./tests/simple\r\n");
}

int main(int argc, char* argv[])
{
	if (argc != 2) {
		usage();
		return -1;
	}
	auto path = argv[1];
	if (!std::filesystem::is_directory(path)) {
		usage();
		return -1;
	}
	for (auto file : std::filesystem::directory_iterator{path}) {
		if (std::filesystem::is_directory(file)) {
			testcase(file, NULL, 0);
		}
	}
	return 0;
}
