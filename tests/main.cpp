#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <malloc.h>
#include <set>
#include <filesystem>

#ifdef _MSC_VER
#include <format>
using std::format;
#else
#include <fmt/core.h>
using fmt::format;
#endif

#include "onnx.h"


static void testcase(const std::filesystem::path& path, onnx::resolver_t** r, int rlen)
{
	onnx::context_t ctx;
	if (ctx.alloc_from_file((path / "model.onnx").string(), r, rlen)) {
		int data_set_index = 0;
		while (1) {
			std::filesystem::path data_set_path = path / format("test_data_set_{}", data_set_index);
			if (!std::filesystem::is_directory(data_set_path)) {
				break;
			}
			int ninput = 0;
			int noutput = 0;
			int okay = 0;
			while (1) {
				auto filepath = data_set_path / format("input_{}.pb", ninput);
				if (!std::filesystem::is_regular_file(filepath)) {
					break;
				}
				if (ninput > ctx.model->graph->n_input) {
					break;
				}
				onnx::tensor_t* t = ctx.search_tensor(ctx.model->graph->input[ninput]->name);
				onnx::tensor_t* o = onnx::tensor_t::alloc_from_file(filepath.string());
				t->apply(o->data, o->ndata * onnx::tensor_type_sizeof(o->type));
				delete o;
				okay++;
				ninput++;
			}
			ctx.run();
			while (1) {
				auto filepath = data_set_path / format("output_{}.pb", noutput);
				if (!std::filesystem::is_regular_file(filepath)) {
					break;
				}
				if (noutput > ctx.model->graph->n_output) {
					break;
				}
				onnx::tensor_t* t = ctx.search_tensor(ctx.model->graph->output[noutput]->name);
				onnx::tensor_t* o = onnx::tensor_t::alloc_from_file(filepath.string());
				if (onnx::tensor_equal(t, o)) {
					okay++;
				}
				delete o;
				noutput++;
			}

			int len = printf("[%s](test_data_set_%d)", path.string().c_str(), data_set_index);
			printf("%*s\r\n", 100 + 12 - 6 - len, ((ninput + noutput == okay) && (okay > 0)) ? "\033[42;37m[OKAY]\033[0m" : "\033[41;37m[FAIL]\033[0m");
			data_set_index++;
		}
	}else {
		int len = printf("[%s]", path.string().c_str());
		printf("%*s\r\n", 100 + 12 - 6 - len, "\033[41;37m[FAIL]\033[0m");
	}
}

static void usage(void)
{
	const char* txt = R"(usage:
    tests <DIRECTORY>
examples:
    tests ./tests/model
    tests ./tests/node
    tests ./tests/pytorch-converted
    tests ./tests/pytorch-operator
    tests ./tests/simple
)";
	printf(txt);
}

int main(int argc, char* argv[])
{
	if (argc != 2) {
		usage();
		return -1;
	}
	std::filesystem::path path{argv[1]};
	if (!std::filesystem::is_directory(path)) {
		usage();
		return -1;
	}
	std::set<std::filesystem::path> m;
	for (auto file : std::filesystem::directory_iterator{path}) {
		if (std::filesystem::is_directory(file)) {
			m.emplace(file);
		}
	}
	for (auto file : m) {
		testcase(file, NULL, 0);
	}
	return 0;
}
