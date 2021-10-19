#include <chrono>
#include <thread>
#include "onnx.h"

using time_point = std::chrono::system_clock::time_point;

struct profiler_t {
	double elapsed = 0;
	long count = 0;
};

static void profiler_dump(std::map<std::string, profiler_t>& m, int count)
{
	double total = 0;
	double mean = 0;
	double fps = 0;

	printf("Profiler analysis:\r\n");
	for (const auto& [key, value] : m) {
		total += value.elapsed;
		printf("%-32s %ld %12.3f(us)\r\n", key.c_str(), value.count, (value.count > 0) ? (value.elapsed / (double)value.count) : 0);
	}
	if (count > 0) {
		mean = total / (double)count;
		fps = (double)1000000.0 / mean;
	}
	printf("----------------------------------------------------------------\r\n");
	printf("Repeat times: %d, Average time: %.3f(us), Frame rates: %.3f(fps)\r\n", count, mean, fps);
}

static void onnx_run_benchmark(onnx::context_t& ctx, int count)
{
	char name[256];
	int cnt = count;

	std::map<std::string, profiler_t> m;
	ctx.run();
	while (count-- > 0) {
		for (auto n : ctx.graph->nodes) {
			int len = sprintf(name, "%s-%d", n->proto->op_type, n->opset);
			auto& p = m[name];
			time_point begin = std::chrono::system_clock::now();
			if (n->reshape()) {
				n->exec();
			}
			time_point end = std::chrono::system_clock::now();
			p.elapsed += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
			++p.count;
		}
	}
	profiler_dump(m, cnt);
}

int main(int argc, char* argv[])
{
	if (argc <= 1) {
		printf("usage:\r\n");
		printf("    benchmark <filename> [count]\r\n");
		return -1;
	}
	int count = 0;
	char* filename = argv[1];
	if (argc >= 3) {
		count = strtol(argv[2], NULL, 0);
	}
	if (count <= 0) {
		count = 1;
	}
	onnx::context_t ctx;
	if (!ctx.alloc_from_file(filename)) {
		return -1;
	}
	onnx_run_benchmark(ctx, count);
	return 0;
}

