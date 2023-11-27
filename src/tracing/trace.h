/*
 * tracing.h
 *
 *  Created on: Jan 19, 2021
 *      Author: teng
 */

#ifndef SRC_TRACING_TRACE_H_
#define SRC_TRACING_TRACE_H_

#include "../geometry/Map.h"
#include "../util/query_context.h"
#include "partitioner.h"
#include "workbench.h"
#include <map>
#include "../tracing/generator.h"

#ifdef USE_GPU
#include "../cuda/mygpu.h"
#endif

using namespace std;

class tracer{
	// the statistic for the data set
	Point *trace = NULL;
	bool owned_trace = false;
	// for query
	configuration *config = NULL;
	partitioner *part = NULL;
	workbench *bench = NULL;
    trace_generator *generator = NULL;
#ifdef USE_GPU
	gpu_info *gpu = NULL;
	workbench *d_bench = NULL;
#endif
public:
	box mbr;
	tracer(configuration *conf, box &b, Point *t, trace_generator *gen);
	tracer(configuration *conf);
	~tracer();
	void dumpTo(const char *path);
	void loadMeta(const char *path);
	void loadData(const char *path, int start_time, int duration);
	void print();
	void print_trace(int oid);
	void print_traces();
	Point *get_trace(){
		return trace;
	}

	void process();

};



#endif /* SRC_TRACING_TRACE_H_ */
