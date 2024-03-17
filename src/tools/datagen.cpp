/*
 * main.cpp
 *
 *  Created on: Jan 11, 2021
 *      Author: teng
 */


#include "../geometry/Map.h"
#include "../util/config.h"
#include <vector>
#include <stdlib.h>
#include "../tracing/generator.h"
#include "../tracing/trace.h"

using namespace std;

int main(int argc, char **argv){

	generator_configuration config = get_generator_parameters(argc, argv);
    //config.cur_duration = 500;
	Map *m = new Map(config.map_path);
	m->print_region();
	trace_generator *gen = new trace_generator(&config,m);
	Point *traces = new Point [config.duration*config.num_objects];
    gen->generate_trace(traces);
    tracer *t = new tracer(&config,*m->getMBR(),traces,gen);
    t->dumpTo(config.trace_path.c_str());

    delete t;
	//free(traces);
	delete gen;
	delete m;
	return 0;
}

