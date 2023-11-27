///*
// * main.cpp
// *
// *  Created on: Jan 11, 2021
// *      Author: teng
// */
//
//
//#include "../geometry/Map.h"
//#include "../util/config.h"
//#include <vector>
//#include <stdlib.h>
//#include "../tracing/generator.h"
//#include "../tracing/trace.h"
//
//using namespace std;
//
//int main(int argc, char **argv){
//
//	generator_configuration config = get_generator_parameters(argc, argv);
//	Map *m = new Map(config.map_path);
//	m->print_region();
//	trace_generator *gen = new trace_generator(&config,m);
//	Point *traces = gen->generate_trace();
////
//	tracer *t = new tracer(&config,*m->getMBR(),traces);
//	t->dumpTo(config.trace_path.c_str());
////	t->print();
////	t->print_trace(0);
//	delete t;
////
//	free(traces);
//	delete gen;
//	delete m;
//	return 0;
//}
//
