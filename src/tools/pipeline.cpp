
#include "../geometry/Map.h"
#include "../util/config.h"
#include <vector>
#include <stdlib.h>
#include "../tracing/generator.h"
#include "../tracing/trace.h"
//#include <unistd.h>
#include <chrono>
#include <thread>

using namespace std;

int main(int argc, char **argv){

    generator_configuration config = get_generator_parameters(argc, argv);
    Map *m = new Map(config.map_path);
    //m->print_region();
    trace_generator *gen = new trace_generator(&config, m);
    tracer *t;
    Point *traces;

    int st = 0;
    for(st=config.start_time;st<config.start_time+config.duration;st+=100){
        config.cur_duration = min((config.start_time+config.duration-st),(uint)100);
        traces = gen->generate_trace();
        if(st==config.start_time){
            t = new tracer(&config, *m->getMBR(), traces);
        }
        else {
            t->trace = traces;
        }
        t->process(st);
        free(traces);
        traces = NULL;
    }
    delete t;
    delete gen;
    delete m;
    return 0;



//    configuration config = get_parameters(argc, argv);
//    tracer *tr = new tracer(&config);
//    //tr->print_trace();
//    tr->process();
//    delete tr;

//	Map *m = new Map(config.map_path);
//	m->print_region();
//	vector<Point *> result;
//	m->navigate(result, new Point(-87.61353222612192,41.75837880179237), new Point(-88.11615669701743,41.94455236873503), 200);
//	print_linestring(result);
//	delete m;
}

