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
    Point *traces = new Point[config.num_objects*100];
    tracer *t = new tracer(&config, *m->getMBR(), traces, gen);
    t->process();


    delete []traces;
    delete gen;
    delete t;
    delete m;
    return 0;
}
