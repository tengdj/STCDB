#include "../geometry/Map.h"
#include "../tracing/generator.h"
#include "../tracing/trace.h"

using namespace std;

int main(int argc, char **argv){
    generator_configuration config = get_generator_parameters(argc, argv);
    //config.load_data = false;
    Map *m = new Map(config.map_path);
    //m->print_region();
    trace_generator *gen = new trace_generator(&config, m);
    Point *traces = new Point[config.num_objects*100];
    tracer *t = new tracer(&config, *m->getMBR(), traces, gen);
    t->process();
    delete t;
    cout<<"delete right"<<endl;
    cerr<<"delete right"<<endl;
    return 0;
}
