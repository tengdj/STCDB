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
    for(int st= config.start_time; st<config.start_time+config.duration; st+=100) {
        config.cur_duration = min((config.start_time+config.duration-st),(uint)100);
        t->loadData(config.trace_path.c_str(), st);
        for(int t=0;t<config.cur_duration;t++){
            std::ofstream outFile("points"+ to_string(st + t)+".csv");
            if(outFile.is_open()){
                cout << st+t << "open" << endl;
            }
            for (int i = 0; i < config.num_objects; i++) {
                outFile << traces[t * config.num_objects + i].x << "," << traces[t * config.num_objects + i].y << "\n";
            }
            outFile.close();
        }
    }

    delete t;
    cout<<"delete right"<<endl;
    cerr<<"delete right"<<endl;
    return 0;
}
