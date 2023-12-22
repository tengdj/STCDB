#include "../geometry/Map.h"
#include "../tracing/generator.h"
#include "../tracing/trace.h"

using namespace std;

int main(int argc, char **argv){
    sorted_run *bg_run = new sorted_run;
    bg_run->first_pid = new uint[bg_run->SSTable_count];
    bg_run->sst = new SSTable[bg_run->SSTable_count];
    for(int i=0;i<bg_run->SSTable_count;i++){
        bg_run->sst[i].kv = new key_value[bg_run->sst->SSTable_kv_capacity];
    }
    ifstream read_meta;
    read_meta.open("../store/first_pid_meta");
    assert(read_meta.is_open());
    for(int i=0;i<bg_run->SSTable_count;i++){
        read_meta.read((char *)&bg_run->first_pid[i], sizeof(uint));
        cout<<bg_run->first_pid[i]<<" ";
    }
    cout<<endl;
    read_meta.close();
    cout<<"before func"<<endl;
    uint pid = 0;
    if(bg_run->search_in_disk(0,500000)){
        cout<<"finally find"<<endl;
    }
    return 0;
}

//bench->bg_run[bench->big_sorted_run_count]->sst = new SSTable[bg_run[0]->SSTable_count];
//search_in_disk(&bench->bg_run[bench->big_sorted_run_count],pid));


//#include "../geometry/Map.h"
//#include "../tracing/generator.h"
//#include "../tracing/trace.h"
//
//using namespace std;
//
//int main(int argc, char **argv){
//
//    generator_configuration config = get_generator_parameters(argc, argv);
//    Map *m = new Map(config.map_path);
//    //m->print_region();
//    trace_generator *gen = new trace_generator(&config, m);
//    Point *traces = new Point[config.num_objects*100];
//    tracer *t = new tracer(&config, *m->getMBR(), traces, gen);
//    cout<<"before process"<<endl;
//    t->searchsst_process();
//
//    delete []traces;
//    delete gen;
//    delete t;
//    delete m;
//    return 0;
//}
