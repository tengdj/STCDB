#include "../geometry/Map.h"
#include "../tracing/generator.h"
#include "../tracing/trace.h"

using namespace std;


int main(int argc, char **argv){         //test write and read in various number of ssd
    clear_cache();
    generator_configuration config = get_generator_parameters(argc, argv);
    cout << "threads" << config.num_threads << endl;

    size_t size_20MB = 20ULL * 1024 * 1024;  // 使用无符号长整型
    size_t total_size = size_20MB * 4000;    // 确保不会溢出
    char* many_ctfs = new char[total_size];
    memset(many_ctfs, 0, size_20MB * 4000);
    struct timeval start_time;
    for(uint raid_count = 1; raid_count <= 2; raid_count++){
        string cmd = "../script/clear_directories.sh";        //sudo!!!
        if(system(cmd.c_str())!=0){
            fprintf(stderr, "Error when clear temp ctf\n");
        }

        uint file_count = 840 * 4 / raid_count;
        clear_cache();
        start_time = get_cur_time();
#pragma omp parallel for num_threads(config.num_threads) collapse(2)
        for(uint i = 0; i < raid_count; i++){
            for(uint j = 0; j < file_count; j++){
                ofstream outFile("/data3/raid0_num" + to_string(i) + "/temp_ctf" + to_string(j), ios::out|ios::binary|ios::trunc);
                if(!outFile.is_open()){
                    cout << i << " " << j << "write error" << endl;
                }
                size_t index = size_20MB * (i * file_count + j);
                outFile.write((char *)&many_ctfs[index], size_20MB);
                outFile.flush();
                outFile.close();
            }
        }
        double time_consume = get_time_elapsed(start_time, true);
        cout << time_consume << endl;

        clear_cache();
        start_time = get_cur_time();
#pragma omp parallel for num_threads(config.num_threads) collapse(2)
        for(uint i = 0; i < raid_count; i++){         //ssd num
            for(uint j = 0; j < file_count; j++){      //840
                ifstream inFile("/data3/raid0_num" + to_string(i) + "/temp_ctf" + to_string(j),  ios::in | ios::binary);
                if(!inFile.is_open()){
                    cout << i << " " << j << "read error" << endl;
                }
                size_t index = size_20MB * (i * file_count + j);
                inFile.read((char *)&many_ctfs[index], size_20MB);
                inFile.close();
            }
        }
        double read_time = get_time_elapsed(start_time, true);
        cout << read_time << endl;

    }
}

