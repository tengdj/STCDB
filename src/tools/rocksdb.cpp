
#include "../tracing/generator.h"
#include "../tracing/trace.h"

#include "rocksdb/c.h"
#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <rocksdb/cache.h>
#include <rocksdb/table.h>
#include <rocksdb/compaction_filter.h>
#include "rocksdb/write_batch.h"

using namespace std;
using namespace ROCKSDB_NAMESPACE;

#define HASH_SIZE 100000
#define OBJECTS_COUNT 10000000
#define ULL_MAX (unsigned long long)1<<62


#if defined(OS_WIN)
#include <Windows.h>
#else

#include <unistd.h>  // sysconf() - get CPU count

#endif

int duration_bit = 12;
int oid_bit = 26;

using namespace std;

workbench * rocksdb_load_meta(const char *path) {
    log("loading meta from %s", path);
    string bench_path = string(path) + "workbench";
    struct timeval start_time = get_cur_time();
    ifstream in(bench_path, ios::in | ios::binary);
    if(!in.is_open()){
        log("%s cannot be opened",bench_path.c_str());
        exit(0);
    }
    generator_configuration * config = new generator_configuration();
    workbench * bench = new workbench(config);
    in.read((char *)config, sizeof(generator_configuration));               //also read meta
    in.read((char *)bench, sizeof(workbench));      //bench->config = NULL
    bench->config = config;
    bench->ctbs = new CTB[config->big_sorted_run_capacity];
    for(int i = 0; i < 100; i++){
        //CTB temp_ctb;
        string CTB_path = string(path) + "CTB" + to_string(i);
        bench->load_CTB_meta(CTB_path.c_str(), i);
    }
    logt("bench meta load from %s",start_time, bench_path.c_str());
    return bench;
}

//class adjacency_node{
//public:
//    uint oid;
//    uint target;
//    uint start;
//};

void parse_and_print_key(uint64_t temp_key){
    uint oid = temp_key >> (oid_bit + duration_bit);
    uint target = (temp_key >> (duration_bit)) & ((1ULL << oid_bit) - 1);
    uint start = temp_key & ((1ULL << duration_bit) - 1);
    cout << " oid:" << oid << " target:" << target << " start:" << start << endl;
}

void parse_and_get(uint64_t temp_key, uint & oid, uint & target, uint & start){
    oid = temp_key >> (oid_bit + duration_bit);
    target = (temp_key >> (duration_bit)) & ((1ULL << oid_bit) - 1);
    start = temp_key & ((1ULL << duration_bit) - 1);
}

string uint32_to_big_endian(uint32_t value) {
    char buffer[4];
    for (int i = 3; i >= 0; --i) {
        buffer[i] = value & 0xFF;
        value >>= 8;
    }
    return string(buffer, 4);
}

string uint64_to_big_endian(uint64_t value) {
    char buffer[8];
    for (int i = 7; i >= 0; --i) {
        buffer[i] = value & 0xFF;
        value >>= 8;
    }
    return string(buffer, 8);
}

string uint128_to_big_endian(__uint128_t value) {
    char buffer[16]; // 128 b = 16 B
    for (int i = 15; i >= 0; --i) {
        buffer[i] = value & 0xFF;
        value >>= 8;
    }
    return string(buffer, 16);
}

uint64_t big_endian_to_uint64(const std::string& input) {
    assert(input.size() == 8 && "Input string must be exactly 8 bytes for uint64_t");

    uint64_t value = 0;
    for (size_t i = 0; i < 8; ++i) {
        value = (value << 8) | static_cast<unsigned char>(input[i]);
    }
    return value;
}

__uint128_t big_endian_to_uint128(const std::string& input) {
    assert(input.size() == 16 && "Input string must be exactly 16 bytes for __uint128_t");

    __uint128_t value = 0;
    for (size_t i = 0; i < 16; ++i) {
        value = (value << 8) | static_cast<unsigned char>(input[i]);
    }
    return value;
}

void print_big_endian_bytes(const std::string& big_endian) {
    for (unsigned char c : big_endian) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (unsigned int)c << " ";
    }
    std::cout << std::endl;
}

void insert_ctb(workbench *bench, uint ctb_id, DB * db){
    rocksdb::WriteBatch batch;
    uint i = ctb_id;
    //bench->ctbs[i].ctfs = new CTF[100];
    for(uint j = 0; j < 100; j++) {
        CTF * ctf = &bench->ctbs[ctb_id].ctfs[j];
        uint8_t * data = reinterpret_cast<uint8_t *>(ctf->keys);
        for(uint k = 0; k < ctf->CTF_kv_capacity; k++){

            key_info temp_ki;
            __uint128_t temp_128 = 0;
            memcpy(&temp_128, data + k * ctf->key_bit / 8, ctf->key_bit / 8);
            uint64_t value_mbr = 0;
            ctf->parse_key(temp_128, temp_ki, value_mbr);
            temp_ki.end += ctf->end_time_min;
            uint start = temp_ki.end - temp_ki.duration;
            uint64_t temp_key = ((uint64_t)temp_ki.oid << (oid_bit + duration_bit)) + ((uint64_t)temp_ki.target << (duration_bit)) + (uint64_t)start;
            batch.Put("key" + uint64_to_big_endian(temp_key), "value" + uint128_to_big_endian(temp_128));
            //Status s = db->Put(WriteOptions(), uint64_to_big_endian(temp_key), uint128_to_big_endian(temp_128));
        }
    }
    timeval write_start = get_cur_time();
    Status s = db->Write(rocksdb::WriteOptions(), &batch);
    double batch_write_time = get_time_elapsed(write_start, true);
    cout << "batch write time: " << batch_write_time << endl;
    if (!s.ok()){
        cerr << s.ToString() << endl;
        assert(s.ok());
    }
}

uint range_query_by_oid(rocksdb::DB* db, uint32_t oid) {
    std::string prefix = uint32_to_big_endian(oid << (32 - oid_bit));       //0~25 is oid, 26~31 is zero

    rocksdb::ReadOptions read_options;
    read_options.fill_cache = false;

    rocksdb::Iterator* it = db->NewIterator(read_options);

//    for (it->Seek(prefix); it->Valid() && it->key().starts_with(prefix); it->Next()) {
//        parse_and_print_key(it->key());
//    }
    uint count = 0;
    for (it->Seek(prefix); it->Valid(); it->Next()) {
        uint k_oid, target, start;
        parse_and_get(big_endian_to_uint64(it->key().ToString()), k_oid, target, start);
        if(k_oid == oid){
            count++;
            //cout << " oid:" << oid << " target:" << target << " start:" << start << endl;
        }
        else{
            //cout << "search finish" << endl;
            break;
        }
    }
    //cout << "find count " << count << endl;
    if (!it->status().ok()) {
        std::cerr << "Error during range query: " << it->status().ToString() << std::endl;
    }
    delete it;
    return count;
}

void print_all_rocksdb(DB * db){
    Iterator* it = db->NewIterator(ReadOptions());
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
        //int key = atoi(it->key().ToString().c_str());
        uint64_t temp_key = big_endian_to_uint64(it->key().ToString());
        parse_and_print_key(temp_key);
    }
    assert(it->status().ok()); // Check for any errors found during the scan
    delete it;
}

void clear_DB(string & kDBPath){
    string kRemoveDirCommand = "rm -rf ";
    string rm_cmd = kRemoveDirCommand + kDBPath;
    int ret = system(rm_cmd.c_str());
    if (ret != 0) {
        fprintf(stderr, "Error deleting %s, code: %d\n", kDBPath.c_str(), ret);
    }
    cout << "clean DB" << endl;
}


void RestartDatabase(DB*& db, string kDBPath, const Options& options) {
//    delete db;
//    db = nullptr;
    Status s = db->Close();
    if (!s.ok()){
        cerr << s.ToString() << endl;
        assert(s.ok());
    }
    s = DB::Open(options, kDBPath, &db);
    if (!s.ok()){
        cerr << s.ToString() << endl;
        assert(s.ok());
    }
}

int main(int argc, char **argv){
    clear_cache();
    string path = "../data/meta/N";
    //workbench * bench = C_load_meta(path.c_str());
    uint max_ctb = 12;
    workbench * bench = load_meta(path.c_str(), max_ctb);
    cout << "bench->ctb_count " << bench->ctb_count << endl;
    cout << "max_ctb " << max_ctb << endl;
    bench->ctb_count = max_ctb;
    for(int i = 0; i < bench->ctb_count; i++) {
        for (int j = 0; j < bench->config->CTF_count; j++) {
            bench->ctbs[i].ctfs[j].keys = nullptr;
        }
    }

    string kDBPath = "/data2/xiang/rocksdb";
    clear_DB(kDBPath);

    // open DB
    Options options;
    options.create_if_missing = true;
    options.max_log_file_size = 1024 * 1024 * 1024;

    rocksdb::ColumnFamilyOptions cf_options;
    cf_options.disable_auto_compactions = true;
    options.table_factory.reset(rocksdb::NewBlockBasedTableFactory());


//    options.disable_auto_compactions = true;
//    options.level0_slowdown_writes_trigger = INT_MAX;
//    options.level0_stop_writes_trigger = INT_MAX;

//    rocksdb::BlockBasedTableOptions table_options;
//    std::shared_ptr<rocksdb::Cache> cache = rocksdb::NewLRUCache(512 * 1024 * 1024);
//    table_options.block_cache = cache;

//    //Concurrent, mainly for multi-thread subcompaction
//    options.allow_concurrent_memtable_write = true;
//    //max_background_jobs = max_background_compactions + max_background_flushes
//    options.max_background_jobs = 128;

    DB *db;
    Status s = DB::Open(options, kDBPath, &db);
    if (s.ok()) {
        cout << "OK opening" << endl;
    }
    if (!s.ok()){
        cerr << s.ToString() << endl;
        assert(s.ok());
    }

    ofstream of;
    of.open("rocks.csv", ios::out|ios::binary|ios::trunc);
    of << "ctbid" << ',' << "total_size(GB)" << ',' << "insert_time(ms)" << ',' << "compaction_time" << ','  << "search_time(ms)" << endl;
    for(uint i = 0; i < 1; i++){
        uint total_ctb_size = 0;
        for(uint j = 0; j < 100; j++) {
            total_ctb_size += bench->ctbs[i].ctfs[j].CTF_kv_capacity;
            bench->load_CTF_keys(i, j);

        }
        struct timeval start_time = get_cur_time();
        insert_ctb(bench, i, db);
        double insert_time = get_time_elapsed(start_time, true);
        cout << "insert_time: " << insert_time << endl;


        rocksdb::CompactRangeOptions compact_options;
        compact_options.exclusive_manual_compaction = false;
        Status status = db->CompactRange(compact_options, nullptr, nullptr);
        if (!status.ok()) {
            std::cerr << "Failed to compact DB: " << status.ToString() << std::endl;
        }

        double compaction_time = get_time_elapsed(start_time, true);
        //sleep(100);

        ofstream q;
        q.open("rocks_search" + to_string(i) + ".csv", ios::out|ios::binary|ios::trunc);
        q << "question number" << ',' << "time_consume(ms)" << ',' << "find_id_count" << endl;
        start_time = get_cur_time();
        for(int k = 0; k < 100; k++) {
            //RestartDatabase(db, path, options);
            clear_cache();
            uint pid = get_rand_number(bench->config->num_objects);
            struct timeval single_start = get_cur_time();
            uint single_count = range_query_by_oid(db, pid);
            double single_consume = get_time_elapsed(single_start, true);
            q << pid << ',' << single_consume << ',' << single_count << endl;
        }
        q.close();
        double search_time = get_time_elapsed(start_time, true);
        of << i << ',' << (double)total_ctb_size / 1024 / 1024 / 1024 * sizeof(__uint128_t) << ',' << insert_time << ',' << compaction_time << ',' << search_time << endl;
    }
    of.close();
    return 0;
}
