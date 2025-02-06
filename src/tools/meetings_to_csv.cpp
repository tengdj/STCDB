#include "../tracing/workbench.h"
using namespace std;

string formatBox(const box& key_box) {
   std::ostringstream oss;
   oss << "POLYGON(("
       << key_box.low[0] << " " << key_box.low[1] << ", "
       << key_box.low[0] << " " << key_box.high[1] << ", "
       << key_box.high[0] << " " << key_box.high[1] << ", "
       << key_box.high[0] << " " << key_box.low[1] << ", "
       << key_box.low[0] << " " << key_box.low[1]  // Close the polygon
       << "))";
   return oss.str();
}

bool areFirstFourDigitsEqual(double a, double b) {
   return std::fabs(a - b) < 0.0001; // 判断两数差值是否小于 0.0001
}

// 检查两点是否接近（前 4 位精度）
bool are_points_equal(double x1, double y1, double x2, double y2) {
   const double epsilon = 0.0001; // 精度
   return std::fabs(x1 - x2) < epsilon && std::fabs(y1 - y2) < epsilon;
}

std::string generate_wkt(box & b) {
   int same_count = 0;
   if(areFirstFourDigitsEqual(b.low[0], b.high[0])) same_count++;
   if(areFirstFourDigitsEqual(b.low[1], b.high[1])) same_count++;
   if (same_count == 2) {
       return "POINT(" + std::to_string(b.low[0]) + " " + std::to_string(b.low[1]) + ")";
   }
   if (same_count == 1){
       return "LINESTRING(" + std::to_string(b.low[0]) + " " + std::to_string(b.low[1]) + ","
                              + std::to_string(b.high[0]) + " " + std::to_string(b.high[1]) + ")";
   }
   //if (same_count == 0){
   return formatBox(b);
}


int main(int argc, char **argv){
    clear_cache();
    string path = "../data/meta/N";
    uint max_ctb = 11;
    workbench * bench = load_meta(path.c_str(), max_ctb, 256);
    for(int i = 0; i < max_ctb; i++) {
        for (int j = 0; j < bench->config->CTF_count; j++) {
            bench->ctbs[i].ctfs[j].keys = nullptr;
        }
    }

//    std::ofstream pg_outFile("pg_meetings"+ to_string(10) +".csv");
//    if(pg_outFile.is_open()){
//        cout << "open" << endl;
//    }
//    pg_outFile << "ctfid" << ',' << "oid" << ',' << "target" << ',' << "start_time" << ',' << "end_time" << ',' << "geom" << endl;

   std::ofstream outFile("mem_meetings"+ to_string(10) + ".csv");
   if(outFile.is_open()){
       cout << "open" << endl;
   }
   outFile << "id" << ',' << "start" << ',' << "end" << ',' << "box1,box2,box3,box4" << ',' << "person2_id" << endl;
   for(int j = 0; j < 100; j++){
        CTF * ctf = &bench->ctbs[10].ctfs[j];
        uint mbr_find_count = 0;
        bench->load_CTF_keys(10, j);
        uint8_t * data = reinterpret_cast<uint8_t *>(ctf->keys);
        for (int i = 0; i < bench->ctbs[10].ctfs[j].CTF_kv_capacity; i++) {
            key_info temp_ki;
            __uint128_t temp_128 = 0;
            memcpy(&temp_128, data + i * ctf->key_bit / 8, ctf->key_bit / 8);
            uint64_t value_mbr = 0;
            ctf->parse_key(temp_128, temp_ki, value_mbr);
            box key_box = ctf->new_parse_mbr(value_mbr);
            outFile << temp_ki.oid << ',' << temp_ki.duration << ',' << temp_ki.end << ','
                << key_box.low[0] << ',' << key_box.low[1] << ',' << key_box.high[0] << ',' 
                << key_box.high[1] << ',' << temp_ki.target << endl;
       }
   }
   outFile.close();
//    pg_outFile.close();


   return 0;
}
