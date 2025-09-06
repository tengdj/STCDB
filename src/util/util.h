#ifndef TENG_UTIL_H_
#define TENG_UTIL_H_

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <string.h>
#include <vector>
#include <thread>
#include <iostream>
#include <stdarg.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <time.h>
#include <sstream>
#include <math.h>
#include <limits.h>
#include <algorithm>
#include <fstream>
#include <set>
#include <atomic>
#include <execution>
#include "unordered_map"
#include "unordered_set"

using namespace std;
typedef unsigned int uint;

namespace{

#define TENG_RANDOM_NUMBER 0315
#define OSM_SRID 4326
#define PI 3.14159265
#define ULL_MAX (unsigned long long)1<<62
#define BLOCK_DIM 1024
// some utility function

#define __BLOOMFILTER_VERSION__ "1.1"
#define __MGAIC_CODE__          (0x01464C42)
#define MIX_UINT64(v)       ((uint32_t)((v>>32)^(v)))

#define DEFAULT_bitmap_edge 256
#define SID_BIT 12
#define OID_BIT 26
#define MBR_BIT 36
#define DURATION_BIT 32
#define END_BIT 32
//26 + 26 + 12 = 64
#define EDGE_REISTICTION 0.008


const double degree_per_meter_latitude = 360.0/(40076.0*1000.0);

    inline double degree_per_meter_longitude(double latitude){
        double absla = fabs(latitude);
        if(absla==90){
            absla = 89;
        }
        assert(absla<90);
        return 360.0/(sin((90-absla)*PI/180)*40076.0*1000.0);
    }

    inline size_t cantorPairing(uint pid1, uint pid2){
        return (size_t)(pid1+pid2)*(pid1+pid2+1)/2+pid2;
    }

    inline pair<uint, uint> InverseCantorPairing1(size_t z){
        size_t w = floor((sqrt(8.0 * z + 1) - 1)/2);
        size_t t = (w*w + w) / 2;
        uint y = (uint)(z - t);
        uint x = (uint)(w - y);
        return pair<uint,uint>(x,y);
    }

    inline int double_to_int(double val){
        int vi = (int)val;
        if(fabs(1.0*(vi+1)-val)<0.00000001){
            vi++;
        }
        return vi;
    }

    inline bool is_number(char ch){
        return ch=='-'||ch=='.'||(ch<='9'&&ch>='0')||ch=='e';
    }

    inline double read_double(const char *input, size_t &offset){
        char tmp[100];
        while(!is_number(input[offset])){
            offset++;
        }
        int index = 0;
        while(is_number(input[offset])){
            tmp[index++] = input[offset++];
        }
        tmp[index] = '\0';
        return atof(tmp);
    }
    inline void skip_space(const char *input, size_t &offset){
        while(input[offset]==' '||input[offset]=='\t'||input[offset]=='\n'){
            offset++;
        }
    }

    inline struct timeval get_cur_time(){
        struct timeval t1;
        gettimeofday(&t1, NULL);
        return t1;
    }
    inline double get_time_elapsed(struct timeval &t1, bool update_start = false){
        struct timeval t2;
        double elapsedTime;
        gettimeofday(&t2, NULL);
        // compute and print the elapsed time in millisec
        elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
        elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
        if(update_start){
            t1 = get_cur_time();
        }
        return elapsedTime;
    }

    inline string time_string(){
        struct timeval tv;
        gettimeofday(&tv, NULL);
        struct tm *nowtm;
        char tmbuf[100];
        char buf[256];
        nowtm = localtime(&tv.tv_sec);
        strftime(tmbuf, sizeof tmbuf, "%H:%M:%S", nowtm);
        sprintf(buf,"%s.%04ld", tmbuf, tv.tv_usec/1000);
        return string(buf);
    }

    static pthread_mutex_t general_lock;
    inline void lock(){
        pthread_mutex_lock(&general_lock);
    }

    inline void unlock(){
        pthread_mutex_unlock(&general_lock);
    }

    static pthread_mutex_t print_lock;
    inline void logt(const char *format, struct timeval &start, ...){
        pthread_mutex_lock(&print_lock);
        va_list args;
        va_start(args, start);
        char sprint_buf[200];
        int n = vsprintf(sprint_buf, format, args);
        va_end(args);
        fprintf(stderr,"%s thread %ld:\t%s", time_string().c_str(), syscall(__NR_gettid),sprint_buf);

        double mstime = get_time_elapsed(start, true);
        if(mstime>1000){
            fprintf(stderr," takes %.4f s\n", mstime/1000);
        }else{
            fprintf(stderr," takes %.4f ms\n", mstime);
        }
        fflush(stderr);

        pthread_mutex_unlock(&print_lock);
    }

    inline void log(const char *format, ...){
        pthread_mutex_lock(&print_lock);
        va_list args;
        va_start(args, format);
        char sprint_buf[200];
        int n = vsprintf(sprint_buf, format, args);
        va_end(args);
        fprintf(stderr,"%s thread %ld:\t%s\n", time_string().c_str(), syscall(__NR_gettid),sprint_buf);
        fflush(stderr);
        pthread_mutex_unlock(&print_lock);
    }

    inline void log_stdout(const char *format, ...){
        pthread_mutex_lock(&print_lock);
        va_list args;
        va_start(args, format);
        char sprint_buf[200];
        int n = vsprintf(sprint_buf, format, args);
        va_end(args);
        fprintf(stdout,"%s thread %ld:\t%s\n", time_string().c_str(), syscall(__NR_gettid),sprint_buf);
        fflush(stdout);
        pthread_mutex_unlock(&print_lock);
    }

    inline void log(){
        pthread_mutex_lock(&print_lock);
        fprintf(stdout,"%s thread %ld:\tterry is good\n", time_string().c_str(),syscall(__NR_gettid));
        fflush(stdout);
        pthread_mutex_unlock(&print_lock);
    }

    inline int get_rand_number(int max_value){
        return rand()%max_value+1;
    }

    inline double get_rand_double(){
        return rand()/(double)RAND_MAX;
    }

    inline bool get_rand_sample(int rate){
        return rand()%100<rate;
    }

    inline bool tryluck(float possibility){
        assert(possibility>=0);
        return possibility>=1.0||(rand()*1.0)/RAND_MAX<possibility;
    }

    inline bool is_dir(const char* path) {
        struct stat buf;
        stat(path, &buf);
        return S_ISDIR(buf.st_mode);
    }

    inline bool is_file(const char* path) {
        struct stat buf;
        stat(path, &buf);
        return S_ISREG(buf.st_mode);
    }

    inline void list_files(const char *path, std::vector<string> &f_list){
        if(is_file(path)){
            f_list.push_back(std::string(path));
            return;
        }
        DIR *dir;
        struct dirent *ent;
        if ((dir = opendir (path)) != NULL) {
            /* print all the files and directories within directory */
            while ((ent = readdir (dir)) != NULL) {
                if(strcmp(ent->d_name,"..")==0||
                        strcmp(ent->d_name,".")==0){
                    continue;
                }
                std::stringstream ss;
                ss<<path<<"/"<<ent->d_name;
                string spath;
                ss>>spath;
                list_files(spath.c_str(), f_list);
            }
            closedir (dir);
        }
    }

    inline long file_size(const char *file){
        struct stat stat_buf;
        int rc = stat(file, &stat_buf);
        return rc == 0 ? stat_buf.st_size : -1;
    }

    inline long file_size(std::vector<string> &f_list){
        long size = 0;
        for(string s:f_list){
            long ls = file_size(s.c_str());
            if(ls>0){
                size += ls;
            }
        }
        return size;
    }

    inline bool file_exist(const char *path) {
      struct stat buffer;
      return (stat(path, &buffer) == 0);
    }

    inline int get_num_threads(){
        return std::thread::hardware_concurrency();
    }

    inline string read_line(){
        string input_line;
        getline(std::cin, input_line);
        return input_line;
    }



    inline void tokenize( const std::string& str, std::vector<std::string>& result,
        const std::string& delimiters = " ,;:\t",
        const bool keepBlankFields=true,
        const std::string& quote="\"\'"
        ){
        // clear the vector
        if (!result.empty()){
            result.clear();
        }

        // you must be kidding
        if (delimiters.empty())
        return ;

        std::string::size_type pos = 0; // the current position (char) in the string
        char ch = 0; // buffer for the current character

        char current_quote = 0; // the char of the current open quote
        bool quoted = false; // indicator if there is an open quote
        std::string token;  // string buffer for the token
        bool token_complete = false; // indicates if the current token is
        // read to be added to the result vector
        std::string::size_type len = str.length();  // length of the input-string

        // for every char in the input-string
        while(len > pos){
            // get the character of the string and reset the delimiter buffer
            ch = str.at(pos);

            bool add_char = true;
            if ( false == quote.empty()){
                // if quote chars are provided and the char isn't protected
                if (std::string::npos != quote.find_first_of(ch)){
                    if (!quoted){
                        quoted = true;
                        current_quote = ch;
                        add_char = false;
                    } else {
                        if (current_quote == ch){
                            quoted = false;
                            current_quote = 0;
                            add_char = false;
                        }
                    }
                }
            }

            if (!delimiters.empty()&&!quoted){
                // if ch is delemiter
                if (std::string::npos != delimiters.find_first_of(ch)){
                    token_complete = true;
                    // don't add the delimiter to the token
                    add_char = false;
                }
            }

            // add the character to the token
            if (add_char){
                token.push_back(ch);
            }

            // add the token if it is complete
            // if ( true == token_complete && false == token.empty() )
            if (token_complete){
                if (token.empty())
                {
                if (keepBlankFields)
                    result.push_back("");
                }
                else
                result.push_back( token );
                token.clear();
                token_complete = false;
            }
            ++pos;
        } // while
        // add the final token
        if ( false == token.empty() ) {
            result.push_back( token );
        } else if(keepBlankFields && std::string::npos != delimiters.find_first_of(ch) ){
            result.push_back("");
        }
    }

    inline bool isdigit(char ch){
        return ch=='-'||ch=='.'||(ch>='0'&&ch<='9');
    }

    inline vector<double> parse_double_values(const std::string &str){
        vector<string> fields;
        vector<double> values;
        tokenize(str,fields," ,()", false);
        for(string s:fields){
            if(isdigit(s.at(0))){
                values.push_back(stod(s));
            }
        }
        fields.clear();
        return values;
    }


    inline void remove_slash(string &str){
        if(str.at(str.size() - 1) == '/'){
            str = str.substr(0, str.size() - 1);
        }
    }

    #define min_equal 0.000001
    inline bool double_equal(double d1, double d2){
        return fabs(d1-d2)<min_equal;
    }

    inline void print_128(__uint128_t x){
        if(x>9)
            print_128(x/10);
        putchar(x%10+'0');
    }

    inline string tostring128(__uint128_t x){
        char ccc[40];
        int i=0;
        while(x>9){
            ccc[i] = x%10 + '0';
            i++;
            x/=10;
        }
        ccc[i] = x + '0';
        i++;                        //real length
        string sss = ccc;
        sss.resize(i);
        reverse(sss.begin(),sss.end());             //#include <algorithm>
        return sss;
    }

    inline __uint128_t stringto128(string sss){
        __uint128_t x=0;
        //char ccc[sss.size()] =sss.data();
        for(int i=0;i<sss.size();i++){
            x=x*10+sss[i]-'0';
        }
        return x;
    }

    inline uint64_t MurmurHash2_x64 ( const void * key, int len, uint32_t seed ){
        const uint64_t m = 0xc6a4a7935bd1e995;
        const int r = 47;

        uint64_t h = seed ^ (len * m);

        const uint64_t * data = (const uint64_t *)key;
        const uint64_t * end = data + (len/8);

        while(data != end)
        {
            uint64_t k = *data++;

            k *= m;
            k ^= k >> r;
            k *= m;

            h ^= k;
            h *= m;
        }

        const uint8_t * data2 = (const uint8_t*)data;

        switch(len & 7)
        {
            case 7: h ^= ((uint64_t)data2[6]) << 48;
            case 6: h ^= ((uint64_t)data2[5]) << 40;
            case 5: h ^= ((uint64_t)data2[4]) << 32;
            case 4: h ^= ((uint64_t)data2[3]) << 24;
            case 3: h ^= ((uint64_t)data2[2]) << 16;
            case 2: h ^= ((uint64_t)data2[1]) << 8;
            case 1: h ^= ((uint64_t)data2[0]);
                h *= m;
        };

        h ^= h >> r;
        h *= m;
        h ^= h >> r;

        return h;
    }

    typedef struct{
        uint32_t dwMagicCode;
        uint32_t dwSeed;
        uint32_t dwCount;

        uint32_t dwMaxItems;
        double dProbFalse;
        uint32_t dwFilterBits;
        uint32_t dwHashFuncs;

        uint32_t dwResv[6];
        uint32_t dwFileCrc;
        uint32_t dwFilterSize;
    }BloomFileHead;

    //#pragma pack()

    static inline void CalcBloomFilterParam(uint32_t n, double p, uint32_t *pm, uint32_t *pk){
        /**
      *  n - Number of items in the filter
      *  p - Probability of false positives, float between 0 and 1 or a number indicating 1-in-p
      *  m - Number of bits in the filter
      *  k - Number of hash functions
      *
      *  f = ln(2) × ln(1/2) × m / n = (0.6185) ^ (m/n)
      * m = -1*n*ln(p)/((ln(2))^2) = -1*n*ln(p)/(ln(2)*ln(2)) = -1*n*ln(p)/(0.69314718055995*0.69314718055995))
      *   = -1*n*ln(p)/0.4804530139182079271955440025
      * k = ln(2)*m/n
     **/
        uint32_t m, k;
        m =(uint32_t) ceil(-1.0 * n * std::log(p) / 0.480453);
        m = (m - m % 64) + 64;                              // 8B glignment
        double double_k = (0.69314 *m /n);

        k = round(double_k);

        *pm = m;
        *pk = k;
    }

    inline void printBits(unsigned char byte) {
        for (int i = 7; i >= 0; --i) {
            cout << ((byte >> i) & 1);
        }
    }

    void clear_cache(){
        string cmd = "echo 'xiangrongkai' | sudo -S sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'";
        if(system(cmd.c_str()) != 0){
            fprintf(stderr, "Error when disabling buffer cache\n");
        }
    }

    template <typename T>
    constexpr uint min_bits_to_store(T max_value) {
        static_assert(std::is_integral<T>::value && std::is_unsigned<T>::value,
                      "Input must be an unsigned integer type.");
        if (max_value == 0) return 1;
        return static_cast<uint>(std::log2(max_value)) + 1; // ⌊log2(value)⌋ + 1
    }

    uint get_key_sid(__uint128_t key){
        return (uint)((key >> (OID_BIT * 2 + DURATION_BIT + END_BIT)) & ((1ULL << SID_BIT) - 1));
    }

    uint get_key_oid(__uint128_t key){
        return (uint)((key >> (OID_BIT + DURATION_BIT + END_BIT)) & ((1ULL << OID_BIT) - 1));
    }

    uint get_key_target(__uint128_t key){
        return (uint)((key >> ( DURATION_BIT + END_BIT)) & ((1ULL << OID_BIT) - 1));
    }

    uint get_key_duration(__uint128_t key){
        return (uint)((key >> END_BIT) & ((1ULL << DURATION_BIT) - 1));
    }

    uint get_key_end(__uint128_t key){
        return (uint)(key & ((1ULL << END_BIT) - 1));
    }

}
#endif


