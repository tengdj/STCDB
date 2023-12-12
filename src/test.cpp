// #include <pthread.h>
// #include <iostream>
// #include <string.h>
// #include <unistd.h>

// using namespace std;

// class tracer{
//     int *bench = NULL;
// public:
//     void process();
// };

// void *sst_dump(void *arg){
//     cout<<"step into"<<endl;
//     int *b = (int *)arg;
//     for(int i=0;i<10000;i++){
//         sleep(1);
//     }
//     cout<<b[3]<<endl;
//     return NULL;
// }

// void tracer::process(){
//     bench = new int[5]{0};
//     bench[3] = 3;
//     if(1){
//         pthread_t bg_thread;
//         int ret;
//         cout<<"before thread"<<endl;
//         if ((ret=pthread_create(&bg_thread,NULL,sst_dump,(void*)bench)) != 0){
//             fprintf(stderr,"pthread_tcreate:%s\n",strerror(ret));
//             exit(1);
//         }
//     }
//     //sst_dump(bench);
// }

// int main() {
//     tracer *t = new tracer;
//     t->process();
//     getchar();
// }


#include <pthread.h>
#include <iostream>
#include <string.h>
#include <unistd.h>
#include <thread>

using namespace std;

class tracer{
    int *bench = NULL;
public:
    void process();
};

void *sst_dump(void *bench){
    int *b = (int *)bench;
    cout<<"step into"<<endl;
    for(int i=0;i<2;i++){
        sleep(1);
    }
    cout<<b[3]<<endl;
    return NULL;
}

void tracer::process(){
    bench = new int[5]{0};
    bench[3] = 3;
    for(int i=0;i<10;i++){
    thread t(sst_dump,bench);
    t.detach();
    cout<<"for "<<i<<endl;
    }
    //delete []bench;
    //sst_dump(bench);
}

int main() {
    tracer *t = new tracer;
    t->process();
getchar();
}
