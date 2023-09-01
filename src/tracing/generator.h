/*
 * generator.h
 *
 *  Created on: Mar 2, 2021
 *      Author: teng
 */

#ifndef SRC_TRACING_GENERATOR_H_
#define SRC_TRACING_GENERATOR_H_

#include <cstddef>
#include <climits>
#include "../geometry/Map.h"
#include "../util/query_context.h"


class gen_core{
public:
	int id;
	Point core;
	vector<int> assigned_tweets;
	vector<pair<int, double>> destination;
};

/*
 *
 * the statistics of the trips parsed from the taxi data
 * each zone
 *
 * */
class ZoneStats{
public:
	int zoneid = 0;
	long count = 0;
	long duration = 0;
	double length = 0.0;
	ZoneStats(int id){
		zoneid = id;
	}
	~ZoneStats(){
	}
};

class Event{
public:
	int timestamp;
	Point coordinate;
};

enum TripType{
	REST = 0,
	WALK = 1,
	DRIVE = 2
};

class Trip {
public:
	Event start;
	Event end;
	TripType type = REST;
	Trip(){};
	Trip(string str);
	void print_trip();
	int duration(){
		return end.timestamp-start.timestamp;
	}
	double speed(){
		return length()/duration();
	}
	double length(){
		return end.coordinate.distance(start.coordinate, true);
	}
	void resize(int max_duration);
};

class Queue {
public:
    Point data[600];
    uint head = 0;
    uint tail = 0;
    uint size = 600;
    inline Queue();
    inline ~Queue();
    void push(Point val);
    void pop();
    inline bool isEmpty();
    inline bool isFull();
    inline int count();
};


//// dynamic
//inline Queue::Queue(std::size_t sz)
//        : data(new int[sz + 1]), head(0), tail(0), size(sz) { }

inline Queue::Queue(){}

inline Queue::~Queue() {
    //delete[] data;
}


inline bool Queue::isEmpty() {
    return head == tail;
}

inline bool Queue::isFull() {                     //never full
    return head == (tail + 1) % (size);
}

inline int Queue::count(){
    if(head<tail)
        return (tail-head);
    if(head==tail)
        return 0;
    if(head>tail)
        return (size - head + tail);
    return 0;
}



class trace_generator{
	Point *tweets = NULL;
	uint *tweets_assign = NULL;
	gen_core *cores = NULL;
	uint tweet_count = 0;
	uint core_count = 0;
public:
    Queue * producer = NULL;
    Map *map = NULL;
    generator_configuration *config = NULL;

	// construct with some parameters
	trace_generator(generator_configuration *conf, Map *m);
	~trace_generator();

	Point get_random_location(int seed = -1);

	//void analyze_trips(const char *path, int limit = 2147483647);
	Point *generate_trace();
	// generate a trace with given duration
	void get_trace(Map *mymap = NULL, int obj= 0);              //default argument missing for parameter 2 of ‘void trace_generator::get_trace(Map*, int)
	int get_core(int seed=-1);
};


#endif /* SRC_TRACING_GENERATOR_H_ */
