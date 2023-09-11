/*
 * generator.h
 *
 *  Created on: Mar 2, 2021
 *      Author: teng
 */

#ifndef SRC_TRACING_GENERATOR_H_
#define SRC_TRACING_GENERATOR_H_

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

class trace_generator{
    Point *tweets = NULL;
    uint *tweets_assign = NULL;
    gen_core *cores = NULL;
    uint tweet_count = 0;
    uint core_count = 0;
public:
    Trace_Meta * meta_data = NULL;
    Map *map = NULL;
    generator_configuration *config = NULL;

    // construct with some parameters
    trace_generator(generator_configuration *conf, Map *m);
    ~trace_generator();

    Point get_random_location(int seed = -1);

    //void analyze_trips(const char *path, int limit = 2147483647);
    void generate_trace(Point * traces);
    // generate a trace with given duration
    vector<Point *> fill_trace(Map *mymap = NULL, int obj = 0);
    int get_core(int seed=-1);

};

#endif /* SRC_TRACING_GENERATOR_H_ */
