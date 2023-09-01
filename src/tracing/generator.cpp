/*
 * trace_generator.cpp
 *
 *  Created on: Feb 1, 2021
 *      Author: teng
 */
//#include <random>
#include "../index/QTree.h"

#include "generator.h"

/*
 *
 * Trip member functions
 *
 * */

Trip::Trip(string str){

	vector<string> cols;
	tokenize(str,cols,",");

	start.timestamp = 0;
	char tmp[2];
	tmp[0] = cols[2][11];
	tmp[1] = cols[2][12];
	start.timestamp += atoi(tmp)*3600;
	tmp[0] = cols[2][14];
	tmp[1] = cols[2][15];
	start.timestamp += atoi(tmp)*60;
	tmp[0] = cols[2][17];
	tmp[1] = cols[2][18];
	start.timestamp += atoi(tmp);
	if(cols[2][20]=='P'){
		start.timestamp += 12*3600;
	}
	end.timestamp = start.timestamp + atoi(cols[4].c_str());

	start.coordinate = Point(atof(cols[18].c_str()),atof(cols[17].c_str()));
	end.coordinate = Point(atof(cols[21].c_str()),atof(cols[20].c_str()));
}

void Trip::print_trip(){
	printf("time: %d to %d\n",start.timestamp,end.timestamp);
	printf("position: (%f %f) to (%f %f)\n",start.coordinate.x,start.coordinate.y,end.coordinate.x,end.coordinate.y);
}

void Trip::resize(int md){
	if(md>0&&duration()>md){
		if(type==REST){
			end.timestamp = start.timestamp+md;
		}else{
			double portion = md*1.0/duration();
			end.coordinate.x = (end.coordinate.x-start.coordinate.x)*portion+start.coordinate.x;
			end.coordinate.y = (end.coordinate.y-start.coordinate.y)*portion+start.coordinate.y;
			end.timestamp = start.timestamp + md+1;
		}
	}
}


/*
 *
 * functions for generating simulated traces of an object
 * based on the real world statistics
 *
 * */


void Queue::push(Point val) {
    if (!isFull()) {
        data[tail] = val;
        tail = (tail + 1) % (size);
    }
    else {
        std::cerr <<" push failed, Queue is full!" << std::endl;
    }
}

void Queue::pop() {
    if (!isEmpty()) {
        //auto temp = data[head];
        head = (head + 1) % (size);
        //return temp;
    }
    else {
        std::cout << "Pop failed, Queue is empty!" << std::endl;
        //return INT_MAX;
    }
}


trace_generator::trace_generator(generator_configuration *conf, Map *m){
	assert(conf && m);

	config = conf;
	map = m;

	ifstream infile(config->meta_path.c_str(), ios::in | ios::binary);
	if(!infile.is_open()){
		log("failed opening %s",config->meta_path.c_str());
		exit(0);
	}

	infile.read((char *)&tweet_count, sizeof(tweet_count));
	tweets = new Point[tweet_count];
	tweets_assign = new uint[tweet_count];
	for(int i=0;i<tweet_count;i++){
		infile.read((char *)&tweets[i], sizeof(Point));
		infile.read((char *)&tweets_assign[i], sizeof(uint));
	}

	infile.read((char *)&core_count, sizeof(core_count));
	cores = new gen_core[core_count];
	for(int i=0;i<core_count;i++){
		cores[i].id = i;
		infile.read((char *)&cores[i].core, sizeof(Point));
		int dest_num = 0;
		infile.read((char *)&dest_num, sizeof(dest_num));
		int d = 0;
		double r = 0.0;
		for(int j=0;j<dest_num;j++){
			infile.read((char *)&d, sizeof(d));
			infile.read((char *)&r, sizeof(r));
			cores[i].destination.push_back(pair<int, double>(d,r));
		}
	}

	for(int i=0;i<tweet_count;i++){
		cores[tweets_assign[i]].assigned_tweets.push_back(i);
	}
	infile.close();
    producer = new Queue[config->num_objects];
}

trace_generator::~trace_generator(){
	map = NULL;
	if(tweets){
		delete []tweets;
	}
	if(tweets_assign){
		delete []tweets_assign;
	}
	if(cores){
		delete []cores;
	}
    if(producer){
        delete []producer;
    }
}

//bool orderzone(ZoneStats *i, ZoneStats *j) { return (i->count>j->count); }
//
//void trace_generator::analyze_trips(const char *path, int limit){
//	struct timeval start = get_cur_time();
//
//	if(total){
//		delete total;
//	}
//	std::ifstream file(path);
//	if(!file.is_open()){
//		log("%s cannot be opened",path);
//		exit(0);
//	}
//	std::string str;
//	//skip the head
//	std::getline(file, str);
//	total = new ZoneStats(0);
//
//	while (std::getline(file, str)&&--limit>0){
//		Trip *t = new Trip(str);
//		t->start.coordinate.x += 0.02*(get_rand_double()-0.5);
//		t->start.coordinate.y += 0.012*(get_rand_double()-0.5);
//		t->end.coordinate.x += 0.02*(get_rand_double()-0.5);
//		t->end.coordinate.y += 0.012*(get_rand_double()-0.5);
//		// a valid trip should be covered by the map,
//		// last for a while and the distance larger than 0
//		if(map->getMBR()->contain(t->start.coordinate)&&
//		   map->getMBR()->contain(t->end.coordinate)&&
//		   t->length()>0&&
//		   t->duration()>0){
//
//			int gids[2];
//			gids[0] = grid->getgridid(&t->start.coordinate);
//			gids[1] = grid->getgridid(&t->end.coordinate);
//			double dist = t->start.coordinate.distance(t->end.coordinate, true);
//			for(int i=0;i<2;i++){
//				int zid = gids[i];
//				int ezid = gids[!i];
//				zones[zid]->count++;
//				zones[zid]->duration += t->duration();
//				zones[zid]->length += dist;
//				total->count++;
//				total->duration += t->duration();
//				total->length += dist;
//			}
//		}
//		delete t;
//	}
//	file.close();
//
//	// reorganize
//	sort(zones.begin(),zones.end(),orderzone);
//	logt("analyze trips in %s",start, path);
//}


/*
 *
 * get the next location according to the distribution of the statistic
 *
 *
 * */


Point trace_generator::get_random_location(int seed){               //下一步位置
	int tid = 0;
	if(seed==-1){
		tid = get_rand_number(tweet_count)-1;
	}else{
		assert(seed<core_count);
		if(cores[seed].assigned_tweets.size()>0){
			tid = get_rand_number(cores[seed].assigned_tweets.size())-1;
			tid = cores[seed].assigned_tweets[tid];
		}else{
			tid = get_rand_number(tweet_count)-1;
		}
	}
	double xval = tweets[tid].x + (0.5-get_rand_double())*100*degree_per_meter_longitude(tweets[tid].y);
	double yval = tweets[tid].y + (0.5-get_rand_double())*100*degree_per_meter_latitude;
	return Point(xval, yval);
}

int trace_generator::get_core(int seed){                            //grid box
	int next_seed = 0;
	if(seed==-1){
		next_seed = get_rand_number(core_count)-1;
		assert(next_seed>=0&&next_seed<core_count);
	}else{
		double target = get_rand_double();
		double cum = 0;
		for(int i=0;i<cores[seed].destination.size();i++){
			cum += cores[seed].destination[i].second;
			if(cum>=target){
				next_seed = cores[seed].destination[i].first;
				break;
			}
		}
	}
	return next_seed;
}

//double random(int max=100){
//    int min = 0;
//    random_device seed;
//    ranlux48 engine(seed());
//    uniform_int_distribution<> distrib(min, max);
//    int randomnum = distrib(engine);
//    return randomnum;
//};

void trace_generator::get_trace(Map *mymap, int obj){
	// use the default map for single thread mode
	if(!mymap){
		mymap = map;
	}
	assert(mymap);
	int cur_core = get_core();
	Point cur_loc;
    if(producer[obj].count()==0){
        cur_loc = get_random_location(cur_core);
    }
    else {
        if(producer[obj].tail == 0){
            cur_loc = producer[obj].data[producer[obj].size-1];
        }
        else{
            cur_loc = producer[obj].data[producer[obj].tail-1];
        }
    }

	//bool rested = false;
	//while(ret.size()<config->duration){
    Point next_loc;
    //uint o = ret.size();
    int restriction = 490 + get_rand_number(100);
    if(tryluck(config->drive_rate)){
        // drive
        cur_core = get_core(cur_core);
        next_loc = get_random_location(cur_core);
        vector<Point *> ret;
        mymap->navigate(ret, &cur_loc, &next_loc, config->drive_speed);
        for(int i=0;i<ret.size()&&producer[obj].count()<restriction;i++){
            producer[obj].push(*ret[i]);
        }
        //cur_loc = next_loc;
        //rested = false;
        //cout<<"drive "<<ret.size()-o<<endl;
    }else if(tryluck(config->walk_rate)){
        //walk
        next_loc = get_random_location(cur_core);
        const double step = config->walk_speed/next_loc.distance(cur_loc, true);
        //for(double portion = 0;portion<1&&producer[obj].size()<config->duration;){                           // not >config->duration   must change
        for(double portion = 0;portion<1&&producer[obj].count()<restriction;){
            double px = cur_loc.x+portion*(next_loc.x - cur_loc.x);
            double py = cur_loc.y+portion*(next_loc.y - cur_loc.y);
            producer[obj].push(Point(px,py));
            portion += step;
        }
        //cur_loc = next_loc;
        //rested = false;
        //cout<<"walk "<<ret.size()-o<<endl;
    }else {
        // stay here
        int dur = config->max_rest_time*get_rand_double();
        for(int i=0;i<dur&&producer[obj].count()<restriction;i++){
            producer[obj].push(Point(cur_loc));
        }
        //rested = true;
        //cout<<"rest "<<ret.size()-o<<endl;
    }
}

void *gentrace_unit(void *arg){
	query_context *ctx = (query_context *)arg;
	trace_generator *gen = (trace_generator *)ctx->target[0];
	Point *result = (Point *)ctx->target[1];
	Map *mymap = gen->map->clone();
	while(true){
		// pick one object for generating
		size_t start = 0;
		size_t end = 0;
		if(!ctx->next_batch(start,end)){
			break;
		}
		for(int obj=start;obj<end;obj++){
			//log("%d",obj);
            while(gen->producer[obj].count()<gen->config->cur_duration+5){               //in case producer is exactly empty
                gen->get_trace(mymap,obj);
            }
			// copy to target
			for(int i=0;i<gen->config->cur_duration;i++){
				result[i*gen->config->num_objects+obj] = gen->producer[obj].data[gen->producer[obj].head];
                gen->producer[obj].pop();
			}
		}
	}
	delete mymap;
	return NULL;
}


Point *trace_generator::generate_trace(){
	srand (time(NULL));
	struct timeval start = get_cur_time();
	Point *ret = (Point *)malloc(config->cur_duration*config->num_objects*sizeof(Point));
	pthread_t threads[config->num_threads];
	query_context tctx;
	tctx.config = config;
	tctx.target[0] = (void *)this;
	tctx.target[1] = (void *)ret;
    //tctx.target[2] = (void *)producer;
	tctx.num_units = config->num_objects;
	tctx.report_gap = 1;
	tctx.num_batchs = 100000;
	for(int i=0;i<config->num_threads;i++){
		pthread_create(&threads[i], NULL, gentrace_unit, (void *)&tctx);
	}
	for(int i = 0; i < config->num_threads; i++ ){
		void *status;
		pthread_join(threads[i], &status);
	}
	logt("generate traces",start);
	return ret;
}
