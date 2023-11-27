/*
 * trace.cpp
 *
 *  Created on: Jan 22, 2021
 *      Author: teng
 */

#include "../tracing/trace.h"

#include "../geometry/Map.h"
#include "../util/config.h"
#include <vector>
#include <stdlib.h>

using namespace std;
#define PI 3.14159265

int main(int argc, char **argv){

	configuration config = get_parameters(argc, argv);
	tracer *tr = new tracer(&config);
	//tr->print_trace();
	tr->process();
	delete tr;

//	Map *m = new Map(config.map_path);
//	m->print_region();
//	vector<Point *> result;
//	m->navigate(result, new Point(-87.61353222612192,41.75837880179237), new Point(-88.11615669701743,41.94455236873503), 200);
//	print_linestring(result);
//	delete m;
	return 0;
}
