/*
 * Parser.cpp
 *
 *  Created on: May 9, 2020
 *      Author: teng
 */



#include "../index/RTree.h"
#include <queue>
#include <fstream>

#define BOX_GLOBAL_MIN 100000.0
#define BOX_GLOBAL_MAX -100000.0


bool PolygonSearchCallback(short * i, box poly_mbr,void* arg){
    vector<pair<short, box>> * ret = (vector<pair<short, box>> *)arg;
    ret->push_back(make_pair(*i, poly_mbr));
    return true;
}

int main(int argc, char** argv) {
    RTree<short *, double, 2, double> poly_rtree;
    cout << "size of rtree" << sizeof(poly_rtree) << endl;

    ifstream file("mbr_in_a_sst.txt");
    string line;
    if (!file.is_open()) {
        cerr << "file cannot opened" << endl;
        return 1;
    }
    box * mbrs = new box[100];
    int mbrs_count = 0;
    while (getline(file, line)) {
        stringstream ss(line);
        string token;
        ss.ignore(9, ' '); // ignore "POLYGON(("
        double number;
        stringstream numStream;
        getline(ss, token, ',');            //low0 low1
        numStream << token;
        numStream >> mbrs[mbrs_count].low[0];
        numStream >> mbrs[mbrs_count].low[1];
        numStream.clear();
        getline(ss, token, ',');            //ignore
        getline(ss, token, ',');            //high0 high1
        numStream << token;
        numStream >> mbrs[mbrs_count].high[0];
        numStream >> mbrs[mbrs_count].high[1];
        numStream.clear();
        mbrs_count++;
    }
    file.close();


    timeval start = get_cur_time();
    for(uint i = 0; i < 100; ++i){
        poly_rtree.Insert(mbrs[i].low, mbrs[i].high, new short(i));
    }
    cout << "Inserted" << endl;

    double mid_x = -87.908503;
    double mid_y = 42.006803;
    double edge_length = 0.1;
    box search_area(mid_x - edge_length/2, mid_y - edge_length/2, mid_x + edge_length/2, mid_y + edge_length/2);
    vector<pair<short, box>> ret;
    poly_rtree.Search(search_area.low, search_area.high, PolygonSearchCallback, (void *)&ret);
    for(int i = 0; i < ret.size(); ++i){
        cout << "ret[" << i << "]:" << ret[i].first << endl;
        ret[i].second.print();
    }

    cout << "size of rtree" << sizeof(poly_rtree) << endl;
	return 0;
}



