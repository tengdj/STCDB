/*
 * geometry.h
 * included the most commonly accessed data structures
 *
 *  Created on: Jan 11, 2021
 *      Author: teng
 */

#ifndef DATAGEN_GEOMETRY_H_
#define DATAGEN_GEOMETRY_H_

#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <math.h>
#include <string>
#include <queue>
#include <iostream>
#include "../util/util.h"
#include "../cuda/cuda_util.cuh"
#include "../cuda/hilbert_curve.cuh"

using namespace std;
class Point{
public:
	double x = 0;
	double y = 0;
	Point(double xx, double yy){
		x = xx;
		y = yy;
	}
	Point(){}
	Point(Point *p){
		x = p->x;
		y = p->y;
	}
	~Point(){}
	double distance(Point &p, bool geography = false){
		return distance(&p, geography);
	}
	double distance(Point *p, bool geography = false){
		double dx = x-p->x;
		double dy = y-p->y;
		if(geography){
			dy = dy/degree_per_meter_latitude;
			dx = dx/degree_per_meter_longitude(y);
		}
		return sqrt(dx*dx+dy*dy);
	}
	bool equals(Point &p){
		return p.x==x&&p.y==y;
	}
	void print(){
		fprintf(stderr,"POINT (%f %f)\n",x,y);
	}

};

class box{
public:
	double low[2] = {100000.0,100000.0};
	double high[2] = {-100000.0,-100000.0};

	box(){}
	box(double lowx, double lowy, double highx, double highy){
		low[0] = lowx;
		low[1] = lowy;
		high[0] = highx;
		high[1] = highy;
	}
	box(box *b){
		low[0] = b->low[0];
		high[0] = b->high[0];
		low[1] = b->low[1];
		high[1] = b->high[1];
	}

    box(__uint128_t value){                   //add
        low[0] = uint_to_float((uint)((value >> 66) & ((1ULL << 22) - 1)));
        low[1] = uint_to_float((uint)((value >> 44) & ((1ULL << 22) - 1)));
        high[0] = uint_to_float((uint)((value >> 22) & ((1ULL << 22) - 1)));
        high[1] = uint_to_float((uint)(value & ((1ULL << 22) - 1)));
        cout<<low[0]<<" "<<low[1]<<" "<<high[0]<<" "<<high[1]<<endl;
    }

	void update(Point p){
		if(low[0]>p.x){
			low[0] = p.x;
		}
		if(high[0]<p.x){
			high[0] = p.x;
		}

		if(low[1]>p.y){
			low[1] = p.y;
		}
		if(high[1]<p.y){
			high[1] = p.y;
		}
	}

	void update(box &b){
		update(Point(b.low[0],b.low[1]));
		update(Point(b.low[0],b.high[1]));
		update(Point(b.high[0],b.low[1]));
		update(Point(b.high[0],b.high[1]));
	}

	bool intersect(box &target){
		return !(target.low[0]>high[0]||
				 target.high[0]<low[0]||
				 target.low[1]>high[1]||
				 target.high[1]<low[1]);
	}
	bool contain(box &target){
		return target.low[0]>=low[0]&&
			   target.high[0]<=high[0]&&
			   target.low[1]>=low[1]&&
			   target.high[1]<=high[1];
	}
	bool contain(Point &p){
		return p.x>=low[0]&&
			   p.x<=high[0]&&
			   p.y>=low[1]&&
			   p.y<=high[1];
	}

	double area(bool geography = false){
		if(!geography){
			return (high[0]-low[0])*(high[1]-low[1]);
		}else{
			double h = (high[1]-low[1])/degree_per_meter_latitude;
			double top = (high[0]-low[0])/degree_per_meter_longitude(high[1]);
			double bot = (high[0]-low[0])/degree_per_meter_longitude(low[1]);
			return (top+bot)*h/2;
		}
	}

	// width and height is in meters
	double height(bool geography = false){
		if(!geography){
			return high[1]-low[1];
		}else{
			return (high[1]-low[1])/degree_per_meter_latitude;
		}
	}
	double width(bool geography = false){
		if(!geography){
			return high[0]-low[0];
		}else{
			return (high[0]-low[0])/degree_per_meter_longitude(low[1]);
		}
	}



	double distance(Point &p, bool geography = false){
		if(this->contain(p)){
			return 0;
		}
		double dx = max(fabs(p.x-(low[0]+high[0])/2) - (high[0]-low[0])/2, 0.0);
		double dy = max(fabs(p.y-(low[1]+high[1])/2) - (high[1]-low[1])/2, 0.0);
		if(geography){
			dy = dy/degree_per_meter_latitude;
			dx = dx/degree_per_meter_longitude(p.y);
		}
		return sqrt(dx * dx + dy * dy);
	}


	double max_distance(Point &p){
		double md = 0;
		double dist = (p.x-low[0])*(p.x-low[0])+(p.y-low[1])*(p.y-low[1]);
		if(dist>md){
			md = dist;
		}
		dist = (p.x-low[0])*(p.x-low[0])+(p.y-high[1])*(p.y-high[1]);
		if(dist>md){
			md = dist;
		}
		dist = (p.x-high[0])*(p.x-high[0])+(p.y-low[1])*(p.y-low[1]);
		if(dist>md){
			md = dist;
		}
		dist = (p.x-high[0])*(p.x-high[0])+(p.y-high[1])*(p.y-high[1]);
		if(dist>md){
			md = dist;
		}
		return sqrt(md);
	}

	void print_vertices(){
		fprintf(stderr,"%f %f, %f %f, %f %f, %f %f, %f %f",
					low[0],low[1],
					high[0],low[1],
					high[0],high[1],
					low[0],high[1],
					low[0],low[1]);
	}

	void print(){
		fprintf(stderr,"POLYGON((");
		print_vertices();
		fprintf(stderr,"))\n");
	}

	void to_squre(bool geography=false){
		double dx = high[0]-low[0];
		double dy = high[1]-low[1];
		int bigger_one = dy>dx;
		double difference = (high[bigger_one]-low[bigger_one])-(high[!bigger_one]-low[!bigger_one]);

		if(geography){
			dy = dy/degree_per_meter_latitude;
			dx = dx/degree_per_meter_longitude(low[1]);
			if(dy>dx){//extend horizontal dimension
				difference = (dy-dx)*degree_per_meter_longitude(low[1]);
			}else{
				difference = (dx-dy)*degree_per_meter_latitude;
			}
			bigger_one = dy>dx;
		}
		low[!bigger_one] = low[!bigger_one]-difference/2;
		high[!bigger_one] = high[!bigger_one]+difference/2;
	}

};


class Grid{

public:
	double step_x = 0;
	double step_y = 0;
	int dimx = 0;
	int dimy = 0;
	box space;
	Grid(box &mbr, double s){
		space = mbr;
		rasterize(s);
	}

	int get_grid_num(){
		return dimx*dimy;
	}
	/*
	 *
	 * member functions for grid class
	 * each grid with width and height s
	 *
	 * */
	void rasterize(double s){
		dimy = space.height(true)/s+1;
		dimx = dimy;
		step_x = space.width()/dimx;
		step_y = space.height()/dimy;
	}

	int getgridid(double x, double y){
		assert(step_x>0&&step_y>0);
		int offsety = (y-space.low[1])/step_y;
		int offsetx = (x-space.low[0])/step_x;
		int gid = dimx*offsety+offsetx;
		assert(gid<=dimx*dimy && gid>=0);
		return gid;
	}

	int getgridid(Point *p){
		return getgridid(p->x, p->y);
	}

	box getgrid(int x, int y){
		double lowx = space.low[0] + x*step_x;
		double highx = lowx + step_x;
		double lowy = space.low[1] + y*step_y;
		double highy = lowy + step_y;
		return box(lowx,lowy,highx,highy);
	}

	box getgrid(Point *p){
		int gid = getgridid(p);
		return getgrid(gid%dimx,gid/dimx);
	}


	inline size_t border_grids(Point *p, double x_buffer, double y_buffer){
		int offsety = (p->y-space.low[1])/step_y;
		int offsetx = (p->x-space.low[0])/step_x;

		bool left = offsetx-1>=0&&offsetx*step_x+space.low[0]>p->x-x_buffer;
		bool right = offsetx+1<dimx&&(offsetx+1)*step_x+space.low[0]<p->x+x_buffer;
		bool bottom = offsety-1>=0&&offsety*step_y+space.low[1]>p->y-y_buffer;
		bool top = offsety+1<dimy&&(offsety+1)*step_y+space.low[1]<p->y+y_buffer;
		size_t gid = 0;
		gid <<= 1;
		gid |= left;
		gid <<= 1;
		gid |= right;
		gid <<= 1;
		gid |= top;
		gid <<= 1;
		gid |= bottom;
		return gid;
	}



	Point get_random_point(int xoff = -1, int yoff = -1){
		double xrand = get_rand_double();
		double yrand = get_rand_double();
		double xval,yval;
		if(xoff==-1||yoff==-1){
			xval = space.low[0]+xrand*(space.high[0]-space.low[0]);
			yval = space.low[1]+yrand*(space.high[1]-space.low[1]);
		}else{
			xval = space.low[0]+xoff*step_x+xrand*step_x;
			yval = space.low[1]+yoff*step_y+yrand*step_y;
		}
		return Point(xval, yval);
	}

	void print(){
		fprintf(stderr,"MULTIPOLYGON(");
		for(int i=0;i<dimx;i++){
			for(int j=0;j<dimy;j++){
				if(i>0||j>0){
					fprintf(stderr,",");
				}
				fprintf(stderr,"((");
				box mbr(space.low[0]+i*step_x,space.low[1]+j*step_y,space.low[0]+(i+1)*step_x,space.low[1]+(j+1)*step_y);
				mbr.print_vertices();
				fprintf(stderr,"))");
			}
		}
		fprintf(stderr,")\n");


	}

};


/*
 * some utility functions shared by other classes
 *
 * */

inline double distance_point_to_segment(double x, double y,
										double x1, double y1,
										double x2, double y2) {
	//the segment is vertical
	if(x1==x2) {
		if(y>max(y1, y2)) {
			return sqrt((x-x1)*(x-x1)+(y-max(y1, y2))*(y-max(y1, y2)));
		}else if(y<min(y1, y2)){
			return sqrt((x-x1)*(x-x1)+(min(y1, y2)-y)*(min(y1, y2)-y));
		}else {
			return fabs(x-x1);
		}
	}

	//the segment is horizontal
	if(y1==y2) {
		if(x>max(x1, x2)) {
			return sqrt((y-y1)*(y-y1)+(x-max(x1, x2))*(x-max(x1, x2)));
		}else if(x<min(x1, x2)){
			return sqrt((y-y1)*(y-y1)+(min(x1, x2)-x)*(min(x1, x2)-x));
		}else {
			return fabs(y-y1);
		}
	}


	double a = (y1-y2)/(x1-x2);
	double b = y1 - a*x1;
	double a1 = -1*(1/a);
	double b1 = y-a1*x;

	double nx = (b1-b)/(a-a1);
	double ny = a1*nx+b1;
	//the cross point is outside the segment
	if(nx>max(x1, x2)||nx<min(x1, x2)) {
		return sqrt(min((x-x1)*(x-x1)+(y-y1)*(y-y1), (x-x2)*(x-x2)+(y-y2)*(y-y2)));
	}else {
		return sqrt((nx-x)*(nx-x)+(ny-y)*(ny-y));
	}

}


inline void print_linestring(vector<Point *> trajectory, double sample_rate=1.0){
	assert(sample_rate<=1&&sample_rate>0);
	fprintf(stderr,"LINESTRING (");
	bool first = true;
	for(int i=0;i<trajectory.size();i++){

		if(tryluck(sample_rate)){
			if(!first){
				fprintf(stderr,",");
			}else{
				first = false;
			}
			fprintf(stderr,"%f %f",trajectory[i]->x,trajectory[i]->y);
		}
	}

	fprintf(stderr,")\n");
}

inline void print_points(vector<Point *> trajectory, uint max_num = INT_MAX){

	double sample_rate = 1.0*max_num/trajectory.size();
	if(sample_rate>1){
		sample_rate = 1.0;
	}

	fprintf(stderr,"MULTIPOINT (");
	bool first = true;
	for(int i=0;i<trajectory.size();i++){

		if(tryluck(sample_rate)){
			if(!first){
				fprintf(stderr,",");
			}else{
				first = false;
			}
			fprintf(stderr,"%f %f",trajectory[i]->x,trajectory[i]->y);
		}
	}

	fprintf(stderr,")\n");
}

inline void print_points(Point *trajectory, size_t num_objects, uint max_num = INT_MAX){
	double sample_rate = 1.0*max_num/num_objects;
	if(sample_rate>1){
		sample_rate = 1.0;
	}
	fprintf(stderr,"MULTIPOINT (");
	bool first = true;
	for(int i=0;i<num_objects;i++){
		if(tryluck(sample_rate)){
			if(!first){
				fprintf(stderr,",");
			}else{
				first = false;
			}
			fprintf(stderr,"%f %f",trajectory[i].x,trajectory[i].y);
		}
	}

	fprintf(stderr,")\n");
}

#endif /* DATAGEN_GEOMETRY_H_ */
