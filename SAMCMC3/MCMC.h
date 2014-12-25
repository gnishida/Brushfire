#pragma once

#include <list>
#include <vector>
#include <time.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#define MAX_DIST 99
#define BF_CLEARED -1
#define NUM_FEATURES 5
#define NUM_PEOPLE_TYPE 10
#define NUM_LAYERS 5

struct Point2D {
	int x;
	int y;
};

struct BF_QueueElement {
	int pos;
	int type;

	BF_QueueElement(int pos, int type): pos(pos), type(type) {}
};

class MCMC {
public:
	MCMC() {}

public:
	void findBestPlan(int** zone, int* city_size);
	void showZone(int city_size, int* zone, char* filename);
	void loadZone(int city_size, int* zone, char* filename);
	void saveZone(int city_size, int* zone, char* filename);
	void dumpZone(int city_size, int* zone);
	void dumpDist(int city_size, int* dist, int featureId);

private:
	float randf();
	float randf(float a, float b);
	int sampleFromCdf(float* cdf, int num);
	int sampleFromPdf(float* pdf, int num);
	bool isOcc(int* obst, int s, int featureId);
	int distance(int city_size, int pos1, int pos2);
	void clearCell(int* dist, int* obst, int s, int featureId);
	void raise(int city_size, std::list<std::pair<int, int> >& queue, int* dist, int* obst, bool* toRaise, int s, int featureId);
	void lower(int city_size, std::list<std::pair<int, int> >& queue, int* dist, int* obst, bool* toRaise, int s, int featureId);
	void updateDistanceMap(int city_size, std::list<std::pair<int, int> >& queue, int* zone, int* dist, int* obst, bool* toRaise);
	void setStore(std::list<std::pair<int, int> >& queue, int* zone, int* dist, int* obst, bool* toRaise, int s, int featureId);
	void removeStore(std::list<std::pair<int, int> >& queue, int* zone, int* dist, int* obst, bool* toRaise, int s, int featureId);
	float min3(float distToStore, float distToAmusement, float distToFactory);
	float computeScore(int city_size, int* zone, int* dist);
	int check(int city_size, int* zone, int* dist);
	void generateZoningPlan(int city_size, int* zone, std::vector<float> zoneTypeDistribution);
	void optimize(int city_size, int max_iterations, int* bestZone);
	void optimize2(int city_size, int max_iterations, int* bestZone);
	
};

