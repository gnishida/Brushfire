#include <list>
#include <vector>
#include <time.h>

#define CITY_SIZE 200
#define BF_MAX_DIST 99
#define BF_CLEARED -1
#define BF_TYPE_RAISE 0
#define BF_TYPE_LOWER 1
#define MAX_ITERATIONS 100
#define NUM_FEATURES 1 // 5

int bf_count = 0;

struct Point2D {
	int x;
	int y;
};

struct BF_QueueElement {
	int pos;
	int type;

	BF_QueueElement(int pos, int type): pos(pos), type(type) {}
};

unsigned int rand(unsigned int* randx) {
    *randx = *randx * 1103515245 + 12345;
    return (*randx)&2147483647;
}

float randf(unsigned int* randx) {
	return rand(randx) / (float(2147483647) + 1);
}

float randf(unsigned int* randx, float a, float b) {
	return randf(randx) * (b - a) + a;
}

int sampleFromCdf(unsigned int* randx, float* cdf, int num) {
	float rnd = randf(randx, 0, cdf[num-1]);

	for (int i = 0; i < num; ++i) {
		if (rnd <= cdf[i]) return i;
	}

	return num - 1;
}

int sampleFromPdf(unsigned int* randx, float* pdf, int num) {
	if (num == 0) return 0;

	float cdf[40];
	cdf[0] = pdf[0];
	for (int i = 1; i < num; ++i) {
		if (pdf[i] >= 0) {
			cdf[i] = cdf[i - 1] + pdf[i];
		} else {
			cdf[i] = cdf[i - 1];
		}
	}

	return sampleFromCdf(randx, cdf, num);
}

void dumpZone(int* zone) {
	printf("<<< Zone Map >>>\n");
	for (int r = 0; r < CITY_SIZE; ++r) {
		for (int c = 0; c < CITY_SIZE; ++c) {
			printf("%2d ", zone[r * CITY_SIZE + c]);
		}
		printf("\n");
	}
	printf("\n");
}

void dumpDist(int* dist) {
	printf("<<< Distance Map >>>\n");
	for (int r = 0; r < CITY_SIZE; ++r) {
		for (int c = 0; c < CITY_SIZE; ++c) {
			printf("%2d ", dist[r * CITY_SIZE + c]);
		}
		printf("\n");
	}
	printf("\n");
}

inline bool isOcc(int* obst, int pos) {
	return obst[pos] == pos;
}

inline int distance(int pos1, int pos2) {
	int x1 = pos1 % CITY_SIZE;
	int y1 = pos1 / CITY_SIZE;
	int x2 = pos2 % CITY_SIZE;
	int y2 = pos2 / CITY_SIZE;

	return abs(x1 - x2) + abs(y1 - y2);
}

void clearCell(int* dist, int* obst, int s) {
	dist[s] = BF_MAX_DIST;
	obst[s] = BF_CLEARED;
}

void raise(std::list<int>& queue, int* dist, int* obst, bool* toRaise, int s) {
	Point2D adj[] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

	int x = s % CITY_SIZE;
	int y = s / CITY_SIZE;

	for (int i = 0; i < 4; ++i) {
		int nx = x + adj[i].x;
		int ny = y + adj[i].y;

		if (nx < 0 || nx >= CITY_SIZE || ny < 0 || ny >= CITY_SIZE) continue;
		int n = ny * CITY_SIZE + nx;

		if (obst[n] != BF_CLEARED && !toRaise[n]) {
			if (!isOcc(obst, obst[n])) {
				clearCell(dist, obst, n);
				toRaise[n] = true;
			}
			queue.push_back(n);
		}
	}

	toRaise[s] = false;
}

void lower(std::list<int>& queue, int* dist, int* obst, bool* toRaise, int s) {
	Point2D adj[] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

	int x = s % CITY_SIZE;
	int y = s / CITY_SIZE;

	int ox = obst[s] % CITY_SIZE;
	int oy = obst[s] / CITY_SIZE;

	for (int i = 0; i < 4; ++i) {
		int nx = x + adj[i].x;
		int ny = y + adj[i].y;

		if (nx < 0 || nx >= CITY_SIZE || ny < 0 || ny >= CITY_SIZE) continue;
		int n = ny * CITY_SIZE + nx;

		if (!toRaise[n]) {
			int d = distance(obst[s], n);
			if (d < dist[n]) {
				dist[n] = d;
				obst[n] = obst[s];
				queue.push_back(n);
			}
		}
	}
}

void updateDistanceMap(std::list<int>& queue, int* zone, int* dist, int* obst, bool* toRaise) {
	while (!queue.empty()) {
		int s = queue.front();
		queue.pop_front();

		if (toRaise[s]) {
			raise(queue, dist, obst, toRaise, s);
		} else if (isOcc(obst, obst[s])) {
			lower(queue, dist, obst, toRaise, s);
		}

		bf_count++;
	}
}

void setStore(std::list<int>& queue, int* zone, int* dist, int* obst, bool* toRaise, int s) {
	// put stores
	obst[s] = s;
	dist[s] = 0;

	queue.push_back(s);
}

void removeStore(std::list<int>& queue, int* zone, int* dist, int* obst, bool* toRaise, int s) {
	clearCell(dist, obst, s);

	toRaise[s] = true;

	queue.push_back(s);
}

/**
 * 計算したdistance mapが正しいか、チェックする。
 */
int check(int* zone, int* dist) {
	int count = 0;

	for (int r = 0; r < CITY_SIZE; ++r) {
		for (int c = 0; c < CITY_SIZE; ++c) {
			int min_dist = BF_MAX_DIST;
			for (int r2 = 0; r2 < CITY_SIZE; ++r2) {
				for (int c2 = 0; c2 < CITY_SIZE; ++c2) {
					if (zone[r2 * CITY_SIZE + c2] == 1) {
						int d = distance(r2 * CITY_SIZE + c2, r * CITY_SIZE + c);
						if (d < min_dist) {
							min_dist = d;
						}
					}
				}
			}

			if (dist[r * CITY_SIZE + c] != min_dist) {
				if (count == 0) {
					printf("e.g. (%d, %d)\n", c, r);
				}
				count++;
			}
		}
	}
	
	if (count > 0) {
		printf("Check results: #error cells = %d\n", count);
	}

	return count;
}

/**
 * ゾーンプランを生成する。
 */
void generateZoningPlan(int* zone, std::vector<float> zoneTypeDistribution) {
	std::vector<float> numRemainings(NUM_FEATURES + 1);
	for (int i = 0; i < NUM_FEATURES + 1; ++i) {
		numRemainings[i] = CITY_SIZE * CITY_SIZE * zoneTypeDistribution[i];
	}

	unsigned int randx = 0;

	for (int r = 0; r < CITY_SIZE; ++r) {
		for (int c = 0; c < CITY_SIZE; ++c) {
			int type = sampleFromPdf(&randx, numRemainings.data(), numRemainings.size());
			zone[r * CITY_SIZE + c] = type;
			numRemainings[type] -= 1;
		}
	}
}

int main() {
	time_t start, end;

	int zone[CITY_SIZE * CITY_SIZE];
	int dist[CITY_SIZE * CITY_SIZE];
	int obst[CITY_SIZE * CITY_SIZE];
	bool toRaise[CITY_SIZE * CITY_SIZE];
	
	// initialize the zone
	std::vector<float> zoneTypeDistribution(6);
	zoneTypeDistribution[0] = 0.5f; // 住宅
	zoneTypeDistribution[1] = 0.2f; // 商業
	zoneTypeDistribution[2] = 0.1f; // 工場
	zoneTypeDistribution[3] = 0.1f; // 公園
	zoneTypeDistribution[4] = 0.05f; // アミューズメント
	zoneTypeDistribution[5] = 0.05f; // 学校・図書館

	// 初期プランを生成
	start = clock();
	generateZoningPlan(zone, zoneTypeDistribution);
	end = clock();
	printf("generateZoningPlan: %lf\n", (double)(end-start)/CLOCKS_PER_SEC);

	//dumpZone(zone);

	// キューのセットアップ
	std::list<int> queue;
	for (int i = 0; i < CITY_SIZE * CITY_SIZE; ++i) {
		toRaise[i] = false;
		if (zone[i] == 1) {
			setStore(queue, zone, dist, obst, toRaise, i);
		} else {
			dist[i] = BF_MAX_DIST;
			obst[i] = BF_CLEARED;
		}
	}

	updateDistanceMap(queue, zone, dist, obst, toRaise);
	//dumpDist(dist);
	check(zone, dist);
	
	bf_count = 0;
	for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
		printf("iter = %d\n", iter);

		queue.clear();

		int s1;
		while (true) {
			s1 = rand() % (CITY_SIZE * CITY_SIZE);
			if (zone[s1] == 1) break;
		}

		int s2;
		while (true) {
			s2 = rand() % (CITY_SIZE * CITY_SIZE);
			if (zone[s2] == 0) break;
		}

		// move a store
		zone[s1] = 0;
		removeStore(queue, zone, dist, obst, toRaise, s1);
		zone[s2] = 1;
		setStore(queue, zone, dist, obst, toRaise, s2);
		updateDistanceMap(queue, zone, dist, obst, toRaise);
		
		if (check(zone, dist) > 0) break;
		//dumpZone(zone);
		//dumpDist(dist);
	}

	printf("avg bf_count = %d\n", bf_count / MAX_ITERATIONS);
	printf("total bf_count = %d\n", bf_count);

	return 0;
}
