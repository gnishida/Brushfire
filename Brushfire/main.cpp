#include <list>
#include <vector>
#include <time.h>

#define CITY_SIZE 200
#define BF_MAX_DIST 99
#define BF_CLEARED -1
#define BF_TYPE_RAISE 0
#define BF_TYPE_LOWER 1
#define MAX_ITERATIONS 1000
#define NUM_FEATURES 5

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

void dumpDist(int* dist, int featureId) {
	printf("<<< Distance Map (featureId = %d) >>>\n", featureId);
	for (int r = 0; r < CITY_SIZE; ++r) {
		for (int c = 0; c < CITY_SIZE; ++c) {
			printf("%2d ", dist[(r * CITY_SIZE + c) * NUM_FEATURES + featureId]);
		}
		printf("\n");
	}
	printf("\n");
}

inline bool isOcc(int* obst, int s, int featureId) {
	return obst[s * NUM_FEATURES + featureId] == s;
}

inline int distance(int pos1, int pos2) {
	int x1 = pos1 % CITY_SIZE;
	int y1 = pos1 / CITY_SIZE;
	int x2 = pos2 % CITY_SIZE;
	int y2 = pos2 / CITY_SIZE;

	return abs(x1 - x2) + abs(y1 - y2);
}

void clearCell(int* dist, int* obst, int s, int featureId) {
	dist[s * NUM_FEATURES + featureId] = BF_MAX_DIST;
	obst[s * NUM_FEATURES + featureId] = BF_CLEARED;
}

void raise(std::list<std::pair<int, int> >& queue, int* dist, int* obst, bool* toRaise, int s, int featureId) {
	Point2D adj[] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

	int x = s % CITY_SIZE;
	int y = s / CITY_SIZE;

	for (int i = 0; i < 4; ++i) {
		int nx = x + adj[i].x;
		int ny = y + adj[i].y;

		if (nx < 0 || nx >= CITY_SIZE || ny < 0 || ny >= CITY_SIZE) continue;
		int n = ny * CITY_SIZE + nx;

		if (obst[n * NUM_FEATURES + featureId] != BF_CLEARED && !toRaise[n]) {
			if (!isOcc(obst, obst[n * NUM_FEATURES + featureId], featureId)) {
				clearCell(dist, obst, n, featureId);
				toRaise[n] = true;
			}
			queue.push_back(std::make_pair(n, featureId));
		}
	}

	toRaise[s] = false;
}

void lower(std::list<std::pair<int, int> >& queue, int* dist, int* obst, bool* toRaise, int s, int featureId) {
	Point2D adj[] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

	int x = s % CITY_SIZE;
	int y = s / CITY_SIZE;

	for (int i = 0; i < 4; ++i) {
		int nx = x + adj[i].x;
		int ny = y + adj[i].y;

		if (nx < 0 || nx >= CITY_SIZE || ny < 0 || ny >= CITY_SIZE) continue;
		int n = ny * CITY_SIZE + nx;

		if (!toRaise[n]) {
			int d = distance(obst[s * NUM_FEATURES + featureId], n);
			if (d < dist[n * NUM_FEATURES + featureId]) {
				dist[n * NUM_FEATURES + featureId] = d;
				obst[n * NUM_FEATURES + featureId] = obst[s * NUM_FEATURES + featureId];
				queue.push_back(std::make_pair(n, featureId));
			}
		}
	}
}

void updateDistanceMap(std::list<std::pair<int, int> >& queue, int* zone, int* dist, int* obst, bool* toRaise) {
	while (!queue.empty()) {
		std::pair<int, int> s = queue.front();
		queue.pop_front();

		if (toRaise[s.first]) {
			raise(queue, dist, obst, toRaise, s.first, s.second);
		} else if (isOcc(obst, obst[s.first * NUM_FEATURES + s.second], s.second)) {
			lower(queue, dist, obst, toRaise, s.first, s.second);
		}

		bf_count++;
	}
}

void setStore(std::list<std::pair<int, int> >& queue, int* zone, int* dist, int* obst, bool* toRaise, int s, int featureId) {
	// put stores
	obst[s * NUM_FEATURES + featureId] = s;
	dist[s * NUM_FEATURES + featureId] = 0;

	queue.push_back(std::make_pair(s, featureId));
}

void removeStore(std::list<std::pair<int, int> >& queue, int* zone, int* dist, int* obst, bool* toRaise, int s, int featureId) {
	clearCell(dist, obst, s, featureId);

	toRaise[s] = true;

	queue.push_back(std::make_pair(s, featureId));
}

/**
 * 計算したdistance mapが正しいか、チェックする。
 */
int check(int* zone, int* dist) {
	int count = 0;

	for (int r = 0; r < CITY_SIZE; ++r) {
		for (int c = 0; c < CITY_SIZE; ++c) {
			for (int k = 0; k < NUM_FEATURES; ++k) {
				int min_dist = BF_MAX_DIST;
				for (int r2 = 0; r2 < CITY_SIZE; ++r2) {
					for (int c2 = 0; c2 < CITY_SIZE; ++c2) {
						if (zone[r2 * CITY_SIZE + c2] - 1 == k) {
							int d = distance(r2 * CITY_SIZE + c2, r * CITY_SIZE + c);
							if (d < min_dist) {
								min_dist = d;
							}
						}
					}
				}

				if (dist[(r * CITY_SIZE + c) * NUM_FEATURES + k] != min_dist) {
					if (count == 0) {
						printf("e.g. (%d, %d) featureId = %d\n", c, r, k);
					}
					count++;
				}
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

	int* zone;
	zone = (int*)malloc(sizeof(int) * CITY_SIZE * CITY_SIZE);
	int* dist;
	dist = (int*)malloc(sizeof(int) * CITY_SIZE * CITY_SIZE * NUM_FEATURES);
	int* obst;
	obst = (int*)malloc(sizeof(int) * CITY_SIZE * CITY_SIZE * NUM_FEATURES);
	bool* toRaise;
	toRaise = (bool*)malloc(CITY_SIZE * CITY_SIZE);
	
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

	// キューのセットアップ
	std::list<std::pair<int, int> > queue;
	for (int i = 0; i < CITY_SIZE * CITY_SIZE; ++i) {
		toRaise[i] = false;
		for (int k = 0; k < NUM_FEATURES; ++k) {
			if (zone[i] - 1 == k) {
				setStore(queue, zone, dist, obst, toRaise, i, k);
			} else {
				dist[i * NUM_FEATURES + k] = BF_MAX_DIST;
				obst[i * NUM_FEATURES + k] = BF_CLEARED;
			}
		}
	}

	updateDistanceMap(queue, zone, dist, obst, toRaise);

	//dumpZone(zone);
	//dumpDist(dist, 4);
	//check(zone, dist);
	
	bf_count = 0;
	for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
		printf("iter = %d\n", iter);

		queue.clear();

		// ２つのセルのゾーンタイプを交換
		int s1;
		while (true) {
			s1 = rand() % (CITY_SIZE * CITY_SIZE);
			if (zone[s1] > 0) break;
		}

		int s2;
		while (true) {
			s2 = rand() % (CITY_SIZE * CITY_SIZE);
			if (zone[s2] == 0) break;
		}

		// move a store
		int featureId = zone[s1] - 1;
		zone[s1] = 0;
		removeStore(queue, zone, dist, obst, toRaise, s1, featureId);
		zone[s2] = featureId + 1;
		setStore(queue, zone, dist, obst, toRaise, s2, featureId);
		updateDistanceMap(queue, zone, dist, obst, toRaise);
		
		//dumpZone(zone);
		//dumpDist(dist, 4);
		//if (check(zone, dist) > 0) break;
	}

	printf("avg bf_count = %d\n", bf_count / MAX_ITERATIONS);
	printf("total bf_count = %d\n", bf_count);

	return 0;
}
