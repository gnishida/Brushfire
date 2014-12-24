/**
 * MCMCを使って、理想的なゾーン計画を探す。
 * 距離マップは、Brushfireアルゴリズムを使って計算する。
 * スコアは、とりあえずシングルスレッドで実装してみよう。容易にGPU版で速度アップできると思われる。。。
 *
 * @author Gen Nishida
 * @date 12/24/2014
 * @version 1.0
 */

#include <list>
#include <vector>
#include <time.h>

#define CELL_LENGTH 20
#define CITY_SIZE 200
#define MAX_DIST 99
#define BF_CLEARED -1
#define MAX_ITERATIONS 1000 //1000
#define NUM_FEATURES 5
#define NUM_PEOPLE_TYPE 1

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
	dist[s * NUM_FEATURES + featureId] = MAX_DIST;
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

float min3(float distToStore, float distToAmusement, float distToFactory) {
	return std::min(std::min(distToStore, distToAmusement), distToFactory);
}

/** 
 * ゾーンのスコアを計算する。
 */
float computeScore(int* zone, int* dist) {
	// 好みベクトル
	float preference[NUM_PEOPLE_TYPE][9];
	preference[0][0] = 0; preference[0][1] = 0; preference[0][2] = 0; preference[0][3] = 0; preference[0][4] = 0; preference[0][5] = 0; preference[0][6] = 0; preference[0][7] = 1.0; preference[0][8] = 0;
	/*
	preference[0][0] = 0; preference[0][1] = 0; preference[0][2] = 0.15; preference[0][3] = 0.15; preference[0][4] = 0.3; preference[0][5] = 0; preference[0][6] = 0.1; preference[0][7] = 0.1; preference[0][8] = 0.2;
	preference[1][0] = 0; preference[1][1] = 0; preference[1][2] = 0.15; preference[1][3] = 0; preference[1][4] = 0.55; preference[1][5] = 0; preference[1][6] = 0.2; preference[1][7] = 0.1; preference[1][8] = 0;
	preference[2][0] = 0; preference[2][1] = 0; preference[2][2] = 0.05; preference[2][3] = 0; preference[2][4] = 0; preference[2][5] = 0; preference[2][6] = 0.25; preference[2][7] = 0.1; preference[2][8] = 0.6;
	preference[3][0] = 0.18; preference[3][1] = 0.17; preference[3][2] = 0; preference[3][3] = 0.17; preference[3][4] = 0; preference[3][5] = 0.08; preference[3][6] = 0.2; preference[3][7] = 0.2; preference[3][8] = 0;
	preference[4][0] = 0.3; preference[4][1] = 0; preference[4][2] = 0.3; preference[4][3] = 0.1; preference[4][4] = 0; preference[4][5] = 0; preference[4][6] = 0.1; preference[4][7] = 0.2; preference[4][8] = 0;
	preference[5][0] = 0.05; preference[5][1] = 0; preference[5][2] = 0.1; preference[5][3] = 0.2; preference[5][4] = 0.1; preference[5][5] = 0; preference[5][6] = 0.1; preference[5][7] = 0.15; preference[5][8] = 0.3;
	preference[6][0] = 0.15; preference[6][1] = 0.1; preference[6][2] = 0; preference[6][3] = 0.15; preference[6][4] = 0; preference[6][5] = 0.1; preference[6][6] = 0.1; preference[6][7] = 0.2; preference[6][8] = 0.2;
	preference[7][0] = 0.2; preference[7][1] = 0; preference[7][2] = 0.25; preference[7][3] = 0; preference[7][4] = 0.15; preference[7][5] = 0; preference[7][6] = 0.1; preference[7][7] = 0.1; preference[7][8] = 0.2;
	preference[8][0] = 0.3; preference[8][1] = 0; preference[8][2] = 0.15; preference[8][3] = 0.05; preference[8][4] = 0; preference[8][5] = 0; preference[8][6] = 0.25; preference[8][7] = 0.25; preference[8][8] = 0;
	preference[9][0] = 0.4; preference[9][1] = 0; preference[9][2] = 0.2; preference[9][3] = 0; preference[9][4] = 0; preference[9][5] = 0; preference[9][6] = 0.2; preference[9][7] = 0.2; preference[9][8] = 0;
	*/

	const float ratioPeople[NUM_PEOPLE_TYPE] = {1.0f};//, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
	const float K[] = {0.002f, 0.002f, 0.001f, 0.002f, 0.001f, 0.001f, 0.001f, 0.001f, 0.001f};

	float score = 0.0f;

	for (int i = 0; i < CITY_SIZE * CITY_SIZE; ++i) {
		if (zone[i] == 0) continue;

		for (int peopleType = 0; peopleType < NUM_PEOPLE_TYPE; ++peopleType) {
			float feature[8];
			feature[0] = exp(-K[0] * dist[i * NUM_FEATURES + 0] * CELL_LENGTH);
			feature[1] = exp(-K[0] * dist[i * NUM_FEATURES + 0] * CELL_LENGTH);
			feature[2] = exp(-K[0] * dist[i * NUM_FEATURES + 0] * CELL_LENGTH);
			feature[3] = exp(-K[0] * dist[i * NUM_FEATURES + 0] * CELL_LENGTH);
			feature[4] = exp(-K[0] * dist[i * NUM_FEATURES + 0] * CELL_LENGTH);
			feature[5] = exp(-K[0] * dist[i * NUM_FEATURES + 0] * CELL_LENGTH);
			feature[6] = 1.0f - exp(-K[6] * min3(dist[i * NUM_FEATURES + 1] * CELL_LENGTH, dist[i * NUM_FEATURES + 3] * CELL_LENGTH, dist[i * NUM_FEATURES + 0] * CELL_LENGTH));
			feature[7] = 1.0f - exp(-K[7] * dist[i * NUM_FEATURES + 1] * CELL_LENGTH);

			for (int x = 0; x < 8; ++x) {
				if (feature[x] < 0 || feature[x] > 1) {
					int hoge = 0;
				}
			}
			
			score += feature[0] * preference[peopleType][0] * ratioPeople[peopleType]; // 店
			score += feature[1] * preference[peopleType][1] * ratioPeople[peopleType]; // 学校
			score += feature[2] * preference[peopleType][2] * ratioPeople[peopleType]; // レストラン
			score += feature[3] * preference[peopleType][3] * ratioPeople[peopleType]; // 公園
			score += feature[4] * preference[peopleType][4] * ratioPeople[peopleType]; // アミューズメント
			score += feature[5] * preference[peopleType][5] * ratioPeople[peopleType]; // 図書館
			score += feature[6] * preference[peopleType][6] * ratioPeople[peopleType]; // 騒音
			score += feature[7] * preference[peopleType][7] * ratioPeople[peopleType]; // 汚染
		}
	}

	return score;
}

/**
 * 計算したdistance mapが正しいか、チェックする。
 */
int check(int* zone, int* dist) {
	int count = 0;

	for (int r = 0; r < CITY_SIZE; ++r) {
		for (int c = 0; c < CITY_SIZE; ++c) {
			for (int k = 0; k < NUM_FEATURES; ++k) {
				int min_dist = MAX_DIST;
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
				dist[i * NUM_FEATURES + k] = MAX_DIST;
				obst[i * NUM_FEATURES + k] = BF_CLEARED;
			}
		}
	}

	updateDistanceMap(queue, zone, dist, obst, toRaise);

	//dumpZone(zone);
	//dumpDist(dist, 4);
	//check(zone, dist);

	float curScore = computeScore(zone, dist);
	
	bf_count = 0;
	for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
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

		float proposedScore = computeScore(zone, dist);

		printf("%lf\n", proposedScore);

		if (proposedScore > curScore) { // accept
			// do nothing
		} else { // reject
			// rollback
			queue.clear();

			zone[s2] = 0;
			removeStore(queue, zone, dist, obst, toRaise, s2, featureId);
			zone[s1] = featureId + 1;
			setStore(queue, zone, dist, obst, toRaise, s1, featureId);
			updateDistanceMap(queue, zone, dist, obst, toRaise);
		}
	}

	printf("avg bf_count = %d\n", bf_count / MAX_ITERATIONS);
	printf("total bf_count = %d\n", bf_count);

	return 0;
}
