/**
 * Simulated annealing機能付きのMCMCを使って、理想的なゾーン計画を探す。
 * 距離マップは、Brushfireアルゴリズムを使って計算する。
 * グリッドサイズを、5x5 -> 10x10 -> 20x20 -> 40x40 -> 80x80 -> 160x160 と拡大していく。
 *
 * 隣のセルと、ゾーンタイプを交換する
 *
 * @author Gen Nishida
 * @date 12/25/2014
 * @version 1.1
 */

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

unsigned int rand(unsigned int* randx) {
    *randx = *randx * 1103515245 + 12345;
    return (*randx)&2147483647;
}

float randf() {
	return (float)rand() / RAND_MAX;
}

float randf(float a, float b) {
	return randf() * (b - a) + a;
}

int sampleFromCdf(float* cdf, int num) {
	float rnd = randf(0, cdf[num-1]);

	for (int i = 0; i < num; ++i) {
		if (rnd <= cdf[i]) return i;
	}

	return num - 1;
}

int sampleFromPdf(float* pdf, int num) {
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

	return sampleFromCdf(cdf, num);
}

void dumpZone(int city_size, int* zone) {
	printf("<<< Zone Map >>>\n");
	for (int r = 0; r < city_size; ++r) {
		for (int c = 0; c < city_size; ++c) {
			printf("%d ", zone[r * city_size + c]);
		}
		printf("\n");
	}
	printf("\n");
}

void dumpDist(int city_size, int* dist, int featureId) {
	printf("<<< Distance Map (featureId = %d) >>>\n", featureId);
	for (int r = 0; r < city_size; ++r) {
		for (int c = 0; c < city_size; ++c) {
			printf("%2d ", dist[(r * city_size + c) * NUM_FEATURES + featureId]);
		}
		printf("\n");
	}
	printf("\n");
}

void showZone(int city_size, int* zone, char* filename) {
	cv::Mat m(city_size, city_size, CV_8UC3);
	for (int r = 0; r < city_size; ++r) {
		for (int c = 0; c < city_size; ++c) {
			cv::Vec3b p;
			if (zone[r * city_size + c] == 0) {
				p = cv::Vec3b(0, 0, 255);
			} else if (zone[r * city_size + c] == 1) {
				p = cv::Vec3b(255, 0, 0);
			} else if (zone[r * city_size + c] == 2) {
				p = cv::Vec3b(64, 64, 64);
			} else if (zone[r * city_size + c] == 3) {
				p = cv::Vec3b(0, 255, 0);
			} else if (zone[r * city_size + c] == 4) {
				p = cv::Vec3b(255, 0, 255);
			} else if (zone[r * city_size + c] == 5) {
				p = cv::Vec3b(0, 255, 255);
			} else {
				p = cv::Vec3b(255, 255, 255);
			}
			m.at<cv::Vec3b>(r, c) = p;
		}
	}

	cv::imwrite(filename, m);
}

void loadZone(int city_size, int* zone, char* filename) {
	FILE* fp = fopen(filename, "r");

	for (int r = 0; r < city_size; ++r) {
		for (int c = 0; c < city_size; ++c) {
			fscanf(fp, "%d,", &zone[r * city_size + c]);
		}
	}

	fclose(fp);
}

void saveZone(int city_size, int* zone, char* filename) {
	FILE* fp = fopen(filename, "w");

	for (int r = 0; r < city_size; ++r) {
		for (int c = 0; c < city_size; ++c) {
			fprintf(fp, "%d,", zone[r * city_size + c]);
		}
		fprintf(fp, "\n");
	}
	fprintf(fp, "\n");

	fclose(fp);
}

inline bool isOcc(int* obst, int s, int featureId) {
	return obst[s * NUM_FEATURES + featureId] == s;
}

inline int distance(int city_size, int pos1, int pos2) {
	int x1 = pos1 % city_size;
	int y1 = pos1 / city_size;
	int x2 = pos2 % city_size;
	int y2 = pos2 / city_size;

	return abs(x1 - x2) + abs(y1 - y2);
}

void clearCell(int* dist, int* obst, int s, int featureId) {
	dist[s * NUM_FEATURES + featureId] = MAX_DIST;
	obst[s * NUM_FEATURES + featureId] = BF_CLEARED;
}

void raise(int city_size, std::list<std::pair<int, int> >& queue, int* dist, int* obst, bool* toRaise, int s, int featureId) {
	Point2D adj[] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

	int x = s % city_size;
	int y = s / city_size;

	for (int i = 0; i < 4; ++i) {
		int nx = x + adj[i].x;
		int ny = y + adj[i].y;

		if (nx < 0 || nx >= city_size || ny < 0 || ny >= city_size) continue;
		int n = ny * city_size + nx;

		if (obst[n * NUM_FEATURES + featureId] != BF_CLEARED && !toRaise[n * NUM_FEATURES + featureId]) {
			if (!isOcc(obst, obst[n * NUM_FEATURES + featureId], featureId)) {
				clearCell(dist, obst, n, featureId);
				toRaise[n * NUM_FEATURES + featureId] = true;
			}
			queue.push_back(std::make_pair(n, featureId));
		}
	}

	toRaise[s * NUM_FEATURES + featureId] = false;
}

void lower(int city_size, std::list<std::pair<int, int> >& queue, int* dist, int* obst, bool* toRaise, int s, int featureId) {
	Point2D adj[] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

	int x = s % city_size;
	int y = s / city_size;

	for (int i = 0; i < 4; ++i) {
		int nx = x + adj[i].x;
		int ny = y + adj[i].y;

		if (nx < 0 || nx >= city_size || ny < 0 || ny >= city_size) continue;
		int n = ny * city_size + nx;

		if (!toRaise[n * NUM_FEATURES + featureId]) {
			int d = distance(city_size, obst[s * NUM_FEATURES + featureId], n);
			if (d < dist[n * NUM_FEATURES + featureId]) {
				dist[n * NUM_FEATURES + featureId] = d;
				obst[n * NUM_FEATURES + featureId] = obst[s * NUM_FEATURES + featureId];
				queue.push_back(std::make_pair(n, featureId));
			}
		}
	}
}

void updateDistanceMap(int city_size, std::list<std::pair<int, int> >& queue, int* zone, int* dist, int* obst, bool* toRaise) {
	while (!queue.empty()) {
		std::pair<int, int> s = queue.front();
		queue.pop_front();

		if (toRaise[s.first * NUM_FEATURES + s.second]) {
			raise(city_size, queue, dist, obst, toRaise, s.first, s.second);
		} else if (isOcc(obst, obst[s.first * NUM_FEATURES + s.second], s.second)) {
			lower(city_size, queue, dist, obst, toRaise, s.first, s.second);
		}
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

	toRaise[s + NUM_FEATURES + featureId] = true;

	queue.push_back(std::make_pair(s, featureId));
}

float min3(float distToStore, float distToAmusement, float distToFactory) {
	return std::min(std::min(distToStore, distToAmusement), distToFactory);
}

/** 
 * ゾーンのスコアを計算する。
 */
float computeScore(int city_size, int* zone, int* dist) {
	int cell_length = 10000 / city_size;

	// 好みベクトル
	float preference[10][8];
	//preference[0][0] = 0; preference[0][1] = 0; preference[0][2] = 0; preference[0][3] = 0; preference[0][4] = 0; preference[0][5] = 0; preference[0][6] = 0; preference[0][7] = 1.0;
	preference[0][0] = 0; preference[0][1] = 0; preference[0][2] = 0.2; preference[0][3] = 0.2; preference[0][4] = 0.2; preference[0][5] = 0; preference[0][6] = 0.1; preference[0][7] = 0.3;
	preference[1][0] = 0; preference[1][1] = 0; preference[1][2] = 0.15; preference[1][3] = 0; preference[1][4] = 0.45; preference[1][5] = 0; preference[1][6] = 0.2; preference[1][7] = 0.2;
	preference[2][0] = 0; preference[2][1] = 0; preference[2][2] = 0.1; preference[2][3] = 0; preference[2][4] = 0; preference[2][5] = 0; preference[2][6] = 0.4; preference[2][7] = 0.5;
	preference[3][0] = 0.15; preference[3][1] = 0.13; preference[3][2] = 0; preference[3][3] = 0.14; preference[3][4] = 0; preference[3][5] = 0.08; preference[3][6] = 0.2; preference[3][7] = 0.3;
	preference[4][0] = 0.3; preference[4][1] = 0; preference[4][2] = 0.3; preference[4][3] = 0.1; preference[4][4] = 0; preference[4][5] = 0; preference[4][6] = 0.1; preference[4][7] = 0.2;
	preference[5][0] = 0.05; preference[5][1] = 0; preference[5][2] = 0.15; preference[5][3] = 0.2; preference[5][4] = 0.15; preference[5][5] = 0; preference[5][6] = 0.15; preference[5][7] = 0.3;
	preference[6][0] = 0.2; preference[6][1] = 0.1; preference[6][2] = 0; preference[6][3] = 0.2; preference[6][4] = 0; preference[6][5] = 0.1; preference[6][6] = 0.1; preference[6][7] = 0.3;
	preference[7][0] = 0.3; preference[7][1] = 0; preference[7][2] = 0.3; preference[7][3] = 0; preference[7][4] = 0.2; preference[7][5] = 0; preference[7][6] = 0.1; preference[7][7] = 0.1;
	preference[8][0] = 0.25; preference[8][1] = 0; preference[8][2] = 0.1; preference[8][3] = 0.05; preference[8][4] = 0; preference[8][5] = 0; preference[8][6] = 0.25; preference[8][7] = 0.35;
	preference[9][0] = 0.25; preference[9][1] = 0; preference[9][2] = 0.2; preference[9][3] = 0; preference[9][4] = 0; preference[9][5] = 0; preference[9][6] = 0.2; preference[9][7] = 0.35;

	const float ratioPeople[10] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
	const float K[] = {0.002f, 0.002f, 0.001f, 0.002f, 0.001f, 0.001f, 0.001f, 0.001f};

	float score = 0.0f;

	int num_zones = 0;
	for (int i = 0; i < city_size * city_size; ++i) {
		if (zone[i] == 0) continue;

		num_zones++;

		for (int peopleType = 0; peopleType < NUM_PEOPLE_TYPE; ++peopleType) {
			float feature[8];
			feature[0] = exp(-K[0] * dist[i * NUM_FEATURES + 0] * cell_length);
			feature[1] = exp(-K[0] * dist[i * NUM_FEATURES + 0] * cell_length);
			feature[2] = exp(-K[0] * dist[i * NUM_FEATURES + 0] * cell_length);
			feature[3] = exp(-K[0] * dist[i * NUM_FEATURES + 0] * cell_length);
			feature[4] = exp(-K[0] * dist[i * NUM_FEATURES + 0] * cell_length);
			feature[5] = exp(-K[0] * dist[i * NUM_FEATURES + 0] * cell_length);
			feature[6] = 1.0f - exp(-K[6] * min3(dist[i * NUM_FEATURES + 1] * cell_length, dist[i * NUM_FEATURES + 3] * cell_length, dist[i * NUM_FEATURES + 0] * cell_length));
			feature[7] = 1.0f - exp(-K[7] * dist[i * NUM_FEATURES + 1] * cell_length);
			
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

	return score / num_zones;
}

/**
 * 計算したdistance mapが正しいか、チェックする。
 */
int check(int city_size, int* zone, int* dist) {
	int count = 0;

	for (int r = 0; r < city_size; ++r) {
		for (int c = 0; c < city_size; ++c) {
			for (int k = 0; k < NUM_FEATURES; ++k) {
				int min_dist = MAX_DIST;
				for (int r2 = 0; r2 < city_size; ++r2) {
					for (int c2 = 0; c2 < city_size; ++c2) {
						if (zone[r2 * city_size + c2] - 1 == k) {
							int d = distance(city_size, r2 * city_size + c2, r * city_size + c);
							if (d < min_dist) {
								min_dist = d;
							}
						}
					}
				}

				if (dist[(r * city_size + c) * NUM_FEATURES + k] != min_dist) {
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
void generateZoningPlan(int city_size, int* zone, std::vector<float> zoneTypeDistribution) {
	std::vector<float> numRemainings(NUM_FEATURES + 1);
	for (int i = 0; i < NUM_FEATURES + 1; ++i) {
		numRemainings[i] = city_size * city_size * zoneTypeDistribution[i];
	}

	for (int r = 0; r < city_size; ++r) {
		for (int c = 0; c < city_size; ++c) {
			int type = sampleFromPdf(numRemainings.data(), numRemainings.size());
			zone[r * city_size + c] = type;
			numRemainings[type] -= 1;
		}
	}

	return;

	// デバッグ用
	// 工場を一番上に持っていく
	// そうすれば、良いゾーンプランになるはず。。。
	for (int r = 2; r < city_size; ++r) {
		for (int c = 0; c < city_size; ++c) {
			if (zone[r * city_size + c] != 2) continue;

			bool done = false;
			for (int r2 = 0; r2 < 2 && !done; ++r2) {
				for (int c2 = 0; c2 < city_size && !done; ++c2) {
					if (zone[r2 * city_size + c2] == 2) continue;

					// 交換する
					int type = zone[r2 * city_size + c2];
					zone[r2 * city_size + c2] = zone[r * city_size + c];
					zone[r * city_size + c] = type;
					done = true;
				}
			}
		}
	}
}

/**
 * bestZoneに、初期ゾーンプランが入っている。
 * MCMCを使って、最適なゾーンプランを探し、bestZoneに格納して返却する。
 */
void optimize(int city_size, int max_iterations, int* bestZone) {
	int* zone = (int*)malloc(sizeof(int) * city_size * city_size);
	int* dist = (int*)malloc(sizeof(int) * city_size * city_size * NUM_FEATURES);
	int* obst = (int*)malloc(sizeof(int) * city_size * city_size * NUM_FEATURES);
	bool* toRaise = (bool*)malloc(city_size * city_size * NUM_FEATURES);

	memcpy(zone, bestZone, sizeof(int) * city_size * city_size);

	// for backup
	int* tmpZone = (int*)malloc(sizeof(int) * city_size * city_size);
	int* tmpDist = (int*)malloc(sizeof(int) * city_size * city_size * NUM_FEATURES);
	int* tmpObst = (int*)malloc(sizeof(int) * city_size * city_size * NUM_FEATURES);

	// キューのセットアップ
	std::list<std::pair<int, int> > queue;
	for (int i = 0; i < city_size * city_size; ++i) {
		for (int k = 0; k < NUM_FEATURES; ++k) {
			toRaise[i * NUM_FEATURES + k] = false;
			if (zone[i] - 1 == k) {
				setStore(queue, zone, dist, obst, toRaise, i, k);
			} else {
				dist[i * NUM_FEATURES + k] = MAX_DIST;
				obst[i * NUM_FEATURES + k] = BF_CLEARED;
			}
		}
	}

	updateDistanceMap(city_size, queue, zone, dist, obst, toRaise);

	//dumpZone(city_size, zone);
	//dumpDist(city_size, dist, 4);
	//check(city_size, zone, dist);

	float curScore = computeScore(city_size, zone, dist);
	float bestScore = curScore;
	memcpy(bestZone, zone, sizeof(int) * city_size * city_size);

	float beta = 1.0f;
	for (int iter = 0; iter < max_iterations; ++iter) {
		queue.clear();

		// バックアップ
		memcpy(tmpZone, zone, sizeof(int) * city_size * city_size);
		memcpy(tmpDist, dist, sizeof(int) * city_size * city_size * NUM_FEATURES);
		memcpy(tmpObst, obst, sizeof(int) * city_size * city_size * NUM_FEATURES);

		// ２つのセルのゾーンタイプを交換
		int s1, s2;
		while (true) {
			s1 = rand() % (city_size * city_size);
			if (zone[s1] > 0) break;
		}
		while (true) {
			s2 = rand() % (city_size * city_size);
			if (zone[s2] == 0) break;
		}

		// move a store
		int featureId = zone[s1] - 1;
		zone[s1] = 0;
		removeStore(queue, zone, dist, obst, toRaise, s1, featureId);
		zone[s2] = featureId + 1;
		setStore(queue, zone, dist, obst, toRaise, s2, featureId);
		updateDistanceMap(city_size, queue, zone, dist, obst, toRaise);
		
		//dumpZone(city_size, zone);
		//dumpDist(city_size, dist, 4);
		//if (check(city_size, zone, dist) > 0) break;

		float proposedScore = computeScore(city_size, zone, dist);

		// ベストゾーンを更新
		if (proposedScore > bestScore) {
			bestScore = proposedScore;
			memcpy(bestZone, zone, sizeof(int) * city_size * city_size);
		}

		//printf("%lf -> %lf (best: %lf)\n", curScore, proposedScore, bestScore);

		if (proposedScore > curScore || randf() < proposedScore / curScore) { // accept
			curScore = proposedScore;
		} else { // reject
			// rollback
			memcpy(zone, tmpZone, sizeof(int) * city_size * city_size);
			memcpy(dist, tmpDist, sizeof(int) * city_size * city_size * NUM_FEATURES);
			memcpy(obst, tmpObst, sizeof(int) * city_size * city_size * NUM_FEATURES);
		}
	}

	printf("city_size: %d, score: %lf\n", city_size, bestScore);

	char filename[256];
	sprintf(filename, "zone_%d.png", city_size);
	showZone(city_size, bestZone, filename);
	//saveZone(city_size, bestZone);
}


/**
 * bestZoneに、初期ゾーンプランが入っている。
 * MCMCを使って、最適なゾーンプランを探し、bestZoneに格納して返却する。
 * 各ステップでは、隣接セルをランダムに選択し、ゾーンを交換する。
 */
void optimize2(int city_size, int max_iterations, int* bestZone) {
	int* zone = (int*)malloc(sizeof(int) * city_size * city_size);
	int* dist = (int*)malloc(sizeof(int) * city_size * city_size * NUM_FEATURES);
	int* obst = (int*)malloc(sizeof(int) * city_size * city_size * NUM_FEATURES);
	bool* toRaise = (bool*)malloc(city_size * city_size * NUM_FEATURES);

	memcpy(zone, bestZone, sizeof(int) * city_size * city_size);

	// for backup
	int* tmpZone = (int*)malloc(sizeof(int) * city_size * city_size);
	int* tmpDist = (int*)malloc(sizeof(int) * city_size * city_size * NUM_FEATURES);
	int* tmpObst = (int*)malloc(sizeof(int) * city_size * city_size * NUM_FEATURES);

	// キューのセットアップ
	std::list<std::pair<int, int> > queue;
	for (int i = 0; i < city_size * city_size; ++i) {
		for (int k = 0; k < NUM_FEATURES; ++k) {
			toRaise[i * NUM_FEATURES + k] = false;
			if (zone[i] - 1 == k) {
				setStore(queue, zone, dist, obst, toRaise, i, k);
			} else {
				dist[i * NUM_FEATURES + k] = MAX_DIST;
				obst[i * NUM_FEATURES + k] = BF_CLEARED;
			}
		}
	}

	updateDistanceMap(city_size, queue, zone, dist, obst, toRaise);

	//dumpZone(city_size, zone);
	//dumpDist(city_size, dist, 4);
	//check(city_size, zone, dist);

	float curScore = computeScore(city_size, zone, dist);
	float bestScore = curScore;
	memcpy(bestZone, zone, sizeof(int) * city_size * city_size);

	float beta = 1.0f;
	int adj[4];
	adj[0] = -1; adj[1] = 1; adj[2] = -city_size; adj[3] = city_size;
	for (int iter = 0; iter < max_iterations; ++iter) {
		queue.clear();

		// バックアップ
		memcpy(tmpZone, zone, sizeof(int) * city_size * city_size);
		memcpy(tmpDist, dist, sizeof(int) * city_size * city_size * NUM_FEATURES);
		memcpy(tmpObst, obst, sizeof(int) * city_size * city_size * NUM_FEATURES);

		// ２つの隣接セルを選択
		int s1, s2;
		while (true) {
			s1 = rand() % (city_size * city_size);
			int u = rand() % 4;
			s2 = s1 + adj[u];

			if (s2 < 0 || s2 >= city_size * city_size) continue;
			if (zone[s1] == zone[s2]) continue;

			int x1 = s1 % city_size;
			int y1 = s1 / city_size;
			int x2 = s2 % city_size;
			int y2 = s2 / city_size;
			if (abs(x1 - x2) + abs(y1 - y2) > 1) continue;

			break;
		}

		// ２つのセルのゾーンタイプを交換
		int f1 = zone[s1] - 1;
		int f2 = zone[s2] - 1;
		zone[s1] = f2 + 1;
		if (f1 >= 0) {
			removeStore(queue, zone, dist, obst, toRaise, s1, f1);
		}
		if (f2 >= 0) {
			setStore(queue, zone, dist, obst, toRaise, s1, f2);
		}
		zone[s2] = f1 + 1;
		if (f2 >= 0) {
			removeStore(queue, zone, dist, obst, toRaise, s2, f2);
		}
		if (f1 >= 0) {
			setStore(queue, zone, dist, obst, toRaise, s2, f1);
		}
		updateDistanceMap(city_size, queue, zone, dist, obst, toRaise);
		
		//dumpZone(city_size, zone);
		//dumpDist(city_size, dist, 4);
		//if (check(city_size, zone, dist) > 0) break;

		float proposedScore = computeScore(city_size, zone, dist);

		// ベストゾーンを更新
		if (proposedScore > bestScore) {
			bestScore = proposedScore;
			memcpy(bestZone, zone, sizeof(int) * city_size * city_size);
		}

		//printf("%lf -> %lf (best: %lf)\n", curScore, proposedScore, bestScore);

		if (proposedScore > curScore || randf() < proposedScore / curScore) { // accept
			curScore = proposedScore;
		} else { // reject
			// rollback
			memcpy(zone, tmpZone, sizeof(int) * city_size * city_size);
			memcpy(dist, tmpDist, sizeof(int) * city_size * city_size * NUM_FEATURES);
			memcpy(obst, tmpObst, sizeof(int) * city_size * city_size * NUM_FEATURES);
		}
	}

	printf("city_size: %d, score: %lf\n", city_size, bestScore);

	char filename[256];
	sprintf(filename, "zone_%d.png", city_size);
	showZone(city_size, bestZone, filename);
	//saveZone(city_size, bestZone);
}

int main() {
	time_t start, end;

	int city_size = 5;

	int* zone;
	zone = (int*)malloc(sizeof(int) * city_size * city_size);
	
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
	generateZoningPlan(city_size, zone, zoneTypeDistribution);
	//loadZone(zone, "zone2.txt");
	end = clock();
	printf("generateZoningPlan: %lf\n", (double)(end-start)/CLOCKS_PER_SEC);

	int max_iterations = 10000;

	for (int layer = 0; layer < NUM_LAYERS; ++layer) {
		if (layer == 0) {
			optimize(city_size, max_iterations, zone);
		} else {
			optimize2(city_size, max_iterations, zone);
		}
		int* tmpZone = (int*)malloc(sizeof(int) * city_size * city_size);
		memcpy(tmpZone, zone, sizeof(int) * city_size * city_size);

		free(zone);

		// ゾーンマップを、たて、よこ、２倍ずつに増やす
		city_size *= 2;
		zone = (int*)malloc(sizeof(int) * city_size * city_size);
		for (int r = 0; r < city_size; ++r) {
			for (int c = 0; c < city_size; ++c) {
				int oldR = r / 2;
				int oldC = c / 2;
				zone[r * city_size + c] = tmpZone[(int)(oldR * city_size * 0.5 + oldC)];
			}
		}

		max_iterations *= 0.5;

		free(tmpZone);
	}
	
	showZone(city_size, zone, "zone_final.png");
	saveZone(city_size, zone, "zone_final.txt");

	free(zone);

	return 0;
}
