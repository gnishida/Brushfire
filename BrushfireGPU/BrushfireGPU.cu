/**
 * Nearest neighbor search
 * マップ内に、店、工場などのゾーンがある確率で配備されている時、
 * 住宅ゾーンから直近の店、工場までのマンハッタン距離を計算する。
 *
 * 各店、工場から周辺に再帰的に距離を更新していくので、O(N)で済む。
 * しかも、GPUで並列化することで、さらに計算時間を短縮できる。
 *
 * shared memoryを使用して高速化できるか？
 */

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <list>
#include <time.h>

#define CELL_LENGTH 100
#define CITY_SIZE 8 //200
#define GPU_BLOCK_SIZE 8 //40
#define GPU_NUM_THREADS 8 //96
#define GPU_BLOCK_SCALE (1.0)
#define NUM_FEATURES 1//5
#define QUEUE_MAX 3999
#define MAX_DIST 99
#define BF_CLEARED -1
#define QUEUE_EMPTY -1
#define MAX_ITERATIONS 1

#define CUDA_CALL(x) {if((x) != cudaSuccess){ \
  printf("CUDA error at %s:%d\n",__FILE__,__LINE__); \
  printf("  %s\n", cudaGetErrorString(cudaGetLastError())); \
  exit(EXIT_FAILURE);}} 


struct ZoneType {
	int type;
	int level;
};

struct ZoningPlan {
	ZoneType zones[CITY_SIZE][CITY_SIZE];
};

struct DistanceMap {
	int distances[CITY_SIZE][CITY_SIZE][NUM_FEATURES];
};

struct Point2D {
	int x;
	int y;
};

__host__ __device__
unsigned int rand(unsigned int* randx) {
    *randx = *randx * 1103515245 + 12345;
    return (*randx)&2147483647;
}

__host__ __device__
float randf(unsigned int* randx) {
	return rand(randx) / (float(2147483647) + 1);
}

__host__ __device__
float randf(unsigned int* randx, float a, float b) {
	return randf(randx) * (b - a) + a;
}

__host__ __device__
int sampleFromCdf(unsigned int* randx, float* cdf, int num) {
	float rnd = randf(randx, 0, cdf[num-1]);

	for (int i = 0; i < num; ++i) {
		if (rnd <= cdf[i]) return i;
	}

	return num - 1;
}

__host__ __device__
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

/**
 * ゾーンプランを生成する。
 */
__host__
void generateZoningPlan(ZoningPlan& zoningPlan, std::vector<float> zoneTypeDistribution) {
	std::vector<float> numRemainings(NUM_FEATURES + 1);
	for (int i = 0; i < NUM_FEATURES + 1; ++i) {
		numRemainings[i] = CITY_SIZE * CITY_SIZE * zoneTypeDistribution[i];
	}

	unsigned int randx = 0;

	for (int r = 0; r < CITY_SIZE; ++r) {
		for (int c = 0; c < CITY_SIZE; ++c) {
			int type = sampleFromPdf(&randx, numRemainings.data(), numRemainings.size());
			zoningPlan.zones[r][c].type = type;
			numRemainings[type] -= 1;
		}
	}
}

__global__
void generateProposal(ZoningPlan* zoningPlan, unsigned int* randx, int2* cell1, int2* cell2) {
	while (true) {
		int x1 = randf(randx, 0, CITY_SIZE);
		int y1 = randf(randx, 0, CITY_SIZE);
		int x2 = randf(randx, 0, CITY_SIZE);
		int y2 = randf(randx, 0, CITY_SIZE);

		//if (zoningPlan->zones[y1][x1].type != 0 || zoningPlan->zones[y2][x2].type != 0) {
		if (zoningPlan->zones[y1][x1].type == 2 || zoningPlan->zones[y2][x2].type == 2) {
			// swap zone
			int tmp_type = zoningPlan->zones[y1][x1].type;
			zoningPlan->zones[y1][x1].type = zoningPlan->zones[y2][x2].type;
			zoningPlan->zones[y2][x2].type = tmp_type;

			*cell1 = make_int2(x1, y1);
			*cell2 = make_int2(x2, y2);

			break;
		}
	}
}

__device__
bool isOcc(int* obst, int pos, int featureId) {
	return obst[pos * NUM_FEATURES + featureId] == pos;
}

__device__ __host__
int distance(int pos1, int pos2) {
	int x1 = pos1 % CITY_SIZE;
	int y1 = pos1 / CITY_SIZE;
	int x2 = pos2 % CITY_SIZE;
	int y2 = pos2 / CITY_SIZE;

	return abs(x1 - x2) + abs(y1 - y2);
}

__device__
void clearCell(int* dist, int* obst, int s, int featureId) {
	dist[s * NUM_FEATURES + featureId] = MAX_DIST;
	obst[s * NUM_FEATURES + featureId] = BF_CLEARED;
}

__device__
void raise(int* queue, unsigned int* queue_end, int* dist, int* obst, bool* toRaise, int s, int featureId) {
	uint2 adj[] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

	int x = s % CITY_SIZE;
	int y = s / CITY_SIZE;

	for (int i = 0; i < 4; ++i) {
		int nx = x + adj[i].x;
		int ny = y + adj[i].y;

		if (nx < 0 || nx >= CITY_SIZE || ny < 0 || ny >= CITY_SIZE) continue;
		int n = ny * CITY_SIZE + nx;

		if (obst[n * NUM_FEATURES + featureId] != BF_CLEARED && !toRaise[n * NUM_FEATURES + featureId]) {
			if (!isOcc(obst, obst[n * NUM_FEATURES + featureId], featureId)) {
				clearCell(dist, obst, n, featureId);
				toRaise[n * NUM_FEATURES + featureId] = true;
			}
			unsigned int queue_index = atomicInc(queue_end, QUEUE_MAX);
			queue[queue_index] = n;
		}
	}

	toRaise[s * NUM_FEATURES + featureId] = false;
}

__device__
void lower(int* queue, unsigned int* queue_end, int* dist, int* obst, bool* toRaise, int s, int featureId) {
	Point2D adj[4] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

	int x = s % CITY_SIZE;
	int y = s / CITY_SIZE;

	for (int i = 0; i < 4; ++i) {
		int nx = x + adj[i].x;
		int ny = y + adj[i].y;

		if (nx < 0 || nx >= CITY_SIZE || ny < 0 || ny >= CITY_SIZE) continue;
		int n = ny * CITY_SIZE + nx;

		if (!toRaise[n * NUM_FEATURES + featureId]) {
			int d = distance(obst[s * NUM_FEATURES + featureId], n);
			if (d < dist[n * NUM_FEATURES + featureId]) {
				dist[n * NUM_FEATURES + featureId] = d;
				obst[n * NUM_FEATURES + featureId] = obst[s * NUM_FEATURES + featureId];

				unsigned int queue_index = atomicInc(queue_end, QUEUE_MAX);
				queue[queue_index] = n;
			}
		}
	}
}

__device__
void setObst(int* queue, unsigned int* queue_end, int* dist, int* obst, bool* toRaise, int s, int featureId) {
	// put stores
	obst[s * NUM_FEATURES + featureId] = s;
	dist[s * NUM_FEATURES + featureId] = 0;

	unsigned int queue_index = atomicInc(queue_end, QUEUE_MAX);
	queue[queue_index] = s;
}

__device__
void removeObst(int* queue, unsigned int* queue_end, int* dist, int* obst, bool* toRaise, int s, int featureId) {
	clearCell(dist, obst, s, featureId);

	toRaise[s * NUM_FEATURES + featureId] = true;

	unsigned int queue_index = atomicInc(queue_end, QUEUE_MAX);
	queue[queue_index] = s;
}

/**
 * 距離マップを計算する
 */
__global__
void computeDistMap(ZoningPlan* zoningPlan, int* dist, int* obst, bool* toRaise, int* queue, unsigned int* queue_begin, unsigned int* queue_end) {
	int featureId = 0;

	__shared__ int lock;
	lock = 0;

	__syncthreads();

	while (true) {
		//do {} while (atomicCAS(&lock, 0, 1));
		if (queue[*queue_begin] == BF_CLEARED) break;
		int queue_index = *queue_begin;
		*queue_begin++;
		lock = 0;
		//int queue_index = atomicInc(queue_begin, QUEUE_MAX);

		int s = queue[queue_index];
		queue[queue_index] = QUEUE_EMPTY;

		if (toRaise[s * NUM_FEATURES + featureId]) {
			raise(queue, queue_end, dist, obst, toRaise, s, featureId);
		} else if (isOcc(obst, obst[s * NUM_FEATURES + featureId], featureId)) {
			lower(queue, queue_end, dist, obst, toRaise, s, featureId);
		}
	}
}

__global__
void initObst(ZoningPlan* zoningPlan, int* dist, int* obst, bool* toRaise, int* queue, unsigned int* queue_begin, unsigned int* queue_end) {
	for (int r = 0; r < CITY_SIZE; ++r) {
		for (int c = 0; c < CITY_SIZE; ++c) {
			for (int k = 0; k < NUM_FEATURES; ++k) {
				clearCell(dist, obst, r * CITY_SIZE + c, k);

				if (zoningPlan->zones[r][c].type > 0) {
					setObst(queue, queue_end, dist, obst, toRaise, r * CITY_SIZE + c, zoningPlan->zones[r][c].type - 1);
				}
			}
		}
	}

}

__global__
void moveObst(ZoningPlan* zoningPlan, int* dist, int* obst, bool* toRaise, int* queue, unsigned int* queue_begin, unsigned int* queue_end, unsigned int* randx) {
	int s1;
	while (true) {
		s1 = randf(randx, 0, CITY_SIZE * CITY_SIZE);
		int x = s1 % CITY_SIZE;
		int y = s1 / CITY_SIZE;
		if (zoningPlan->zones[y][x].type == 1) {
			zoningPlan->zones[y][x].type = 0;
			break;
		}
	}

	int s2;
	while (true) {
		s2 = randf(randx, 0, CITY_SIZE * CITY_SIZE);
		int x = s2 % CITY_SIZE;
		int y = s2 / CITY_SIZE;
		if (zoningPlan->zones[y][x].type == 0) {
			zoningPlan->zones[y][x].type = 1;
			break;
		}
	}

	setObst(queue, queue_end, dist, obst, toRaise, s2, 0);
	removeObst(queue, queue_end, dist, obst, toRaise, s1, 0);
}

__device__
float min3(int distToStore, int distToAmusement, int distToFactory) {
	return min(min(distToStore, distToAmusement), distToFactory);
}

__global__
void computeScore(ZoningPlan* zoningPlan, DistanceMap* distanceMap, float* devScore, float* devScores) {
	int num_strides = (GPU_BLOCK_SIZE * GPU_BLOCK_SIZE + GPU_NUM_THREADS - 1) / GPU_NUM_THREADS;

	__shared__ float sScore;
	sScore = 0.0f;
	__syncthreads();

	__shared__ float preference[10][9];
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

	//const float ratioPeople[10] = {0.06667, 0.06667, 0.06667, 0.21, 0.09, 0.09, 0.09, 0.12, 0.1, 0.1};
	const float ratioPeople[10] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

	float K[] = {0.002, 0.002, 0.001, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001};

	float lScore = 0.0f;

	int r0 = blockIdx.y * GPU_BLOCK_SIZE;
	int c0 = blockIdx.x * GPU_BLOCK_SIZE;
	for (int i = 0; i < num_strides; ++i) {
		float tmpScore = 0.0f;
		int r1 = (i * GPU_NUM_THREADS + threadIdx.x) / GPU_BLOCK_SIZE;
		int c1 = (i * GPU_NUM_THREADS + threadIdx.x) % GPU_BLOCK_SIZE;

		// 対象ブロックの外ならスキップ
		if (r1 >= GPU_BLOCK_SIZE * GPU_BLOCK_SCALE || c1 >= GPU_BLOCK_SIZE * GPU_BLOCK_SCALE) continue;

		// city範囲の外ならスキップ
		if (r0 + r1 < 0 || r0 + r1 >= CITY_SIZE || c0 + c1 < 0 && c0 + c1 >= CITY_SIZE) continue;

		// 住宅ゾーン以外なら、スキップ
		if (zoningPlan->zones[r0 + r1][c0 + c1].type > 0) continue;

		//for (int peopleType = 0; peopleType < 10; ++peopleType) {
		for (int peopleType = 0; peopleType < 1; ++peopleType) {
			tmpScore += exp(-K[0] * distanceMap->distances[r0 + r1][c0 + c1][0] * CELL_LENGTH) * preference[peopleType][0] * ratioPeople[peopleType]; // 店
			tmpScore += exp(-K[1] * distanceMap->distances[r0 + r1][c0 + c1][4] * CELL_LENGTH) * preference[peopleType][1] * ratioPeople[peopleType]; // 学校
			tmpScore += exp(-K[2] * distanceMap->distances[r0 + r1][c0 + c1][0] * CELL_LENGTH) * preference[peopleType][2] * ratioPeople[peopleType]; // レストラン
			tmpScore += exp(-K[3] * distanceMap->distances[r0 + r1][c0 + c1][2] * CELL_LENGTH) * preference[peopleType][3] * ratioPeople[peopleType]; // 公園
			tmpScore += exp(-K[4] * distanceMap->distances[r0 + r1][c0 + c1][3] * CELL_LENGTH) * preference[peopleType][4] * ratioPeople[peopleType]; // アミューズメント
			tmpScore += exp(-K[5] * distanceMap->distances[r0 + r1][c0 + c1][4] * CELL_LENGTH) * preference[peopleType][5] * ratioPeople[peopleType]; // 図書館
			tmpScore += (1.0f - exp(-K[6] * min3(distanceMap->distances[r0 + r1][c0 + c1][0] * CELL_LENGTH, distanceMap->distances[r0 + r1][c0 + c1][3] * CELL_LENGTH, distanceMap->distances[r0 + r1][c0 + c1][1] * CELL_LENGTH))) * preference[peopleType][6] * ratioPeople[peopleType]; // 騒音
			tmpScore += (1.0f - exp(-K[7] * distanceMap->distances[r0 + r1][c0 + c1][1] * CELL_LENGTH)) * preference[peopleType][7] * ratioPeople[peopleType]; // 汚染
		}
		lScore += tmpScore;

		devScores[(r0 + r1) * CITY_SIZE + c0 + c1] = tmpScore;
	}

	//atomicAdd(&sScore, lScore);

	__syncthreads();

	//atomicAdd(devScore, sScore);
}

__global__
void acceptProposal(ZoningPlan* zoningPlan, float* score, float* proposalScore, int2* cell1, int2* cell2, int* result) {
	if (*proposalScore > *score) {
		*score = *proposalScore;
		*result = 1;
	} else {
		// プランを元に戻す
		int tmp_type = zoningPlan->zones[cell1->y][cell1->x].type;
		zoningPlan->zones[cell1->y][cell1->x].type = zoningPlan->zones[cell2->y][cell2->x].type;
		zoningPlan->zones[cell2->y][cell2->x].type = tmp_type;
		*result = 0;
	}
}

/**
 * デバッグ用に、スコアを表示する。
 */
__host__
void showDevScore(int* devScore) {
	float score;
	CUDA_CALL(cudaMemcpy(&score, devScore, sizeof(float), cudaMemcpyDeviceToHost));
	printf("Score: %lf\n\n", score);
}

/**
 * デバッグ用に、ゾーンプランを表示する。
 */
__host__
void showDevZoningPlan(ZoningPlan* zoningPlan) {
	ZoningPlan plan;

	CUDA_CALL(cudaMemcpy(&plan, zoningPlan, sizeof(ZoningPlan), cudaMemcpyDeviceToHost));
	printf("<<< Zone Map >>>\n");
	for (int r = CITY_SIZE - 1; r >= 0; --r) {
		for (int c = 0; c < CITY_SIZE; ++c) {
			printf("%2d,", plan.zones[r][c].type);
		}
		printf("\n");
	}
	printf("\n");
}

/**
 * デバッグ用に、距離マップを表示する。
 */
__host__
void showDevDistMap(int* dist, int feature_id) {
	int* hostDist;

	hostDist = (int*)malloc(sizeof(int) * CITY_SIZE * CITY_SIZE * NUM_FEATURES);
	CUDA_CALL(cudaMemcpy(hostDist, dist, sizeof(DistanceMap), cudaMemcpyDeviceToHost));
	printf("<<< Distance Map >>>\n");
	for (int r = CITY_SIZE - 1; r >= 0; --r) {
		for (int c = 0; c < CITY_SIZE; ++c) {
			printf("%2d,", hostDist[(r * CITY_SIZE + c) * NUM_FEATURES + feature_id]);
		}
		printf("\n");
	}
	printf("\n");

	free(hostDist);
}

/**
 * キューの内容を表示する。
 *
 * queueEndは、次の要素を格納する位置を示す。つまり、queueEndの１つ手前までが、有効なキューの値ということだ。
 */
__host__
void showDevQueue(int* devQueue, unsigned int* devQueueEnd) {
	int* queue;
	int queueEnd;

	queue = (int*)malloc(sizeof(int) * (QUEUE_MAX + 1));
	CUDA_CALL(cudaMemcpy(queue, devQueue, sizeof(int) * (QUEUE_MAX + 1), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(&queueEnd, devQueueEnd, sizeof(int), cudaMemcpyDeviceToHost));
	printf("<<< Queue >>>\n");
	for (int i = 0; i < queueEnd; ++i) {
		if (queue[i] >= 0) {
			int x = queue[i] % CITY_SIZE;
			int y = queue[i] / CITY_SIZE;
			printf("%2d,%2d\n", x, y);
		} else {
			printf("--,--\n");
		}
	}
	printf("\n");

	free(queue);
}

__host__
int check(ZoningPlan* devZone, int* devDist) {
	ZoningPlan zone;
	int* dist;
	dist = (int*)malloc(sizeof(int) * CITY_SIZE * CITY_SIZE * NUM_FEATURES);

	CUDA_CALL(cudaMemcpy(&zone, devZone, sizeof(ZoningPlan), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(dist, devDist, sizeof(DistanceMap), cudaMemcpyDeviceToHost));

	int count = 0;

	for (int r = 0; r < CITY_SIZE; ++r) {
		for (int c = 0; c < CITY_SIZE; ++c) {
			for (int k = 0; k < NUM_FEATURES; ++k) {
				int min_dist = MAX_DIST;
				for (int r2 = 0; r2 < CITY_SIZE; ++r2) {
					for (int c2 = 0; c2 < CITY_SIZE; ++c2) {
						if (zone.zones[r2][c2].type == 0) continue;

						if (zone.zones[r2][c2].type - 1 == k) {
							int d = distance(r2 * CITY_SIZE + c2, r * CITY_SIZE + c);
							if (d < min_dist) {
								min_dist = d;
							}
						}
					}
				}

				if (dist[r * CITY_SIZE + c] != min_dist) {
					count++;
				}
			}
		}
	}
	
	if (count > 0) {
		printf("Check results: #error cells = %d\n", count);
	}

	free(dist);

	return count;
}

int main()
{
	time_t start, end;

	// ホストバッファを確保
	ZoningPlan* hostZoningPlan = (ZoningPlan*)malloc(sizeof(ZoningPlan));
	int* hostDist = (int*)malloc(sizeof(int) * CITY_SIZE * CITY_SIZE * NUM_FEATURES);
	int* hostObst = (int*)malloc(sizeof(int) * CITY_SIZE * CITY_SIZE * NUM_FEATURES);
	bool* hostToRaise = (bool*)malloc(sizeof(bool) * CITY_SIZE * CITY_SIZE * NUM_FEATURES);
	int* hostQueue = (int*)malloc(sizeof(int) * (QUEUE_MAX + 1));
	unsigned int hostQueueBegin;
	unsigned int hostQueueEnd;

	// ホストバッファの初期化
	for (int i = 0; i < CITY_SIZE * CITY_SIZE * NUM_FEATURES; ++i) {
		hostDist[i] = MAX_DIST;
		hostObst[i] = BF_CLEARED;
		hostToRaise[i] = false;
	}


	// デバイスバッファを確保
	ZoningPlan* devZoningPlan;
	CUDA_CALL(cudaMalloc((void**)&devZoningPlan, sizeof(ZoningPlan)));
	int* devDist;
	CUDA_CALL(cudaMalloc((void**)&devDist, sizeof(int) * CITY_SIZE * CITY_SIZE * NUM_FEATURES));
	int* devObst;
	CUDA_CALL(cudaMalloc((void**)&devObst, sizeof(int) * CITY_SIZE * CITY_SIZE * NUM_FEATURES));
	bool* devToRaise;
	CUDA_CALL(cudaMalloc((void**)&devToRaise, sizeof(bool) * CITY_SIZE * CITY_SIZE * NUM_FEATURES));
	int* devQueue;
	CUDA_CALL(cudaMalloc((void**)&devQueue, sizeof(int) * (QUEUE_MAX + 1)));
	unsigned int* devQueueBegin;
	CUDA_CALL(cudaMalloc((void**)&devQueueBegin, sizeof(unsigned int)));
	unsigned int* devQueueEnd;
	CUDA_CALL(cudaMalloc((void**)&devQueueEnd, sizeof(unsigned int)));


	std::vector<float> zoneTypeDistribution(6);
	zoneTypeDistribution[0] = 0.5f; // 住宅
	zoneTypeDistribution[1] = 0.2f; // 商業
	zoneTypeDistribution[2] = 0.1f; // 工場
	zoneTypeDistribution[3] = 0.1f; // 公園
	zoneTypeDistribution[4] = 0.05f; // アミューズメント
	zoneTypeDistribution[5] = 0.05f; // 学校・図書館
	
	// 初期プランを生成
	generateZoningPlan(*hostZoningPlan, zoneTypeDistribution);
	
	// 初期プランをデバイスバッファへコピー
	CUDA_CALL(cudaMemcpy(devZoningPlan, hostZoningPlan, sizeof(ZoningPlan), cudaMemcpyHostToDevice));

	// デバッグ用
	if (CITY_SIZE <= 100) {
		showDevZoningPlan(devZoningPlan);
	}




	unsigned int* devRand;
	CUDA_CALL(cudaMalloc((void**)&devRand, sizeof(unsigned int)));
	CUDA_CALL(cudaMemset(devRand, 0, sizeof(unsigned int)));

	// 現在プランのスコア
	float* devScore;
	CUDA_CALL(cudaMalloc((void**)&devScore, sizeof(float)));
	CUDA_CALL(cudaMemset(devScore, 0, sizeof(float)));

	// 提案プランのスコア
	float* devProposalScore;
	CUDA_CALL(cudaMalloc((void**)&devProposalScore, sizeof(float)));

	// 交換セル
	int2* devCell1;
	CUDA_CALL(cudaMalloc((void**)&devCell1, sizeof(int2)));
	int2* devCell2;
	CUDA_CALL(cudaMalloc((void**)&devCell2, sizeof(int2)));

	//
	int* devResult;
	CUDA_CALL(cudaMalloc((void**)&devResult, sizeof(int)));



	float* devScores;
	CUDA_CALL(cudaMalloc((void**)&devScores, sizeof(float) * CITY_SIZE * CITY_SIZE));









	printf("start...\n");


	// マルチスレッドで、直近の店までの距離を計算
	/*
	start = clock();
	for (int iter = 0; iter < 1; ++iter) {
		// 提案プランを生成
		generateProposal<<<1, 1>>>(devZoningPlan, devRand, devCell1, devCell2);

		// 距離マップを計算
		computeDistMap<<<dim3(CITY_SIZE / GPU_BLOCK_SIZE, CITY_SIZE / GPU_BLOCK_SIZE), GPU_NUM_THREADS>>>(devZoningPlan, devDistanceMap);
		cudaDeviceSynchronize();

		// スコアを計算
		CUDA_CALL(cudaMemset(devProposalScore, 0, sizeof(float)));
		computeScore<<<dim3(CITY_SIZE / GPU_BLOCK_SIZE, CITY_SIZE / GPU_BLOCK_SIZE), GPU_NUM_THREADS>>>(devZoningPlan, devDistanceMap, devProposalScore, devScores);
		cudaDeviceSynchronize();

		// accept
		acceptProposal<<<1, 1>>>(devZoningPlan, devScore, devProposalScore, devCell1, devCell2, devResult);
	}
	end = clock();
	printf("computeDistanceToStore GPU: %lf\n", (double)(end-start)/CLOCKS_PER_SEC);
	*/

	// キューを初期化
	for (int i = 0; i < QUEUE_MAX + 1; ++i) {
		hostQueue[i] = QUEUE_EMPTY;
	}
	hostQueueBegin = 0;
	hostQueueEnd = 0;
	CUDA_CALL(cudaMemcpy(devQueue, hostQueue, sizeof(int) * (QUEUE_MAX + 1), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(devQueueBegin, &hostQueueBegin, sizeof(unsigned int), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(devQueueEnd, &hostQueueEnd, sizeof(unsigned int), cudaMemcpyHostToDevice));

	// 初期化
	initObst<<<1, 1>>>(devZoningPlan, devDist, devObst, devToRaise, devQueue, devQueueBegin, devQueueEnd);
	cudaThreadSynchronize();
	computeDistMap<<<dim3(CITY_SIZE / GPU_BLOCK_SIZE, CITY_SIZE / GPU_BLOCK_SIZE), GPU_NUM_THREADS>>>(devZoningPlan, devDist, devObst, devToRaise, devQueue, devQueueBegin, devQueueEnd);
	cudaThreadSynchronize();

	showDevDistMap(devDist, 0);

	check(devZoningPlan, devDist);

	for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
		// キューを初期化
		CUDA_CALL(cudaMemset(devQueue, BF_CLEARED, sizeof(int) * (QUEUE_MAX + 1)));
		CUDA_CALL(cudaMemset(devQueueBegin, 0, sizeof(unsigned int)));
		CUDA_CALL(cudaMemset(devQueueEnd, 0, sizeof(unsigned int)));

		// 店を１つ移動する
		moveObst<<<1, 1>>>(devZoningPlan, devDist, devObst, devToRaise, devQueue, devQueueBegin, devQueueEnd, devRand);
		cudaThreadSynchronize();
		computeDistMap<<<dim3(CITY_SIZE / GPU_BLOCK_SIZE, CITY_SIZE / GPU_BLOCK_SIZE), GPU_NUM_THREADS>>>(devZoningPlan, devDist, devObst, devToRaise, devQueue, devQueueBegin, devQueueEnd);
		cudaThreadSynchronize();

		if (check(devZoningPlan, devDist) > 0) break;
	}

	showDevZoningPlan(devZoningPlan);
	showDevDistMap(devDist, 0);
	showDevQueue(devQueue, devQueueEnd);

	// release device buffer
	cudaFree(devZoningPlan);
	cudaFree(devDist);
	cudaFree(devQueue);
	cudaFree(devQueueBegin);
	cudaFree(devQueueEnd);

	// release host buffer
	free(hostZoningPlan);
	free(hostDist);
	free(hostQueue);

	//cudaDeviceReset();
}
