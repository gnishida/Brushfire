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

#define CELL_LENGTH 200
#define CITY_SIZE 5 //200
#define GPU_BLOCK_SIZE 5 //40
#define GPU_NUM_THREADS 2 //96
#define GPU_BLOCK_SCALE (1.0)
#define NUM_FEATURES 1 //5
#define QUEUE_MAX 4999
#define MAX_DIST 99
#define BF_CLEARED -1
#define QUEUE_EMPTY -1
#define MAX_ITERATIONS 1000

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
void raise(int2* queue, unsigned int* queue_end, int* dist, int* obst, bool* toRaise, int s, int featureId) {
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
			queue[queue_index] = make_int2(n, featureId);
		}
	}

	toRaise[s * NUM_FEATURES + featureId] = false;
}

__device__
void lower(int2* queue, unsigned int* queue_end, int* dist, int* obst, bool* toRaise, int s, int featureId) {
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
				queue[queue_index] = make_int2(n, featureId);
			}
		}
	}
}

__device__
void setObst(int2* queue, unsigned int* queue_end, int* dist, int* obst, bool* toRaise, int s, int featureId) {
	// put stores
	obst[s * NUM_FEATURES + featureId] = s;
	dist[s * NUM_FEATURES + featureId] = 0;

	unsigned int queue_index = atomicInc(queue_end, QUEUE_MAX);
	queue[queue_index] = make_int2(s, featureId);
}

__device__
void removeObst(int2* queue, unsigned int* queue_end, int* dist, int* obst, bool* toRaise, int s, int featureId) {
	clearCell(dist, obst, s, featureId);

	toRaise[s * NUM_FEATURES + featureId] = true;

	unsigned int queue_index = atomicInc(queue_end, QUEUE_MAX);
	queue[queue_index] = make_int2(s, featureId);
}

/**
 * 距離マップを計算する
 */
__global__
void computeDistMap(ZoningPlan* zoningPlan, int* dist, int* obst, bool* toRaise, int2* queue, unsigned int* queue_begin, unsigned int* queue_end) {
	__shared__ int lock;
	lock = 0;

	//__syncthreads();

	while (true) {
		do {} while (atomicCAS(&lock, 0, 1));
		if (queue[*queue_begin].x == QUEUE_EMPTY) {
			lock = 0;
			break;
		}
		int queue_index = *queue_begin;
		(*queue_begin)++;
		if (*queue_begin > QUEUE_MAX) *queue_begin = 0;
		lock = 0;

		int s = queue[queue_index].x;
		int featureId = queue[queue_index].y;
		//queue[queue_index] = QUEUE_EMPTY;

		if (toRaise[s * NUM_FEATURES + featureId]) {
			raise(queue, queue_end, dist, obst, toRaise, s, featureId);
		} else if (isOcc(obst, obst[s * NUM_FEATURES + featureId], featureId)) {
			lower(queue, queue_end, dist, obst, toRaise, s, featureId);
		}
	}
}

__global__
void initObst(ZoningPlan* zoningPlan, int* dist, int* obst, bool* toRaise, int2* queue, unsigned int* queue_begin, unsigned int* queue_end) {
	for (int r = 0; r < CITY_SIZE; ++r) {
		for (int c = 0; c < CITY_SIZE; ++c) {
			for (int k = 0; k < NUM_FEATURES; ++k) {
				clearCell(dist, obst, r * CITY_SIZE + c, k);
				toRaise[(r * CITY_SIZE + c) * NUM_FEATURES + k] = false;

				if (zoningPlan->zones[r][c].type > 0) {
					setObst(queue, queue_end, dist, obst, toRaise, r * CITY_SIZE + c, zoningPlan->zones[r][c].type - 1);
				}
			}
		}
	}

}

__global__
void moveObst(ZoningPlan* zoningPlan, int* dist, int* obst, bool* toRaise, int2* queue, unsigned int* queue_begin, unsigned int* queue_end, unsigned int* randx) {
	int s1;
	while (true) {
		s1 = randf(randx, 0, CITY_SIZE * CITY_SIZE);
		int x = s1 % CITY_SIZE;
		int y = s1 / CITY_SIZE;
		if (zoningPlan->zones[y][x].type > 0) break;
	}

	int s2;
	while (true) {
		s2 = randf(randx, 0, CITY_SIZE * CITY_SIZE);
		int x = s2 % CITY_SIZE;
		int y = s2 / CITY_SIZE;
		if (zoningPlan->zones[y][x].type == 0) break;
	}

	int x1 = s1 % CITY_SIZE;
	int y1 = s1 / CITY_SIZE;
	int featureId = zoningPlan->zones[y1][x1].type - 1;
	zoningPlan->zones[y1][x1].type = 0;
	int x2 = s2 % CITY_SIZE;
	int y2 = s2 / CITY_SIZE;
	zoningPlan->zones[y2][x2].type = featureId + 1;

	setObst(queue, queue_end, dist, obst, toRaise, s2, featureId);
	removeObst(queue, queue_end, dist, obst, toRaise, s1, featureId);
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
	for (int r = 0; r < CITY_SIZE; ++r) {
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
	CUDA_CALL(cudaMemcpy(hostDist, dist, sizeof(int) * CITY_SIZE * CITY_SIZE * NUM_FEATURES, cudaMemcpyDeviceToHost));
	printf("<<< Distance Map (feature=%d) >>>\n", feature_id);
	for (int r = 0; r < CITY_SIZE; ++r) {
		for (int c = 0; c < CITY_SIZE; ++c) {
			printf("%2d,", hostDist[(r * CITY_SIZE + c) * NUM_FEATURES + feature_id]);
		}
		printf("\n");
	}
	printf("\n");

	free(hostDist);
}

/**
 * デバッグ用に、obstマップを表示する。
 */
__host__
void showDevObst(int* obst, int feature_id) {
	int* hostObst;

	hostObst = (int*)malloc(sizeof(int) * CITY_SIZE * CITY_SIZE * NUM_FEATURES);
	CUDA_CALL(cudaMemcpy(hostObst, obst, sizeof(int) * CITY_SIZE * CITY_SIZE * NUM_FEATURES, cudaMemcpyDeviceToHost));
	printf("<<< Obst Map (feature=%d) >>>\n", feature_id);
	for (int r = 0; r < CITY_SIZE; ++r) {
		for (int c = 0; c < CITY_SIZE; ++c) {
			printf("%2d,", hostObst[(r * CITY_SIZE + c) * NUM_FEATURES + feature_id]);
		}
		printf("\n");
	}
	printf("\n");

	free(hostObst);
}

/**
 * キューの内容を表示する。
 *
 * queueEndは、次の要素を格納する位置を示す。つまり、queueEndの１つ手前までが、有効なキューの値ということだ。
 */
__host__
void showDevQueue(int2* devQueue, unsigned int* devQueueEnd) {
	int2* queue;
	int queueEnd;

	queue = (int2*)malloc(sizeof(int2) * (QUEUE_MAX + 1));
	CUDA_CALL(cudaMemcpy(queue, devQueue, sizeof(int2) * (QUEUE_MAX + 1), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(&queueEnd, devQueueEnd, sizeof(int), cudaMemcpyDeviceToHost));
	printf("<<< Queue >>>\n");
	for (int i = 0; i < queueEnd; ++i) {
		if (queue[i].x >= 0) {
			int x = queue[i].x % CITY_SIZE;
			int y = queue[i].x / CITY_SIZE;
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
	CUDA_CALL(cudaMemcpy(dist, devDist, sizeof(int) * CITY_SIZE * CITY_SIZE * NUM_FEATURES, cudaMemcpyDeviceToHost));

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

				if (dist[(r * CITY_SIZE + c) * NUM_FEATURES + k] != min_dist) {
					if (count == 0) {
						printf("e.g. (%d, %d) featureId=%d\n", c, r, k);
					}
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

	start = clock();

	// ホストバッファを確保
	ZoningPlan* hostZoningPlan = (ZoningPlan*)malloc(sizeof(ZoningPlan));

	// デバイスバッファを確保
	ZoningPlan* devZoningPlan;
	CUDA_CALL(cudaMalloc((void**)&devZoningPlan, sizeof(ZoningPlan)));
	int* devDist;
	CUDA_CALL(cudaMalloc((void**)&devDist, sizeof(int) * CITY_SIZE * CITY_SIZE * NUM_FEATURES));
	int* devObst;
	CUDA_CALL(cudaMalloc((void**)&devObst, sizeof(int) * CITY_SIZE * CITY_SIZE * NUM_FEATURES));
	bool* devToRaise;
	CUDA_CALL(cudaMalloc((void**)&devToRaise, sizeof(bool) * CITY_SIZE * CITY_SIZE * NUM_FEATURES));
	int2* devQueue;
	CUDA_CALL(cudaMalloc((void**)&devQueue, sizeof(int2) * (QUEUE_MAX + 1)));
	unsigned int* devQueueBegin;
	CUDA_CALL(cudaMalloc((void**)&devQueueBegin, sizeof(unsigned int)));
	unsigned int* devQueueEnd;
	CUDA_CALL(cudaMalloc((void**)&devQueueEnd, sizeof(unsigned int)));
	unsigned int* devRand;
	CUDA_CALL(cudaMalloc((void**)&devRand, sizeof(unsigned int)));

	// 初期プランを生成
	std::vector<float> zoneTypeDistribution(6);
	zoneTypeDistribution[0] = 0.5f; // 住宅
	zoneTypeDistribution[1] = 0.2f; // 商業
	zoneTypeDistribution[2] = 0.1f; // 工場
	zoneTypeDistribution[3] = 0.1f; // 公園
	zoneTypeDistribution[4] = 0.05f; // アミューズメント
	zoneTypeDistribution[5] = 0.05f; // 学校・図書館
	
	generateZoningPlan(*hostZoningPlan, zoneTypeDistribution);
	
	// 初期プランをデバイスバッファへコピー
	CUDA_CALL(cudaMemcpy(devZoningPlan, hostZoningPlan, sizeof(ZoningPlan), cudaMemcpyHostToDevice));

	// 初期プランを表示
	if (CITY_SIZE <= 20) {
		showDevZoningPlan(devZoningPlan);
	}

	// 乱数を初期化
	CUDA_CALL(cudaMemset(devRand, 0, sizeof(unsigned int)));

	// キューを初期化
	CUDA_CALL(cudaMemset(devQueue, QUEUE_EMPTY, sizeof(int2) * (QUEUE_MAX + 1)));
	CUDA_CALL(cudaMemset(devQueueBegin, 0, sizeof(unsigned int)));
	CUDA_CALL(cudaMemset(devQueueEnd, 0, sizeof(unsigned int)));

	// 各種マップを初期化
	initObst<<<1, 1>>>(devZoningPlan, devDist, devObst, devToRaise, devQueue, devQueueBegin, devQueueEnd);
	cudaThreadSynchronize();
	computeDistMap<<<1, GPU_NUM_THREADS>>>(devZoningPlan, devDist, devObst, devToRaise, devQueue, devQueueBegin, devQueueEnd);
	cudaThreadSynchronize();

	//showDevDistMap(devDist, 1);
	//showDevObst(devObst, 1);

	//check(devZoningPlan, devDist);

	for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
		// キューを初期化
		CUDA_CALL(cudaMemset(devQueue, QUEUE_EMPTY, sizeof(int2) * (QUEUE_MAX + 1)));
		CUDA_CALL(cudaMemset(devQueueBegin, 0, sizeof(unsigned int)));
		CUDA_CALL(cudaMemset(devQueueEnd, 0, sizeof(unsigned int)));

		// 店を１つ移動する
		moveObst<<<1, 1>>>(devZoningPlan, devDist, devObst, devToRaise, devQueue, devQueueBegin, devQueueEnd, devRand);
		cudaThreadSynchronize();
		computeDistMap<<<1, 1>>>(devZoningPlan, devDist, devObst, devToRaise, devQueue, devQueueBegin, devQueueEnd);
		cudaThreadSynchronize();

		//if (check(devZoningPlan, devDist) > 0) break;
	}

	//showDevZoningPlan(devZoningPlan);
	//showDevDistMap(devDist, 4);
	//showDevObst(devObst, 0);

	//showDevQueue(devQueue, devQueueEnd);

	// release device buffer
	cudaFree(devZoningPlan);
	cudaFree(devDist);
	cudaFree(devQueue);
	cudaFree(devQueueBegin);
	cudaFree(devQueueEnd);
	cudaFree(devRand);

	// release host buffer
	free(hostZoningPlan);

	//cudaDeviceReset();

	end = clock();
	printf("computeDistanceToStore GPU: %lf\n", (double)(end-start)/CLOCKS_PER_SEC);
}
