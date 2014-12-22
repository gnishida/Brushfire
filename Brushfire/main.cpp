#include <list>

#define GRID_SIZE 20
#define BF_MAX_DIST 99
#define BF_CLEARED -1
#define BF_TYPE_RAISE 0
#define BF_TYPE_LOWER 1
#define MAX_ITERATIONS 1000

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

void dumpZone(int* zone) {
	printf("<<< Zone Map >>>\n");
	for (int r = 0; r < GRID_SIZE; ++r) {
		for (int c = 0; c < GRID_SIZE; ++c) {
			printf("%2d ", zone[r * GRID_SIZE + c]);
		}
		printf("\n");
	}
	printf("\n");
}

void dumpDist(int* dist) {
	printf("<<< Distance Map >>>\n");
	for (int r = 0; r < GRID_SIZE; ++r) {
		for (int c = 0; c < GRID_SIZE; ++c) {
			printf("%2d ", dist[r * GRID_SIZE + c]);
		}
		printf("\n");
	}
	printf("\n");
}

inline bool isOcc(int* obst, int pos) {
	return obst[pos] == pos;
}

inline int distance(int pos1, int pos2) {
	int x1 = pos1 % GRID_SIZE;
	int y1 = pos1 / GRID_SIZE;
	int x2 = pos2 % GRID_SIZE;
	int y2 = pos2 / GRID_SIZE;

	return abs(x1 - x2) + abs(y1 - y2);
}

void clearCell(int* dist, int* obst, int s) {
	dist[s] = BF_MAX_DIST;
	obst[s] = BF_CLEARED;
}

void raise(std::list<int>& queue, int* dist, int* obst, bool* toRaise, int s) {
	Point2D adj[] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

	int x = s % GRID_SIZE;
	int y = s / GRID_SIZE;

	for (int i = 0; i < 4; ++i) {
		int nx = x + adj[i].x;
		int ny = y + adj[i].y;

		if (nx < 0 || nx >= GRID_SIZE || ny < 0 || ny >= GRID_SIZE) continue;
		int n = ny * GRID_SIZE + nx;

		if (obst[n] != BF_CLEARED && !toRaise[n]) {
			if (!isOcc(obst, obst[n])) {
				clearCell(dist, obst, n);
				toRaise[n] = true;
				//queue.push_back(n);
			}
			queue.push_back(n);
		}
	}

	toRaise[s] = false;
}

void lower(std::list<int>& queue, int* dist, int* obst, bool* toRaise, int s) {
	Point2D adj[] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

	int x = s % GRID_SIZE;
	int y = s / GRID_SIZE;

	int ox = obst[s] % GRID_SIZE;
	int oy = obst[s] / GRID_SIZE;

	for (int i = 0; i < 4; ++i) {
		int nx = x + adj[i].x;
		int ny = y + adj[i].y;

		if (nx < 0 || nx >= GRID_SIZE || ny < 0 || ny >= GRID_SIZE) continue;
		int n = ny * GRID_SIZE + nx;

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
	zone[s] = 1;

	// put stores
	obst[s] = s;
	dist[s] = 0;

	queue.push_back(s);
}

void removeStore(std::list<int>& queue, int* zone, int* dist, int* obst, bool* toRaise, int s) {
	zone[s] = 0;

	clearCell(dist, obst, s);

	toRaise[s] = true;

	queue.push_back(s);
}

/**
 * 計算したdistance mapが正しいか、チェックする。
 */
void check(int* zone, int* dist) {
	int count = 0;

	for (int r = 0; r < GRID_SIZE; ++r) {
		for (int c = 0; c < GRID_SIZE; ++c) {
			int min_dist = BF_MAX_DIST;
			for (int r2 = 0; r2 < GRID_SIZE; ++r2) {
				for (int c2 = 0; c2 < GRID_SIZE; ++c2) {
					if (zone[r2 * GRID_SIZE + c2] == 1) {
						int d = distance(r2 * GRID_SIZE + c2, r * GRID_SIZE + c);
						if (d < min_dist) {
							min_dist = d;
						}
					}
				}
			}

			if (dist[r * GRID_SIZE + c] != min_dist) {
				count++;
			}
		}
	}
	
	if (count > 0) {
		printf("Check results: #error cells = %d\n", count);
	}
}

int main() {
	//Point2D store_loc[] = {{0, 0}, {3, 4}};
	int zone[GRID_SIZE * GRID_SIZE];
	int dist[GRID_SIZE * GRID_SIZE];
	int obst[GRID_SIZE * GRID_SIZE];
	bool toRaise[GRID_SIZE * GRID_SIZE];
	
	// initialize the zone
	for (int r = 0; r < GRID_SIZE; ++r) {
		for (int c = 0; c < GRID_SIZE; ++c) {
			zone[r * GRID_SIZE + c] = 0;
			dist[r * GRID_SIZE + c] = BF_MAX_DIST;
			obst[r * GRID_SIZE + c] = BF_CLEARED;
			toRaise[r * GRID_SIZE + c] = false;
		}
	}

	// initialize dist map
	std::list<int> queue;
	int s;
	for (int i = 0; i < 20; ++i) {
		while (true) {
			s = rand() % (GRID_SIZE * GRID_SIZE);
			if (zone[s] == 0) break;
		}

		setStore(queue, zone, dist, obst, toRaise, s);
	}
	dumpZone(zone);
	updateDistanceMap(queue, zone, dist, obst, toRaise);
	dumpDist(dist);

	bf_count = 0;
	for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
		int s2;
		while (true) {
			s2 = rand() % (GRID_SIZE * GRID_SIZE);
			if (zone[s2] == 0) break;
		}

		// move a store
		removeStore(queue, zone, dist, obst, toRaise, s);
		setStore(queue, zone, dist, obst, toRaise, s2);
		updateDistanceMap(queue, zone, dist, obst, toRaise);
		check(zone, dist);
		//dumpZone(zone);
		//dumpDist(dist);
		s = s2;
	}

	printf("avg bf_count = %d\n", bf_count / MAX_ITERATIONS);
	printf("total bf_count = %d\n", bf_count);

	return 0;
}
