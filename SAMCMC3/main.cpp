#include "MCMC.h"

int main() {
	int* zone;
	int size;

	MCMC mcmc;
	mcmc.findBestPlan(&zone, &size);

	mcmc.showZone(size, zone, "zone.png");
	mcmc.saveZone(size, zone, "zone.txt");

	free(zone);

	return 0;
}
