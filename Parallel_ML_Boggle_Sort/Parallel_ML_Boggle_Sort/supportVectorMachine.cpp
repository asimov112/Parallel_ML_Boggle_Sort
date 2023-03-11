#include "supportVectorMachine.h"
#include "hingeLoss.h"
#include <vector>
#include <iostream>
#include <omp.h>

double getSVMCost(std::vector<double>& x1, std::vector<double>& x2, std::vector<int>& y, double w1, double w2, double b, double db = 0, double dw1 = 0, double dw2 = 0) {
	int n{ static_cast<int>(y.size()) };
	double cost{0.0}, loss{0.0};
	
	#pragma omp parallel for default(shared) private(n) schedule(dynamic)
	for (int i = 0; i < n; ++i) {
		loss = getHingeLoss(x1[i], x2[i], y[i], w1, w2, b);
		cost += loss;
		if (loss > 0) {
			dw1 += (-1 * (x1[i] * y[i]));
			dw2 += (-1 * (x2[i] * y[i]));
			db += (-y[i]);
		}
	}

	cost /= n;
	dw1 /= n;
	dw2 /= n;
	db /= n;
	return cost;
}
void trainSVM(std::vector<double>& x1, std::vector<double>& x2, std::vector<int>& y) {
	// Learning rate, threshold assignment, weight(s) multiplier
	double lrate{ 0.0005 }, threshold{ 0.001 }, w1{ 1 }, w2{ 1 };
	double b{ 0 }, dw1{ 0 }, dw2{ 0 }, db{ 0 }, cost{ 0 };
	int iteration{ 0 };

	#pragma omp parrallel default(shared) schedule(dynamic) 
	{
		while (true) {
			cost = getSVMCost(x1, x2, y, w1, w2, b, dw1, dw2, db);
			if (iteration % 1000 == 0) {
				std::cout << "Iter: " << iteration << " cost = " << cost << " dw1 = " << dw1 << " dw2 = " << dw2 << " db = " << db << std::endl;
			}
			if ((abs(dw1) < threshold) && (abs(dw2) < threshold) && abs((db) < threshold)) {
				std::cout << "y = " << w1 << "* x1 + " << w2 << "* x2 + " << b << std::endl;
				break;
			}
			w1 -= lrate * dw1;
			w2 -= lrate * dw2;
			b -= lrate * db;
			++iteration;
		}
	}
}
