#include "hingeLoss.h"

double getHingeLoss(double x1, double x2, int y, double w1, double w2, double b) {
	double loss{ 0 };
	(y == 1) ? loss = 1 - (w1 * x1 + w2 * x2 + b) : loss = 1 + (w1 * x1 + w2 * x2 + b);
	if (loss < 0)
		loss = 0;
	return loss;
}