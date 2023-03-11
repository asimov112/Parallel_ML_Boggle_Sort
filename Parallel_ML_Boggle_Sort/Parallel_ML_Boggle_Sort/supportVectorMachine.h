#ifndef SUPPORTVECTORMACHINE_H
#define SUPPORTVECTORMACHINE_H

#include <vector>

double getSVMCost(std::vector<double>& x1, std::vector<double>& x2, std::vector<int>& y, double w1, double w2, double b, double dw1, double dw2, double db);
void trainSVM(std::vector<double>& x1, std::vector<double>& x2, std::vector<int>& y);

#endif
