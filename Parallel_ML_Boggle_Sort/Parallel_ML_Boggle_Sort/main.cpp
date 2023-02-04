// Parallel_ML_Boggle_Sort.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <iterator>
#include "hingeLoss.h"
#include "outlierDetection.h"
#include "supportVectorMachine.h"

using namespace std;

int main(void)
{
	try {
		vector<double> x1{ 35,27,19,25,26,45,46,48,47,29,27,28,27,30,28,23,27,18 };
		vector<double> x2{ 20,57,76,33,52,26,28,29,49,43,137,44,90,49,84,20,54,44 };
		//Clasification resultant
		vector<int> Y{ -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1 };
		trainSVM(x1, x2, Y);
	}
	catch (exception e) {
		printf("Exception occurred: %s", e.what());
	}
	return 0;
}


