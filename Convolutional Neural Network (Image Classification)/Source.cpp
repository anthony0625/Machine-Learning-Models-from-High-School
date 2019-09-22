/*
Convolutional Neural Network for Handwritten Digit Image Recognition

Single Convolutional Layered 12 Kernel(3x3) Proof of Concept
The CNN has been made in a more general form, where the information is moved between layers as rank 3 tensors rather than matrices. This will allow for the addition or removal of convolutional kernels as well as changing the size of acceptable kernels to be much simpler. The same is done for the Neural Network section, using rank 3 tensors for the hidden layers rather than individual matrices. The Convolutional Kernels used in this CNN are also treated as hyperparameters and are thus, not subject to change as a result of the calculated cost.



Comments:
Because of the 3x3 size of Convolutional Kernels, there is no pooling procedure put into place, though the structure of the CNN allows for an easily integratable pooling sequence if desired.
*/

#include <vector>
#include <iostream>
#include <string>
#include <thread>
#include <algorithm>
#include <fstream>
#include <string>

using namespace std;

const int stride = 1;

const int HIDDEN_LAYERS = 1;
const int FLATTENED_INPUTS = 108;
const int HIDDEN_NODES_1 = 20;
const int OUTPUTS = 10;
const int IMAGE_HEIGHT = 28;
const int IMAGE_WIDTH = 28;

vector<double> actual;
vector<double>* actualP = &actual;

//Input
vector<vector<vector<int>>> input = { { { 2,2,2,2,2 },{ 3,3,3,3,3 },{ 4,4,4,4,4 },{ 5,5,5,5,5 },{ 6,6,6,6,6 } } };
vector<vector<vector<int>>>* inputP = &input;

vector<vector<vector<int>>> input1(1, vector<vector<int>>(28, vector<int>(28, 1)));
vector<vector<vector<int>>>* input1P = &input1;

//First layer convolutional kernels
vector<vector<vector<int>>> ConvKernels1 = { { { 0,1,0 },{ 0,1,0 },{ 0,1,0 } },
{ { 0,0,0 },{ 1,1,1 },{ 0,0,0 } },
{ { 1,0,0 },{ 1,0,0 },{ 1,1,1 } },
{ { 1,1,1 },{ 0,0,1 },{ 0,0,1 } },
{ { 0,1,0 },{ 1,1,1 },{ 0,1,0 } },
{ { 1,0,1 },{ 0,1,0 },{ 1,0,1 } },
{ { 1,1,1 },{ 1,1,1 },{ 1,1,1 } },
{ { 0,1,0 },{ 1,0,1 },{ 0,1,0 } },
{ { 1,1,1 },{ 1,0,0 },{ 1,0,0 } },
{ { 0,0,1 },{ 0,0,1 },{ 1,1,1 } },
{ { 1,1,0 },{ 1,1,0 },{ 1,1,0 } },
{ { 1,1,1 },{ 1,1,1 },{ 0,0,0 } } };
vector<vector<vector<int>>>* ConvKernels1P = &ConvKernels1;

//Output of convolutional layer
vector<vector<vector<int>>> ConvOut1(ConvKernels1P->size(), vector<vector<int>>(inputP->at(0).size() - 1 - (int)(ConvKernels1P->at(0).size() / 2), vector<int>(inputP->at(0).size() - 1 - (int)(ConvKernels1P->at(0).size() / 2), 1)));
vector<vector<vector<int>>>* ConvOut1P = &ConvOut1;

//Flattened Layer
vector<int> flatIn = {};
vector<int>* flatInP = &flatIn;

//Weights for Hidden Layers for Neural Network Section
vector<vector<vector<double>>> weights(HIDDEN_LAYERS + 1);
vector<vector<vector<double>>>* weightsP = &weights;

//Final Output Vector
vector<double> flatOut = { 0,0,0,0,0,0,0,0,0,0 };
vector<double>* flatOutP = &flatOut;

vector<double> cost(flatOutP->size());
vector<double>* costP = &cost;

double randDouble() {
	return (((double)rand()) / (((double)RAND_MAX) + 30000000.0));
}
//Randomized Generation of Weights
//At the smallest level, weights of the same origin(starting node) are grouped together in the same vector
void initialize() {
	vector<double> temp;
	for (int i = 0; i < FLATTENED_INPUTS; i++) {
		for (int j = 0; j < HIDDEN_NODES_1; j++) {
			temp.push_back(randDouble());
		}
		weights[0].push_back(temp);
		temp.clear();
	}
	for (int i = 0; i < HIDDEN_NODES_1; i++) {
		for (int j = 0; j < OUTPUTS; j++) {
			temp.push_back(randDouble());
		}
		weights[1].push_back(temp);
		temp.clear();
	}
}



int dot(vector<vector<vector<int>>>* tenA, vector<vector<vector<int>>>* tenB, int startX, int startY, int startZ1, int startZ2) {
	int sum = 0;
	for (int i = 0; i < tenB->at(startZ2).size(); i++) {
		for (int j = 0; j < tenB->at(startZ2).at(i).size(); j++) {
			sum += tenA->at(startZ1).at(startY + i).at(startX + j) * tenB->at(startZ2).at(i).at(j);
			//cout << tenA->at(0).at(startY + j).at(startX + k) * tenB->at(i).at(j).at(k);
		}
		//cout << endl;;
	}
	return sum;
}

void convolute(vector<vector<vector<int>>>* tenA, vector<vector<vector<int>>>* tenB, vector<vector<vector<int>>>* tenC) {
	for (int i = 0; i < tenB->size(); i++) {
		for (int j = 0; j < tenC->at(i).size(); j += stride) {
			for (int k = 0; k < tenC->at(i).at(j).size(); k += stride) {
				tenC->at(i).at(j).at(k) = dot(tenA, tenB, k, j, 0, i);
			}
		}
	}
}

void flatten(vector<vector<vector<int>>>* input, vector<int>* output) {
	for (int i = 0; i < input->size(); i++) {
		for (int j = 0; j < input->at(i).size(); j++) {
			for (int k = 0; k < input->at(i).at(j).size(); k++) {
				output->push_back(input->at(i).at(j).at(k));
			}
		}
	}
}



double reLU(double x) {
	if (x > 0.0) return x;
	else return 0.025 * x;
}

//Progresses the input through the entire neural network portion
void feedForward(vector<int>* input, vector<vector<vector<double>>>* weights, vector<double>* output) {
	vector<vector<double>>results(HIDDEN_LAYERS + 1);
	vector<double>temp;
	results[0].resize(HIDDEN_NODES_1);
	results[1].resize(OUTPUTS);
	//Read List Vertically
	//Input to first Hidden Layer
	for (int j = 0; j < HIDDEN_NODES_1; j++) {
		for (int i = 0; i < FLATTENED_INPUTS; i++) {
			results[0][j] += ((double)input->at(i)) * weights->at(0).at(i).at(j);
		}
	}
	//ReLU
	for (int i = 0; i < results[0].size(); i++) {
		results[0][i] = reLU(results[0][i]);

	}
	//First Hidden Layer to Output
	for (int j = 0; j < OUTPUTS; j++) {
		for (int i = 0; i < HIDDEN_NODES_1; i++) {
			results[1][j] += (results[0][i] * weights->at(1).at(i).at(j));
		}
	}
	for (int i = 0; i < results[1].size(); i++) {
		results[1][i] = reLU(results[1][i]);
	}
	output->clear();
	output->resize(OUTPUTS);
	for (int i = 0; i < OUTPUTS; i++) {
		output->at(i) = results[1][i];
	}
}

void calculateCost(vector<double>* result, vector<double>* actual, vector<double>* cost) {
	for (int i = 0; i < result->size(); i++) {
		cost->at(i) = actual->at(i) - result->at(i);
	}
}


//Diagnosis Functions
void printTensor(vector<vector<vector<int>>>* ten) {
	for (int i = 0; i < ten->size(); i++) {
		for (int j = 0; j < ten->at(i).size(); j++) {
			for (int k = 0; k < ten->at(i).at(j).size(); k++) {
				cout << ten->at(i).at(j).at(k);
				cout << " ";
			}
			cout << endl;
		}
		cout << endl;
	}
}

void printTensor(vector<vector<vector<double>>>* ten) {
	for (int i = 0; i < ten->size(); i++) {
		for (int j = 0; j < ten->at(i).size(); j++) {
			for (int k = 0; k < ten->at(i).at(j).size(); k++) {
				cout << ten->at(i).at(j).at(k);
				cout << " ";
			}
			cout << endl;
		}
		cout << endl;
	}
}

//Convert CSV data into usable tensor
void convert(string input, vector<vector<vector<int>>>* output, vector<double>* answer) {
	output->clear();
	answer->clear();
	//Converting the data from string format into a vector of integers
	int i = 0, j = 0;
	vector<int> temp;


	for (int i = 0;i < IMAGE_HEIGHT;i++) {
		for (int j = 0;j < IMAGE_WIDTH;j++) {
			//output->at(i).at(j) = input
		}
	}
}

int main() {
	ifstream ip("train.csv");
	string data;
	if (!ip.is_open()) cout << "ERROR: File Open" << endl;
	ip >> data;
	ip >> data;
	ip.close();
	//convert(data, inputP, actualP);

	initialize();//initializes random weights

	//No Pooling sequence is used
	convolute(inputP, ConvKernels1P, ConvOut1P);
	flatten(ConvOut1P, flatInP);

	//Standard Neural Network
	feedForward(flatInP, weightsP, flatOutP);
	for (int i = 0; i < flatOut.size(); i++) {
		cout << flatOut[i] << endl;
	}
	calculateCost(flatOutP, actualP, costP);
	cout << data << endl;
	cout << endl;
	data = data.substr(0,data.find(",",data.find(",")+1));
	cout << data << endl;
	//for(int i=0; label)
	int a;
	cin >> a;
	return 0;
}