#include <vector>
#include <iostream>
#include <string>
#include <thread>
#include <algorithm>
#include <fstream>
#include <string>

using namespace std;

const int HIDDEN_LAYERS = 1;
const int INPUTS = 784;
const int HIDDEN_NODES_1 = 200;
const int OUTPUTS = 10;

//Input
vector<int> input(784, 0);
vector<int>* inputP = &input;

//Weights for Hidden Layers for Neural Network Section
vector<vector<vector<double>>> weights(HIDDEN_LAYERS + 1);
vector<vector<vector<double>>>* weightsP = &weights;

//Final Output Vector
vector<double> output = { 0,0,0,0,0,0,0,0,0,0 }; 
vector<double>* outputP = &output;

vector<double> actual(outputP->size());
vector<double>* actualP = &actual;

vector<double> cost(outputP->size());
vector<double>* costP = &cost;

vector<vector<double>> bias(HIDDEN_LAYERS + 1);
vector<vector<double>>* biasP = &bias;

//Vectors containing partial derivatives
//vector<vector<vector<double>>> activationPartial;
//vector<vector<vector<double>>>* activationPartialP = &activationPartial;
vector<vector<vector<double>>> gradient((HIDDEN_NODES_1+1)*2);
vector<vector<vector<double>>>* gradientP = &gradient;
/*GRADIENT VECTORS
0 - LAYER 1 WEIGHTS gradient.at(0).size() = 784
1 - LAYER 1 BIASES gradient.at(1).size() = 1
2 - LAYER 2 WEIGHTS gradient.at(2).size() = 200
3 - LAYER 2 BIASES gradient.at(3).size() = 1

0 - LAYER 1 WEIGHTS gradient.at(0)[n].size() = 200
1 - LAYER 1 BIASES gradient.at(1)[0].size() = 200
2 - LAYER 2 WEIGHTS gradient.at(2)[n].size() = 10
3 - LAYER 2 BIASES gradient.at(3)[0].size() = 10
*/
vector<vector<double>>results((HIDDEN_LAYERS + 1)*2); //Seperating original sum and reLU'd sum



double randDouble() {
	return (((double)rand()) / (((double)RAND_MAX) + 30000000.0));
}

//Randomized Generation of Weights and Biases, also Resizes some vectors
void initialize() {
	results[0].resize(HIDDEN_NODES_1);
	results[1].resize(HIDDEN_NODES_1);
	results[2].resize(OUTPUTS);
	results[3].resize(OUTPUTS);
	bias[0].resize(HIDDEN_NODES_1);
	bias[1].resize(OUTPUTS);

	vector<double> temp;
	for (int i = 0; i < INPUTS; i++) {
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

	for (int i = 0; i < HIDDEN_NODES_1;i++) {
		biasP->at(0).push_back(randDouble());
	}
	for (int i = 0;i < OUTPUTS;i++) {
		biasP->at(1).push_back(randDouble());
	}
}

double reLU(double x) {
	if (x > 0.0) return x;
	else return 0.025 * x;
}

//Progresses the input through the entire neural network portion
void feedForward(vector<int>* input, vector<vector<vector<double>>>* weights, vector<double>* output) {
	
	vector<double>temp;
	
	//Read List Vertically
	//Input to first Hidden Layer
	for (int j = 0; j < HIDDEN_NODES_1; j++) {
		for (int i = 0; i < INPUTS; i++) {
			results[0][j] += ((double)input->at(i)) * weights->at(0).at(i).at(j);
		}
	}
	//Add biases
	for (int i = 0;i < HIDDEN_NODES_1;i++) {
		results[0][i] += biasP->at(0).at(i);
	}
	//ReLU
	for (int i = 0; i < results[0].size(); i++) {
		results[1][i] = reLU(results[0][i]);
	}
	//First Hidden Layer to Output
	for (int j = 0; j < OUTPUTS; j++) {
		for (int i = 0; i < HIDDEN_NODES_1; i++) {
			results[2][j] += (results[0][i] * weights->at(1).at(i).at(j));
		}
	}
	//Add biases
	for (int i = 0;i < OUTPUTS;i++) {
		results[2][i] += biasP->at(1).at(i);
	}
	//ReLU
	for (int i = 0; i < results[1].size(); i++) {
		results[3][i] = reLU(results[1][i]);
	}
	output->clear();
	output->resize(OUTPUTS);
	for (int i = 0; i < OUTPUTS; i++) {
		output->at(i) = results[1][i];
	}
}

void backpropagate() {
	//Need to clear and initialize gradient vectors before use each time
	int n = 0;
	for (int i = 0;i < HIDDEN_NODES_1;i++) {//Starting with i as the smallest vectors of the weights are grouped by the node of origin, not of destination
		for (int j = 0;j < OUTPUTS;j++) {//
			if (results[2][j] > 0) n = 1;//Derivative of reLU
			if (results[2][j] <= 0) n = 0.025;
			gradientP->at(2).at(j).at(i) = (2 * (actual[i] - results[3][i])) * (n) * (results[1][i]);//GRAD 2
		}
	}
	for (int i = 0;i < OUTPUTS;i++) {
		gradientP->at(3).at(0).at(i) = (2 * (actual[i] - results[3][i])) * (n) * (1);//GRAD 3
	}
	for (int k = 0;k < OUTPUTS;k++) {
		for (int i = 0;i < INPUTS;i++) {
			for (int j = 0;j < HIDDEN_NODES_1;j++) {
				//Simplification: n=1
				gradientP->at(0).at(j).at(i) += (2 * (actual[i] - results[3][k])) * (1) * (weightsP->at(1)[j][k]) * (input[i]); //Using += to sum across multiple costs  GRAD 0
				gradientP->at(1).at(0).at(j) += (2 * (actual[i] - results[3][k])) * (n) * (weightsP->at(1)[j][k]) * (1);//Using += for the same reason as stated above  GRAD 1 
			}
		}
	}

	//Applying the gradient to the weights and biases
	for (int i = 0;i < HIDDEN_NODES_1;i++) {//UPDATE LAYER 1 BIASES (1ST HIDDEN LAYER)
		biasP->at(0)[i] += gradientP->at(1)[0][i];
	}
	for (int i = 0;i < OUTPUTS;i++) {//UPDATE LAYER 2 BIASES (OUTPUT)
		biasP->at(1)[i] += gradientP->at(3)[0][i];
	}
	for (int i = 0;i < INPUTS;i++) {
		for (int j = 0;j < HIDDEN_NODES_1;j++) {
			weightsP->at(0)[j][i] += gradientP->at(0)[j][i];
		}
	}
	for (int i = 0;i < HIDDEN_NODES_1;i++) {
		for (int j = 0;j < OUTPUTS;j++) {
			weightsP->at(1)[j][i] += gradientP->at(2)[j][i];
		}
	}
}

void calculateCost(vector<double>* result, vector<double>* actual, vector<double>* cost) {
	for (int i = 0; i < result->size(); i++) {
		cost->at(i) = (actual->at(i) - result->at(i)) * (actual->at(i) - result->at(i));
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

int main() {
	initialize();//initializes random weightsz1
	
	feedForward();//Pushes the input through the network to get an output
	
	calculateCost();//Calculates the squared error of the output
	
	backpropagate();//Calculates the gradient of the network based off of the cost to update the weights and biases

	//Simple sequence to keep the terminal from automatically exiting upon completion
	int a;
	cin >> a;
	return 0;
}