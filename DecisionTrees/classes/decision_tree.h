#include "node.h"


using namespace std;

class DecisionTree{
	private:
		DecisionNode *root;
		string method; //gini or information gain
		vector<string> headers;
		vector<double> errors;
	public:
		double_matrix parse_data(string filename);
		void print_tree(int spaces);
		DecisionTree(string filename,string method);
		void generatePDF();
};

void DecisionTree::generatePDF() {
	fstream file("graph.vz", fstream::out | fstream::trunc);
	if (file.is_open()) {
		file << "digraph G\n";
		file << "{\n";
		this->root->printAllNodes(file,headers);
		this->root->printNodesConnections(file);
		file << "}\n";
		file.close();
		system("dot -Tpdf graph.vz -o graph.pdf");
	}
}

DecisionTree::DecisionTree(string filename,string method){
	double_matrix data = parse_data(filename);
	this->method = method;
	this->root = new DecisionNode(data,0,method);
	this->root->calculate_error(this->errors);
}

void DecisionTree::print_tree(int spaces){
	cout << "ROOT:" << endl;
	this->root->print_node(spaces);
}

double_matrix DecisionTree::parse_data(string filename){
	double_matrix data;
	fstream file;
	file.open(filename, ios::in);
	string line;
	int x = 0, y = 0;

	getline(file, line);
	istringstream ss_header(line);
	string token;

	while(getline(ss_header, token, ',')) {
		headers.push_back(token);
	}

	while(getline(file,line)){
		vector<double> single_row;
		istringstream ss(line);

		while(getline(ss, token, ',')) {
			if(isdigit(token[0])){
				double value = floorf(stod(token) * 100) / 100;
				single_row.push_back(value);
			} else {
				if(token == "Male")		
					single_row.push_back(0.0);
				else 
					single_row.push_back(1.0);
			}
		}
		data.push_back(single_row);
	}
	file.close();
	return data;
}