#include "node.h"

using namespace std;

class DecisionTree{
	private:
		DecisionNode *root;
		string method; //gini or information gain
		vector<string> headers;
		int_matrix confusion_matrix;
		float_matrix data;
	public:
		float_matrix parse_data(string filename);
		void print_tree(int spaces);
		DecisionTree(string filename,string method);
		void generatePDF();
		void get_confusion_matrix();
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
	data = parse_data(filename);
	this->method = method;
	this->root = new DecisionNode(data,0,method);
}

void DecisionTree::print_tree(int spaces){
	cout << "ROOT:" << endl;
	this->root->print_node(spaces);
}

float_matrix DecisionTree::parse_data(string filename){
	float_matrix data;
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

	headers.push_back("Male");
	headers.push_back("Female");

	while(getline(file,line)){
		vector<long> single_row;
		istringstream ss(line);

		while(getline(ss, token, ',')) {
			if(isdigit(token[0])){
				//float value = floorf(stod(token) * 100) / 100;
				//float value = roundoff(stof(token),2);
				stringstream out;
				out << fixed << setprecision(2) << stof(token);
				long value = stof(out.str());
				//long value = static_cast<long>(stof(token));
				single_row.push_back(value);
			} else {
				if(token == "Male\r")		
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

void DecisionTree::get_confusion_matrix(){
	int true_positive, true_negative, false_positive, false_negative;
	int right_error=0, left_error=0;
	for(int i=0; i<2; i++){
		vector<int> row(2,0);
		this->confusion_matrix.push_back(row);
	}
	for(int i=0; i<this->root->rows; i++){
		this->root->calculate_error(confusion_matrix,data,i);
	}
	
	cout << " "<< "F" << " " << "M" << endl;
	cout << "F" << confusion_matrix[0][0] << " " << confusion_matrix[0][1] << endl;
	cout << "M" << confusion_matrix[1][0] << " " << confusion_matrix[1][1] << endl;
}