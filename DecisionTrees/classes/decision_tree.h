#include "node.h"


using namespace std;

class DecisionTree{
	private:
		DecisionNode *root;
		string method; //gini or information gain
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
		this->root->printAllNodes(file);
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
	while(getline(file, line))
	{
		vector<double> single_row;
		string temp = "";
		string::const_iterator it = line.begin();
		while (it != line.end())
		{
			if(*it == ',')
			{
				if(isdigit(temp[0])){
					double value = floorf(stod(temp) * 100) / 100;
					single_row.push_back(value);
				}
				temp = "";
				y++;
			} else temp += *it;
			it++;
		}
		if(!temp.empty()){
			if(temp == "Male")		
				single_row.push_back(1.0);
			else 
				single_row.push_back(2.0);
			temp = "";
		}
		y = 0;
		x++;
		data.push_back(single_row);
    }
	file.close();
	return data;
}