#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include <map>
#include <iomanip>
#include <utility>
#include <fstream>

using namespace std;

typedef vector<vector<double>> double_matrix;

class DecisionNode{
	private:
		int id;
		int rows;
		int columns;
		DecisionNode *left;
		DecisionNode *right;
		double_matrix data;
		int threshold; //split
		int feature; // which column has best split
		vector<pair<double_matrix,double_matrix>> splits;
		vector<int> middle_values;
		string method;

		void set_best_feature();
		double_matrix sort_column(int column_index);
		void split(vector<double> gains);
		double gini_impurity(double_matrix branch);
		double gini_gain(pair<double_matrix,double_matrix> branches);
		double information_entropy(double_matrix branch);
		double information_gain(pair<double_matrix,double_matrix> branches);
		vector<pair<double,double>> get_probabilities(double_matrix branch);
		pair<double_matrix,double_matrix> set_branches(int middle, double_matrix sorted_column,int column_index);
	
	public:
		DecisionNode(double_matrix data,int id,string method);
		void print_node(int spaces);
		void printNodesConnections(fstream &file);
		void printAllNodes(fstream &file);
};

void DecisionNode::printNodesConnections(fstream &file){
	if(this->left != nullptr){
		file << "\"" << this << "\"->";
		file << "\"" << left << "\";\n"; 
		this->left->printNodesConnections(file);
	}
	if(this->right != nullptr){
		file << "\"" << this << "\"->";
		file << "\"" << right << "\";\n"; 
		this->right->printNodesConnections(file);
	}
}

void DecisionNode::printAllNodes(fstream &file){
	file << "\"" << this << "\" [\n";
	file << "\tlabel = \"" << this->id <<"\\n column: "<< this->feature << "\\n threshold: "<< this->threshold << " \"\n]\n";
	if(this->left != nullptr){
		this->left->printAllNodes(file);
	}
	if(this->right != nullptr){
		this->right->printAllNodes(file);
	}
}

DecisionNode::DecisionNode(double_matrix dataset,int id,string method){
	this->data = dataset;
	this->rows = this->data.size();
	this->columns = this->data[0].size();
	this->left = nullptr;
	this->right = nullptr;
	this->id = id;
	this->method = method;
	set_best_feature();
}

void DecisionNode::set_best_feature(){
	vector<double> gini_gains;
	vector<double> information_gains;
	for(int i=0; i<columns-1; i++){
		double_matrix sorted_column = sort_column(i);
		double min = sorted_column[0][i];
		double max = sorted_column[rows-1][i];
		int middle = max-min/2;
		this->middle_values.push_back(middle);
		pair<double_matrix,double_matrix> branches = set_branches(middle,sorted_column,i);
		this->splits.push_back(branches);
		if(this->method=="gini")
			gini_gains.push_back(gini_gain(branches));
		else
			information_gains.push_back(information_gain(branches));
	}

	if(this->method=="gini")
		split(gini_gains);
	else
		split(information_gains);
}

double_matrix DecisionNode::sort_column(int column_index){
	double_matrix sorted_column = this->data;
	for(int i=0; i<rows; i++){		
		for(int j=i+1; j<rows; j++){
			double num1 = sorted_column[i][column_index];
			double num2 = sorted_column[j][column_index];
			if(sorted_column[i][column_index] > sorted_column[j][column_index]){
				vector<double> temp = sorted_column[i];
				sorted_column[i] = sorted_column[j];
				sorted_column[j] = temp;
			}
		}
	}
	return sorted_column;
}

pair<double_matrix,double_matrix> DecisionNode::set_branches(int middle, double_matrix sorted_column, int column_index){
	double_matrix left_branch, right_branch;
	for(int i=0; i<rows; i++){
		if(sorted_column[i][column_index] <= middle)
			left_branch.push_back(sorted_column[i]);
		else
			right_branch.push_back(sorted_column[i]);
	}
	return {left_branch,right_branch};
}

double DecisionNode::gini_gain(pair<double_matrix,double_matrix> branches){
	double left_weight = branches.first.size()/(double)rows;
	double right_weight = branches.second.size()/(double)rows;
	double g_initial = gini_impurity(this->data);
	double g_left = gini_impurity(branches.first);
	double g_right = gini_impurity(branches.second);
	return g_initial - left_weight*g_left - right_weight*g_right;
}

double DecisionNode::gini_impurity(double_matrix branch){
	double gini_sum = 0.0;
	vector<pair<double,double>> probs = get_probabilities(branch);
	int c = probs.size(); //number of classes
	for(int i=0; i<c; i++){
		gini_sum += probs[i].second*(1-probs[i].second);
	}
	return gini_sum;
}

double DecisionNode::information_gain(pair<double_matrix,double_matrix> branches){
	double left_weight = branches.first.size()/(double)rows;
	double right_weight = branches.second.size()/(double)rows;
	double e_initial = information_entropy(this->data);
	double e_left = information_entropy(branches.first);
	double e_right = information_entropy(branches.second);
	return e_initial - left_weight*e_left - right_weight*e_right;
}

double DecisionNode::information_entropy(double_matrix branch){
	double entropy_sum = 0.0;
	vector<pair<double,double>> probs = get_probabilities(branch);
	int c = probs.size(); //number of classes
	for(int i=0; i<c; i++){
		entropy_sum += probs[i].second*log2(probs[i].second);
	}
	return -(entropy_sum);
}

vector<pair<double,double>> DecisionNode::get_probabilities(double_matrix branch){
	int size = branch.size();
	map<float,int> freq;
    int i = 0;
    while (i<size){
		float current = data[i][columns-1];
		if(freq.empty() || freq.find(current)==freq.end())
			freq.insert({current,1});
		else
			freq[current] += 1;

        i++;
    }

	vector<pair<double,double>> probs;
	for(auto it=freq.begin(); it!=freq.end(); it++){
		probs.push_back({it->first,it->second/(float)size});
	}
	return probs;
}

void DecisionNode::split(vector<double> gains){
	float max = 0;
	int max_index = 0;
	int zero_counter = 0;
    for (int i=0; i<columns-1; ++i){
		if(gains[i]==0) zero_counter++;
        else if (gains[i] > max) {
            max = gains[i];
			max_index = i;
        }
    }
	
	this->feature = max_index+1;
	this->threshold = this->middle_values[max_index];

	//if all gains==0 or all splits are the same
	if(zero_counter!=columns-1 && 
	adjacent_find(this->middle_values.begin(),this->middle_values.end(),not_equal_to<>()) 
	!= this->middle_values.end()){
		this->left = new DecisionNode(this->splits[max_index].first,this->id+1,this->method);
		this->right = new DecisionNode(this->splits[max_index].second,this->id+1,this->method);
	}
}

void DecisionNode::print_node(int spaces){
	cout << "ID: " << this->id << " ";
	cout << "Threshold: " << this->threshold << " ";
	cout << "Feature: " << this->feature << " ";
	cout << endl;

	if(this->left){
		cout << setw(this->left->id*spaces) << "Left child: ";
		this->left->print_node(spaces);
	}
	if(this->right){
		cout << setw(this->right->id*spaces) << "Right child: ";
		this->right->print_node(spaces);
	}
}

