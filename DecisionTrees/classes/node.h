#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include <map>
#include <iomanip>
#include <utility>
#include <fstream>

using namespace std;

long roundoff(long value, unsigned char prec){
	long pow_10 = pow(10.0f, (long)prec);
  	return round(value * pow_10) / pow_10;
}

typedef vector<vector<long>> float_matrix;
typedef vector<vector<int>> int_matrix;

class DecisionNode{
	private:
		int id;
		int rows;
		int columns;
		DecisionNode *left;
		DecisionNode *right;
		float_matrix data;
		long threshold; //split
		int feature; // which column has best split
		vector<pair<float_matrix,float_matrix>> splits;
		vector<long> middle_values;
		string method;
		
		friend class DecisionTree;

		void set_best_feature();
		float_matrix sort_column(int column_index);
		void split(vector<long> gains);
		long gini_impurity(float_matrix branch);
		long gini_gain(pair<float_matrix,float_matrix> branches);
		long information_entropy(float_matrix branch);
		long information_gain(pair<float_matrix,float_matrix> branches);
		vector<pair<long,long>> get_probabilities(float_matrix branch);
		pair<float_matrix,float_matrix> set_branches(int middle, float_matrix sorted_column,int column_index);

	public:
		DecisionNode(float_matrix data,int id,string method);
		void print_node(int spaces);
		void printNodesConnections(fstream &file);
		void printAllNodes(fstream &file, vector<string> headers);
		void calculate_error(int_matrix &confusion_matrix, float_matrix full_data, int example_index);
};


DecisionNode::DecisionNode(float_matrix dataset,int id,string method){
	this->data = dataset;
	this->rows = this->data.size();
	this->columns = this->data[0].size();
	this->left = nullptr;
	this->right = nullptr;
	this->id = id;
	this->method = method;
	set_best_feature();
}

void DecisionNode::calculate_error(int_matrix &confusion_matrix, float_matrix full_data, int example_index){
	long column_value = full_data[example_index][this->feature];
	if(column_value <= this->threshold){
		if(this->left && this->left->left && this->left->right)
			this->left->calculate_error(confusion_matrix,full_data,example_index);
		else{
			if(full_data[example_index][columns-1]==0)
				confusion_matrix[1][1]++;
			else
				confusion_matrix[0][1]++;
		}
	}
	else{
		if(this->right && this->right->left && this->right->right)
			this->right->calculate_error(confusion_matrix,full_data,example_index);
		else{
			if(full_data[example_index][columns-1]==1)
				confusion_matrix[0][0]++;
			else
				confusion_matrix[1][0]++;
		}
	}
}

void DecisionNode::set_best_feature(){
	vector<long> gini_gains;
	vector<long> information_gains;
	for(int i=0; i<columns-1; i++){
		float_matrix sorted_column = sort_column(i);
		long min = sorted_column[0][i];
		long max = sorted_column[rows-1][i];
		long middle = (long) (max-min)/2.0;
		this->middle_values.push_back(middle);
		pair<float_matrix,float_matrix> branches = set_branches(middle,sorted_column,i);
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

float_matrix DecisionNode::sort_column(int column_index){
	float_matrix sorted_column = this->data;
	for(int i=0; i<rows; i++){		
		for(int j=i+1; j<rows; j++){
			long num1 = sorted_column[i][column_index];
			long num2 = sorted_column[j][column_index];
			if(sorted_column[i][column_index] > sorted_column[j][column_index]){
				vector<long> temp = sorted_column[i];
				sorted_column[i] = sorted_column[j];
				sorted_column[j] = temp;
			}
		}
	}
	return sorted_column;
}

pair<float_matrix,float_matrix> DecisionNode::set_branches(int middle, float_matrix sorted_column, int column_index){
	float_matrix left_branch, right_branch;
	for(int i=0; i<rows; i++){
		if(sorted_column[i][column_index] <= middle)
			left_branch.push_back(sorted_column[i]);
		else
			right_branch.push_back(sorted_column[i]);
	}
	return {left_branch,right_branch};
}

long DecisionNode::gini_gain(pair<float_matrix,float_matrix> branches){
	int all_size = this->data.size(); //
	int left_size = branches.first.size(); //
	int right_size = branches.second.size(); //
	long left_weight = branches.first.size()/(long)rows;
	long right_weight = branches.second.size()/(long)rows;
	long g_initial = gini_impurity(this->data);
	long g_left = gini_impurity(branches.first);
	long g_right = gini_impurity(branches.second);
	long left = left_weight*g_left; //
	long right = right_weight*g_right; //
	long final_gain = g_initial - left - right; //
	return g_initial - left_weight*g_left - right_weight*g_right;
}

long DecisionNode::gini_impurity(float_matrix branch){
	long gini_sum = 0.0;
	vector<pair<long,long>> probs = get_probabilities(branch);
	int c = probs.size(); //number of classes
	for(int i=0; i<c; i++){
		gini_sum += probs[i].second*(1-probs[i].second);
	}
	return gini_sum;
}

long DecisionNode::information_gain(pair<float_matrix,float_matrix> branches){
	long left_weight = branches.first.size()/(long)rows;
	long right_weight = branches.second.size()/(long)rows;
	long e_initial = information_entropy(this->data);
	long e_left = information_entropy(branches.first);
	long e_right = information_entropy(branches.second);
	return e_initial - left_weight*e_left - right_weight*e_right;
}

long DecisionNode::information_entropy(float_matrix branch){
	long entropy_sum = 0.0;
	vector<pair<long,long>> probs = get_probabilities(branch);
	int c = probs.size(); //number of classes
	for(int i=0; i<c; i++){
		entropy_sum += probs[i].second*log2(probs[i].second);
	}
	return -(entropy_sum);
}

vector<pair<long,long>> DecisionNode::get_probabilities(float_matrix branch){
	int size = branch.size();
	map<long,int> freq;
    int i = 0;
    while (i<size){
		long current = branch[i][columns-1];
		if(freq.empty() || freq.find(current)==freq.end())
			freq.insert({current,1});
		else
			freq[current] += 1;

        i++;
    }

	vector<pair<long,long>> probs;
	for(auto it=freq.begin(); it!=freq.end(); it++){
		probs.push_back({it->first,it->second/(long)size});
	}
	return probs;
}

void DecisionNode::split(vector<long> gains){
	long max = 0;
	int max_index = 0;
	int zero_counter = 0;
    for (int i=0; i<columns-1; ++i){
		if(gains[i]==0) zero_counter++;
        else if (gains[i] > max) {
            max = gains[i];
			max_index = i;
        }
    }
	
	this->feature = max_index;
	this->threshold = this->middle_values[max_index];

	//if all gains==0 or all splits are the same
	if(zero_counter!=columns-1 && 
	adjacent_find(this->middle_values.begin(),this->middle_values.end(),not_equal_to<>()) 
	!= this->middle_values.end()){
		this->left = new DecisionNode(this->splits[max_index].first,this->id+1,this->method);
		this->right = new DecisionNode(this->splits[max_index].second,this->id+1,this->method);
	} else {
		this->left->feature == columns;
		this->right->feature == columns+1;
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

void DecisionNode::printAllNodes(fstream &file, vector<string> headers){
	file << "\"" << this << "\" [\n";
	file << "\tlabel = \"" << this->id <<"\\n column: "<< headers[this->feature] << "\\n threshold: "<< this->threshold << " \"\n]\n";
	if(this->left != nullptr){
		this->left->printAllNodes(file, headers);
	}
	if(this->right != nullptr){
		this->right->printAllNodes(file,headers);
	}
}

