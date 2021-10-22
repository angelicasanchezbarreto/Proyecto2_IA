#include <iostream>
#include <string>
#include "classes/decision_tree.h"

using namespace std;

int main() {
	DecisionTree decision_tree("gender_classification.csv","gini");
	//decision_tree.print_tree(5);
	decision_tree.generatePDF();
	decision_tree.get_confusion_matrix();
}
