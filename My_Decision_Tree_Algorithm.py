import numpy as np
import pandas as pd

'''
Steps to create algorithm:
1. Split dataset based on attribute
2. Calculate gini index
3. Create tree
'''
class Condition:

    def __init__(self, attribute, value):
        self.attribute = attribute
        self.value = value

    def compare(self, example):
        return example[self.attribute] >= self.value

class Predict_Node:

    def __init__(self, data):
        self.predictions = prediction_class_count(data)

class Decision_Node:

    def __init__(self, condition, true_nodes, false_nodes):
        self.condition = condition
        self.true_nodes = true_nodes
        self.false_nodes = false_nodes

def prediction_class_count(data):
    classes = {}
    for row in data:
        predict_class = row[-1]
        if predict_class not in classes:
            classes[predict_class] = 0
        classes[predict_class] += 1
    return classes

def split_on_condition(data, condition):
    true, false = [], []
    for row in data:
        if condition.compare(row):
            true.append(row)
        else:
            false.append(row)
    return true, false

def gini_index(data):
    classes = prediction_class_count(data)
    impurity = 1
    for cl in classes:
        probability = classes[cl] / float(len(data))
        impurity -= probability ** 2
    return impurity

def info_gain(left, right, uncertainty):
    prob = float(len(left)) / (len(left) + len(right))
    return uncertainty - prob * gini_index(left) - (1 - prob) * gini_index(right)

def split(data):
    best_gain = 0
    best_condition = None
    uncertainty = gini_index(data)
    len_row = len(data[0]) - 1

    for attribute in range(len_row):
        values = set([row[attribute] for row in data])
        for val in values:

            condition = Condition(attribute, val)
            true, false = split_on_condition(data, condition)
            gain = info_gain(true, false, uncertainty)

            if gain >= best_gain:
                best_gain = gain
                best_condition = condition

    return best_gain, best_condition

def build_tree(data):
    gain, condition = split(data)
    if gain == 0:
        return Predict_Node(data)

    true, false = split_on_condition(data, condition)

    true_nodes = build_tree(true)
    false_nodes = build_tree(false)

    return Decision_Node(condition, true_nodes, false_nodes)

def classify(data, node):
    if isinstance(node, Predict_Node):
        return node.predictions

    if node.condition.compare(data):
        return classify(data, node.true_nodes)
    else:
        return classify(data, node.false_nodes)

def print_predict_nodes(counts):
    total = sum(counts.values()) * 1.0
    probability = {}
    for value in counts.keys():
        probability[value] = str(int(counts[value] / total * 100)) + "%"
    return probability

dataset = pd.read_csv('HR_comma_sep.csv')
dataset["sales"] = dataset["sales"].astype('category').cat.codes
one_coding = pd.get_dummies(dataset["salary"])
del dataset["salary"]
dataset["low"] = one_coding["low"]
dataset["medium"] = one_coding["medium"]
dataset["high"] = one_coding["high"]

turnover = dataset["left"]
del dataset["left"]
dataset["left"] = turnover

dataset = dataset.astype(object)
dataset = dataset.as_matrix()

percentage = np.random.rand(len(dataset)) < 0.8
train = dataset[percentage]
test = dataset[~percentage]

tree = build_tree(train)

correct = 0
for row in test:
    dic = classify(row,tree)
    for key in dic.keys():
        if row[-1] == key:
            correct += 1

accuracy = correct / float(len(test)) * 100.0
print(accuracy)

for row in test:
    print ("Actual: %s. Predicted: %s" % (row[-1], classify(row, tree)))
