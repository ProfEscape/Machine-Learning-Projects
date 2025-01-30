import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Function to calculate entropy
def entropy(y):
    counts = Counter(y)
    total = len(y)
    return -sum((count / total) * np.log2(count / total) for count in counts.values())

# Function to calculate information gain
def information_gain(y, left_y, right_y):
    parent_entropy = entropy(y)
    n = len(y)
    n_left, n_right = len(left_y), len(right_y)
    weighted_avg_entropy = (n_left / n) * entropy(left_y) + (n_right / n) * entropy(right_y)
    return parent_entropy - weighted_avg_entropy

# Function to split data based on a feature and threshold
def split_data(X, y, feature, threshold):
    left_idx = X[:, feature] <= threshold
    right_idx = X[:, feature] > threshold
    return X[left_idx], X[right_idx], y[left_idx], y[right_idx]

# Decision tree node
class DecisionNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Predicted value (used for leaf nodes)

# Recursive function to build the tree
def build_tree(X, y, max_leaf_nodes=3, depth=0):
    # Stop if leaf nodes are reached or there is only one class
    if len(set(y)) == 1 or max_leaf_nodes <= 1:
        return DecisionNode(value=Counter(y).most_common(1)[0][0])
    
    best_gain = 0
    best_split = None

    # Iterate through all features and thresholds
    for feature in range(X.shape[1]):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            _, _, left_y, right_y = split_data(X, y, feature, threshold)
            if len(left_y) == 0 or len(right_y) == 0:
                continue
            gain = information_gain(y, left_y, right_y)
            if gain > best_gain:
                best_gain = gain
                best_split = (feature, threshold)

    # If no good split is found, return a leaf node
    if not best_split:
        return DecisionNode(value=Counter(y).most_common(1)[0][0])

    # Split data
    feature, threshold = best_split
    left_X, right_X, left_y, right_y = split_data(X, y, feature, threshold)

    # Recursively build left and right subtrees
    left_node = build_tree(left_X, left_y, max_leaf_nodes - 1, depth + 1)
    right_node = build_tree(right_X, right_y, max_leaf_nodes - 1, depth + 1)
    return DecisionNode(feature, threshold, left_node, right_node)

# Predict function
def predict_tree(node, X_row):
    if node.value is not None:  # If leaf node
        return node.value
    if X_row[node.feature] <= node.threshold:
        return predict_tree(node.left, X_row)
    else:
        return predict_tree(node.right, X_row)

# Evaluate function
def evaluate_tree(tree, X_test, y_test):
    y_pred = [predict_tree(tree, X_row) for X_row in X_test]
    
    # Calculate metrics
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return y_pred, conf_matrix, accuracy, f1

def calculate_metrics(confusion_matrix):
    TN, FP, FN, TP = confusion_matrix.ravel()  # Extract values from the matrix
    
    # Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    # Precision
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    
    # Recall
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    # F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1_score


# Load dataset
data = pd.read_csv("heart_2020_cleaned.csv")  
data = data.head(40000)  #Take the first 4000 samples

# Encode categorical variables
data['HeartDisease'] = data['HeartDisease'].apply(lambda x: 1 if x == 'Yes' else 0)
for col in ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'Asthma', 'KidneyDisease', 'SkinCancer']:
    data[col] = data[col].astype('category').cat.codes

# Split into training and testing sets (4:1 ratio)
train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)
X_train, y_train = train_data.drop(columns=['HeartDisease']).values, train_data['HeartDisease'].values
X_test, y_test = test_data.drop(columns=['HeartDisease']).values, test_data['HeartDisease'].values

# Train the decision tree
max_leaf_nodes = 3
decision_tree = build_tree(X_train, y_train, max_leaf_nodes=max_leaf_nodes)

# Evaluate the model
y_pred, conf_matrix, accuracy, f1 = evaluate_tree(decision_tree, X_test, y_test)
accuracy, precision, recall, f1_score = calculate_metrics(conf_matrix)

print(f"\nEvaluation Parameters with Maximum Leaf Nodes = {max_leaf_nodes}: \n")
print(f"Confusion Matrix:\n{conf_matrix}\n")
print(f"Accuracy: {accuracy * 100:.2f}%\n")
print(f"F1 Score: {f1:.2f}\n")

print("Custom metrics calculations:")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1_score * 100:.2f}%")

# Function to plot confusion matrix
def plot_confusion_matrix(confusion_matrix, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

# Class names
class_names = ["No Heart Disease", "Heart Disease"]

# Plot the confusion matrix
plot_confusion_matrix(conf_matrix, class_names)