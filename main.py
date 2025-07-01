import random
import numpy as np
import time 

# Small Dataset Results:
#   - Forward: Feature Subset: {3, 5}, Acc: 0.92
#   - Backward: Feature Subset: {2, 4, 5, 7, 10}, Acc: 0.83
# Large Dataset Results:
#   - Forward: Feature Subset: {27, 1}, Acc: 0.995
#   - Backward: Feature Subset: {27}, Acc: 0.847
# Titanic Dataset Results:
#   - Forward: Feature Subset: {2}, Acc: 0.78
#   - Backward: Feature Subset: {2}, Acc: 0.78


class Validator:

  def __init__(self, data):
    self.data = data


  def evaluate(self, feature_list):
    labels = self.data[:,0] 
    data_features = self.data[:, feature_list] 
    nn_classifier = Classifier(data_features, labels)
    #nn_classifier.normalize()
    accuracy = nn_classifier.train()
    return accuracy

class Classifier:
  def __init__(self,data_features, labels):
    self.data_features = data_features #  N X F
    self.labels = labels 
    self.predictions = None

  def train(self,):
    count = 0
    predictions = []
    for i in range(len(self.data_features)):
      predicted_label = self.test(i)
      predictions.append(predicted_label)
      if predicted_label == self.labels[i]:
        count+= 1
    self.predictions = predictions
    accuracy = count / len(self.data_features)
    #print(f"Accuracy: {accuracy}%\n")
    return accuracy


  def test(self, index):
    test_features = self.data_features[index, :] 
    train_features = np.delete(self.data_features, index, 0) 

    squared_diff = (train_features - test_features) **2 
    sum = np.sum(squared_diff, axis=1) 
    distances = np.sqrt(sum) 
    nearest_neighbor_index = np.argmin(distances)
    train_labels = np.delete(self.labels, index)
    return train_labels[nearest_neighbor_index]


def forward_selection(data, total_features):
    features_subset = []
    final_set = []
    temp_subset = []
    max_accuracy = 0

    for i in range(total_features):
        new_feature = None
        local_max_acc = 0
        local_feature = None

        for j in range(1, total_features + 1):
            if j not in features_subset:
                temp_subset = features_subset + [j]
                accuracy = Validator(data).evaluate(temp_subset)
                print(f"Evaluating subset {temp_subset}, Accuracy: {accuracy * 100:.2f}%")

                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    new_feature = j
                if accuracy > local_max_acc:
                    local_max_acc = accuracy
                    local_feature = j

        if new_feature is not None:
            features_subset.append(new_feature)
            final_set = features_subset[:]  
            print(f"Adding feature {new_feature} to the final set\n")
        else:
            features_subset.append(local_feature)
            print(f"Adding feature {local_feature} to the local subset\n")

    print("Finished Forward Selection")
    print(f"Final selected features: {final_set}, Max Accuracy: {max_accuracy * 100:.2f}%")
    return final_set

def backward_selection(data, total_features):
    features_subset = list(range(1, total_features + 1))
    final_set = list(range(1, total_features + 1))
    max_accuracy =Validator(data).evaluate(features_subset)
    print(f"Evaluating subset {features_subset}, Accuracy: {max_accuracy * 100:.2f}%\n")


    for i in range(total_features - 1):
        loose_feature = None
        local_max_acc = 0
        local_feature = None

        for j in range(1, total_features + 1):
            if j in features_subset:
                temp_subset = [f for f in features_subset if f != j]
                accuracy = Validator(data).evaluate(temp_subset)
                print(f"Evaluating subset {temp_subset}, Accuracy: {accuracy * 100:.2f}%")

                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    loose_feature = j
                if accuracy > local_max_acc:
                    local_max_acc = accuracy
                    local_feature = j

        if loose_feature is not None:
            features_subset = [f for f in features_subset if f != loose_feature]
            final_set = features_subset[:]
            print(f"Removing feature {loose_feature} from the final set\n")
        else:
            features_subset = [f for f in features_subset if f != local_feature]
            print(f"Removing feature {local_feature} from the local subset\n")

    print("Finished Backward Selection")
    print(f"Final selected features: {final_set}, Max Accuracy: {max_accuracy * 100:.2f}%")
    return final_set


def arya_selection (data, total_features):
  accuracies = []
  for i in range(1, total_features + 1):
    temp_accuracy = Validator(data).evaluate([i])
    accuracies.append(temp_accuracy)
    
  top_5 = np.argsort(accuracies)[-5:]  
  print(f"Final selected features: {list(top_5)}")

''''
def normalize(total_features):
    features_subset = list(range(1, total_features + 1))
    min_values = np.min(self.data_features, axis=0)  
    max_values = np.max(self.data_features, axis=0)  
    data_range = max_values - min_values  

    data_range[data_range == 0] = 1
    self.data_features = (self.data_features - min_values) / data_range
'''

def normalize(data):
    features = data[:, 1:] 
    
    min_values = np.min(features, axis=0)
    max_values = np.max(features, axis=0)
    data_range = max_values - min_values

    data_range[data_range == 0] = 1
    
    normalized_features = (features - min_values) / data_range
    
    data = np.hstack((data[:, :1], normalized_features))  
    
if __name__ == "__main__":
    print("Welcome to Fatima & Arya's Feature Selection Algorithm.\n")
    dataset_path = input("Type the name of the file to test: ")

    data = np.loadtxt(dataset_path)
    #normalize(data)

    print("\n\nType the number of the algorithm you want to run.")
    print("1) Forward Selection")
    print("2) Backward Elimination")
    print("3) Arya's Special Algorithm.\n")
    test_num = int(input())

    total_features = len(data[0]) -1
    instances = len(data)

    print(f"\nThis dataset has {total_features}, with {instances} instances!\n")

    print("Training data...\n")
    start_time = time.time()

    if test_num == 1:
      feats = forward_selection(data, total_features)
      end_time = time.time()
      print(f"Time to train: {end_time-start_time}\n\n")

    if test_num == 2:
      feats = backward_selection(data, total_features)
      end_time = time.time()
      print(f"Time to train: {end_time-start_time}\n\n")

    if test_num == 3:
      feats = arya_selection(data, total_features)
      end_time = time.time()
      print(f"Time to train: {end_time-start_time}\n\n")
