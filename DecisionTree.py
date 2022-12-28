import time
import random
import math
import logging, sys

class DecisionTreeNode:
    def __init__(self, data, classes):
        self.subtree = []
        self.data = data
        self.classes = [0 for _ in range(len(classes))]


class DecisionTree:    
    def __init__(self, verbose):
        self.train_tree = DecisionTreeNode(None, [])
        
        #Logging type based on the user setting
        match verbose:
            case 1:
                self.logging = logging.DEBUG
            case _:
                self.logging = logging.INFO
        logging.basicConfig(stream=sys.stderr, level=self.logging, format='%(message)s')
    

    #Remove line that have incomplete sentence by checking the length of the line
    def data_cleaning(self, data):
        #Store the index of the data that we will remove
        remove_index = []

        #Loop thru each line of the data
        for index in range(len(data)):
            #Grab the index where the underscore is
            word_index = int(data[index][1])

            #If the total length of that line less than the index
            #And also the ___ not in the index then we will remove this index
            if((len(data[index]) - 3 < word_index) or ("_" not in data[index][word_index + 2])):
                remove_index.append(index)

        #Reverse because we will pop starting from the end of the index
        #If we pop starting from the beginning of the index then the index of the data will be different
        remove_index.sort(reverse=True)
        [data.pop(index) for index in remove_index]

        return self
    

    #Extract features off from the data
    def split_features(self, data, shuffle=False):
        if(shuffle):
            random.shuffle(data)
        
        #class/label
        y = [words[0] for words in data]

        #For storing features
        total_feature_count = 3
        x = [[] for _ in range(total_feature_count)]
        sentences = []

        #Go thru each line of the data to extract features
        for line in data:
            sentence = line[2:]
            sentences.append(sentence)
            
            #feature 0: position/index of the word
            x[0].append(int(line[1]))
            #feature 1: prev word (Only if the prev word exist, otherwise None)
            x[1].append(sentence[int(line[1]) - 1] if int(line[1]) > 0 else None)
            #feature 2: next word (Only if the next word exist by checking if it reach end of sentence, otherwise None)
            x[2].append(sentence[int(line[1]) + 1] if int(line[1]) < (len(sentence) - 1) else None)
        return x, y, sentences  
    
    
    #Get the feature order for tree height based on the information gain
    def __index_order__(self, x, y):
        #Calculate the entropy for the total classes/labels
        label_count = dict()
        class_entropy = 0
        for label in self.total_classes:
            count = y.count(label)
            label_count[label] = count
            logging.debug('[Entropy] {}: {}/{}'.format(label, count, len(y)))
            class_entropy += -(count/len(y)) * math.log((count/len(y)), 2)
        logging.debug('[Entropy] Classes/Labels: {:.6f}'.format(class_entropy))
        
        #Calculate information gain for each feature
        information_gain = []
        #Loop through each feature
        for i in range(len(x)):
            feature_info_gain = {}
            #Loop through each data of a feature
            for j in range(len(x[i])):
                #Check if the data exist in this feature
                if(x[i][j] not in feature_info_gain):
                    feature_info_gain[x[i][j]] = [0 for _ in range(len(self.total_classes))]
                
                #Increase the count of this data based on classes/labels we got from y
                #So we can use it to calculate expected entropy
                feature_info_gain[x[i][j]][self.total_classes.index(y[j])] += 1
            
            #Loop through each data that contain the count of each class/label
            #Calculate excepted entropy
            expected_entropy = 0
            for key in feature_info_gain:
                data_entropy = 0
                total_count_data = sum(feature_info_gain[key])
                for each in feature_info_gain[key]:
                    if each > 0:
                        data_entropy += -(each/total_count_data) * math.log((each/total_count_data), 2)
                    
                expected_entropy += (total_count_data/len(x[i])) * data_entropy
            
            logging.debug('[Entropy] Feature {} expected entropy: {:.6f}'.format(i, expected_entropy))
            
            #Add calculate information gain for each feature and add them to the list
            information_gain.append(class_entropy - expected_entropy)
            logging.debug('[Entropy] Feature {} information gain: {:.6f}'.format(i, class_entropy - expected_entropy))
            
        #We got all information gain and entropy. Done calculated
        logging.debug('[Entropy] Done calculated')
        
        #Put the feature index order based on the highest information gain to lowest
        index_order = []
        for i in range(len(information_gain)):
            info_copy = information_gain[i:]
            index = information_gain.index(max(info_copy), i, len(information_gain))
            information_gain[i], information_gain[index] = information_gain[index], information_gain[i]
            index_order.append(index)
        
        logging.debug('[Entropy] Root will be feature {}'.format(index_order[0]))
        return index_order


    #Training the model
    def fit(self, x, y):
        #Timer to keep track how long it takes to train the model
        time_start = time.time()

        #Get all the classes (no-duplication)
        self.total_classes = list(set(y))
        #Set root classes
        self.train_tree.classes = [0 for _ in range(len(self.total_classes))]
        
        #Get the lowest index order as a root
        self.index_order = self.__index_order__(x, y)
        
        #Loop each data
        for i in range(len(x[0])):
            #Output
            print('[Training] Completed {}/{}\033[A'.format(i, len(x[0])))

            #Set the node back to the root for next/new data
            node = self.train_tree

            #Get the class index
            class_index = self.total_classes.index(y[i])

            #Increase the counter in the root for the class/label
            self.train_tree.classes[class_index] += 1

            #Loop each feature based on the __index_order__ which feature will goes first
            for j in self.index_order:
                #Grab the node address/reference based on the training data
                node_loc = [eachnode for eachnode in node.subtree if x[j][i] == eachnode.data]

                #If node doesn't exist then we will make a new node
                if not len(node_loc):
                    temp = DecisionTreeNode(x[j][i], self.total_classes)
                    node.subtree.append(temp)
                    node = temp
                #Otherwise, we will go into the node (Traversal just like binary tree traversal)
                else:
                    node = node_loc[0]

                #Increase the node's class counter
                node.classes[class_index] += 1
        
        time_end = time.time()
        self.train_time = time_end - time_start
        
        print('                                                                            \033[A')
        return self
        
        
    #Predict using the model
    def predict(self, x):
        #Time to keep track how long to predict
        time_start = time.time()

        #Store the class/label that we predict
        self.test_pred = []

        #Loop thru each x data just like how training works
        for i in range(len(x[0])):
            print('[Predict] Completed {}/{}\033[A'.format(i, len(x[0])))

            #Store previous nodes
            prevnode = [];

            #First node is the root
            node = self.train_tree
            prevnode.append(node)

            for j in self.index_order[:-1]:
                node_loc = [eachnode for eachnode in node.subtree if x[j][i] == eachnode.data]
                if len(node_loc):
                    prevnode.append(node)
                    node = node_loc[0]
                else:
                    break

            #In case all classes/labels contain same probablity/counter
            #Then we will use the previous height probablity/counter of the tree
            counter = -1
            j = 0
            while(j < len(node.classes)):
                if(node.classes[j] == counter):
                    node = prevnode[-1]
                    prevnode.pop()
                    j = 0
                    counter = -1
                    continue
                counter = node.classes[j]
                j += 1

            #Add the class/label based on label index we used for training
            #Also based on the highest counter (Which it uses for features that didn't exist in the model; Probability)
            self.test_pred.append(self.total_classes[node.classes.index(max(node.classes))])
        
        time_end = time.time()
        self.test_time = time_end - time_start

        print('                                                                            \033[A')
        return self.test_pred
    
    
    #Return the score (accuracy, precision, recall, matrix) by comparing the predict with the true label
    def metric_score(self, y_true, y_pred):
        #Count total correct classes/labels
        correct = 0
        for i in range(len(y_true)):
            if(y_pred[i] in y_true[i]):
                correct += 1
                
        #Confusion matrix
        confusion_matrix = [[0 for _ in range(len(self.total_classes))] for _ in range(len(self.total_classes))]
        for i in range(len(y_true)):
            confusion_matrix[self.total_classes.index(y_true[i])][self.total_classes.index(y_pred[i])] += 1

        return { "correct": correct,
                "confusion_matrix": confusion_matrix,
                "accuracy": correct / len(y_true),
                "precision": confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[1][0]),
                "recall": confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1])}
        
