from DecisionTree import DecisionTree
import json


#Load configs
configs = json.load(open('config.json'))


#Extract and data set
data_train = open(configs['filepath']['train'], encoding=configs['file_encoding'])
data_train = data_train.readlines()

data_dev = open(configs['filepath']['dev'], encoding=configs['file_encoding'])
data_dev = data_dev.readlines()

data_test = open(configs['filepath']['test'], encoding=configs['file_encoding'])
data_test = data_test.readlines()


#2D data
data_train_2D = [data.split(" ") for data in data_train]
data_dev_2D = [data.split(" ") for data in data_dev]
data_test_2D = [data.split(" ") for data in data_test]


########################
# Decision tree
########################
DT = DecisionTree(configs['verbose_level'])

#Data cleaning removing broken sentence or missing info from the data
DT.data_cleaning(data_train_2D)
DT.data_cleaning(data_dev_2D)
DT.data_cleaning(data_test_2D)
print('[Info] Data cleaning completed')

#Getting features
X_train, Y_train, Sentences_train = DT.split_features(data_train_2D, configs['shuffle_data'])
X_dev, Y_dev, Sentences_dev = DT.split_features(data_dev_2D, configs['shuffle_data'])
X_test, Y_test, Sentences_test = DT.split_features(data_test_2D, configs['shuffle_data'])
print('[Info] Extracted features')

#Learning Curve
#Training the model based on the percentage of training sample
learning_scores = []
for percentage in configs['learning_sample_rate']:
    print('\n[Learning] Training with {:.0f}% samples'.format(percentage * 100))
    
    #Get samples %
    train_sample_count = int(len(X_train[0]) * percentage)
    X_train_sample = [row[:train_sample_count] for row in X_train]
    Y_train_sample = Y_train[:train_sample_count]
    
    #Train the model
    DT_model_sample = DT.fit(X_train_sample, Y_train_sample)
    #Predict
    Y_test_pred_sample = DT_model_sample.predict(X_test)
    #Scores
    Y_test_score_sample = DT.metric_score(Y_test, Y_test_pred_sample)
    
    learning_scores.append(Y_test_score_sample)
    print('[Learning] Test Accuracy: {:.4f}'.format(Y_test_score_sample['accuracy']))
#Calculate the accuracy average of the learning curve
learning_avg_list = [each['accuracy'] for each in learning_scores]
print('\n[Learning] Test Average Accuracy: {:.4f}'.format(sum(learning_avg_list)/len(learning_avg_list)))

#Final run with a complete training sample
print('\n[Info] Final Run')
DT_model = DT.fit(X_train, Y_train)
print('[Info] Final training model completed')

#Predict
Y_train_pred = DT_model.predict(X_train)
train_score = DT.metric_score(Y_train, Y_train_pred)
print('Train Accuracy:\t\t{:.4f}'.format(train_score['accuracy']))
print('Train Precision:\t{:.4f}'.format(train_score['precision']))
print('Train Recall:\t\t{:.4f}'.format(train_score['recall']))
print('Train Matrix:\t\t{}'.format(train_score['confusion_matrix']))

Y_dev_pred = DT_model.predict(X_dev)
dev_score = DT.metric_score(Y_dev, Y_dev_pred)
print('\nDev Accuracy:\t\t{:.4f}'.format(dev_score['accuracy']))
print('Dev Precision:\t\t{:.4f}'.format(dev_score['precision']))
print('Dev Recall:\t\t{:.4f}'.format(dev_score['recall']))
print('Dev Matrix:\t\t{}'.format(dev_score['confusion_matrix']))

Y_test_pred = DT_model.predict(X_test)
test_score = DT.metric_score(Y_test, Y_test_pred)
print('\nTest Accuracy:\t\t{:.4f}'.format(test_score['accuracy']))
print('Test Precision:\t\t{:.4f}'.format(test_score['precision']))
print('Test Recall:\t\t{:.4f}'.format(test_score['recall']))
print('Test Matrix:\t\t{}'.format(test_score['confusion_matrix']))
