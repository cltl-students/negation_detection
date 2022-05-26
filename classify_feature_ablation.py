import sklearn
import csv
import gensim
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report

trainfile = sys.argv[1]
testfile = sys.argv[2]
outputfile = sys.argv[3]

def extract_features_token_only_and_labels(conllfile):
    '''Function that extracts features and gold label from preprocessed conll (here: tokens only).
    :param conllfile: path to the (preprocessed) conll file
    :type conllfile: string
    :return features: a list of dictionaries, with key-value pair providing the value for the feature `token' for individual instances
    :return labels: a list of gold labels of individual instances
    '''
    
    features = []
    labels = []
    conllinput = open(conllfile, 'r')
    csvreader = csv.reader(conllinput, delimiter=',',quotechar='|')
    for row in csvreader:
        if len(row) > 0: # == 20:
            #structuring feature value pairs as key-value pairs in a dictionary
            #the first column in the conll file represents tokens
            feature_value = {'Token': row[0]}
            features.append(feature_value)
            #The last column provides the gold label
            labels.append(row[-1])

    return features, labels

def create_vectorizer_and_classifier(features, labels):
    '''
    Function that takes feature-value pairs and gold labels as input and trains a logistic regression classifier
    :param features: feature-value pairs
    :param labels: gold labels
    :type features: a list of dictionaries
    :type labels: a list of strings
    '''
    
    vec = DictVectorizer()
    model = SVC()
    tokens_vectorized = vec.fit_transform(features)
    model.fit(tokens_vectorized, labels)
    
    return model, vec

#extract features and labels:
feature_values, labels = extract_features_token_only_and_labels(trainfile) 
#create vectorizer and trained classifier:
lr_classifier, vectorizer = create_vectorizer_and_classifier(feature_values, labels)

def get_predicted_and_gold_labels_token_only(testfile, vectorizer, classifier):
    '''
    Function that extracts features and runs classifier on a test file returning predicted and gold labels
    
    :param testfile: path to the (preprocessed) test file
    :param vectorizer: vectorizer in which the mapping between feature values and dimensions is stored
    :param classifier: the trained classifier
    :type testfile: string
    :type vectorizer: DictVectorizer
    :type classifier: SVM()
    :return predictions: list of output labels provided by the classifier on the test file
    :return goldlabels: list of gold labels as included in the test file
    '''
    
    #we use the same function as above (guarantees features have the same name and form)
    sparse_feature_reps, goldlabels = extract_features_token_only_and_labels(testfile)
    
    #we need to use the same fitting as before, so now we only transform the current features according to this mapping (using only transform)
    test_features_vectorized = vectorizer.transform(sparse_feature_reps)
    predictions = classifier.predict(test_features_vectorized)
    
    return predictions, goldlabels

def print_confusion_matrix(predictions, goldlabels):
    '''
    Function that prints out a confusion matrix
    
    :param predictions: predicted labels
    :param goldlabels: gold standard labels
    :type predictions, goldlabels: list of strings
    '''
    
    #based on example from https://datatofish.com/confusion-matrix-python/ 
    data = {'Gold':    goldlabels, 'Predicted': predictions    }
    df = pd.DataFrame(data, columns=['Gold','Predicted'])

    confusion_matrix = pd.crosstab(df['Gold'], df['Predicted'], rownames=['Gold'], colnames=['Predicted'])
    print (confusion_matrix)

def print_precision_recall_fscore(predictions, goldlabels):
    '''
    Function that prints out precision, recall and f-score
    
    :param predictions: predicted output by classifier
    :param goldlabels: original gold labels
    :type predictions, goldlabels: list of strings
    '''
    
    report = classification_report(goldlabels, predictions, digits = 3)
    print('metrics:')
    print(report)

predictions, goldlabels = get_predicted_and_gold_labels_token_only(testfile, vectorizer, lr_classifier)
print_confusion_matrix(predictions, goldlabels)
print_precision_recall_fscore(predictions, goldlabels)

# the functions with multiple features and analysis

#defines the column in which each feature is located (note: you can also define headers and use csv.DictReader)
feature_to_index = {'Event': 0, 'Previous token one': 1, 'Previous token two': 2, 'Previous token three': 3, 
                    'Next token one': 4, 'Next token two': 5, 'Next token three': 6, 'POS previous token one': 7,
                   'POS previous token two': 8, 'POS previous token three': 9, 'POS next token one': 10,
                   'POS next token two': 11, 'POS next token three': 12, 'Negcue prev token one': 13, 
                    'Negcue prev token two': 14, 'Negcue prev token three': 15, 'Negcue next token one': 16,
                   'Negcue next token two': 17, 'Negcue next token three': 18, 'Negcue in sentence': 19,
                    'The negcue in sentence': 20, 'Negcue ambiguous': 21}


def extract_features_and_gold_labels(conllfile, selected_features):
    '''Function that extracts features and gold label from preprocessed conll (here: tokens only).
    
    :param conllfile: path to the (preprocessed) conll file
    :type conllfile: string
    
    
    :return features: a list of dictionaries, with key-value pair providing the value for the feature `token' for individual instances
    :return labels: a list of gold labels of individual instances
    '''
    
    features = []
    labels = []
    conllinput = open(conllfile, 'r')
    #delimiter indicates we are working with a tab separated value (default is comma)
    csvreader = csv.reader(conllinput, delimiter=',',quotechar='|')
    for row in csvreader:
        if len(row) > 0: #== 5:
            #structuring feature value pairs as key-value pairs in a dictionary
            #the first column in the conll file represents tokens
            feature_value = {}
            for feature_name in selected_features:
                row_index = feature_to_index.get(feature_name)
                feature_value[feature_name] = row[row_index]
            features.append(feature_value)
            #The last column provides the gold label (= the correct answer). 
            labels.append(row[-1])
    return features, labels

def get_predicted_and_gold_labels(testfile, vectorizer, classifier, selected_features):
    '''
    Function that extracts features and runs classifier on a test file returning predicted and gold labels
    :param testfile: path to the (preprocessed) test file
    :param vectorizer: vectorizer in which the mapping between feature values and dimensions is stored
    :param classifier: the trained classifier
    :type testfile: string
    :type vectorizer: DictVectorizer
    :type classifier: SVC()
    :return predictions: list of output labels provided by the classifier on the test file
    :return goldlabels: list of gold labels as included in the test file
    '''
    
    #we use the same function as above (guarantees features have the same name and form)
    features, goldlabels = extract_features_and_gold_labels(testfile, selected_features)
    test_features_vectorized = vectorizer.transform(features)
    predictions = classifier.predict(test_features_vectorized)
    
    outputfile = sys.argv[4]

    outfile = open(outputfile, 'w')
    counter = 0
    for line in open(testfile, 'r'):
        if len(line.rstrip('\n').split()) > 0:
            outfile.write(line.rstrip('\n') + ',' + predictions[counter] + '\n')
            counter += 1
    outfile.close()
    
    return predictions, goldlabels

#define which from the available features will be used (names must match key names of dictionary feature_to_index)
all_features = ['Event', 'Previous token one', 'Previous token two', 'Previous token three', 
                    'Next token one', 'Next token two', 'Next token three', 'POS previous token one',
                   'POS previous token two', 'POS previous token three', 'POS next token one',
                   'POS next token two', 'POS next token three', 'Negcue prev token one', 
                    'Negcue prev token two', 'Negcue prev token three', 'Negcue next token one',
                   'Negcue next token two', 'Negcue next token three', 'Negcue in sentence', 'The negcue in sentence', 
                'Negcue ambiguous', ]

sparse_feature_reps, labels = extract_features_and_gold_labels(trainfile, all_features)
lr_classifier, vectorizer = create_vectorizer_and_classifier(sparse_feature_reps, labels)
#when applying our model to new data, we need to use the same features
predictions, goldlabels = get_predicted_and_gold_labels(testfile, vectorizer, lr_classifier, all_features)
print_confusion_matrix(predictions, goldlabels)
print_precision_recall_fscore(predictions, goldlabels)

# example of system with just one additional feature
#define which from the available features will be used (names must match key names of dictionary feature_to_index)
selected_features = ['Event', 'Negcue prev token one', 
                    'Negcue prev token two', 'Negcue prev token three']

feature_values, labels = extract_features_and_gold_labels(trainfile, selected_features)
#we can use the same function as before for creating the classifier and vectorizer
lr_classifier, vectorizer = create_vectorizer_and_classifier(feature_values, labels)
#when applying our model to new data, we need to use the same features
predictions, goldlabels = get_predicted_and_gold_labels(testfile, vectorizer, lr_classifier, selected_features)
print_confusion_matrix(predictions, goldlabels)
print_precision_recall_fscore(predictions, goldlabels)
