import pandas as pd
import glob
import os.path
from itertools import combinations
from sklearn.metrics import cohen_kappa_score, confusion_matrix
categories = ["NegCue", "NegEvent", "_"]

''''Code from Language as Data lab 3.2 Evaluate Annotations'''

def collect_files_1():
    '''
        	Collect all the annotation files in the specified directory
        	and put them in a dictionary with each annotator and their annotations.
        	:param dir: path to the directory the annotationsheets are placed in
        	:type dir: string
        	:returns: dictionary with annotations per annotator
        	'''
    annotations = {}
    for sheet in glob.glob("Annotator_1/*.tsv"):
        filename, extension = os.path.basename(sheet).split(".")
        prefix, annotator= filename.split("_")

        # Read in annotations
        annotation_data = pd.read_csv(sheet, sep="\t", header=0, keep_default_na=False)
        annotations[annotator] = annotation_data["Annotation"]

    for annotator_a, annotator_b in combinations(annotations.keys(), len(annotations.keys())):
        # calculate the agreement percentage
        agreement = [anno1 == anno2 for anno1, anno2 in  zip(annotations[annotator_a], annotations[annotator_b])]
        percentage = sum(agreement)/len(agreement)
        print(annotator_a, annotator_b, )
        print("Percentage Agreement: %.2f" %percentage)
        #calculate cohen's kappa
        kappa = cohen_kappa_score(annotations[annotator_a], annotations[annotator_b], labels=categories)
        print("Cohen's Kappa: %.2f" %kappa)
        #provide the confusion matrix
        confusions = confusion_matrix(annotations[annotator_a], annotations[annotator_b], labels=categories)
        #print(confusions)
        matrix= pd.DataFrame(confusions, index=categories, columns=categories)
        print(matrix)
        #write results to a txt
        #with open(f'{dir}/annotation_evaluation.txt', 'w', encoding='utf8') as outfile:
            #outfile.write("Percentage Agreement: %.2f\n" %percentage)
            #outfile.write("Cohen's Kappa: %.2f\n" %kappa)
            #outfile.write(matrix.to_markdown())

collect_files_1()

def collect_files_2():
    '''
        	Collect all the annotation files in the specified directory
        	and put them in a dictionary with each annotator and their annotations.
        	:param dir: path to the directory the annotationsheets are placed in
        	:type dir: string
        	:returns: dictionary with annotations per annotator
        	'''
    annotations = {}
    for sheet in glob.glob("Annotator_2/*.tsv"):
        filename, extension = os.path.basename(sheet).split(".")
        prefix, annotator= filename.split("_")

        # Read in annotations
        annotation_data = pd.read_csv(sheet, sep="\t", header=0, keep_default_na=False)
        annotations[annotator] = annotation_data["Annotation"]

    for annotator_a, annotator_b in combinations(annotations.keys(), len(annotations.keys())):
        # calculate the agreement percentage
        agreement = [anno1 == anno2 for anno1, anno2 in  zip(annotations[annotator_a], annotations[annotator_b])]
        percentage = sum(agreement)/len(agreement)
        print(annotator_a, annotator_b, )
        print("Percentage Agreement: %.2f" %percentage)
        #calculate cohen's kappa
        kappa = cohen_kappa_score(annotations[annotator_a], annotations[annotator_b], labels=categories)
        print("Cohen's Kappa: %.2f" %kappa)
        #provide the confusion matrix
        confusions = confusion_matrix(annotations[annotator_a], annotations[annotator_b], labels=categories)
        #print(confusions)
        matrix= pd.DataFrame(confusions, index=categories, columns=categories)
        print(matrix)
        #write results to a txt
        #with open(f'{dir}/annotation_evaluation.txt', 'w', encoding='utf8') as outfile:
            #outfile.write("Percentage Agreement: %.2f\n" %percentage)
            #outfile.write("Cohen's Kappa: %.2f\n" %kappa)
            #outfile.write(matrix.to_markdown())

collect_files_2()

def collect_files_3():
    '''
        	Collect all the annotation files in the specified directory
        	and put them in a dictionary with each annotator and their annotations.
        	:param dir: path to the directory the annotationsheets are placed in
        	:type dir: string
        	:returns: dictionary with annotations per annotator
        	'''
    annotations = {}
    for sheet in glob.glob("Annotator_3/*.tsv"):
        filename, extension = os.path.basename(sheet).split(".")
        prefix, annotator= filename.split("_")

        # Read in annotations
        annotation_data = pd.read_csv(sheet, sep="\t", header=0, keep_default_na=False)
        annotations[annotator] = annotation_data["Annotation"]

    for annotator_a, annotator_b in combinations(annotations.keys(), len(annotations.keys())):
        # calculate the agreement percentage
        agreement = [anno1 == anno2 for anno1, anno2 in  zip(annotations[annotator_a], annotations[annotator_b])]
        percentage = sum(agreement)/len(agreement)
        print(annotator_a, annotator_b, )
        print("Percentage Agreement: %.2f" %percentage)
        #calculate cohen's kappa
        kappa = cohen_kappa_score(annotations[annotator_a], annotations[annotator_b], labels=categories)
        print("Cohen's Kappa: %.2f" %kappa)
        #provide the confusion matrix
        confusions = confusion_matrix(annotations[annotator_a], annotations[annotator_b], labels=categories)
        #print(confusions)
        matrix= pd.DataFrame(confusions, index=categories, columns=categories)
        print(matrix)
        #write results to a txt
        #with open(f'{dir}/annotation_evaluation.txt', 'w', encoding='utf8') as outfile:
            #outfile.write("Percentage Agreement: %.2f\n" %percentage)
            #outfile.write("Cohen's Kappa: %.2f\n" %kappa)
            #outfile.write(matrix.to_markdown())

collect_files_3()