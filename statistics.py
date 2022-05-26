import nltk
from collections import Counter
import sys
import glob
import os
import spacy
import csv
import sys
import pandas 
import numpys as np

def error_analysis_stats():
    """calculates number of false positives and false negatines with
    implicitit negation and ambiguity"""
    
    #read in outputfile and create pandas dataframe
    df = pd.read_csv('sys.argv[1]', usecols= ['Event', 'Previous token one', 'Previous token two', 'Previous token three', 
                    'Next token one', 'Next token two', 'Next token three', 'POS previous token one',
                   'POS previous token two', 'POS previous token three', 'POS next token one',
                   'POS next token two', 'POS next token three', 'Negcue prev token one', 
                    'Negcue prev token two', 'Negcue prev token three', 'Negcue next token one',
                   'Negcue next token two', 'Negcue next token three', 'Negcue in sentence',
                    'The negcue in sentence', 'Negcue ambiguous', 'Negation', 'Negated']) 
    
    #get the mismatch columns
    df1 = df.loc[~(df['Negated'] == df['Negation'])]

    #define false positive columns with implicit negation and ambitguity
    false_positives = df1[(df1['Negation'] == 'Negated') & (df1['Negated'] == 'NotNegated')]
    fp_ambi = false_positives[(false_positives['Negcue ambiguous'] == '1')]
    fp_unambi = false_positives[(false_positives['Negcue ambiguous'] == 'None')]
    fp_implicit = false_positives[(false_positives['Negcue in sentence'] == 'None')]

    #get length of cases per category
    print(len(false_positives.index))
    print(len(fp_ambi.index))
    print(len(fp_unambi.index))
    print(len(fp_implicit.index))

    #define false positive columns with ambitguity
    false_negatives = df1[(df1['Negation'] == 'NotNegated') & (df1['Negated'] == 'Negated')]
    fn_ambi = false_negatives[(false_negatives['Negcue ambiguous'] == '1')]
    fn_unambi = false_negatives[(false_negatives['Negcue ambiguous'] == 'None')]
    
    #get length of cases per category   
    print()                            
    print(len(false_negatives.index))
    print(len(fn_ambi.index))
    print(len(fn_unambi.index))

error_analysis_stats()

def count_negcues():
    """count negation cues from VUmc corpus from given corpus"""

    corpus = sys.argv[1]
    txt = glob.glob(os.path.join(corpus + sys.argv[2]))
    count = Counter()
    
    for infile in txt:
        with open(infile, errors='ignore') as doc:
            counts = nltk.word_tokenize(doc.read().lower())
            count.update(counts)

    print ("'geen' occurred", count['geen'], "times")
    print ("'niet' occurred", count['niet'], "times")
    print ("'niets' occurred", count['niets'], "times")
    print ("'nooit' occurred", count['nooit'], "times")
    print ("'niks' occurred", count['niks'], "times")
    print ("'zonder' occurred", count['zonder'], "times")
    print ("'niemand' occurred", count['niemand'], "times")
    print ("'weigeren' occurred", count['weigeren'], "times")
    print ("'prevent' occurred", count['prevent'], "times")
    print ("'noch' occurred", count['noch'], "times")
    print ("'afzien' occurred", count['afzien'], "times")
    print ("'zag af' occurred", count['zag af'], "times")
    print ("'af te zien' occurred", count['af te zien'], "times")
    print ("'ziet af' occurred", count['ziet af'], "times")
    print ("'zie af' occurred", count['zie af'], "times")
    print ("'zien af' occurred", count['zien af'], "times")
    print ("'zagen af' occurred", count['zagen af'], "times")
    print ("'afgezien' occurred", count['afgezien'], "times")
    print ("'blanco' occurred", count['blanco'], "times")
    print ("'nee' occurred", count['nee'], "times")
    print ("'negatief' occurred", count['negatief'], "times")
    print ("'niet aanwezig' occurred", count['niet aanwezig'], "times")
    print ("'niet meer' occurred", count['niet meer'], "times")
    print ("'noch, noch' occurred", count['noch, noch'], "times")
    print ("'ontbreken' occurred", count['ontbreken'], "times")
    print ("'ontbreek' occurred", count['ontbreek'], "times")
    print ("'ontbreekt' occurred", count['ontbreekt'], "times")
    print ("'ontbrak' occurred", count['ontbrak'], "times")
    print ("'opheffen' occurred", count['opheffen'], "times")
    print ("'hef op' occurred", count['hef op'], "times")
    print ("'heft op' occurred", count['heft op'], "times")
    print ("'opgeheven' occurred", count['opgeheven'], "times")
    print ("'staken' occurred", count['staken'], "times")
    print ("'staak' occurred", count['staak'], "times")
    print ("'staakt' occurred", count['staakt'], "times")
    print ("'gestaakt' occurred", count['gestaakt'], "times")
    print ("'staakte' occurred", count['staakte'], "times")
    print ("'staakten' occurred", count['staakten'], "times")
    print ("'stop' occurred", count['stop'], "times")
    print ("'stoppen' occurred", count['stoppen'], "times")
    print ("'stopt' occurred", count['stopt'], "times")
    print ("'gestopt' occurred", count['gestopt'], "times")
    print ("'stopte' occurred", count['stopte'], "times")
    print ("'stopten' occurred", count['stopten'], "times")
    print ("'sluit uit' occurred", count['sluit uit'], "times")
    print ("'sluiten uit' occurred", count['sluiten'], "times")
    print ("'uitgesloten' occurred", count['uitgesloten'], "times")
    print ("'sloot uit' occurred", count['sloot uit'], "times")
    print ("'sloten uit' occurred", count['sloten uit'], "times")
    print ("'uitsluiten' occurred", count['uitsluiten'], "times")
    print ("'verdwijn' occurred", count['verdwijn'], "times")
    print ("'verdwijnt' occurred", count['verdwijnt'], "times")
    print ("'verdwijnen' occurred", count['verdwijnen'], "times")
    print ("'verdween' occurred", count['verdween'], "times")
    print ("'verdwenen' occurred", count['verdwenen'], "times")
    print ("'verdwijnen' occurred", count['verdwijnen'], "times")
    print ("'vervangen' occurred", count['vervangen'], "times")
    print ("'vervang' occurred", count['vervang'], "times")
    print ("'vervangt' occurred", count['vervangt'], "times")
    print ("'verving' occurred", count['verving'], "times")
    print ("'vervingen' occurred", count['vervingen'], "times")
    print ("'verwijderen' occurred", count['verwijderen'], "times")
    print ("'verwijder' occurred", count['verwijder'], "times")
    print ("'verwijdert' occurred", count['verwijdert'], "times")
    print ("'verdwijderden' occurred", count['verwijderden'], "times")
    print ("'verwijderde' occurred", count['verwijderde'], "times")
    print ("'weigeren' occurred", count['weigeren'], "times")
    print ("'weiger' occurred", count['weiger'], "times")
    print ("'weigert' occurred", count['weigert'], "times")
    print ("'weigerde' occurred", count['weigerde'], "times")
    print ("'weigerden' occurred", count['weigerden'], "times")
    print ("'geweigerd' occurred", count['geweigerd'], "times")
    print ("'zonder' occurred", count['zonder'], "times")
    print ("'ontbraken' occurred", count['ontbraken'], "times")
    print ("'hefte op' occurred", count['hefte op'], "times")
    print ("'hefde op' occurred", count['hefde op'], "times")
    print ("'hefden op' occurred", count['hefden op'], "times")
    print ("'nog niet' occurred", count['nog niet'], "times")
    print ("'nog geen' occurred", count['nog geen'], "times")
    print ("'negatief' occurred", count['negatief'], "times")
    
count_negcues()

def count_tokens_sentences():
"""count the number of tokens and sentences in a given corpus"""

    corpus = sys.argv[1]

    with open(corpus) as f:
        content = f.read()
        tokenized = nltk.word_tokenize(content)
        sents = nltk.sent_tokenize(tokenized)
        print(len(tokenized))
        print(len(sents))

count_tokens_sentences()
