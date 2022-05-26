import glob
import os
import nltk
import re
import csv
import spacy
import itertools
from nltk.corpus import alpino as alp
from nltk.tag import UnigramTagger, BigramTagger
training_corpus = alp.tagged_sents()
unitagger = UnigramTagger(training_corpus)
bitagger = BigramTagger(training_corpus, backoff=unitagger)
pos_tag = bitagger.tag
from collections import Counter

def process_ann():
    """Preprocess and exctract features from corpus"""
    
    corpus = 'EMCDutchClinicalCorpus'
    txt = glob.glob(os.path.join(corpus + '/DL/*.txt'))
    inputfolder = glob.glob(os.path.join(corpus + '/DL/*.ann'))

    #open annotations and get the Negated and Notnegated offsets from .ann files
    lines = list()
    for infile in inputfolder:
        with open(infile) as f:
            for line in f:
                line = line.rstrip().split('\t')
                if len(line) > 0:
                    offset = line[1]
                    events = line[2]
                    if 'Negated' in offset:
                        negevents = []
                        split = offset.split(" ")
                        negevents.append(split)
                        
                        #get the word from txt file for every offset ints in negevents
                        for i in negevents:
                            textfile = open(os.path.splitext(infile)[0] + '.txt', errors="ignore")
                            offsetword = textfile.read()
                            offsetword = offsetword.lower()
                            offsetword = re.sub(r'[^a-zA-Z0-9\s]', ' ', offsetword)
                            offsetword = offsetword[int(i[1]):int(i[2])]
                            offsetword = offsetword.split()
                            
                        #reduce events with multiple words to one word events, convert to string and list
                            multiple_words = []
                            list_result = []
                            if len(offsetword) > 1:
                                multiple_words.append(offsetword)
                                for word in multiple_words:
                                    offsetword = [word[0]]
                            for string in offsetword:
                                offsetword = string
                                list_result.append(string)
                                
                            #open textfiles and read per line
                            text_lines = open(os.path.splitext(infile)[0] + '.txt', errors="ignore")
                            lines_text = text_lines.readlines()
                            for line in lines_text:
                                #print(line)
                                line = line.strip()
                                line = line.lower()
                                line = re.sub(r'[^a-zA-Z0-9\s]', ' ', line)
                                line = nltk.word_tokenize(line)
  
                            #add 3x 'None' at beginning and end of sentences for previous and next token extraction
                                line.append('None')
                                line.append('None')
                                line.append('None')
                                line.insert(0, 'None')
                                line.insert(1, 'None')
                                line.insert(2, 'None')
                                
                                #get matches with text files and offsetwords
                                intersect = list(set(list_result) & set(line))
                                for match in intersect:
                                    item = match
                                    item = item.lower()
                                    item = re.sub(r'[^a-zA-Z0-9\s]', ' ', item)
                                    index = line.index(item)

                                    prev_tokens_one = []
                                    prev_tokens_two = []
                                    prev_tokens_three = []
                                    prev_tokens_four = []
                                    next_tokens_one = []
                                    next_tokens_two = []
                                    next_tokens_three = []
                                    pos_prev_one = []
                                    pos_prev_two = []
                                    pos_prev_three = []
                                    pos_prev_four = []
                                    pos_next_one = []
                                    pos_next_two = []
                                    pos_next_three = []

                                        #extract previous and next tokens
                                    prev_index_one = index - 1
                                    previous_token_one = line[prev_index_one]
                                    item_one = nltk.word_tokenize(previous_token_one)
                                    item_prev_one = pos_tag(item_one)
                                    prev_tokens_one.append(previous_token_one)
                                    pos_prev_one.append(item_prev_one)

                                    prev_index_two = index - 2
                                    previous_token_two = line[prev_index_two]
                                    item_two = nltk.word_tokenize(previous_token_two)
                                    item_prev_two = pos_tag(item_two)
                                    pos_prev_two.append(item_prev_two)
                                    prev_tokens_two.append(previous_token_two)

                                    prev_index_three = index - 3
                                    previous_token_three = line[prev_index_three]
                                    item_three = nltk.word_tokenize(previous_token_three)
                                    item_prev_three = pos_tag(item_three)
                                    pos_prev_three.append(item_prev_three)
                                    prev_tokens_three.append(previous_token_three)

                                    prev_index_four = index - 4
                                    previous_token_four = line[prev_index_four]
                                    item_four = nltk.word_tokenize(previous_token_four)
                                    item_prev_four = pos_tag(item_four)
                                    pos_prev_four.append(item_prev_four)
                                    prev_tokens_four.append(previous_token_four)

                                    next_index_one = index + 1
                                    next_token_one = line[next_index_one]
                                    item_n_one = nltk.word_tokenize(next_token_one)
                                    item_next_one = pos_tag(item_n_one)
                                    pos_next_one.append(item_next_one)
                                    next_tokens_one.append(next_token_one)

                                    next_index_two = index + 2
                                    next_token_two = line[next_index_two]
                                    item_n_two = nltk.word_tokenize(next_token_two)
                                    item_next_two = pos_tag(item_n_two)
                                    pos_next_two.append(item_next_two)
                                    next_tokens_two.append(next_token_two)

                                    next_index_three = index + 3
                                    next_token_three = line[next_index_three]
                                    item_n_three = nltk.word_tokenize(next_token_three)
                                    item_next_three = pos_tag(item_n_three)
                                    pos_next_three.append(item_next_three)
                                    next_tokens_three.append(next_token_three)

                                    neg_col = []
                                    neg_prev_one = []
                                    neg_prev_two = []
                                    neg_prev_three = []
                                    neg_prev_four = []
                                    neg_next_one = []
                                    neg_next_two = []
                                    neg_next_three = []
                                    neg_list_ambi = []
                                    neg_list = []
                                    
                                    #list of negation cues minus the one-occurence ones

                                    neg_cues = ['staken', 'stop', 'stoppen', 'verdwijnen', 'vervangen', 'verwijderen', 'weigeren', 'geen', 'niet aanwezig', 'nooit', 'zonder', 'negatief', 'afzien', 'verdwenen', 'ontbrak', 'nee', 'niet', 'staken', 'stoppen', 'uitsluiten', 'uitgesloten,' 'gestopt']
                                    neg_cues_ambi = ['niet', 'stoppen', 'uitsluiten', 'uitgesloten,','gestopt']

                                      #create column: 1 for negated, 0 for not negated
                                    for neg in negevents:
                                         neg = neg[0]

                                    if neg == 'Negated':
                                        neg_col.append('1')
                                    else:
                                        neg_col.append('0')
                                    
                                    #if there is 1 negcue in the sentence, insert '1', more than one '>1', otherwise '0
                                    if any(neg in line for neg in neg_cues):
                                        intersect_list = list(set(neg_cues) & set(line))
                                        str_negcue = ','.join(intersect_list)
                                        if len(intersect_list) > 1:
                                            neg_list.append('>1')
                                        else:
                                            neg_list.append('1')
                                    else:
                                        str_negcue = 'None'
                                        neg_list.append('None')

                                    #if the neg cue in the sentence is ambiguous insert '1', otherwise 'none'
                                    if any(neg_sent_ambi in line for neg_sent_ambi in neg_cues_ambi):
                                        intersect_sent_ambi = list(set(neg_cues_ambi) & set(line))
                                        if len(intersect_sent_ambi) > 1:
                                            neg_list_ambi.append('>1')
                                        else:
                                            neg_list_ambi.append('1')
                                    else:
                                        neg_list_ambi.append('None')

                                    #if there is neg cue in one of the previous/next tokens, show in new column
                                    for item in prev_tokens_one:
                                        if item in neg_cues:
                                            neg_prev_one.append(item)
                                        else:
                                            neg_prev_one.append("None")
                                            
                                    for item in prev_tokens_two:
                                        if item in neg_cues:
                                            neg_prev_two.append(item)
                                        else:
                                            neg_prev_two.append("None")

                                    for item in prev_tokens_three:
                                        if item in neg_cues:
                                            neg_prev_three.append(item)
                                        else:
                                            neg_prev_three.append("None")

                                    for item in prev_tokens_four:
                                        if item in neg_cues:
                                            neg_prev_four.append(item)
                                        else:
                                            neg_prev_four.append("None")

                                    for item in next_tokens_one:
                                        if item in neg_cues:
                                            neg_next_one.append(item)
                                        else:
                                            neg_next_one.append("None")

                                    for item in next_tokens_two:
                                        if item in neg_cues:
                                            neg_next_two.append(item)
                                        else:
                                            neg_next_two.append("None")

                                    for item in next_tokens_three:
                                        if item in neg_cues:
                                            neg_next_three.append(item)
                                        else:
                                            neg_next_three.append("None")

                                    data = {'Event': offsetword,
                                                        'Previous token one': prev_tokens_one[0],
                                                            'Previous token two': prev_tokens_two[0],
                                                            'Previous token three': prev_tokens_three[0],
                                                            'Next token one': next_tokens_one[0],
                                                            'Next token two': next_tokens_two[0],
                                                            'Next token three': next_tokens_three[0],
                                                            'POS previous token one': pos_prev_one[0],
                                                            'POS previous token two': pos_prev_two[0],
                                                            'POS previous token three': pos_prev_three[0],
                                                            'POS next token one': pos_next_one[0],
                                                            'POS next token two': pos_next_two[0],
                                                            'POS next token three': pos_next_three[0],
                                                            'Negcue prev token one': neg_prev_one[0],
                                                            'Negcue prev token two': neg_prev_two[0],
                                                            'Negcue prev token three': neg_prev_three[0],
                                                            'Negcue next token one': neg_next_one[0],
                                                            'Negcue next token two': neg_next_two[0],
                                                            'Negcue next token three': neg_next_three[0],
                                                            'Negcue in sentence': neg_list[0],
                                                            'The negcue in sentence': str_negcue,  
                                                            'Negcue ambiguous': neg_list_ambi[0],
                                                            'Negation': neg}                                                       

                                    lines.append(data)

                                    header = data.keys()

                                    f = csv.writer(open('corpus_EMC.csv', 'w'), delimiter=(','))
                                    f.writerow(header)

                                    for dic in lines:
                                        values_list = list()
                                        for key, value in dic.items():
                                            values_list.append(value)

                                        f.writerow(values_list)

process_ann()
