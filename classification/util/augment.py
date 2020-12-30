from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
import nltk
from nltk.stem import WordNetLemmatizer
import random
import argparse
from .lexicon_config import *
from .lexicon_utils import *


def negator_preprocess(text):
    text = text.replace("ain ' t", "ain't")
    text = text.replace("aren ' t", "aren't")
    text = text.replace("can ' t", "can't")
    text = text.replace("couldn ' t", "couldn't")
    text = text.replace("daren ' t", "daren't")
    text = text.replace("didn ' t", "didn't")
    text = text.replace("doesn ' t", "doesn't")
    text = text.replace("don ' t", "don't")
    text = text.replace("hadn ' t", "hadn't")
    text = text.replace("hasn ' t", "hasn't")
    text = text.replace("haven ' t", "haven't")
    text = text.replace("isn ' t", "isn't")
    text = text.replace("mightn ' t", "mightn't")
    text = text.replace("mustn ' t", "mustn't")
    text = text.replace("needn ' t", "needn't")
    text = text.replace("weren ' t", "weren't")
    text = text.replace("wasn ' t", "wasn't")
    text = text.replace("shouldn ' t", "shouldn't")
    text = text.replace("won ' t", "won't")
    text = text.replace("wouldn ' t", "wouldn't")
    text = text.replace("oughtn ' t", "oughtn't")
    text = text.replace("shan ' t", "shan't")
    text = text.replace("' ll", "'ll")
    text = text.replace("' d", "'d")
    text = text.replace("' v", "'v")
    return text


def get_wn_params(word_pos):
    adj_pos_list = ['JJ' ,'JJS' ,'JJR']
    rb_pos_list = ['RB' , 'RBS' , 'RBR'] 
    verb_pos_list = ['VB' ,'VBZ' , 'VBD' , 'VBN' , 'VBG' , 'VBP']
    if word_pos[1] in adj_pos_list:
        param = ['s', 'a']
    elif word_pos[1] in rb_pos_list:
        param = ['r']
    elif word_pos[1] in verb_pos_list:
        param = ['v']
    return word_pos[0], param


def penn_to_wn(tag):
    if tag.startswith('J') :
        return [wn.ADJ, wn.ADJ_SAT]
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()    
    tag_dict = {"J": wn.ADJ,
                "N": wn.NOUN,
                "V": wn.VERB,
                "R": wn.ADV}
    return tag_dict.get(tag, wn.NOUN)


def tokenize_and_tag(sent):
    sent_temp = []
    negator_temp = []
    
    for token in sent.split(' '):
        if token in NEGATOR:
            sent_temp.append('#')
            negator_temp.append(token)
        else:
            sent_temp.append(token)    
    
    sent_temp = ' '.join(sent_temp)
    tokens = nltk.word_tokenize(sent_temp)
    token_pos_list = nltk.pos_tag(tokens)
    
    if len(negator_temp) > 0:
        for neg_tok in negator_temp:
            for idx, (tok, pos) in enumerate(token_pos_list):
                if tok == '#':
                    token_pos_list[idx] =(neg_tok, 'NEGATOR')
                    break
                    
    lemmatized_tokens = get_lemmatized_sent(sent_temp)

    if len(negator_temp) >0:
        for neg_tok in negator_temp:
            for idx ,tok in enumerate(lemmatized_tokens):
                if tok == '#':
                    lemmatized_tokens[idx] = neg_tok
                    break
    
    assert len(token_pos_list) == len(lemmatized_tokens)
    return token_pos_list, lemmatized_tokens


def get_lemmatized_sent(sent):
    tokens_list = nltk.word_tokenize(sent)
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(sent)]


def get_antonyms(word):
    antonyms_list = []
    for syn in wn.synsets(word):
        for term in syn.lemmas():
            if term.antonyms():
                antonyms_list.append(term.antonyms()[0].name())
        for sim_syn in syn.similar_tos():
            for term in sim_syn.lemmas():
                if term.antonyms():
                    antonyms_list.append(term.antonyms()[0].name())
    return list(antonyms_list)



def get_antonyms_with_tag(word, tag):
    antonyms_list = []
    tag_list = None
    if isinstance(tag, list):
        tag_list = tag
    else:
        tag_list = penn_to_wn(tag)
#     print(word, tag)
    for syn in wn.synsets(word):
        if syn.pos() in tag_list:
            for term in syn.lemmas():
                if term.antonyms():
                    antonyms_list.append(term.antonyms()[0].name())
            for sim_syn in syn.similar_tos():
                for term in sim_syn.lemmas():
                    if term.antonyms():
                        antonyms_list.append(term.antonyms()[0].name())
    return list(antonyms_list)



def augment_antonym_lexicons(golden_lexicons):
    augmented_lexicons = {0:set(), 1:set()}
    for label in golden_lexicons:
        label = int(label)
        for word, tag in golden_lexicons[label]:
            antonyms = get_antonyms_with_tag(word, tag)
            reverse_label = 1 - label
            for antonym in antonyms:
                augmented_lexicons[reverse_label].add(antonym)
            augmented_lexicons[label].add(word)
    return augmented_lexicons
    


def augment_syn_lexicons(golden_lexicons):
    candidate = {}
    for label in golden_lexicons:
        label = int(label)
        if label not in candidate:
            candidate[label] = set()
        for word, tag in golden_lexicons[label]:
            synonyms = get_synonyms(word, tag)
            for synonym in synonyms:
                candidate[label].add(synonym)
    return candidate



def get_synonyms(origin_word, tag):
    synonyms = set()
    tag_list = penn_to_wn(tag)
    for syn in wn.synsets(origin_word):
        if syn.pos() in tag_list:
            for term in syn.lemmas():
                term = term.name().replace("_", " ").replace("-"," ").lower()
                term = "".join([char for char in term if char in ' qwertyuiopasdfghjklzxcvbnm'])
                synonyms.add(term)
    if origin_word in synonyms:
        synonyms.remove(origin_word)
    return list(synonyms)



def get_hypernyms(origin_word, tag):
    hypernyms = []
    tag_list = penn_to_wn(tag)
    for syn in wn.synsets(origin_word):
        for hyp in syn.hypernyms():
            hypernyms.extend([word.replace("_"," ").replace("-"," ") for word in hyp.lemma_names()])
    if origin_word in hypernyms:
        hypernyms.remove(origin_word)
    return list(set(hypernyms))



def get_hyponyms(origin_word, tag):
    hyponyms = []
    tag_list = penn_to_wn(tag)
    for syn in wn.synsets(origin_word):
        for hyp in syn.hyponyms():
            hyponyms.extend(hyp.lemma_names())
    hyponyms = [word.replace("_"," ").replace("-"," ") for word in hyponyms]
    if origin_word in hyponyms:
        hyponyms.remove(origin_word)
    return list(set(hyponyms))



def sample_antonym(token, origin_label):
    '''
    origin label과 반대되는 클래스의 단어집합과 학습 데이터와 교차하는 단어 중 임의로 샘플링
    '''
    origin_label = int(origin_label)
    intersection = None
    result = ''
    if origin_label == 0:
        # sample from intersection btw positive lexicons and traning dataset
        result = random.choice(list(POS_LEXICONS))
    else:
        # sample from intersection btw positive lexicons and training dataset 
        result = random.choice(list(NEG_LEXICONS))
    return result


def gen_reverse_sent_with_lexicons(data, lexicons):
    origin_label = data[0]
    origin_text = data[1]
    origin_text = negator_preprocess(origin_text)
    reversed_label = 1 - origin_label
    reversed_sent = []
    token_pos_list, lemmatized_tokens = tokenize_and_tag(origin_text)
    assert len(token_pos_list) == len(lemmatized_tokens)
    i = 0
    lexicons = lexicons[0] | lexicons[1]
    
    while i != len(token_pos_list):
        token = token_pos_list[i][0]
        tag = penn_to_wn(token_pos_list[i][1])
        lemmatized_token = lemmatized_tokens[i]
        if token in NEGATOR:
            i += 1
            while i != len(token_pos_list) and token_pos_list[i][0] not in END_WORDS and token_pos_list[i][0] not in NEGATOR:
                reversed_sent.append(token_pos_list[i][0])
                i += 1
        elif (wn.morphy(token) in lexicons or token in lexicons) and (token_pos_list[i][1].startswith('J') or token_pos_list[i][1].startswith('N') or token_pos_list[i][1].startswith('R') or token_pos_list[i][1].startswith('V')):
            antonyms = get_antonyms_with_tag(token, token_pos_list[i][1])
            lemmtized_antonyms = get_antonyms_with_tag(lemmatized_token, token_pos_list[i][1])
            morphy_token = wn.morphy(token)
            morphy_antonyms = []
            if morphy_token is not None:
                morphy_antonyms = get_antonyms_with_tag(morphy_token, token_pos_list[i][1])
            candidate_list = list(set(antonyms + lemmtized_antonyms + morphy_antonyms))
            antonym = None
            if len(candidate_list) != 0:
                antonym = random.choice(candidate_list)
            else:
                antonym = "not " + token
            reversed_sent.append(antonym)
            i += 1
        else:
            reversed_sent.append(token)
            i += 1
    reversed_sent = ' '.join(reversed_sent)
    return reversed_label, reversed_sent
    
    
def gen_reverse_sent(data):
    origin_label = data[0]
    origin_text = data[1].lower()
    
    reversed_sent = []
    reversed_label = 1 - int(origin_label)
    token_pos_list, lemmatized_tokens = tokenize_and_tag(origin_text)
    i = 0
    #print('token_pos', token_pos_list)
#     print('lemmatized_tokens',  ' '.join(lemmatized_tokens))
    
    while i != len(token_pos_list):
        token = token_pos_list[i][0].lower()
        tag = penn_to_wn(token_pos_list[i][1])
        lemmatized_token = lemmatized_tokens[i]
        if token in NEGATOR:
            i += 1 
            while i != len(token_pos_list) and token_pos_list[i][0] not in END_WORDS and token_pos_list[i][0] not in NEGATOR:
                reversed_sent.append(token_pos_list[i][0])
                i += 1
        elif token in SENTI_LEXICONS: #SENTI_LEXICON 재정의 필요 
            antonyms = set(get_antonyms(token)) | set(get_antonyms(lemmatized_token))
            candidate = []
            candidate.extend(antonyms)
            antonym = ""
            if len(candidate) == 0:
                antonym = sample_antonym(token, origin_label) # sample random antonym from lexicon 
            else:
                antonym = random.choice(candidate)
            reversed_sent.append(antonym)
            i += 1
        else:
            reversed_sent.append(token)
            i += 1
    reversed_sent = ' '.join(reversed_sent)
    return reversed_label, reversed_sent



def synonym_replacement(words, n): 
    # code from eda paper
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in STOP_WORDS]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word, '')
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            #print("replaced", random_word, "with", synonym)
            num_replaced += 1
        if num_replaced >= n: #only replace up to n words
            break
    #this is stupid but we need it, trust me
    sentence = ' '.join(new_words)
    return sentence



def gen_similiar_sent(data):    
    origin_label = data[0]
    origin_text = data[1].lower()
    token_pos_list, lemmatized_tokens = tokenize_and_tag(origin_text)

    similar_text = []
    augment_action = [0, 1] #0.Not change #1.change  
    
    for idx, (token, tag) in enumerate(token_pos_list):
        lemma_token = lemmatized_tokens[idx]
        if token in STOP_WORDS:
            similar_text.append(token)
            continue
        action = random.choice(augment_action)
        if action != 0: # Not change
            similar_text.append(token)
        else: # Change 
            action = random.random()
            if action > 0.5: # synonym
                candidate = get_synonyms(token, tag) + get_synonyms(lemma_token, tag)
                candidate = list(set(candidate))
                if len(candidate) == 0:
                    similar_text.append(token)
                    continue
                synonym = random.choice(candidate)
                similar_text.append(synonym)
            else: # hypernym or hyponym
                candidate = []
                candidate.extend(get_hypernyms(token, tag))
                candidate.extend(get_hypernyms(lemma_token, tag))
                candidate.extend(get_hyponyms(token, tag))
                candidate.extend(get_hyponyms(lemma_token, tag))
                candidate = list(set(candidate))
                if len(candidate) == 0: 
                    similar_text.append(token)
                    continue
                hypword = random.choice(candidate)
                similar_text.append(hypword)
    similar_text = ' '.join(similar_text)
    return origin_label, similar_text
