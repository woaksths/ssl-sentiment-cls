#!/usr/bin/env python
# coding: utf-8

# 1. sentiword 어떻게 찾냐? -> 1. opinion_lexicion(positive, negative)  2. vader_lexicon 3. sentiwordNet 
    # opinion_lexicon + vader_lexicon 
    # sentiwordNet은 단순 wordNet에 pos, neg, obj score를 나타내서. 향후 어떻게 활용할지 고민.
    
# 2. reversed doc 구하는 logic (pos tag 사용)
    # 반대의 레이블을 만들어내는 상황 규칙 정하기 
    # reversed doc 구할 때, origin doc과 완전 일치한다면 ? 
    
# 3. perturbed samples에 대해서는 hypernym, synonym, hyponym, neighbor_word
    # 같은 레이블 perturbed sample 할때는 모든 tokens에 대해서 위 처리를 해도 무방
    
# 4. perturbed samples에 대해서만 meta learning을 통해 generalization for unseen data

# 5. binary classification 방법 이외의 다중 분류 문제에 접근하기 위한 방법으로 attention을 통해 사전을 구축하고, 
    # 사전에 매칭되는 단어들을 다른 클래스의 단어로 바꾸고, 레이블도 해당 클래스의 레이블로 바꾸는 방법 관련     


from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
import nltk

NEGATOR = {"aint", "arent","cannot","cant","couldnt","darent", "didnt","doesnt","ain't","aren't",
        "can't","couldn't","daren't","didn't", "doesn't","dont","hadnt", "hasnt","havent","isnt",
        "mightnt","mustnt","neither","don't","hadn't","hasn't","haven't","isn't",
        "mightn't","mustn't","neednt","needn't","never","none","nope","nor",
        "not","nothing","nowhere","oughtnt","shant","shouldnt","uhuh","wasnt",
        "werent","oughtn't","shan't","shouldn't","uh-uh", "wasn't","weren't","without",
        "wont","wouldnt","won't","wouldn't","rarely","seldom","despite","no"}
CONTRAST = ['but', 'however', 'But', 'However']
END_WORDS = ['.', ',', '!', '?', '...']


STOP_WORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 
			'ours', 'ourselves', 'you', 'your', 'yours', 
			'yourself', 'yourselves', 'he', 'him', 'his', 
			'himself', 'she', 'her', 'hers', 'herself', 
			'it', 'its', 'itself', 'they', 'them', 'their', 
			'theirs', 'themselves', 'what', 'which', 'who', 
			'whom', 'this', 'that', 'these', 'those', 'am', 
			'is', 'are', 'was', 'were', 'be', 'been', 'being', 
			'have', 'has', 'had', 'having', 'do', 'does', 'did',
			'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
			'because', 'as', 'until', 'while', 'of', 'at', 
			'by', 'for', 'with', 'about', 'against', 'between',
			'into', 'through', 'during', 'before', 'after', 
			'above', 'below', 'to', 'from', 'up', 'down', 'in',
			'out', 'on', 'off', 'over', 'under', 'again', 
			'further', 'then', 'once', 'here', 'there', 'when', 
			'where', 'why', 'how', 'all', 'any', 'both', 'each', 
			'few', 'more', 'most', 'other', 'some', 'such', 'no', 
			'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
			'very', 's', 't', 'can', 'will', 'just', 'don', 
			'should', 'now', '']

def tokenize_and_tag(sent):
    token_list = nltk.word_tokenize(sent)
    term_pos_list = nltk.pos_tag(token_list)
    return term_pos_list


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


def get_synonyms(word):
    synonyms = set()
    for syn in wn.synsets(word):
        for term in syn.lemmas():
            term = term.name().replace("_", " ").replace("-"," ").lower()
            term = "".join([char for char in term if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(term)
    if word in synonyms:
        synonyms.remove(word)
    if len(synonyms) == 0:
        synonyms = [word]
    return list(synonyms)


def get_hypernyms(word, tag):
    hypernyms = []    
    for syn in wn.synsets(word):
        for hyp in syn.hypernyms():
            hypernyms.extend([word.replace("_"," ").replace("-"," ") for word in hyp.lemma_names()])
 
    hypernyms = [word.replace("_"," ").replace("-"," ") for word in hypernyms]
    if len(hypernyms) == 0:
        hypernyms = [word]
    return list(set(hypernyms))
    

def get_hyponyms(word, tag):
    hyponyms = []
    for syn in wn.synsets(word):
        if syn.pos() != 'n':
            continue
        for word in syn.hyponyms():
            hyponyms.extend(word.lemma_names())
    
    hyponyms = [word.replace("_"," ").replace("-"," ") for word in hyponyms]
    if len(hyponyms) == 0:
        hyponyms = [word]
    return hyponyms


def get_pair_dataset(path):
    with open(path,'r') as rf:
        dataset = rf.read().split('\n')
        pair_d = []
        for d in dataset:
            if d.strip() == '':
                continue
            label, data = d.split('\t')
            pair_d.append((label,data))
    return pair_d


def gen_reverse_sent(data, senti_lexicon):
    origin_label = data[0]
    origin_text = data[1]
    tokens_list = nltk.word_tokenize(origin_text)
    pos_token_list = nltk.pos_tag(tokens_list)
    reversed_sent = []
    reversed_label = 1 - int(origin_label)
    i = 0
    
    # 치환단어 tagging 형용사, 부사 
    while i != len(pos_token_list):        
        token = pos_token_list[i][0].lower()
        tag = penn_to_wn(pos_token_list[i][1])
        if token in NEGATOR:
            i += 1 
            while i != len(pos_token_list) and pos_token_list[i][0] not in END_WORDS and pos_token_list[i][0] not in NEGATOR:
                reversed_sent.append(pos_token_list[i][0])
                i += 1
        elif token in senti_lexicon:
            antonym_word = get_antonyms(token)
            if len(antonym_word) == 0:
                antonym_word =  token
            else:
                antonym_word = antonym_word[0]
            reversed_sent.append(antonym_word)
            i += 1
        else:
            reversed_sent.append(token)
            i += 1
    reversed_sent = ' '.join(reversed_sent)
    return reversed_label, reversed_sent


def gen_similiar_sent(data):    
    import random 
    origin_label = data[0]
    origin_text = data[1]
    token_list = nltk.word_tokenize(origin_text)
    per_token_list = nltk.pos_tag(token_list)
    
    transformed_text = []
    transformed_label = origin_label
    
    for token, tag in per_token_list:
        tag = penn_to_wn(tag)
        if token in STOP_WORDS:
            transformed_text.append(token)
            continue

        rand_val = random.random()
        if rand_val <= 0.5:
            tokens = get_synonyms(token)
        else:
            tokens = get_synonyms(token)
        
        token = random.choice(tokens)
        transformed_text.append(token)
            
    transformed_text = ' '.join(transformed_text)
    return transformed_label, transformed_text
    

def get_senti_lexicon():
    # sentiword_lexicon
    sentiword_lexicon = swn.all_senti_synsets()

    # opinion_lexicon
    from nltk.corpus import opinion_lexicon
    opinion_lexicon = opinion_lexicon.words()

    # vader_lexicon
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sentiment_analyzer = SentimentIntensityAnalyzer()
    vader_lexicon = sentiment_analyzer.lexicon
    vader_lexicon = list(vader_lexicon.keys())

    # Merge lexicon
    senti_lexicon = set(opinion_lexicon) | set(vader_lexicon)
    return senti_lexicon


# load dataset
origin_dataset = 'sst2_train_500.txt'  
dataset = get_pair_dataset(origin_dataset)

with open('augmented_' + origin_dataset,'w') as fw:
    for idx, data in enumerate(dataset):
        origin_label, origin_text = data[0], data[1]
        reverse_label, reverse_text = gen_reverse_sent(data, get_senti_lexicon())
        similar_label, similar_text = gen_similiar_sent(data)
        
        origin = origin_text + '\t ' + str(origin_label) +'\n'
        similar = similar_text + '\t ' + str(similar_label) +'\n'
        reverse = reverse_text + '\t ' +  str(reverse_label) +'\n'
        
        if origin_text == reverse_text:
            continue
        
        fw.write(origin)
        fw.write(similar)
        fw.write(reverse)