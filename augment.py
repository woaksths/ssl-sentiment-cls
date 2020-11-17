# 1. sentiword 구성 방법 -> 1. opinion_lexicion(positive, negative)  2. vader_lexicon 3. sentiwordNet 
    # opinion_lexicon + vader_lexicon  
    # well defined 된 lexicon 추가 관련 고민 
    # lexicon 구성 시에 pos, neg lexicon 별도로 합치는 것을 원칙으로

# 2. reversed doc 구하는 logic (pos tag 사용)
    # Issue1. reversed doc 구할 때, origin doc과 완전 일치하는 문제 
    # Issue2. 학습 데이터 내에 위 단어에 대응하는 antynom을 구하지 못했다면,학습 데이터 내 존재하는 단어중에 임의의 부정의 단어로 random replacement 작업을 진행

# 3. WordNet 치환 operation 1. unchaged 2. synonyms 3. hyponyms 4. hypernyms 5. antonyms(if 문장에 sentiword 존재)        

# 4. perturbed samples-meta learning

# 5. binary classification 방법 이외의 다중 분류 문제에 접근 (If possible) 
    # attention을 통해 사전을 구축하고, 
    # 사전에 매칭되는 단어들을 다른 클래스의 단어로 바꾸고, 레이블도 해당 클래스의 레이블로 바꾸는 방법 관련     

from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
import nltk
from augmentation.lexicon_utils import get_pair_dataset, get_word_statistics, get_senti_lexicon
from augmentation.lexicon_config import NEGATOR, CONTRAST, END_WORDS, STOP_WORDS, WORD_STATS, POS_LEXICONS, NEG_LEXICONS, SENTI_LEXICONS 
from nltk.stem import WordNetLemmatizer
import random


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


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()    
    tag_dict = {"J": wn.ADJ,
                "N": wn.NOUN,
                "V": wn.VERB,
                "R": wn.ADV}
    return tag_dict.get(tag, wn.NOUN)


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


def get_synonyms(origin_word, tag):
    synonyms = set()
    tag_list = penn_to_wn(tag)
    for syn in wn.synsets(origin_word):
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
        intersection = POS_LEXICONS & set(WORD_STATS)
        intersection = {word: WORD_STATS[word] for word in intersection}
        intersection = sorted(intersection.items(), key=lambda x: x[1],reverse=True)
        intersection = list(intersection)[0:50]
        result = random.choice(intersection)[0]
    else:
        # sample from intersection btw positive lexicons and training dataset 
        intersection = NEG_LEXICONS & set(WORD_STATS)
        intersection = {word: WORD_STATS[word] for word in intersection}
        intersection = sorted(intersection.items(), key=lambda x: x[1],reverse=True)
        intersection = list(intersection)[0:50]
        result = random.choice(intersection)[0]
    return result


def gen_reverse_sent(data):
    origin_label = data[0]
    origin_text = data[1]
    reversed_sent = []
    reversed_label = 1 - int(origin_label)
    token_pos_list = tokenize_and_tag(origin_text)
    lemmatized_tokens = get_lemmatized_sent(origin_text)
    i = 0
    while i != len(token_pos_list):
        token = token_pos_list[i][0].lower()
        tag = penn_to_wn(token_pos_list[i][1])
        lemmatized_token = lemmatized_tokens[i]
        if token in NEGATOR:
            i += 1 
            while i != len(token_pos_list) and token_pos_list[i][0] not in END_WORDS and token_pos_list[i][0] not in NEGATOR:
                reversed_sent.append(token_pos_list[i][0])
                i += 1
        elif token in SENTI_LEXICONS: 
            antonyms = set(get_antonyms(token)) | set(get_antonyms(lemmatized_token))
            candidate = []
            candidate.extend(antonyms)
            valid_candidate = {word:WORD_STATS[word] for word in candidate if word in WORD_STATS}
            antonym = ""
            if len(valid_candidate) == 0:
                antonym = sample_antonym(token, origin_label) # sample random antonym from lexicon 
            else:
                valid_candidate = sorted(valid_candidate.items(), key=lambda x: x[1],reverse=True)[0:5]
                antonym = random.choice(valid_candidate)[0]
            reversed_sent.append(antonym)
            i += 1
        else:
            reversed_sent.append(token)
            i += 1
    reversed_sent = ' '.join(reversed_sent)
    return reversed_label, reversed_sent


def gen_similiar_sent(data):    
    origin_label = data[0]
    origin_text = data[1]
    token_list = nltk.word_tokenize(origin_text)
    token_pos_list = nltk.pos_tag(token_list)
    lemmatized_tokens = get_lemmatized_sent(origin_text)

    similar_text = []
    augment_action = [0, 1] #0. Not change #1.change  
    
    for idx, (token, tag) in enumerate(token_pos_list):
        lemma_token = lemmatized_tokens[idx]
        if token in STOP_WORDS:
            continue
        action = random.choice(augment_action)
        if action == 0: # Noy change
            similar_text.append(token)
        else: # Change 
            action = random.random()
            if action > 0.5: # synonym
                candidate = get_synonyms(token, tag) + get_synonyms(lemma_token, tag)
                valid_candidate = set(candidate) & set(WORD_STATS)
                valid_candidate = {word:WORD_STATS[word] for word in valid_candidate}
                valid_candidate = sorted(valid_candidate.items(), key=lambda x: x[1],reverse=True)[0:10]
                if len(valid_candidate) == 0:
                    similar_text.append(token)
                    continue
                synonym = random.choice(valid_candidate)[0]
                similar_text.append(synonym)
            else: # hypernym or hyponym
                candidate = []
                candidate.extend(get_hypernyms(token, tag))
                candidate.extend(get_hypernyms(lemma_token, tag))
                candidate.extend(get_hyponyms(token, tag))
                candidate.extend(get_hyponyms(lemma_token, tag))
                valid_candidate = set(candidate) & set(WORD_STATS)
                valid_candidate = {word:WORD_STATS[word] for word in valid_candidate}
                valid_candidate = sorted(valid_candidate.items(), key=lambda x: x[1], reverse=True)[0:10]
                if len(valid_candidate) == 0: 
                    similar_text.append(token)
                    continue
                hypword = random.choice(valid_candidate)[0]
                similar_text.append(hypword)
    similar_text = ' '.join(similar_text)
    return origin_label, similar_text
    
    
if __name__ == '__main__':
    # load dataset
    origin_dataset = 'temp_dataset.txt'  
    dataset = get_pair_dataset(origin_dataset)

    with open('augmented_' + origin_dataset,'w') as fw:
        for idx, data in enumerate(dataset):
            origin_label, origin_text = data[0], data[1]
            reverse_label, reverse_text = gen_reverse_sent(data)
            similar_label, similar_text = gen_similiar_sent(data)
            
            print('origin_text:', origin_text)            
            print('similar_text:', similar_text)
            print('reverse_text:', reverse_text)
            print()
            origin = origin_text + '\t ' + str(origin_label) +'\n'
            similar = similar_text + '\t ' + str(similar_label) +'\n'
            reverse = reverse_text + '\t ' +  str(reverse_label) +'\n'
            