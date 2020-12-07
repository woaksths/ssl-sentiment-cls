from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
import nltk
from augmentation.lexicon_utils import get_pair_dataset, get_word_statistics, get_senti_lexicon
from augmentation.lexicon_config import NEGATOR, CONTRAST, END_WORDS, STOP_WORDS, WORD_STATS, POS_LEXICONS, NEG_LEXICONS, SENTI_LEXICONS 
from nltk.stem import WordNetLemmatizer
import random
import argparse


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
    return [wn.ADJ, wn.ADJ_SAT, wn.NOUN, wn.ADV, wn.VERB]


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
        result = random.choice(list(POS_LEXICONS))
    else:
        # sample from intersection btw positive lexicons and training dataset 
        result = random.choice(list(NEG_LEXICONS))
    return result


def gen_reverse_sent(data):
    origin_label = data[0]
    origin_text = data[1].lower()
    
    reversed_sent = []
    reversed_label = 1 - int(origin_label)
    token_pos_list, lemmatized_tokens = tokenize_and_tag(origin_text)
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
    new_words = sentence.split(' ')
    return new_words



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
    
    
if __name__ == '__main__':    
    # load dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', action='store', dest='file_name', help='Path to file_name ') 
    parser.add_argument('--type', action='store', dest='type', help='augment type')
    opt = parser.parse_args()
   
    file_name = opt.file_name
    new_file_name = '/'.join(file_name.split('/')[:-1]) + '/augmented_{}_'.format(opt.type)+file_name.split('/')[-1]
    dataset = get_pair_dataset(file_name)
    alpha_sr = [0.1, 0.2, 0.3]
    
    with open(new_file_name,'w') as fw:
        for idx, data in enumerate(dataset):
            origin_label, origin_text = data[0], data[1]
            origin_text, lemmatized_tokens = tokenize_and_tag(origin_text.lower())
            origin_text = ' '.join([val[0] for val in origin_text])
            
            # reverse dataset
            reverse_label, reverse_text = gen_reverse_sent(data)
            
            # similar dataset
            # similar_label, similar_text = gen_similiar_sent(data)
            alpha = random.choice(alpha_sr)
            n_sr = max(1, int(alpha* len(origin_text.split(' '))))
            similar_text = synonym_replacement(origin_text.split(' '), n_sr)
            similar_text = ' '.join(similar_text)
            
            origin_text = origin_text.strip()
            reverse_text = reverse_text.strip()
            similar_text = similar_text.strip()
            
            origin = str(origin_label) + '\t' + origin_text
            similar = str(origin_label) + '\t' + similar_text
            reverse = str(reverse_label) + '\t' +  reverse_text
            
            if opt.type == 'sym':
                if origin_text == similar_text:
                    fw.write(origin+'\n')
                else:
                    fw.write(origin +'\n')
                    fw.write(similar+'\n')
            elif opt.type == 'ant':
                if origin_text == reverse_text:
                    fw.write(origin +'\n')
                else:
                    fw.write(origin +'\n')
                    fw.write(reverse +'\n')
            elif opt.type == 'all':            
                if reverse_text == origin_text and similar_text != origin_text:
                    fw.write(origin +'\n')
                    fw.write(similar + '\n')
                elif similar_text == origin_text and reverse_text != origin_text:
                    fw.write(origin +'\n')
                    fw.write(reverse +'\n')
                elif reverse_text == similar_text:
                    fw.write(origin +'\n')
                else: 
                    fw.write(origin +'\n')
                    fw.write(similar + '\n')
                    fw.write(reverse +'\n')
            elif opt.type == 'origin':
                fw.write(origin +'\n')