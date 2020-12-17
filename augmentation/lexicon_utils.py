import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import sentiwordnet as swn


def get_word_statistics(file_name):
    '''
    @param file_name: file_name should have total dataset including [labeled + unlabeled] and [training + test+ valid ] 
    '''
    word_stats = {}
    with open(file_name, 'r') as rf:
        dataset = rf.read().split('\n')
        for d in dataset:
            if d.strip() == '':
                continue
            label, text = d.split('\t')
            tokens_list = nltk.word_tokenize(text)
            for token in tokens_list:
                if token in word_stats:
                    word_stats[token] += 1
                else:
                    word_stats[token] =1    
    return word_stats


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


def get_lexicon(fname):
    with open(fname, 'r') as rf:
        lexicons = rf.read().split('\n')
    return set(lexicons)


def get_senti_lexicon():
    # opinion_lexicon
    from nltk.corpus import opinion_lexicon
    opinion_pos = opinion_lexicon.positive()
    opinion_neg = opinion_lexicon.negative()
    
    # vader_lexicon 
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sentiment_analyzer = SentimentIntensityAnalyzer()
    vader_lexicon = sentiment_analyzer.lexicon
    vader_pos = set()
    vader_neg =  set()
    for d in vader_lexicon:
        if vader_lexicon[d] >= 0.5: # threshold 조정 필요 ?
            vader_pos.add(d)
        elif  vader_lexicon[d] <= -0.5:
            vader_neg.add(d)
    
    # finance lexcion
    finance_pos = get_lexicon('../lexicons/finance_pos.txt')
    finance_neg = get_lexicon('../lexicons/finance_neg.txt')
        
    # hu-liu lexicon
    hu_liu_pos = get_lexicon('../lexicons/hu_liu_pos.txt')
    hu_liu_neg = get_lexicon('../lexicons/hu_liu_neg.txt')
    
    # harvard lexicon
    harvard_neg = get_lexicon('../lexicons/harvard_neg.txt')
    
    pos_lexicon = set(opinion_pos) | vader_pos | finance_pos | hu_liu_pos
    neg_lexicon = set(opinion_neg) | vader_neg | finance_neg | hu_liu_neg | harvard_neg
    senti_lexicon = pos_lexicon | neg_lexicon
    
    return pos_lexicon, neg_lexicon, senti_lexicon
