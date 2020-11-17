from .lexicon_utils import get_senti_lexicon, get_word_statistics

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

WORD_STATS = get_word_statistics('dataset/pros_cons/total_dataset.txt')
POS_LEXICONS, NEG_LEXICONS, SENTI_LEXICONS = get_senti_lexicon()

