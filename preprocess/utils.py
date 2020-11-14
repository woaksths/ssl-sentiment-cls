import re
import nltk
#cleaning up text
def get_only_chars(line):
    clean_line = ""
    line = line.lower()
    line = line.replace("-", " ") #replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    
    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm.!? ':
            clean_line += char
        else:
            clean_line += ' '
    tokens_list = nltk.word_tokenize(clean_line)
    clean_line = ' '.join(tokens_list)
    clean_line = clean_line.strip()
    
    return clean_line