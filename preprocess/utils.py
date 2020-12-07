import re
import nltk

#cleaning up text
def get_only_chars(line):
    clean_line = ""
    line = line.lower()
    line = line.replace(" 's", " is")
    line = line.replace("-", " ") #replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    
    for char in line:
        if char in "qwertyuiopasdfghjklzxcvbnm'., ": # ., 넣어야하나...
            clean_line += char
        else:
            clean_line += ' '
    clean_line = re.sub(' +', ' ', clean_line)
    tokens_list = clean_line.split(' ')
    clean_line = ' '.join(tokens_list)
    clean_line = clean_line.strip()
    return clean_line