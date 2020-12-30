import torchtext
from util.augment import augment_antonym_lexicons, gen_reverse_sent, gen_reverse_sent_with_lexicons
from util.lexicon_utils import get_senti_lexicon

# set up fields
TEXT = torchtext.data.Field(lower=True, include_lengths=True, batch_first=True,
                            use_vocab=True, sequential=True)
LABEL = torchtext.data.Field(sequential=False, use_vocab=False, batch_first=True)


def get_dataset(path):
    # read dataset from file
    with open(path, 'r') as rf:
        dataset = rf.read().split('\n')
        dataset = [data for data in dataset if data.strip()!='']
    return dataset


def examples_from_dataset(dataset, max_len):
    # make exampeles that lenght of text is less than max_len 300
    examples =[]
    for data in dataset:
        if data.strip() =='':
            continue
        label, text = data.split('\t')
        example = torchtext.data.Example.fromlist((text, label),[('text', TEXT),('label', LABEL)])
        example.text = example.text[:max_len]
        examples.append(example)
    return examples


def glove_to_example(glove_words):
    # word_list
    text = ' '.join(glove_words)
    label = 1
    example = torchtext.data.Example.fromlist((text, label),[('text', TEXT),('label', LABEL)])
    return example



def sampling_initial_dataset(total_dataset, class_num, num_per_class):
    train_dict = {cls_idx:0 for cls_idx in range(class_num)}
    valid_dict = {cls_idx:0 for cls_idx in range(class_num)}
    train_set = []
    valid_set = []
    # sample balanced train
    while True:
        if len(train_set) == int(class_num*num_per_class):
            break
        data = total_dataset.pop(0)
        if data.strip() =='':
            continue
        label, text = data.split('\t')
        label = int(label)
        if train_dict[label] < num_per_class:
            train_set.append(data)
            train_dict[label] +=1
        elif train_dict[label] >= num_per_class:
            total_dataset.append(data)
    # sample balanced dev   --> train 뽑는 코드와 동일 ... 함수로 뺴기 나중에
    while True:
        if len(valid_set) == int(class_num*num_per_class):
            break
        data = total_dataset.pop(0)
        if data.strip() =='':
            continue
        label, text = data.split('\t')
        label = int(label)
        if valid_dict[label] < num_per_class:
            valid_set.append(data)
            valid_dict[label] +=1
        elif valid_dict[label] >= num_per_class:
            total_dataset.append(data)
    return train_set, valid_set, total_dataset
    
    
def write_sampled_dataset(path, dataset):
    with open(path, 'w') as fw:
        for d in dataset:
            fw.write(d +'\n')
            
            
def get_golden_lexicons(path):
    lexicons = dict()
    with open(path, 'r') as rf:
        dataset = rf.read().split('\n')
        for d in dataset:
            label, word_set = d.split('\t')
            label = int(label)
            word_set = word_set.split(' ')
            word_set = [word for word in word_set if word.strip() !='']
            word_set = set(word_set)
            if label in lexicons:
                for word in word_set:
                    lexicons[label].add(word)
            else:
                lexicons[label] = set()
                for word in word_set:
                    lexicons[label].add(word)
    return lexicons


def get_annotated_word_tag(path, class_num=None):
    word_tag = {label: list() for label in range(class_num)}
    with open(path, 'r') as rf:
        dataset = rf.read().split('\n')
        for d in dataset:
            label = d.split('\t')[0]
            label = int(label)
            word_set = d.split('\t')[1:]
            for data in word_set:
                if data.strip() =='':
                    continue
                word, tag = data.replace("'","").replace("(","").replace(")","").split(',')
                word = word.strip()
                tag = tag.strip()
                word_tag[label].append((word,tag))
    return word_tag


def sample_unbalanced_dataset(dataset, ratio_0, ratio_1):
    print('sample_unbalanced_dataset', len(dataset))
    class_per_num = len(dataset)*0.5
    num_0 = int(class_per_num*ratio_0)
    num_1 = int(class_per_num*ratio_1)
    cnt_0 = 0
    cnt_1 = 0 
    unbalanced_dataset = []
    print('class_per_num', class_per_num)
    print('label_0 >> {}, label_1 >> {}'.format(num_0, num_1))
    for data in dataset:
        label, text = data.split('\t')
        label = int(label)
        if label == 0:
            if cnt_0 < num_0:
                unbalanced_dataset.append(data)
                cnt_0 += 1
        elif label == 1:
            if cnt_1 < num_1:
                unbalanced_dataset.append(data)
                cnt_1 += 1
    return unbalanced_dataset


def augment_reverse(dataset):
    reverse_dataset = []
    for data in dataset:
        origin_label, origin_text = data.split('\t')
        origin_label = int(origin_label)
        
        reverse_label, reverse_text = gen_reverse_sent_with_lexicons((origin_label, origin_text), get_senti_lexicon())
        
        print('origin_label', origin_label)
        print('origin_Text', origin_text)
        print('reverse_label', reverse_label, type(reverse_label))
        print('reverse_Text', reverse_text, type(reverse_text))
        print('\n')
        reverse_data = str(reverse_label) +'\t' +reverse_text
        reverse_dataset.append(reverse_data)
    return reverse_dataset