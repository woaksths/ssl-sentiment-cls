from utils import *

def get_good_stuff(line):
    idx = line.find('s>')
    good = line[idx+2:-8]
    return get_only_chars(good)


def clean_file(con_file, pro_file, output_train, output_valid, output_test):
    train_writer = open(output_train, 'w')
    valid_writer = open(output_valid, 'w')
    test_writer = open(output_test, 'w')
    con_lines = open(con_file, 'r').readlines()
    
    valid_ratio = int(len(con_lines) * 0.1)
    test_ratio = int(len(con_lines) * 0.1)
    train_ratio = int(len(con_lines) * 0.8)
    
    # write cons test
    for line in con_lines[:test_ratio]:
        content = get_good_stuff(line)
        if len(content) >= 8:
            test_writer.write('0\t' + content + '\n')
    # write cons valid
    for line in con_lines[test_ratio:test_ratio+valid_ratio]:
        content = get_good_stuff(line)
        if len(content) >= 8:
            valid_writer.write('0\t' + content + '\n')
    # write cons train
    for line in con_lines[test_ratio + valid_ratio : ]:
        content = get_good_stuff(line)
        if len(content) >= 8:
            train_writer.write('0\t' + content + '\n')
    
    pro_lines = open(pro_file, 'r').readlines()
    # write pros test
    for line in pro_lines[:test_ratio]:
        content = get_good_stuff(line)
        if len(content) >= 8:
            test_writer.write('1\t' + content + '\n')
    # write pros vlaid
    for line in pro_lines[test_ratio:test_ratio+valid_ratio]:
        content = get_good_stuff(line)
        if len(content) >= 8:
            valid_writer.write('1\t' + content + '\n')
    #write pros train
    for line in pro_lines[test_ratio+valid_ratio:]:
        content = get_good_stuff(line)
        if len(content) >= 8:
            train_writer.write('1\t' + content + '\n')

            
if __name__ == '__main__':
    con_file = '../dataset/pros_cons/IntegratedCons.txt'
    pro_file = '../dataset/pros_cons/IntegratedPros.txt'
    output_train = '../dataset/pros_cons/train.txt'
    output_dev = '../dataset/pros_cons/dev.txt'
    output_test = '../dataset/pros_cons/test.txt'
    clean_file(con_file, pro_file, output_train, output_dev, output_test)
    