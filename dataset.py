import numpy as np 

def build_file(word_to_id,l,vocab_size=0,one_hot=True):
    x=list()
    y=list()
    x.append(word_to_id["start_token"])
    for i in l:
        x.append(word_to_id[i])
        y.append(word_to_id[i])
    y.append(word_to_id["end_token"])
    if one_hot==True:
        if(vocab_size==0):
            vocab_size = len(set(l))
        y = one_hot_func(y,vocab_size)
    return x,y

def one_hot_func(l,vocab_size):
    rows = len(l)
    col = vocab_size
    arr = np.atleast_2d(np.zeros((rows,col)))
    count=0
    for i in l:
        arr[count,i] = 1
        count +=1
    return arr

def read_file(fname,remove_words=['\n']):
    with open(fname) as f:
        l = f.read().split(' ')
        for i in l:
            if(i in remove_words):
                l.remove(i)
    return l                

def tokenize_file(l):    
        s = set(l)
        vocab_size = len(s)+2
        word_to_id = dict()
        id_to_word = dict()
        for count,word in enumerate(s):
            word_to_id[word] = count+1
            id_to_word[count+1] = word
        word_to_id["start_token"] = 0
        word_to_id["end_token"] = count+2
        id_to_word[0] = "start_token"
        id_to_word[count+2] = "end_token"
        return word_to_id,id_to_word,vocab_size,l