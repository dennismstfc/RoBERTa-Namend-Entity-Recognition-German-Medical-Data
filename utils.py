import pickle
import os
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords

DATA_PATH = "./data/given_data/"
SAVE_PATH = "./data/generated_data/"

'''
Simple function that reads .txt-files of a given path.
'''
def read_txt(fname):
    if not os.path.exists(fname):
        print(f"{fname} doesn't exist or can't be found. Try with another filename")
    else:
        with open(fname, "r", encoding="utf-8") as f:
            arr = f.readlines()
        return arr

'''
Simple function that reads .txt-files of a given path.
'''
def read_txt_full(fname):
    if not os.path.exists(fname):
        print(f"{fname} doesn't exist or can't be found. Try with another filename")
    else:
        with open(fname, "r", encoding="utf-8") as f:
            arr = f.read()
        return arr


'''
Simple function that writes to a .txt-file with a given array and filename.
'''
def write_arr_txt(fname, arr):
    with open(fname, "w") as f:
            f.write("\n".join(arr) + "\n")

'''
Simple function that reads a .pickle-file. In pickle files you can save any
kind of python datatypes such as e.g. lists, dictionaries, etc.
'''
def read_pickle(fname):
    if not os.path.exists(fname):
        print(f"{fname} doesn't exist or can't be found. Try with another filename")
    else:
        with open(fname, "rb") as f:
            arr = pickle.load(f)
        return arr


'''
Simple function that stores any kind of python datatype into a .pickle-file. This
type of file can be loaded later on again.
'''
def save_pickle(fname, arr):
    with open(fname, "wb") as f:
        pickle.dump(arr, f)


'''
Deletes a file of a given filename.
'''
def delete_file(fname):
    if not os.path.exists(fname):
        print(f"{fname} doesn't exist or can't be found. Try with another filename")
    else:
        os.remove(fname)

'''
Returns the content of the .ann-file.
'''
def get_ann(fname):
    # reading ann file
    content = read_txt(fname)
    
    ann_data = []
    for el in content:
        # splitting every word in the element
        tmp = el.split("\t")
        tmp = [el.split(" ") for el in tmp]
        
        # flatten list
        tmp = [el for sublist in tmp for el in sublist]
        
        # remove \n
        tmp = [el.strip() for el in tmp]
        ann_data.append(tmp)

    return ann_data


stemmer = PorterStemmer()
'''
This functions takes a sentence as input and returns the stemmed version.
'''
def stem_sentence(sentence):
    words = []
    for word in sentence.split(" "):
        words.append(stemmer.stem(word, to_lowercase=True))
    return " ".join(words)


'''
This functions takes a word as input and returns the stemmed version.
'''
def stem(word):
    return stemmer.stem(word, to_lowercase=True)


'''
Simple function that returns true if a given word is a stop word.
'''
def should_get_skipped(act_word):
    try:
        stop_words = stopwords.words("german")
    except:
        nltk.download('stopwords')
        stop_words = stopwords.words("german")

    if act_word in stop_words:
        return True
    
    return False

'''
This function returns the filenames without the ending, so that the .txt and .ann 
files can be read later on.
'''
def get_filenames_without_ending(file_path):
    subfolders = os.listdir(file_path)
    subfolders = [file_path  + el + "/" for el in subfolders]
    fnames = []

    for act_subfolder in subfolders:
        tmp_files = os.listdir(act_subfolder)
        tmp_files = [act_subfolder + el for el in tmp_files]

        for act_file in tmp_files:
            if act_file.endswith(".txt"):
              fnames.append(act_file[:-4])
        
    return fnames