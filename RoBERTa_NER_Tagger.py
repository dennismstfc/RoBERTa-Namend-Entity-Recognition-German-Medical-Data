# imports from pip packages
import os
import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from simpletransformers.ner import NERModel, NERArgs

# selfmade functions import
from utils import get_filenames_without_ending
from utils import read_txt
from utils import get_ann
from utils import stem
from utils import should_get_skipped
from utils import save_pickle, read_pickle
from utils import DATA_PATH, SAVE_PATH

def pipeline(model, checkpoint, epochs, random_state=111):
    print("##############################################################")
    print(f"loading pipeline {model} with checkpoint {checkpoint} with {epochs} epochs...")
    print("checking if data was saved before...")

    if os.path.exists(SAVE_PATH + "X.pickle") and os.path.exists(SAVE_PATH + "y.pickle"):
        print("could find saved files...")
        print("loading...")
        X = read_pickle(SAVE_PATH + "X.pickle")
        y = read_pickle(SAVE_PATH + "y.pickle")
    else:
        print("starting to load all files...")
        print("this could take a while...")
        # loading data
        subfolders = os.listdir(DATA_PATH)
        subfolders = [DATA_PATH  + el + "/" for el in subfolders]
        fnames = get_filenames_without_ending(DATA_PATH)
        
        # X is the text and y the ann data
        X = []
        y = []

        for act_file in fnames:
            tmp_txt = get_ann(act_file + ".txt")
            tmp_ann = get_ann(act_file + ".ann")
            X.append(tmp_txt)
            y.append(tmp_ann)

        print("saving the data into pickle files...")
        save_pickle(SAVE_PATH + "X.pickle", X)
        save_pickle(SAVE_PATH + "y.pickle", y)
        

    print(f"amount of loaded files: {len(X)}")
    print("preprocessing the text...")
    # this is actually a very ugly way, but i was in a hurry... :D
    X = [" ".join(el) for el in X]
    X = [" ".join(el.split("-")) for el in X]
    X = [" ".join(el.split("/")) for el in X]
    X = [" ".join(el.split(",")) for el in X]
    X = [" ".join(el.split("\n")) for el in X]
    X = [" ".join(el.split("\t")) for el in X]
    X = [" ".join(el.split("(")) for el in X]
    X = [" ".join(el.split(")")) for el in X]
    X = [re.sub(" +", " ", el) for el in X]
    
    print("preprocessing the according label...")
    
    # cleaning the label little bit, seperate words, that are concatenated by '-'
    # Lungen-Karzinom -> Lungen Karzinom
    clean_y = []
    for act_y in y:
        tmp_row = []
        for act_row in act_y:
            tmp = " ".join(act_row)
            if "-" in tmp:
                tmp = " ".join(tmp.split("-"))
                tmp_row.append(tmp.split(" "))
            else:
                tmp_row.append(act_row)
        tmp_row = list(filter(None, tmp_row))
        clean_y.append(tmp_row)


    # mapping the golden standard to the new data form
    labeled_sentences = []
    sentence_tags = []
    for act_X, act_y in zip(X, clean_y):
        tmp_sentence = []
        tmp_tags = []
        label = {}
        stemmed_label = {}

        # creating the label stemmed and not stemmed
        for act_ann in act_y:
            for i in range(4, len(act_ann)):
                if should_get_skipped(act_ann[i]):
                    continue

                label[act_ann[i]] = act_ann[1]
                stemmed_label[stem(act_ann[i])] = stem(act_ann[1])
        

        # check for every word in the actual sentence if the word matches with a label
        # do this for stemmed and non stemmed words
        # if the word could be found normal or stemmed, then label it as O
        for act_word in act_X.split(" "):
            if act_word in label.keys():
                tmp_sentence.append((act_word, label[act_word]))
                tmp_tags.append(label[act_word])

            elif stem(act_word) in stemmed_label.keys():
                tmp_sentence.append((act_word, stemmed_label[stem(act_word)]))
                tmp_tags.append(stemmed_label[stem(act_word)])

            else:
                tmp_sentence.append((act_word, "O"))
                tmp_tags.append("O")
        
        sentence_tags.append(tmp_tags)
        labeled_sentences.append(tmp_sentence)

    print(f"the labeled sentences are now in following structure: {labeled_sentences[0]}")

    # getting all unique tags
    tags = set()
    for act_y in y:
        for act_ann in act_y:
            tags.add(act_ann[1])
    
    tags = list(tags)
    tags.append("O")

    print(f"following entities are can be found in the text: {tags}")
    print("prepraring the train data, so that it can be given as input for RoBERTa...")

    # creating tags to index datastructure
    tags_to_index = {t:i for i, t in enumerate(tags)}

    # creating tokenized sentences, splitting every word
    X_tokenized = [act_X.split(" ") for act_X in X]

    # creating sentence number array
    sentence_no = list(range(len(X)))

    # creating the dataframe
    tmp_tokenized = []
    tmp_tags = []
    sentence_no = []

    for idx_sentence, act_row in enumerate(labeled_sentences):
        for idx, content in enumerate(act_row):
            tmp_tokenized.append(content[0])
            tmp_tags.append(content[1])
            sentence_no.append(idx_sentence)

    print(np.array(tmp_tokenized).shape)
    print(np.array(tmp_tags).shape)
    print(np.array(sentence_no).shape)


    ner_data = pd.DataFrame()
    ner_data["words"] = tmp_tokenized
    ner_data["labels"] = tmp_tags
    ner_data["sentence_id"] = sentence_no

    print(ner_data.head())
    print(f"before: {ner_data['labels'].unique()}")
    
    # now there are some tags, that are because of stemming wrong... let's fix that# fixing tags
    ner_data["labels"] = ner_data["labels"].map({
        "physiolog": "Physiology",
        "Physiology": "Physiology",
        "disord": "Disorders",
        "Disorders": "Disorders",
        "chemicals_drug": "Chemicals_Drugs",
        "Chemicals_Drugs": "Chemicals_Drugs",
        "living_b": "Living_Beings",
        "Living_Beings": "Living_Beings",
        "devic": "Devices",
        "Devices": "Devices",
        "procedur": "Procedures",
        "Procedures": "Procedures",
        "anatomical_structur": "Anatomical_Structure",
        "Anatomical_Structure": "Anatomical_Structure",
        "O": "O",
        "tnm": "TNM",
        "TNM": "TNM"
        })

    label = ner_data["labels"].unique().tolist()
    print(f"after: {label}")


    # creating train test eval split
    print("creating train test eval split...")
    X = ner_data[["sentence_id", "words"]]
    y = ner_data["labels"].str.upper()

    x_train, x_test, y_train, y_test = train_test_split(X,y, test_size =0.2, random_state=random_state)
    x_test, x_val, y_test, y_val = train_test_split(x_test,y_test, test_size =0.5, random_state=random_state)

    print(f"training size: {len(x_train)}")
    print(f"test size: {len(x_test)}")
    print(f"eval size: {len(x_val)}")

    #building up train data and test data
    train_data = pd.DataFrame({"sentence_id":x_train["sentence_id"],"words":x_train["words"],"labels":y_train})
    test_data = pd.DataFrame({"sentence_id":x_test["sentence_id"],"words":x_test["words"],"labels":y_test})
    val_data = pd.DataFrame({"sentence_id":x_val["sentence_id"],"words":x_val["words"],"labels":y_val})


    print("defining hyperparamters...")
    args = NERArgs()
    args.num_train_epochs = epochs
    args.learning_rate = 1e-4
    args.overwrite_output_dir =True
    args.train_batch_size = 16
    args.eval_batch_size = 16
    args.labels_list = [el.upper() for el in tags]
    args.save_best_model = True
    args.save_model_every_epoch = False

    print("loading model...")
    model = NERModel(model, checkpoint ,args =args, use_cuda=True)
    
    print(f"training the model for {args.num_train_epochs} epochs...")
    model.train_model(train_data, eval_data = test_data,acc=accuracy_score)

    print("evaluating the model with the test data...")
    result, model_outputs, preds_list = model.eval_model(test_data)

    for key in result.keys():
        print(f"{key}: {result[key]}")

    print("evaluating the model with the eval data...")
    result, model_outputs, preds_list = model.eval_model(val_data)

    for key in result.keys():
        print(f"{key}: {result[key]}")
    
    print("##############################################################\n\n\n")


if __name__ == "__main__":
    try:
        pipeline("roberta", "roberta-base", 5)
    except:
        print("didn't work...")
    
    try:
        pipeline("roberta", "roberta-base", 15)
    except:
        print("didn't work...")

    try:
        pipeline("roberta", "roberta-base", 100)
    except:
        print("didn't work...")

    try:
        pipeline("roberta", "roberta-base", 500)
    except:
        print("didn't work...")
    

    try:
        pipeline("bert", "bert-base-cased", 5)
    except:
        print("didn't work...")

    try:
        pipeline("bert", "bert-base-cased", 15)
    except:
        print("didn't work...")

    try:
        pipeline("bert", "bert-base-cased", 100)
    except:
        print("didn't work...")

    try:
        pipeline("bert", "bert-base-cased", 500)
    except:
        print("didn't work...")

