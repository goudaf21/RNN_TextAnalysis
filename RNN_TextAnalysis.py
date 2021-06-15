import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

#Read in csv file into pandas data frame
df = pd.read_csv("consumer_complaints.csv")

y = "product"
x = "consumer_complaint_narrative"

#Extract complaints and products into a new dataframe
new_df = df[[x, y]]

#Check how many values are null and count them
print("Null values in the new dataframe:\n", new_df.isnull().sum())

#Check how many product values we have
print("Number of product values: ", new_df[y].count())

#Drop all empty values, reset the index, and drop them
#drop null values
new_df.dropna(axis=0, inplace=True)

#reset the index
new_df.reset_index(drop=True, inplace=True)

#Recount product values
print("Recount product values: ", new_df[y].count())

#Create a clean text function
def clean_text(text):

    #Lower case the text
    text = text.str.lower()

    #Compile pattern to remove all other characters
    special_chrs = "[!@#$%^&*()-+\|{}';:/?.=>,<\t\n]"
    #special_chrs = "!'#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n"

    #Sub the regular expresion with a "" character.
    text = text.str.replace(special_chrs, "")

    #Remove x from the text characters with a "" character.
    text = text.str.replace("x", "")

    #Split the text
    text = text.str.split()

    #nltk.download('stopwords')
    stop = stopwords.words('english')

    #For each word check if its a word and its an alphanumeric
    for index, complaint  in enumerate(text):
        update_complaint = []
        for word in complaint:
            if word not in stop: update_complaint.append(word)
        #join the list and update dataframe
        text[index] = " ".join(update_complaint[:])

    #Return the clean text
    return text

#Apply clean text to the complaints
new_df[x] = clean_text(new_df[x])

#Define maximum number of words in our vocabulary to 50000
max_vocab = 50000

#Define maximum number of words within each complaint document to 250
max_word = 250

#Define maximum number of words within each embedding to 100
max_emb = 100

#Implement Tokenizer object with num_words, filters, lower, split, and char_level
tokenizer = Tokenizer(num_words=max_vocab,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,char_level=False,split=' ')

#Fit Tokenizer object on the text
tokenizer.fit_on_texts(new_df[x])

#Get the word index from tokenizer object
word_index=tokenizer.word_index

#Print number of unique tokens found
print(len(word_index))

#Get a text to sequences representation of the complaints
seq=tokenizer.texts_to_sequences(new_df[x])

#Pad the sequences with the max length
pad_sequences(seq, maxlen=max_word)

#Print the shape of the data
print(new_df.shape)

#Print the first example of the tokenizer object to the sequences to text
print(tokenizer.sequences_to_texts(seq)[0])
