#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer
import pickle, os
from termcolor import colored
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.preprocessing.sequence import pad_sequences
MAX_LENGTH = 500
MAX_WORDS = 20000
EMBENDING_DIM = 100
from termcolor import colored
# In[2]:


pkl_filename = "tokenizer.pkl"
with open(pkl_filename, 'rb') as file:
    tokenizer = pickle.load(file)


# In[3]:


model = load_model("sentiment.model")
os.system("clear")
print("Enter any review")


# In[ ]:


while True:
    que = input(">")
    twt = [que]
    test_sequences = tokenizer.texts_to_sequences(twt)
    test_data = pad_sequences(test_sequences,maxlen=MAX_LENGTH)
    predictions = model.predict(test_data)[0]
    print(predictions[0])
    if predictions > 0.5:
        print(colored("positive", 'green'))
    else:
        print(colored("negative", "red"))

