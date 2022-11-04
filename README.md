# Named Entity Recognition

## AIM

To develop an LSTM-based model for recognizing the named entities in the text.

## PROBLEM STATEMENT AND DATASET
Using LSTM model to identify the Named Entities in a given sentence.
The dataset used has a number of sentences, and each words have their tags. 
We vectorize these words using Embedding techniques to train our model.

## DESIGN STEPS

### Step 1:
Import the necessary packages.

### Step 2:
Read the dataset, and fill the null values using forward fill. 

### Step 3:
Create a list of words, and tags. Also find the number of unique words and tags in the dataset.

### Step 4:
Create a dictionary for the words and their Index values. Do the same for the tags as well.

### Step 5:
Now we move to moulding the data for training and testing. We do this by padding the sequences. This is done to acheive the same length of input data.

### Step 6:
We build a build a model using Input, Embedding, Bidirectional LSTM, Spatial Dropout, Time Distributed Dense Layers.

### Step 7:
We compile the model and fit the train sets and validation sets. 

### Step 8:
We plot the necessary graphs for analysis. 

### Step 8:
A custom prediction is done to test the model manually.

## PROGRAM
```python3
import tensorflow as tf
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras import layers
from keras.models import Model

data=pd.read_csv("ner_dataset.csv",encoding="latin1")
data=data.fillna(method="ffill")
data.head()

words=list(data['Word'].unique())
words.append("ENDPAD")
tags=list(data['Tag'].unique())
num_words = len(words)
num_tags = len(tags)

class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

getter = SentenceGetter(data)
sentences = getter.sentences
word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}

import matplotlib.pyplot as plt
plt.hist([len(s) for s in sentences], bins=50)
plt.show()

X1 = [[word2idx[w[0]] for w in s] for s in sentences]
max_len = 60
X = sequence.pad_sequences(maxlen=max_len,
                  sequences=X1, padding="post",
                  value=num_words-1)
y1 = [[tag2idx[w[2]] for w in s] for s in sentences]
y = sequence.pad_sequences(maxlen=max_len,
                  sequences=y1,
                  padding="post",
                  value=tag2idx["O"])

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=1)
input_word = layers.Input(shape=(max_len,))
embedding_layer=layers.Embedding(num_words,60)(input_word)
drop_layer=layers.SpatialDropout1D(0.2)(embedding_layer)
b_layer=layers.Bidirectional(layers.LSTM(units=100,return_sequences=True,recurrent_dropout=0.2))(drop_layer)
output=layers.TimeDistributed(layers.Dense(num_tags,activation='softmax'))(b_layer)
model = Model(input_word, output)
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

history = model.fit(x=X_train,y=y_train, validation_data=(X_test,y_test),epochs=3)

metrics = pd.DataFrame(model.history.history)
metrics.iloc[metrics.shape[0]-1,:]
plt.figure(figsize=(15, 5))
plt.subplot(1,2,1)
plt.plot(metrics[['accuracy','val_accuracy']])
plt.legend(["Training Accuracy","Validation Accuracy"])
plt.title("Accuracy vs Test Accuracy")
plt.subplot(1,2,2)
plt.plot(metrics[['loss','val_loss']])
plt.legend(["Training Loss","Validation Loss"])
plt.title("Loss vs Test Loss")
plt.show()

```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![image](https://user-images.githubusercontent.com/65499285/197232302-88fc13ba-85dd-4ff6-a282-382d67cfa27d.png)

<br><br><br><br><br><br><br><br><br><br><br><br><br>
### Sample Text Prediction
![image](https://user-images.githubusercontent.com/65499285/197232140-77b888f0-1cc5-40e0-835d-a054e5415243.png)

## RESULT
Thus, we have built a model for Named Entity Recognition successfully. 
