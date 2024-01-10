#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install tqdm


# In[72]:


get_ipython().system('pip install keras')


# In[74]:


get_ipython().system('pip install tensorflow')


# In[76]:


pip install --upgrade tensorflow keras


# In[1]:


import dask.dataframe as dd
from dask.diagnostics import Profiler, ResourceProfiler, ProgressBar, visualize
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
from tqdm import tqdm

# Function to normalize text
def normalize_text(text):
    # Tokenization
    tokens = word_tokenize(text)

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Sentence Segmentation
    sentences = sent_tokenize(text)

    return pd.Series({'tokens': tokens, 'sentences': sentences})

# Read CSV with increased blocksize and reduced sample
ddf = dd.read_csv('D:\Language&AI\Assign.Data\lai-data\political_leaning.csv', blocksize="4MB", sample=33)

# Optimize Tokenization and Lemmatization Functions
normalized_columns = ddf['post'].map(normalize_text, meta=('x', 'object'))

# Concatenate the original DataFrame with the normalized columns
ddf_normalized = dd.concat([ddf, normalized_columns], axis=1)

# Increase parallelism by adjusting the number of workers
num_processes = 4  # Adjust this based on your machine's resources

# Use tqdm to display progress
with ProgressBar():
    ddf_normalized = dd.compute(ddf_normalized, scheduler='processes', num_workers=num_processes)

# Persist the intermediate result
#ddf_normalized = ddf_normalized[0].persist()

# Dask Profiling
with Profiler() as prof, ResourceProfiler() as rprof:
    # Continue with further computations if needed
    result = ddf_normalized  # Add your subsequent operations here

visualize([prof, rprof])


# In[19]:


# Size of vocabulary

import dask.dataframe as dd
from dask.diagnostics import Profiler, ResourceProfiler, ProgressBar, visualize
from nltk import WordNetLemmatizer, FreqDist
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
from tqdm import tqdm

# Read CSV with increased blocksize and reduced sample
ddf = dd.read_csv('D:\Language&AI\Assign.Data\lai-data\political_leaning.csv', blocksize="4MB", sample=33)
text_data = ddf['post']
# Number of examples
num_examples = len(ddf)

# Number of examples per class
examples_per_class = ddf['political_leaning'].value_counts()

# Tokenize the text data
all_tokens = [word_tokenize(text) for text in text_data]

# Vocabulary size
vocabulary_size = len(set([word.lower() for tokens in all_tokens for word in tokens]))

# Display summary statistics
print(f"Number of Examples: {num_examples}")
print("\nExamples per Class:")
print(examples_per_class)
print(f"\nVocabulary Size: {vocabulary_size}")

# Use tqdm to display progress
with ProgressBar():
    ddf_normalized = dd.compute(ddf_normalized, scheduler='processes', num_workers=num_processes)


# In[4]:


# Word2Vec training, logistic regression, Naive Bayes

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec

# Encode political leaning labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(ddf_normalized[0].iloc[:, 2])

# Assuming ddf_normalized[0].iloc[:, 3] contains your tokenized text data
data = [ddf_normalized[0].iloc[:, 3].iloc[index][0] for index in range(len(ddf_normalized[0]))]

# Split data
data_train, data_test, labels_train, labels_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)

# Train Word2Vec model
word2vec_model = Word2Vec(sentences=data_train, vector_size=50, window=5, min_count=1, workers=4)

# Vectorize training data using Word2Vec embeddings
X_train_w2v = np.array([np.mean([word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv.key_to_index], axis=0) for tokens in data_train])

# Vectorize test data using Word2Vec embeddings
X_test_w2v = np.array([
    np.mean([word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv.key_to_index], axis=0)
    for tokens in data_test
])

# Naive Bayes - Bag-of-Words representation
from sklearn.feature_extraction.text import CountVectorizer

# Convert tokenized text to string
data_train_str = [' '.join(tokens) for tokens in data_train]
data_test_str = [' '.join(tokens) for tokens in data_test]

# Vectorize using CountVectorizer
vectorizer = CountVectorizer()
X_train_nb = vectorizer.fit_transform(data_train_str)
X_test_nb = vectorizer.transform(data_test_str)

# Logistic Regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_w2v, labels_train)
predictions_logreg = logreg.predict(X_test_w2v)
accuracy_logreg = accuracy_score(labels_test, predictions_logreg)
precision_logreg = precision_score(labels_test, predictions_logreg, average='weighted')
recall_logreg = recall_score(labels_test, predictions_logreg, average='weighted')
f1_logreg = f1_score(labels_test, predictions_logreg, average='weighted')

print("Logistic Regression Metrics:")
print(f"Accuracy: {accuracy_logreg}")
print(f"Precision: {precision_logreg}")
print(f"Recall: {recall_logreg}")
print(f"F1 Score: {f1_logreg}")

# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_nb, labels_train)
predictions_nb = nb.predict(X_test_nb)
accuracy_nb = accuracy_score(labels_test, predictions_nb)
precision_nb = precision_score(labels_test, predictions_nb, average='weighted')
recall_nb = recall_score(labels_test, predictions_nb, average='weighted')
f1_nb = f1_score(labels_test, predictions_nb, average='weighted')

print("\nNaive Bayes Metrics:")
print(f"Accuracy: {accuracy_nb}")
print(f"Precision: {precision_nb}")
print(f"Recall: {recall_nb}")
print(f"F1 Score: {f1_nb}")


# In[8]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer

# Encode political leaning labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(ddf_normalized[0].iloc[:, 2])

# Assuming ddf_normalized[0].iloc[:, 3] contains your tokenized text data
data = [ddf_normalized[0].iloc[:, 3].iloc[index][0] for index in range(len(ddf_normalized[0]))]

# Split data
data_train, data_test, labels_train, labels_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)

# Train Word2Vec model
word2vec_model = Word2Vec(sentences=data_train, vector_size=50, window=5, min_count=1, workers=4)

# Vectorize training data using Word2Vec embeddings
X_train_w2v = np.array([np.mean([word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv.key_to_index], axis=0) for tokens in data_train])

# Vectorize test data using Word2Vec embeddings
X_test_w2v = np.array([
    np.mean([word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv.key_to_index], axis=0)
    for tokens in data_test
])

# Naive Bayes - Bag-of-Words representation
data_train_str = [' '.join(tokens) for tokens in data_train]
data_test_str = [' '.join(tokens) for tokens in data_test]

# Vectorize using CountVectorizer
vectorizer = CountVectorizer()
X_train_nb = vectorizer.fit_transform(data_train_str)
X_test_nb = vectorizer.transform(data_test_str)

# Logistic Regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_w2v, labels_train)
predictions_logreg = logreg.predict(X_test_w2v)
accuracy_logreg = accuracy_score(labels_test, predictions_logreg)
precision_logreg = precision_score(labels_test, predictions_logreg, average='weighted')
recall_logreg = recall_score(labels_test, predictions_logreg, average='weighted')
f1_logreg = f1_score(labels_test, predictions_logreg, average='weighted')

print("Logistic Regression Metrics:")
print(f"Accuracy: {accuracy_logreg}")
print(f"Precision: {precision_logreg}")
print(f"Recall: {recall_logreg}")
print(f"F1 Score: {f1_logreg}")

# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_nb, labels_train)
predictions_nb = nb.predict(X_test_nb)
accuracy_nb = accuracy_score(labels_test, predictions_nb)
precision_nb = precision_score(labels_test, predictions_nb, average='weighted')
recall_nb = recall_score(labels_test, predictions_nb, average='weighted')
f1_nb = f1_score(labels_test, predictions_nb, average='weighted')

print("\nNaive Bayes Metrics:")
print(f"Accuracy: {accuracy_nb}")
print(f"Precision: {precision_nb}")
print(f"Recall: {recall_nb}")
print(f"F1 Score: {f1_nb}")

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train_w2v, labels_train)
predictions_rf = rf.predict(X_test_w2v)
accuracy_rf = accuracy_score(labels_test, predictions_rf)

# Get the mapping of numerical labels to original class labels
label_mapping = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))
class_labels_rf = [label_mapping[label] for label in predictions_rf]

# Confusion Matrix for Random Forest
cm_rf = confusion_matrix(labels_test, predictions_rf)
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train_w2v, labels_train)
predictions_dt = dt.predict(X_test_w2v)
accuracy_dt = accuracy_score(labels_test, predictions_dt)

# Get the mapping of numerical labels to original class labels
class_labels_dt = [label_mapping[label] for label in predictions_dt]

# Confusion Matrix for Decision Tree
cm_dt = confusion_matrix(labels_test, predictions_dt)
sns.heatmap(cm_dt, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Decision Tree Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print(f"Random Forest Accuracy: {accuracy_rf}")
print(f"Random Forest Precision: {rf_precision}")
print(f"Random Forest Recall: {rf_recall}")
print(f"Random Forest F1 Score: {rf_f1}")
print(f"Random Forest Confusion Matrix:\n{rf_confusion_matrix}")

print(f"Decision Tree Accuracy: {accuracy_dt}")
print(f"Decision Tree Precision: {dt_precision}")
print(f"Decision Tree Recall: {dt_recall}")
print(f"Decision Tree F1 Score: {dt_f1}")
print(f"Decision Tree Confusion Matrix:\n{dt_confusion_matrix}")


# In[9]:


print(f"Random Forest Accuracy: {accuracy_rf}")
print(f"Random Forest Precision: {rf_precision}")
print(f"Random Forest Recall: {rf_recall}")
print(f"Random Forest F1 Score: {rf_f1}")
print(f"Random Forest Confusion Matrix:\n{rf_confusion_matrix}")

print(f"Decision Tree Accuracy: {accuracy_dt}")
print(f"Decision Tree Precision: {dt_precision}")
print(f"Decision Tree Recall: {dt_recall}")
print(f"Decision Tree F1 Score: {dt_f1}")
print(f"Decision Tree Confusion Matrix:\n{dt_confusion_matrix}")


# In[6]:


# Encode political leaning labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(ddf_normalized[0].iloc[:, 2])

# Assuming ddf_normalized[0].iloc[:, 3] contains your tokenized text data
data = [ddf_normalized[0].iloc[:, 3].iloc[index][0] for index in range(len(ddf_normalized[0]))]


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns

# Logistic Regression - Confusion Matrix Heatmap
sns.heatmap(confusion_matrix(labels_test, predictions_logreg), annot=True, fmt="d", cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Naive Bayes - Confusion Matrix Heatmap
sns.heatmap(confusion_matrix(labels_test, predictions_nb), annot=True, fmt="d", cmap="Blues")
plt.title("Naive Bayes Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Random Forest - Confusion Matrix Heatmap
sns.heatmap(rf_confusion_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Decision Tree - Confusion Matrix Heatmap
sns.heatmap(dt_confusion_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# In[14]:


import pandas as pd
from nltk import FreqDist
from nltk.tokenize import word_tokenize

# Assuming ddf_normalized[0].iloc[:, 3] contains your tokenized text data
text_data = [ddf_normalized[0].iloc[:, 3].iloc[index][0] for index in range(len(ddf_normalized[0]))]

# Number of examples
num_examples = len(text_data)

# Number of examples per class
examples_per_class = ddf_normalized[0].iloc[:, 2].value_counts()

# Tokenize the text data
all_tokens = [word_tokenize(text) for text in text_data]

# Vocabulary size
vocabulary_size = len(set([word.lower() for tokens in all_tokens for word in tokens]))

# Display summary statistics
print(f"Number of Examples: {num_examples}")
print("\nExamples per Class:")
print(examples_per_class)
print(f"\nVocabulary Size: {vocabulary_size}")


# In[ ]:


# Bidirectional LSTM - first try -not working

# Bidirectional LSTM with Attention
# Assuming each document in data_train and data_test has varying lengths

# Pad sequences to ensure uniform length
max_len = 100  # Choose a reasonable value based on your data
X_train_lstm = pad_sequences(
    [
        [word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv.key_to_index]
        for tokens in data_train
    ],
    maxlen=max_len,
    padding='post'
)
X_test_lstm = pad_sequences(
    [
        [word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv.key_to_index]
        for tokens in data_test
    ],
    maxlen=max_len,
    padding='post'
)

# Define Bidirectional LSTM model with Attention
model = Sequential()
model.add(Embedding(input_dim=len(word2vec_model.wv.key_to_index), output_dim=50, input_length=max_len))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Attention())
model.add(Dense(3, activation='softmax'))  # Adjust output size for 3 classes
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_lstm, labels_train, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate LSTM model
predictions_lstm = model.predict(X_test_lstm)
predictions_lstm = np.argmax(predictions_lstm, axis=1)
accuracy_lstm = accuracy_score(labels_test, predictions_lstm)
print(f"LSTM Accuracy: {accuracy_lstm}")


# In[56]:


# Bidirectional LSTM - second try -not wroking


from keras.layers import Input, Embedding, Bidirectional, LSTM, Dense
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

# Assuming each document in data_train and data_test has varying lengths

# Pad sequences to ensure uniform length
max_len = 100  # Choose a reasonable value based on your data
X_train_lstm = pad_sequences(
    [
        [word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv.key_to_index]
        for tokens in data_train
    ],
    maxlen=max_len,
    padding='post'
)

# Make sure the input data has the correct shape
X_train_lstm = X_train_lstm.reshape((X_train_lstm.shape[0], max_len, 50))  # Assuming word vectors are of size 50

# Define the model
input_layer = Input(shape=(max_len, 50))  # Assuming each word has a vector representation of size 50
bidirectional_lstm = Bidirectional(LSTM(64, return_sequences=True))(input_layer)
dense_layer = Dense(10, activation='softmax')(bidirectional_lstm)

# Create the model
model = Model(inputs=input_layer, outputs=dense_layer)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_lstm, labels_train, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate LSTM model
X_test_lstm = pad_sequences(
    [
        [word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv.key_to_index]
        for tokens in data_test
    ],
    maxlen=max_len,
    padding='post'
)
# Make sure the input data has the correct shape
X_test_lstm = X_test_lstm.reshape((X_test_lstm.shape[0], max_len, 50))

predictions_lstm = model.predict(X_test_lstm)


# In[59]:


pip install transformers


# In[5]:


pip install transformers tensorflow dask nltk tqdm scikit-learn


# In[12]:


pip install torch


# In[11]:


# BERT -not wrking

import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load BERT Tokenizer and Model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = TFBertModel.from_pretrained("bert-base-uncased")

# Tokenize and pad the sequences using encode_plus
max_length = 512  # Set the desired max length
batch_size = 8  # Adjust this based on your machine's resources

def batch_generator(data, labels, batch_size):
    for i in range(0, len(data), batch_size):
        batch_data = data[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]
        tokenized_batch = tokenizer(batch_data, max_length=max_length, return_tensors="tf", padding=True, truncation=True, return_attention_mask=True)
        batch_embeddings = bert_model(tokenized_batch.input_ids, attention_mask=tokenized_batch.attention_mask)['pooler_output'].numpy()
        yield np.concatenate(batch_embeddings), batch_labels

# Get BERT embeddings for training data
X_train_bert = np.concatenate([batch[0] for batch in batch_generator(data_train, labels_train, batch_size)])

# Get BERT embeddings for test data
X_test_bert = np.concatenate([batch[0] for batch in batch_generator(data_test, labels_test, batch_size)])

# Continue with Logistic Regression or other classifiers
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_bert, labels_train)
predictions_logreg = logreg.predict(X_test_bert)
accuracy_logreg = accuracy_score(labels_test, predictions_logreg)
print(f"Logistic Regression Accuracy: {accuracy_logreg}")


# In[4]:


pip install transformers --upgrade


# In[64]:


x = []
for i in range(len(ddf_normalized[0].iloc[:, 3])-1):
    x.append(ddf_normalized[0].iloc[:, 3].iloc[i][0])


# In[29]:


ddf_normalized[0].iloc[:, 3].iloc[:][0]


# In[158]:


# PPMI vetorizer - issues

import dask.dataframe as dd
import dask.array as da
import numpy as np
import pandas as pd

# Extract 'tokens' column from the computed result
tokens_column = ddf_normalized_computed[0][0].iloc[:, 3]

# Function to flatten the list of lists and pad each list to a fixed length
max_length = 100  # Choose an appropriate value based on your data
flatten_and_pad = lambda l: [sublist + [''] * (max_length - len(sublist)) for sublist in l]

# Flatten and pad the 'tokens' column
flat_tokens = flatten_and_pad(tokens_column.apply(list).tolist())

# Create a Dask DataFrame with a single column named 'tokens'
ddf_tokens = dd.from_pandas(pd.DataFrame({'tokens': flat_tokens}), npartitions=1)

# PPMI Vectorization Class
class PPMIVectorizer(object):
    def __init__(self, alpha=0.75, context_window=5):
        self.word_freqs = Counter()
        self.context_freqs = Counter()
        self.co_occ_freqs = Counter()
        self.alpha = alpha
        self.context_window = context_window
        self.word_cols = {}
        self.context_cols = {}

    def get_counts(self, words):
        for window in zip(*[words[i:] for i in range(self.context_window)]):
            middle = int((len(window) - 1) / 2)
            context = window[:middle] + window[middle + 1:]
            word = window[middle]
            self.word_freqs[word] += 1
            for context_word in context:
                self.context_freqs[context_word] += 1
                self.co_occ_freqs[(word, context_word)] += 1

    def get_ppmi(self):
        sum_context = sum(self.context_freqs.values())
        sum_total = sum(self.word_freqs.values()) + sum_context
        for (w, c), wc_freq in self.co_occ_freqs.items():
            P_wc = wc_freq / sum_total
            P_w = self.word_freqs[w] / sum_total
            P_alpha_c = (self.context_freqs[c] ** self.alpha / sum_context ** self.alpha)
            ppmi = max(np.log2(P_wc / (P_w * P_alpha_c)), 0)
            self.co_occ_freqs[(w, c)] = ppmi

    def set_cols(self):
        self.context_cols = {w: i for i, w in enumerate(sorted(self.context_freqs.keys()))}
        self.word_cols = {}
        for w, c in self.co_occ_freqs.keys():
            if not self.word_cols.get(w):
                self.word_cols[w] = []
            self.word_cols[w].append((w, c))

    def get_vec(self, words):
        vec = csr_matrix((1, len(self.context_cols)))
        try:
            for word in words:
                for pair in self.word_cols.get(tuple(word), []):
                    w, c = pair
                    vec[0, self.context_cols[c]] = self.co_occ_freqs[(w, c)]
        except KeyError:
                pass
        return vec

    def check_tokens(self, document):
        if not isinstance(document, list):
            return document.split(' ')
        else:
            return document

    def fit(self, X):
        for document in X:
            self.get_counts(self.check_tokens(document))
        self.get_ppmi()
        self.set_cols()

    def transform(self, X):
        result = []
        for document in X:
            vectors = [self.get_vec(self.check_tokens(document))]
            result.append(csr_matrix(np.sum(vectors, axis=0)))
        return result

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# Initialize PPMIVectorizer
ppmi_vectorizer = PPMIVectorizer()

# Fit and transform with PPMIVectorizer
X_ppmi = ppmi_vectorizer.transform(ddf_tokens['tokens'])

# Display PPMI values for the first document
print("PPMI Values for the first document:")
print(X_ppmi[0])


# In[176]:


##### The word tokens are at:
ddf_normalized_computed[0][0].iloc[:, 3].iloc[0][0]

##### The sentencestokens are at:
#ddf_normalized_computed[0][0].iloc[:, 3].iloc[0][1]


# In[8]:


pip install nltk


# In[7]:


pip install dask


# In[10]:


ddf_normalized


# In[12]:


# Assuming ddf_normalized is your Dask DataFrame
summary_stats = ddf.describe().compute()

# Print the summary statistics
print(summary_stats)


# In[6]:


print(data.loc[0, 'post'])


# In[ ]:


# Vector Representation using PPMI
class PPMIVectorizer(object):
    def __init__(self, alpha=0.75, context_window=5):
        self.word_freqs = Counter()
        self.context_freqs = Counter()
        self.co_occ_freqs = Counter()
        self.alpha = alpha
        self.context_window = context_window
        self.word_cols = {}
        self.context_cols = {}

    def get_counts(self, document):
        for window in zip(*[document[i:] for i in range(self.context_window)]):
            middle = int((len(window) - 1) / 2)
            context = window[:middle] + window[middle + 1:]
            word = window[middle]
            self.word_freqs[word] += 1
            for context_word in context:
                self.context_freqs[context_word] += 1
                self.co_occ_freqs[(word, context_word)] += 1
        return self.co_occ_freqs

    def get_ppmi(self):
        sum_context = sum(self.context_freqs.values())
        sum_total = sum(self.word_freqs.values()) + sum_context
        for (w, c), wc_freq in self.co_occ_freqs.items():
            P_wc = wc_freq / sum_total
            P_w = self.word_freqs[w] / sum_total
            P_alpha_c = (self.context_freqs[c] ** self.alpha / sum_context ** self.alpha)
            ppmi = max(np.log2(P_wc / (P_w * P_alpha_c)), 0)
            self.co_occ_freqs[(w, c)] = ppmi
        return self.co_occ_freqs

    def set_cols(self):
        self.context_cols = {w: i for i, w in enumerate(sorted(self.context_freqs.keys()))}
        self.word_cols = {}
        for w, c in self.co_occ_freqs.keys():
            if not self.word_cols.get(w):
                self.word_cols[w] = []
            self.word_cols[w].append((w, c))

    def get_vec(self, word):
        vec = [0.0] * len(self.context_cols)
        try:
            for (w, c) in self.word_cols[word]:
                vec[self.context_cols[c]] = self.co_occ_freqs[(w, c)]
        except KeyError:
            pass
        return vec

    def check_tokens(self, document):
        if not isinstance(document, list):
            return document.split(' ')
        else:
            return document

    def fit(self, X):
        for document in X:
            self.get_counts(self.check_tokens(document))
        self.get_ppmi()
        self.set_cols()

    def transform(self, X):
        return [np.sum([self.get_vec(w) for w in self.check_tokens(document)]) for document in X]

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# Example Usage
ppmi_vectorizer = PPMIVectorizer()
X_ppmi = ppmi_vectorizer.fit_transform(df['post'])

# Display PPMI values for the first document
print("PPMI Values for the first document:")
print(X_ppmi[0])

