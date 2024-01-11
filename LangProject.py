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


# In[23]:


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
examples_per_class = ddf['political_leaning'].value_counts().compute()

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


# In[30]:


# Vocabulary size and visualizations

import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from nltk.tokenize import word_tokenize
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read CSV with increased blocksize and reduced sample
ddf = dd.read_csv('D:\Language&AI\Assign.Data\lai-data\political_leaning.csv', blocksize="4MB", sample=33)
text_data = ddf['post']

# Number of examples
num_examples = len(ddf)

# Number of examples per class
examples_per_class = ddf['political_leaning'].value_counts().compute()

# Tokenize the text data
all_tokens = [word_tokenize(text) for text in text_data]

# Vocabulary size
vocabulary_size = len(set([word.lower() for tokens in all_tokens for word in tokens]))

# Display summary statistics
print(f"Number of Examples: {num_examples}")
print("\nExamples per Class:")
print(examples_per_class)
print(f"\nVocabulary Size: {vocabulary_size}")

# Visualizations
plt.figure(figsize=(12, 6))

# Plot examples per class
plt.subplot(1, 2, 1)
sns.barplot(x=examples_per_class.index, y=examples_per_class.values)
plt.title('Examples per Class')
plt.xlabel('Political Leaning')
plt.ylabel('Number of Examples')


plt.tight_layout()
plt.show()


# In[34]:


# Visualizations
plt.figure(figsize=(12, 6))

# Plot examples per class
plt.subplot(1, 2, 1)
sns.barplot(x=examples_per_class.index, y=examples_per_class.values)
plt.title('Examples per Class')
plt.xlabel('Political Leaning')
plt.ylabel('Number of Examples')


plt.tight_layout()
plt.show()


# In[ ]:


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


# In[32]:


get_ipython().system('pip install wordcloud')


# In[33]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from wordcloud import WordCloud

# Visualize Logistic Regression Confusion Matrix
cm_logreg = confusion_matrix(labels_test, predictions_logreg)
sns.heatmap(cm_logreg, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Visualize Naive Bayes Confusion Matrix
cm_nb = confusion_matrix(labels_test, predictions_nb)
sns.heatmap(cm_nb, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Naive Bayes Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Classification Report for Logistic Regression
print("Logistic Regression Classification Report:")
print(classification_report(labels_test, predictions_logreg, target_names=label_encoder.classes_))

# Classification Report for Naive Bayes
print("\nNaive Bayes Classification Report:")
print(classification_report(labels_test, predictions_nb, target_names=label_encoder.classes_))

# Visualize Word Cloud for each political leaning class
for political_leaning in label_encoder.classes_:
    # Extract text for the current class
    class_text = ' '.join([text for text, label in zip(data_test_str, labels_test) if label_encoder.classes_[label] == political_leaning])
    
    # Generate Word Cloud
    wordcloud = WordCloud(width = 800, height = 400, random_state=42, max_font_size=100).generate(class_text)
    
    # Plot Word Cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title(f'Word Cloud for {political_leaning} Class')
    plt.show()


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


# In[59]:


pip install transformers


# In[5]:


pip install transformers tensorflow dask nltk tqdm scikit-learn


# In[12]:


pip install torch


# In[4]:


pip install transformers --upgrade


# In[8]:


pip install nltk


# In[7]:


pip install dask


# In[10]:


ddf_normalized


# In[6]:


print(data.loc[0, 'post'])


# In[21]:


pip install prettytable


# In[22]:


from prettytable import PrettyTable

# Logistic Regression Metrics
lr_metrics = ['Logistic Regression', 0.4956757228968289, 0.48743957389577086, 0.4956757228968289, 0.4787612740959953]

# Naive Bayes Metrics
nb_metrics = ['Naive Bayes', 0.6409539617366996, 0.678915764156701, 0.6409539617366996, 0.6378053838190366]

# Random Forest Metrics
rf_metrics = ['Random Forest', 0.6975626801782127, 0.7175985516934213, 0.7060365161177601, 0.6989070653799844]

# Decision Tree Metrics
dt_metrics = ['Decision Tree', 0.5411898313968725, 0.5384659604515172, 0.5369092338604001, 0.537609107554384]

# Create a PrettyTable
table = PrettyTable()
table.field_names = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
table.add_row(lr_metrics)
table.add_row(nb_metrics)
table.add_row(rf_metrics)
table.add_row(dt_metrics)

# Print the table
print(table)


# In[ ]:




