#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import sklearn
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from nltk.corpus import stopwords

stop = stopwords.words('english')
g_patent = pd.read_csv("/home/ubuntu/flaskapp/output.csv")
g_patent.head()

# In[2]:


pattern = ['Abstract\n\n', 'Abstract\n', '\n']
for i in pattern:
    g_patent.Abstract = g_patent.Abstract.str.replace(i, '')

# In[3]:


g_patent.head()
g_patent.shape

# In[4]:


g_patent = g_patent[g_patent.Title != 'Title']

# In[5]:


g_patent.shape

# In[6]:


pattern = ['Classifications\n\n', 'Classifications\n', '\n']
for i in pattern:
    g_patent.Classification = g_patent.Classification.str.replace(i, '')
g_patent.head()
g_patent['Title'].head()

# In[7]:


g_patent['Current Assignee'] = g_patent['Current Assignee'].str.replace('\n', '')
g_patent.head()

# In[8]:


# g_patent['Inventors'] = g_patent.Inventors.str.replace('[','')
# g_patent['Inventors'] = g_patent.Inventors.str.replace(']','')
# g_patent['Inventors'] = g_patent.Inventors.str.replace('\' ','')
# g_patent['Inventors'] = g_patent.Inventors.str.replace('\'','')

patterns = ['[', ']', '\' ', ' \'', '\'']
for i in patterns:
    g_patent['Inventors'] = g_patent.Inventors.str.replace(i, '')

# In[9]:


g_patent.head()

# In[10]:


g_patent = g_patent.dropna(how='all', axis=1)
g_patent = g_patent.dropna(how='all', axis=0)

# In[11]:


g_patent = g_patent.drop_duplicates(subset='Patent Number', keep='first')

# In[12]:

print("HI1")

g_patent.shape

# In[13]:
print("HI2")


vectorizer = TfidfVectorizer(analyzer='word')
print("HI3")

def foo(obj):
	temp = obj.copy()
	for idx in range(len(obj)):
		if obj[idx] != obj[idx]:
			temp[idx] = ''
	return temp


tfidf_matrix = vectorizer.fit_transform(foo(g_patent['Title'].values))
print("HI4")

#comping cosine similarity matrix using linear_kernal of sklearn
cosine_similarity = linear_kernel(tfidf_matrix, tfidf_matrix)
print("HI5")
g_patent = g_patent.reset_index(drop=True)
print("HI6")

indices = pd.Series(g_patent['Title'].index)


# In[19]:


# Function to get the most similar Patents
def recommend(index, method):
    id = indices[index]
    # Get the pairwise similarity scores of all patents compared to that patent,
    # sorting them and getting top 5
    similarity_scores = list(enumerate(method[id]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:6]

    # Get the patents index
    patents_index = [i[0] for i in similarity_scores]

    # Return the top 5 most similar patents using integar-location based indexing (iloc)
    return g_patent['Title'].iloc[patents_index]


# In[20]:


# input the index of the patent

# In[21]:


g_patent['All Content'] = g_patent['Title'] + g_patent['Inventors'] + g_patent['Current Assignee'] + g_patent[
    'Abstract']

# In[22]:


tfidf_all_content = vectorizer.fit_transform(foo(g_patent['All Content'].values))
tfidf_all_content.shape

# In[23]:


# comping cosine similarity matrix using linear_kernal of sklearn
cosine_similarity_all_content = linear_kernel(tfidf_all_content, tfidf_all_content)


# Recommendind Patents based on Title Search
g_patent_data = g_patent[g_patent['Title'].notnull()].copy()
g_patent_data = g_patent_data[g_patent_data['Title'].map(len) > 5]

tfidf_des = vectorizer.fit_transform(foo(g_patent_data['Title'].values))
# comping cosine similarity matrix using linear_kernal of sklearn
cosine_sim_des = linear_kernel(tfidf_des, tfidf_des)
# cosine_sim_des
indices_t = pd.Series(g_patent_data['Title'])
# indices_t
inddict_title = indices_t.to_dict()
inddict_title = dict((v, k) for k, v in inddict_title.items())


# In[25]:


def recommend_cosine(title):
    id = inddict_title[title]
    # Get the pairwise similarity scores of all patents compared to that patent,
    # sorting them and getting top 5
    similarity_scores = list(enumerate(cosine_sim_des[id]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:6]
    # print(similarity_scores)
    # Get the books index
    patents_index = [i[0] for i in similarity_scores]

    # Return the top 5 most similar patents using integar-location based indexing (iloc)
    return g_patent_data.iloc[patents_index]


# In[26]:



# # Euclidean Distance
#     from sklearn.feature_extraction.text import TfidfVectorizer
#     from sklearn.metrics.pairwise import euclidean_distances
#
#     v = TfidfVectorizer()
#     X = v.fit_transform(your_documents)
#     D = euclidean_distances(X)
#     Now D[i, j] is the Euclidean distance between document vectors X[i] and X[j].

# In[27]:


from sklearn.metrics.pairwise import euclidean_distances

# In[28]:


# tfidf_des is the vectorizer fn on the Title column
D = euclidean_distances(tfidf_des)


# In[29]:


def recommend_euclidean_distance(title):
    ind = inddict_title[title]
    distance = list(enumerate(D[ind]))
    distance = sorted(distance, key=lambda x: x[1])
    distance = distance[1:6]
    # Get the patents index
    patents_index = [i[0] for i in distance]

    # Return the top 5 most similar paptents using integar-location based indexing (iloc)
    return g_patent_data.iloc[patents_index]


# In[30]:





# finding similarity using Pearson Corelation
from scipy.stats import pearsonr

tfidf_des_array = tfidf_des.toarray()


# In[32]:


def recommend_pearson(title):
    ind = inddict_title[title]
    correlation = []
    for i in range(len(tfidf_des_array)):
        correlation.append(pearsonr(tfidf_des_array[ind], tfidf_des_array[i])[0])
    correlation = list(enumerate(correlation))
    sorted_corr = sorted(correlation, reverse=True, key=lambda x: x[1])[1:6]
    patent_index = [i[0] for i in sorted_corr]
    return g_patent_data.iloc[patent_index]


# In[33]:




# ## So, all 3 methods of similarity: (1) Cosine (2)Euclidean distance (3) Pearson corelation gave us the same resuls.
# We are proceeding with cosine similarity. The next funtion takes 2 arguments.
# Argument 1. Title, Author or Current Assignee - The actual value on which the search should be based.
# Argument 2. 'Title' or 'Inventors' or 'Assignee' - The user has to specify the argument against which he wants to search.

# In[34]:


def recommend_patents(value, criteria):
    if criteria == 'Title':
        tfidf_des = vectorizer.fit_transform(foo(g_patent_data['Title'].values))
        indices = pd.Series(g_patent_data['Title'])
    elif criteria == 'Inventors':
        tfidf_des = vectorizer.fit_transform(foo(g_patent_data['Inventors'].values))
        indices = pd.Series(g_patent_data['Inventors'])
    elif criteria == 'Assignee':
        tfidf_des = vectorizer.fit_transform(foo(g_patent_data['Current Assignee'].values))
        indices = pd.Series(g_patent_data['Current Assignee'])

    cosine_sim_des = linear_kernel(tfidf_des, tfidf_des)

    inddict = indices.to_dict()
    inddict = dict((v, k) for k, v in inddict.items())

    id = inddict[value]
    # Get the pairwise similarity scores of all patents compared to that patent,
    # sorting them and getting top 5
    similarity_scores = list(enumerate(cosine_sim_des[id]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:6]
    # print(similarity_scores)
    # Get the books index
    patents_index = [i[0] for i in similarity_scores]

    # Return the top 5 most similar patents using integar-location based indexing (iloc)
    return g_patent_data.iloc[patents_index]


# # Next improved function handles the scenariors of random search.
# If a Title doesnot exists our function will be able to recommend users based on keywords user searched.

# In[44]:


def recommend_patents(value, criteria):
    df = g_patent_data
    if criteria == 'Title':
        if value not in df['Title']:
            mod_df = g_patent_data.append({'Title': value}, ignore_index=True)
            df = mod_df
        tfidf_des = vectorizer.fit_transform(foo(df['Title'].values))
        indices = pd.Series(df['Title'])
    elif criteria == 'Inventors':
        if value not in df['Inventors']:
            mod_df = g_patent_data.append({'Inventors': value}, ignore_index=True)
            df = mod_df
        tfidf_des = vectorizer.fit_transform(foo(df['Inventors'].values))
        indices = pd.Series(df['Inventors'])
    elif criteria == 'Assignee':
        if value not in df['Inventors']:
            mod_df = g_patent_data.append({'Current Assignee': value}, ignore_index=True)
            df = mod_df
        tfidf_des = vectorizer.fit_transform(foo(df['Current Assignee'].values))
        indices = pd.Series(df['Current Assignee'])

    cosine_sim_des = linear_kernel(tfidf_des, tfidf_des)

    inddict = indices.to_dict()
    inddict = dict((v, k) for k, v in inddict.items())

    id = inddict[value]
    # Get the pairwise similarity scores of all patents compared to that patent,
    # sorting them and getting top 5
    similarity_scores = list(enumerate(cosine_sim_des[id]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:6]
    # print(similarity_scores)
    # Get the books index
    patents_index = [i[0] for i in similarity_scores]

    # Return the top 5 most similar patents using integar-location based indexing (iloc)
    return df.iloc[patents_index]






