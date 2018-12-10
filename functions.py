
# coding: utf-8

# In[ ]:


#importing basic libs
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np


#import libs for scraping
from bs4 import BeautifulSoup as bs
import requests
import re

#from pymongo import MongoClient

#import libs for diagnostics
import time
from IPython.display import clear_output

#importing libs for clustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

#import libs for cleaning and stemming of the descriptions
from nltk.corpus import stopwords
stop_words = set(stopwords.words("italian"))
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('italian')

#importling libs for the wordcloud
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


'''
Below you can find our basic functions, e.g. for saving dataframes or dictionarys. 
'''

#function to save a pandas DataFrame to csv
def dfToCsv(df,filename):
    df.to_csv(filename+'.csv', sep='\t')
    
#function to read our vocabulary file from disk    
def getVocabulary(vocabularyFile):
        with open(vocabularyFile, newline='') as file:
            reader = csv.reader(file, delimiter='\t')
            vocabulary = list(map(tuple, reader))
        file.close()
        return(vocabulary)

#function to save our vocabulary file to disk        
def saveVocabulary(vocabulary,fileName="output.csv"):
    with open(fileName,'wb') as vfile:
        for i in vocabulary.keys():
            vfile.write(str(i).encode())
            vfile.write(str('\t').encode())
            vfile.write(str(vocabulary[i]).encode())
            vfile.write('\n'.encode())
        vfile.close()
        


'''
Below you can find the functions we used for the scraping of the details of the appartments and the descriptions.
'''

def getDetails(url, i):
    #getting up beautifulsoup and requests so we can crawl the url passed to the function
    content = requests.get(url)
    soup = bs(content.text, 'html.parser')

    #Checking what class contains the information that we want to retrieve. 
    #At some point different classes are used in the web page so this step is necessary.
    classes = ["listing-item vetrina js-row-detail", "listing-item star js-row-detail","listing-item top js-row-detail", "listing-item premium js-row-detail"]
    for item in classes:
        if item in content.text:
            use_class = item
        
    #creating empty list to be filled with a list of details of all apartement listet on the current page
    detlist =[]
    
    for app in soup.find_all(class_=use_class):

        #creating empty list to fill with the details of the appartement in app
        dets =[]
        
        #Retrieving the link of the apartement thats contained in the class "titolo text-primary"
        #by searching for the first occurence of 'href'.
        link_class = app.find(class_="titolo text-primary")
        link = link_class.find('a', href=True)['href']
        
        #Skipping the listsings that are advertisments or something like it
        if link.startswith('/nuove_costruzioni/'):
            continue
            
        #appending the link to the list of details 
        dets.append(link)
        
        #Retrieving the price of the apartement. It is contained in the second item in the list that beautifulsoup
        price = app.find(class_="listing-features list-piped")
        #appending the price to the list of details
        dets.append(price.contents[1].text.strip().split(" ")[1])

        #getting the information on the other attributes of the appartement (e.g. bagni, piano )
        for details in app.find_all(class_="lif__data"):
            det = details.contents[0]
            dets.append(det.get_text().strip())
        
        #getting the description of the current listing and appending it to the list of details we want to scrape.
        #will only be executed if the listing provides all the information we need.
        if len(dets) == 6:
            desc = getDescription(link, i)
            if desc is None:
                continue
            else: 
                dets.append(desc)
            
        #appending the list of details for the current apartement to the list of the
        #details of the other appartement on the current page of the search result.
        #We don't add if there is some information missing that we are going to need.
        if len(dets) == 7:
            detlist.append(dets)
    
    return detlist

def getDescription(link, i):
    #setting timer so we dont get blocked. We only need it in this if statement as we are not getting the description
    #if some other information (e.g. bagni) is missing.
    time.sleep(2)
    #list with links we couldn't get the description from (diagnostic purposes)
    dead_desc= []
    
    #getting up beautifulsoup and requests so we can crawl the url passed to the function
    desc_cont = requests.get(link)
    desc_soup = bs(desc_cont.text, 'html.parser')
    
    #getting the description of the houses
    desc = desc_soup.find('div',attrs={'class':'col-xs-12 description-text text-compressed','role':'contentinfo','aria-expanded':'false'})
    
    #if the description exists the functions cleans and then returns it, otherwise it appends the announcement at the list of the failed descriptions
    if desc is not None:
        desc = desc.get_text()  
        desc=re.sub('\n',' ',desc)
        desc=re.sub('_',' ',desc)
        desc=re.sub('\t','',desc)
        print(i, data.shape[0], link, desc)
        return desc
    else: 
        dead_desc.append([i, link])
        return None
    
'''
Below you can find the functions we used for the processing of the text in the descriptions. We used them e.g. for the calculation of the tfidf or the creation of our word clouds.
'''

def cleanData(rawData, lang='italian'):
        tokenizer = RegexpTokenizer(r'\w+')
        stop_words = set(stopwords.words(lang))

        # get words lowercased
        t0 = rawData.lower()
        # remove puctuations
        t1 = tokenizer.tokenize(t0)

        # remove stop words
        t2 =[]
        t2 = [t1[i] for i in range(0,len(t1)) if t1[i] not in stop_words]

        # stemm words
        t3 = [stemmer.stem(t2[i]) for i in range(0, len(t2))]

        # remove nummbers and strings starting with numbers
        t4 = [t3[i] for i in range(0, len(t3)) if t3[i][0].isdigit()==False]

        return(t4)

#modified cleaning function for the word cloud
def cleanDataWc(rawData, lang='italian'):
        tokenizer = RegexpTokenizer(r'\w+')
        stop_words = set(stopwords.words(lang))
       
        # get words lowercased
        t0 = rawData.lower()
        
        # remove puctuations
        t1 = tokenizer.tokenize(t0)

        # remove stop words
        t2 =[]
        t2 = [t1[i] for i in range(0,len(t1)) if t1[i] not in stop_words]


        # remove nummbers and strings starting with numbers
        t3 = [t2[i] for i in range(0, len(t2)) if t2[i][0].isdigit()==False]

        return(t3)
    
'''
The functions we used for the calculation of the TfIdf matrix.
'''    
#creating the two dictionaries for the words of all the description    
def TermDictionaries(ColumnOfDerscriptions):
    Term_Id = {}
    Inv_Dict = {}
    num = 0
    #cleaning each description
    for i in ColumnOfDerscriptions:
        app = cleanData(i, lang='italian')
        
        #adding the unique words at the dictionaries, updating them: if the word is already in the Term_Id vocabulary the function only update the list of Inv_Dict, otherwise both dictionaries are updated assigning a key to the word and creating the list of announcements where it appears
        for w in set(app):
            if w not in Term_Id:
                Inv_Dict[len(Term_Id)] = [num]
                Term_Id[w]=len(Term_Id)        
            else:
                Inv_Dict[Term_Id.get(w)].append(num)
        num += 1
    return(Term_Id,Inv_Dict)

#calculating the Idf score
def Idf(word,Inv_Dict,Term_Id,Data):
    key = Term_Id[word]
    Idf = {}
    n = len(Data)
    Idf = math.log10(n/len(Inv_Dict[key]))
    return Idf

#calculating the Tf score
def Tf(word,ann):
    Tf = ann.count(word)/len(ann)
    return Tf        

'''
The function we used to calculate the jaccard similary.
'''

def compute_jaccard(user1_vals, user2_vals):
    #given two arrays, the function transforms them into set to calculate the intersetion and the union
    user1_vals = set(user1_vals)
    user2_vals = set(user2_vals)
    intersection = user1_vals.intersection(user2_vals)
    union = user1_vals.union(user2_vals)
    #calculating the Jaccard Similarity
    jaccard = len(intersection)/float(len(union))
    return jaccard      

'''
Functions used for our own k-means calculation
'''

#function to calculate the euclidean distance between two points
def eucDist(x, y):
    if len(x) != len(y):
        print("The given points aren't in the same dimension")
    else:
        sq_dist = 0
        for i in range(len(x)):
            sq_dist += (x[i] - y[i])**2
        sq_dist = math.sqrt(sq_dist)
        return sq_dist

#function to get k random points to be used at centroids
def get_centroids(matrix, k):
    old_centroids = []

    #get k random numbers in the range of the length of the array
    cents = np.random.randint(len(matrix), size=k)
    #use the k random numbers to take the rows matching the random values of for k
    centroids = matrix[cents,:]
    if len(centroids) !=k:
        while old_centroids.all(centroids) == True:
            cents = np.random.randint(len(matrix), size=k)
            centroids = matrix[cents,:]
        old_centroids.append(centroids)
    else:
        old_centroids.append(centroids)
    return centroids


#function to calculate the sum of distances of all points to the centroids, only smallest distance to one of the centroids to find the smallest overall sum of distances.
def getMinDistanceCentroids(matrix, centroids):
    dist = 0
    for point in matrix:
        dists = []
        for centroid in centroids:
            dists.append(eucDist(point, centroid))
        dists_a = np.asarray(dists)
        dist += min(dists_a,key=float)
    return dist

#function to create a list that cointains the number of the centroid to which the current point is the closest. 
def getMinClusters(matrix,centroids):
    clust = []
    for point in matrix:
        dists = []
        i = 0
        for centroid in centroids:
            dists.append(eucDist(point, centroid))
        clust.append(dists.index(min(dists)))
    return clust

#main function to calculate the clusters
def getClusters(matrix, k_clusters, iterations):
    old_centroids = []
    min_centroids = []
    dist_sum = 0
    min_dist = 0
    for i in range(iterations):
        #get new centroids
        centroids = get_centroids(matrix, k_clusters)
        #calculate the sum of distances of all points to the current centroid
        dist_sum = getMinDistanceCentroids(matrix, centroids)
        #for the first iteration set the minimum sum of distances to the first sum of distances and the same for min_centroids
        if i == 0:
            min_dist = dist_sum
            min_centroids = centroids
        #if the sum of minimum distances is smaller then the one we saved before, we update the minimum sum and the minimum centroids
        if dist_sum < min_dist:
            min_centroids = centroids
            min_dist = dist_sum
    #after all iterations are done, we get a list from 1 to n (number of points to cluster). Every item in the list gets assigned the value of the centroid which it is the closest to. 
    min_clust = getMinClusters(matrix, min_centroids)
    return min_clust, min_centroids


