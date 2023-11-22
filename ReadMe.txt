Objectives
Estimate and analyze geolocation tweets of hot topics extracted based on specific locations.
To collect a considerable amount of geolocation tweets for public Twitter users.
To estimate and analyze the tweets collected based on their subject matter and the location they issued from.
Build a prediction model for the after-mentioned objective-based deep neural network.

Towards the end of chapter 1

Methodology
To extract the features of the location-specific tweets and their related properties, DNN was used to acquire regional features while the latter can extract global features from tweet text. As a result, location-specific characteristics may be retrieved quickly by using this deep learning approach. The three attributes employed to complete the prediction job were the screen name, tweet content, and user profile location.

How data was collected

This work studies a multi-city classification of text data in Deep learning by creating a topic-building model. The goal is to classify the tweets for each location into a specific category. Different cities might have similar tweets, the problem can be solved by DNNs through a set of algorithms. Similar tweets can be easily placed in the related category when this algorithm is in place.

The model used comprised 15000 tweets retrieved from 15 cities in Saudi Arabia. These cities are Riyadh, Mecca, Medina, Abha, Jeddah, Al Khobar, Jubail, Taif, Dammam, Dhahran, Al Qatif, Tabuk, and Najran.


Steps
Preparing the text data for training.
We started by creating a custom corpus of the text data collected from 15 cities.
The 15 datasets were compiled and saved in a folder as City_Datasets.csv. 
We then ran each dataset through a Data processing program and saved the processed dataset


Datasets Preprocessing
Our model, which depends on TF-IDF vectors, counts the exact spellings of terms in a document. Hence, texts that carried the same meaning would have had different TF-IDF vector representations if they were spelled differently or applied different words. This might be challenging in our documents as our model seek similarity comparison since it relies on the counts of tokens. 
We, therefore, examined the normalization approaches such as stemming and lemmatization to preprocess our datasets with a collection of words with similar spellings and meanings. Our model labeled each of these datasets with their lemma or stem and then processed these new tokens instead of the original words used on Twitter.
All the 15 datasets were passed through our first data processing program named as Preprocessingcsv.py.

Our models were consequently ready for DNNs training.

Our task was to build a model that would extract topics from each of the datasets. Specifically, the work looked at the unique keywords or topics and decided on the most rightful. This revolved around dataset tagging and clustering. We included numerical representations of words that capture the importance or information content of the words they represent. 

Similarly, we used three natural language processing principles to represent the meaning of words in a document, namely:

i) Bags of words(BOG) - vectors of word frequencies
ii) Frequency of words
III)Term frequency times inverse document frequency(TF-IDF) vectors - scores of words that represent their performance


Algorithm for determining topics
The algorithm used the judgement call of words with similar contributions to a topic. Our model performed a full search by creating indexes on the full text. There are two commonly used  LSA models in NLP; Linear Discriminant Analysis(LDA) and Latent Dirichlet Allocation(LDiA).  The former breaks down texts into a single topic while the latter can create a specified number of topics. Since LDA is unidimensional, it uses a supervised algorithm that is more accurate than LDiA.  Topics were set to 3 per each topic model, and the results for Mecca were as follows.

2022-05-17 20:42:50.001 LdaModel lifecycle event {'msg': 'trained LdaModel<num_terms=2666, num_topics=3, decay=0.5, chunksize=2000> in 21.18s', 'datetime': '2022-05-17T20:42:50.001022', 'gensim': '4.2.0', 'python': '3.7.13 (default, Apr  7 2022, 05:38:27) \n[GCC 10.2.1 20210110]', 'platform': 'Linux-5.4.170+-x86_64-with-debian-11.3', 'event': 'created'}
2022-05-17 20:42:50.001 topic #0 (0.333): 0.061*"mecca" + 0.060*"rt" + 0.054*"umrah" + 0.054*"photo" + 0.054*"trip" + 0.053*"brother" + 0.053*"local" + 0.053*"really" + 0.053*"photographer" + 0.053*"catch"
2022-05-17 20:42:50.001 topic #1 (0.333): 0.038*"n" + 0.020*"s" + 0.020*"rt" + 0.016*"afp" + 0.015*"stone" + 0.012*"mecca" + 0.012*"क" + 0.010*"islam" + 0.010*"site" + 0.009*"ramadan"
2022-05-17 20:42:50.001 topic #2 (0.333): 0.053*"mecca" + 0.039*"t" + 0.037*"co" + 0.035*"rt" + 0.033*"http" + 0.029*"n" + 0.014*"brother" + 0.014*"use" + 0.014*"shade" + 0.013*"sister"

The bigger the number before the dot product of the TF-IDF vector, the closer the word is to the topic analysis. For example, for the first topic(topic #0), the words that form the topic analysis have the following weights:
mecca : 0.061*, rt: 0.060*, photo: 0.054*, photographer: 053*, local: 053*
We can conclusively state that the topic in Mecca is mostly photography.
Also, we can compare the three topics and tell how close their meanings are. Based on the results returned by our topic vectors, we can see how similar the three topics are.

We leveraged a useful Latent Semantic Approach(LSA) TF-IDF matrix that represented the meanings that words shared in common. Our machine learning model could tell which words belonged together despite the transformation of two different languages (English and Arabic).

Although calculating the frequency or using Bag Of Words techniques are straightforward approaches that determine the topics by occurrences of words, TF-IDF vectors have better results. The text information gathered by performing a BOW is enough to provide the original topic/intent of a model but some words are not quite informative tokens. Twitter’s data can have a lot of informal text whose spelling may vary broadly. This was examined through the normalization approach covered below. Furthermore, Twitter’s hashtags capability was handy in our machine learning approach as we used it for our self-labeled dataset.

Our option focused on the LDA(Linear discriminant analysis) algorithm where we broke down the datasets into choices of three topics.

Below is an overview of our LDA classifier in action.

 Code here


From the three-topic option, the weighting of words helped to analyze the best topic. This helped us determine words that are directly the opposite of the topic. Therefore, we made the selection of the topic vectors manually by hand.

Gensim
This library is primarily used for topic modeling and finding document similarities. We used it to calculate and print the coherence of each topic. It comes with a useful inbuilt library(LDA) - a model that statistically explains why some data parts are similar. This model ensured that we could update new datasets(15 Datasets), in real time each at a time during the topic analysis. 



Discussion
NLP researchers found an algorithm for revealing the meaning of word combinations and computing vectors to represent this meaning known as the latent semantic analysis (LSA). This tool not only represents the meaning of words as vectors but also represents the meaning of entire documents. We used this technology to create semantic and topic vectors. The semantic vectors were set to identify the words that best represent the subject of a dataset. With this vector of words and their relative importance, we were able to achieve the most meaningful words for our documents.
Most of the words our topic model returned were found to be predominant in the dataset. For instance, In Mecca, the following were the eight heavily used words.


But we needed to score the meanings, the topics, and the words used. Our word-to-word TF-IDF vector or matrix helped us to identify the importance of words. 
TF-IDF vectors (term frequency-inverse document frequency vectors) assisted in our estimation of the importance of words in a chunk of text. TF-IDF vectors and matrices tell how important each word is to the overall meaning of a bit of text in a document collection.
These TF-IDF “importance” scores worked not only for words but also for short sentences, throughout our datasets. In this case, our topic model returned these words as the best topics.







Convert JSON files into a csv format by visiting data.page.(Our data has been converted)

Preprocess the data and save it using the Preprocessing.py app above.

Run the csv file through the (Topic Modeling algorithm)- only change the location/name of the dataset


