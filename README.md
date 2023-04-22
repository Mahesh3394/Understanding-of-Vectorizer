# Understanding-of-Vectorizer

A vectorizer is a tool used in machine learning to convert textual or categorical data into numerical vectors that can be understood and processed by algorithms. The process of vectorization involves transforming raw data into a structured format that can be used for modeling and analysis. There are several types of vectorizers, including CountVectorizer, TF-IDF Vectorizer, and Word2Vec, each with its unique features and benefits. CountVectorizer is used to convert a collection of text documents to a matrix of token counts, while TF-IDF Vectorizer assigns weights to each term based on their importance in the document. Word2Vec is a neural network-based technique that creates a dense representation of words in a vector space. Vectorization is an essential step in many machine learning tasks, including text classification, clustering, and recommendation systems.

## BOW Vectorizer:

- Import the necessary libraries (e.g., numpy, pandas).
- Define a function to tokenize the text data (e.g., split the text into individual words).
- Create a vocabulary by iterating through all the text data and adding unique words to a set.
- Convert the vocabulary set to a list and sort it alphabetically.
- Create a dictionary where the keys are the words in the vocabulary and the values are their corresponding index positions in the sorted list.
- Create a numpy array of zeros with the dimensions (number of documents, length of vocabulary).
- Iterate through each document in the text data and tokenize it.
- For each token in the document, increment the corresponding index position in the numpy array by 1.
- Return the numpy array as the bag-of-words representation of the text data.

## TFIDF Vectorizer:

- Follow steps 1-5 from the BOW vectorizer description.
- Create a numpy array of zeros with the dimensions (number of documents, length of vocabulary).
- Iterate through each document in the text data and tokenize it.
- For each token in the document, increment the corresponding index position in the numpy array by 1.
- Calculate the term frequency (TF) for each word in each document by dividing the frequency of the word by the total number of words in the document.
- Calculate the inverse document frequency (IDF) for each word in the vocabulary by dividing the total number of documents by the number of documents containing the word, taking the logarithm of that quotient, and adding 1.
- Multiply the TF and IDF values for each word in each document to get the TF-IDF value.
- Return the numpy array as the TF-IDF representation of the text data.

