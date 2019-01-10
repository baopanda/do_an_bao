from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

corpus = [
   'This is the first document.',
   'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]
print(corpus)
vectorizer = TfidfVectorizer()
transformed_x_train = vectorizer.fit_transform(corpus).toarray()
print(transformed_x_train)