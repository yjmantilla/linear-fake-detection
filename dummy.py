from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["The cat sat", "The cat sat on the mat"]

cv = CountVectorizer()
tfidf = TfidfVectorizer(use_idf=False)

count_matrix = cv.fit_transform(docs)
tfidf_matrix = tfidf.fit_transform(docs)

print("CountVectorizer output:")
print(count_matrix.toarray())

print("\nTfidfVectorizer output (use_idf=False):")
print(tfidf_matrix.toarray())


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Sample dataset
texts = [
    "The cat sat on the mat",
    "The dog chased the cat",
    "The bird flew over the house",
    "The fish swam in the pond",
    "The cat caught the mouse",
    "The dog barked at the mailman",
    "The bird built a nest",
    "The fish jumped out of the water"
]
labels = [0, 0, 1, 1, 0, 0, 1, 1]  # 0 for ground animals, 1 for flying/swimming

# Split the data
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.25, random_state=42)

# Define vectorizers
count_vec = CountVectorizer()
tfidf_vec = TfidfVectorizer()
tfidf_no_idf_vec = TfidfVectorizer(use_idf=False)
tfidf_no_norm_vec = TfidfVectorizer(use_idf=False, norm=None)

# Function to train and evaluate
def evaluate_vectorizer(vectorizer, name):
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train_vec, y_train)
    nb_pred = nb.predict(X_test_vec)
    nb_acc = accuracy_score(y_test, nb_pred)
    
    # SVM
    svm = SVC()
    svm.fit(X_train_vec, y_train)
    svm_pred = svm.predict(X_test_vec)
    svm_acc = accuracy_score(y_test, svm_pred)
    
    print(f"{name}:")
    print(f"  Naive Bayes Accuracy: {nb_acc:.2f}")
    print(f"  SVM Accuracy: {svm_acc:.2f}")
    print(f"  Sparsity: {X_train_vec.nnz / np.prod(X_train_vec.shape):.2%}")
    print()

# Evaluate each vectorizer
evaluate_vectorizer(count_vec, "CountVectorizer")
evaluate_vectorizer(tfidf_vec, "TfidfVectorizer")
evaluate_vectorizer(tfidf_no_idf_vec, "TfidfVectorizer (no IDF)")
evaluate_vectorizer(tfidf_no_norm_vec, "TfidfVectorizer (no IDF, no norm)")