import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict

class NonOverlappedListModel:
    def __init__(self, docs_folder):
        self.docs_folder = docs_folder
        self.documents = {}
        self.preprocessed_docs = {}
        self.term_doc_map = defaultdict(set)
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def preprocess(self, text):
        """Preprocess text: tokenize, remove stop words, and stem."""
        tokens = word_tokenize(text.lower())
        filtered_tokens = [self.stemmer.stem(word) for word in tokens if word.isalnum() and word not in self.stop_words]
        return filtered_tokens

    def load_documents(self):
        """Load and preprocess documents from the specified folder."""
        for filename in os.listdir(self.docs_folder):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.docs_folder, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    self.documents[filename] = content
                    preprocessed_terms = self.preprocess(content)
                    self.preprocessed_docs[filename] = preprocessed_terms
                    for term in preprocessed_terms:
                        self.term_doc_map[term].add(filename)

    def retrieve_documents(self, terms):
        """Retrieve documents for each term and return a non-overlapping set."""
        non_overlapping_docs = set()
        for term in terms:
            stemmed_term = self.stemmer.stem(term.lower())
            if stemmed_term in self.term_doc_map:
                non_overlapping_docs.update(self.term_doc_map[stemmed_term])
        return non_overlapping_docs

# Example Usage
if __name__ == "__main__":
    # Initialize NonOverlappedListModel with document folder path
    nol_model = NonOverlappedListModel(docs_folder="Docs")
    
    # Load and preprocess documents
    print("Loading documents...")
    nol_model.load_documents()
    print("Documents loaded successfully!")
    
    # Specify terms of interest
    terms_of_interest = ["machine learning", "data visualization"]
    
    # Retrieve non-overlapping documents
    results = nol_model.retrieve_documents(terms=terms_of_interest)
    
    # Display results
    print(f"\nNon-Overlapping Documents for Terms: {', '.join(terms_of_interest)}")
    if results:
        for doc in results:
            print(doc)
    else:
        print("No documents found for the specified terms.")
