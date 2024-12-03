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

    def preprocess_text(self, text):
        """Preprocess text by tokenizing, removing stop words, and stemming."""
        tokens = word_tokenize(text.lower())
        filtered_tokens = [self.stemmer.stem(word) for word in tokens if word.isalnum() and word not in self.stop_words]
        return filtered_tokens

    def load_document(self, filename):
        """Load and preprocess a single document."""
        file_path = os.path.join(self.docs_folder, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            preprocessed_terms = self.preprocess_text(content)
            return content, preprocessed_terms

    def build_term_document_map(self):
        """Create a mapping of terms to the documents they appear in."""
        for filename in os.listdir(self.docs_folder):
            if filename.endswith(".txt"):
                content, preprocessed_terms = self.load_document(filename)
                self.documents[filename] = content
                self.preprocessed_docs[filename] = preprocessed_terms
                for term in preprocessed_terms:
                    self.term_doc_map[term].add(filename)

    def retrieve_documents_for_term(self, term):
        """Retrieve documents containing all the specified terms."""
        # Preprocess the input term to handle multiple words
        stemmed_terms = self.preprocess_text(term)
        
        # If no terms remain after preprocessing, return an empty set
        if not stemmed_terms:
            return set()
        
        # Initialize the result with the documents of the first term
        docs = self.term_doc_map.get(stemmed_terms[0], set())
        
        # Intersect with documents of subsequent terms
        for stemmed_term in stemmed_terms[1:]:
            docs = docs.intersection(self.term_doc_map.get(stemmed_term, set()))
        
        return docs

    def retrieve_non_overlapping_documents(self, terms):
        """Retrieve a non-overlapping set of documents for multiple terms."""
        non_overlapping_docs = {}  # Dictionary to store documents and their matching terms

        for term in terms:
            term_docs = self.retrieve_documents_for_term(term)  # Get documents for the term
            for doc in term_docs:
                # Initialize the list if the document is not already in the dictionary
                if doc not in non_overlapping_docs:
                    non_overlapping_docs[doc] = []
                # Append the term to the list of matched terms for the document
                non_overlapping_docs[doc].append(term)
        
        return non_overlapping_docs


# Example Usage
def main():
    # Initialize the model
    nol_model = NonOverlappedListModel(docs_folder="Docs")
    
    # Build term-document map
    print("Building term-document map...")
    nol_model.build_term_document_map()
    print("Term-document map built successfully!")
    
    # Specify terms of interest
    terms_of_interest = ["machine learning", "artificial intelligence"]
    
    # Retrieve non-overlapping documents
    print(f"\nRetrieving documents for terms: {', '.join(terms_of_interest)}")
    results = nol_model.retrieve_non_overlapping_documents(terms_of_interest)
    
    print(f"\nNon-Overlapping Documents")
    if results:
        for doc, matched_terms in results.items():
            # Display document name and the matched terms
            print(f"Document: {doc}, Matches: {', '.join(matched_terms)}")
    else:
        print("No documents found for the specified terms.")

if __name__ == "__main__":
    main()
