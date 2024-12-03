import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# nltk.download('punkt')
# nltk.download('stopwords')

class BinaryIndependenceModel:
    def __init__(self, docs_folder):
        self.docs_folder = docs_folder
        self.documents = {}
        self.term_document_matrix = {}
        self.preprocessed_query = []
        self.term_list = []
        self.preprocessed_docs = {}
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
                    self.preprocessed_docs[filename] = self.preprocess(content)
        self.build_term_document_matrix()

    def build_term_document_matrix(self):
        """Build a binary term-document matrix."""
        # Collect all unique terms from preprocessed documents
        self.term_list = sorted(set(term for terms in self.preprocessed_docs.values() for term in terms))
        # Initialize the term-document matrix
        for doc, terms in self.preprocessed_docs.items():
            self.term_document_matrix[doc] = [1 if term in terms else 0 for term in self.term_list]

    def preprocess_query(self, query):
        """Preprocess the query and generate a binary vector."""
        self.preprocessed_query = self.preprocess(query)
        return [1 if term in self.preprocessed_query else 0 for term in self.term_list]

    def score_documents(self, query_vector):
        """Score documents based on Dice similarity."""
        scores = {}
        for doc, doc_vector in self.term_document_matrix.items():
            intersection = sum(min(q, d) for q, d in zip(query_vector, doc_vector))
            query_sum = sum(query_vector)
            doc_sum = sum(doc_vector)
            dice_score = (2 * intersection) / (query_sum + doc_sum) if (query_sum + doc_sum) != 0 else 0
            scores[doc] = dice_score
        return scores


    def rank_documents(self, scores):
        """Rank documents based on scores in descending order."""
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def retrieve_top_k_documents(self, query, k=3):
        """Retrieve the top-K documents for the query."""
        query_vector = self.preprocess_query(query)
        scores = self.score_documents(query_vector)
        ranked_docs = self.rank_documents(scores)
        return ranked_docs[:k]

if __name__ == "__main__":
    # Initialize BIM with document folder path
    bim = BinaryIndependenceModel(docs_folder="Docs")
    
    # Load and preprocess documents
    print("Loading documents...")
    bim.load_documents()
    print("Documents loaded successfully!")
    
    # Interactive query loop
    print("\nWelcome to the Document Retrieval System!")
    print("Type your query to retrieve documents or 'exit' to quit.")
    
    while True:
        # Get user query
        user_query = input("\nEnter your query: ").strip()
        if user_query.lower() == "exit":
            print("Exiting the Document Retrieval System. Goodbye!")
            break
        
        # Retrieve Top-K documents
        top_k = 3
        results = bim.retrieve_top_k_documents(user_query, k=top_k)
        
        # Display results
        if results:
            print(f"\nTop-{top_k} Documents for Query: '{user_query}'")
            for rank, (doc, score) in enumerate(results, start=1):
                print(f"Rank {rank}: {doc} with Score: {score:.4f}")
        else:
            print(f"No relevant documents found for the query: '{user_query}'.")

