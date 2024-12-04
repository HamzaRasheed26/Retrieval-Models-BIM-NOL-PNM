import os
import json
from itertools import combinations
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class ProximalNodesModel:
    def __init__(self):
        # Initialize an empty graph structure
        self.graph = {}
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def add_node(self, node):
        """Add a node to the graph if it doesn't exist."""
        if node not in self.graph:
            self.graph[node] = set()

    def add_edge(self, node1, node2):
        """Add an edge between two nodes."""
        self.add_node(node1)
        self.add_node(node2)
        self.graph[node1].add(node2)
        self.graph[node2].add(node1)

    def build_network(self, docs_folder):
        """Build the network by processing structured documents in JSON format."""
        for filename in os.listdir(docs_folder):
            if filename.endswith(".json"):
                file_path = os.path.join(docs_folder, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    document = json.load(file)
                    doc_id = document["title"]
                    self.add_node(filename)  # Add document as a node
                    
                    # Process terms from document content
                    all_terms = set()
                    for section in document["sections"]:
                        terms = self.preprocess(section["content"])
                        all_terms.update(terms)
                        for term in terms:
                            self.add_edge(term, filename)  # Link terms to the document
                    
                        # Add relationships between terms
                        for i in range(len(terms) - 1):
                            term1, term2 = terms[i], terms[i + 1]
                            # for term1, term2 in combinations(all_terms, 2):
                            self.add_edge(term1, term2)

    def preprocess(self, text):
        """Preprocess text by tokenizing, removing stop words, and stemming."""
        tokens = word_tokenize(text.lower())
        filtered_tokens = [self.stemmer.stem(word) for word in tokens if word.isalnum() and word not in self.stop_words]
        return filtered_tokens

    def retrieve_connected_documents(self, proximal_nodes):
        """Retrieve documents directly or indirectly connected to the given proximal nodes."""
        visited = set()
        connected_documents = set()

        # Breadth-first search to explore the graph
        queue = list(proximal_nodes)
        while queue:
            current_node = self.preprocess(queue.pop(0))[0]
            if current_node not in visited:
                visited.add(current_node)
                for neighbor in self.graph.get(current_node, []):
                    if neighbor.endswith('.txt') or neighbor.endswith('.json'):
                        connected_documents.add(neighbor)
                    else:
                        queue.append(neighbor)
        return connected_documents

    def present_results(self, proximal_nodes):
        """Present the documents connected to the proximal nodes."""
        results = self.retrieve_connected_documents(proximal_nodes)
        print(f"\nRelevant Documents for Proximal Nodes: {', '.join(proximal_nodes)}")
        if results:
            for doc in results:
                print(f"Document: {doc}")
        else:
            print("No documents found for the specified proximal nodes.")

# Example Usage
def main():
    # Initialize the ProximalNodesModel
    model = ProximalNodesModel()

    # Folder containing structured documents
    docs_folder = "Structured_Docs"
    
    # Build the network
    print("Building the network...")
    model.build_network(docs_folder)
    print("Network built successfully!")
    
    # Define proximal nodes
    proximal_nodes = ["nasa", "space", "exploration"]
    
    # Present results
    model.present_results(proximal_nodes)

if __name__ == "__main__":
    main()
