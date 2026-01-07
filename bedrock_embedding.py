# Embeddings capture the meaning of a piece of text in a series of numbers called a vector.
# We can then use these vectors to determine how similar pieces of text are to each other.
# We can use a vector database to store these embeddings and perform fast similarity searches. 
# Embeddings paired with a vector database are a core component of retrieval-augmented generation

import json
import boto3
from numpy import dot
from numpy.linalg import norm

# Define the function to get an embedding from Amazon Bedrock

def get_embedding(text):
    session = boto3.Session()
    bedrock = session.client(service_name='bedrock-runtime') #creates a Bedrock client
    
    response = bedrock.invoke_model(
        body=json.dumps({ "inputText": text }), 
        modelId="amazon.titan-embed-text-v2:0", 
        accept="application/json",
        contentType="application/json"
    )
    
    response_body = json.loads(response['body'].read())
    return response_body['embedding']
    
# Define the classes to store embeddings and the comparison results

class EmbedItem:
    def __init__(self, text):
        self.text = text
        self.embedding = get_embedding(text)

class ComparisonResult:
    def __init__(self, text, similarity):
        self.text = text
        self.similarity = similarity

# Define the function to compare the similarity of two vectors.
# This implements the Cosine Similarity  equation

def calculate_similarity(a, b): #See Cosine Similarity: https://en.wikipedia.org/wiki/Cosine_similarity
    return dot(a, b) / (norm(a) * norm(b))

# Build a list of embeddings from the items.txt file.

#Build the list of embeddings to compare
items = []

with open("items.txt", "r") as f:
    text_items = f.read().splitlines()

for text in text_items:
    items.append(EmbedItem(text))

# Compare embeddings and display lists to show how similar or different the various texts are.
#   A similarity value of 1 means exactly the same.
#   The smaller the similarity, the less similar are the embeddings.

for e1 in items:
    print(f"Closest matches for '{e1.text}'")
    print ("----------------")
    cosine_comparisons = []
    
    for e2 in items:
        similarity_score = calculate_similarity(e1.embedding, e2.embedding)
        
        cosine_comparisons.append(ComparisonResult(e2.text, similarity_score)) #save the comparisons to a list
        
    cosine_comparisons.sort(key=lambda x: x.similarity, reverse=True) # list the closest matches first
    
    for c in cosine_comparisons:
        print("%.6f" % c.similarity, "\t", c.text)
    
    print()

