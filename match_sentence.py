import csv
import requests
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the index from the index.json file
def load_index(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    index = []
    for outer_index, (outer_key, inner_dict) in enumerate(data.items()):
        sentence = inner_dict.get("sentence")
        embedding = np.array(inner_dict.get("embedding"))
        index.append({"sentence": sentence, "embedding": embedding})
    
    return index

# Calculate cosine similarity between query embedding and index embeddings
def calculate_similarity(query_embedding, index_embeddings):
    similarities = cosine_similarity([query_embedding], index_embeddings)
    return similarities[0]

# Find top k matches
def find_top_matches(query_embedding, index, k=5):
    similarities = calculate_similarity(query_embedding, [item['embedding'] for item in index])
    top_indices = np.argsort(similarities)[::-1][:k]
    top_matches = [(index[i]['sentence'], similarities[i]) for i in top_indices]
    return top_matches

# Your OpenAI API key
api_key = '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'

# The API URL for embeddings
url = "https://api.openai.com/v1/embeddings"

# The headers to provide, including your API key for authorization
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Read input queries
queries = [
    "Selada Hijau 500 gr",
    "Jamur Enoki 500 Gram",
    "Cabe Hijau Besar 500 Gram",
    "Ayam Negeri Utuh 1 Kg",
    "Kacang Hijau 250 gr",
    "Tomat 10 Kilo"
]

# Load index from index.json
index = load_index("/tmp/index.json")  # Adjust path as necessary

csv_file_path = "/tmp/results.csv"  # Adjust path as necessary
with open(csv_file_path, mode='w', newline='') as csvfile:
    fieldnames = ['Query', 'Sentence Number', 'Match Sentence', 'Similarity']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Find top matches for each query
    for query in queries:
        data = {
            "model": "text-embedding-ada-002",
            "input": query
        }

        response = requests.post(url, headers=headers, data=json.dumps(data))
        response_json = response.json()

        # Extract query embeddings
        query_embedding = response_json['data'][0]['embedding']

        top_matches = find_top_matches(query_embedding, index, k=5)

        # Write results to CSV
        for rank, (sentence, similarity) in enumerate(top_matches, start=1):
            writer.writerow({'Query': query, 'Sentence Number': rank, 'Match Sentence': sentence, 'Similarity': similarity})
        writer.writerow({})  # Add an empty row to separate queries

print(f"Results saved to {csv_file_path}")
