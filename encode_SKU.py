import requests
import json

# Your OpenAI API key
api_key = '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'

def write_index(sentences, embeddings, path="/tmp/index.json"):
    index = {}
    for i, (s, e) in enumerate(zip(sentences, embeddings)):
        index[i] = {
            "sentence": s,
            "embedding": e
        }
    with open(path, 'w') as f:
        json.dump(index, f, indent=2)


# The API URL for embeddings
url = "https://api.openai.com/v1/embeddings"

# The headers to provide, including your API key for authorization
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Read input sentence from tmp/sku.txt
with open("sku.txt", "r") as file:  # Adjust the path to your file
    sentences_list = file.read().splitlines()

embeddings_list = []

# The data payload with the model and input text
for sentence in sentences_list:
    data = {
        "model": "text-embedding-ada-002",
        "input": sentence
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    # Making the POST request to the API
    if response.status_code == 200:
        try:
            embeddings_list.append(response.json()["data"][0]["embedding"])
        except KeyError:
            print(f"Embedding not found in the response for '{sentence}': {response.json()}")
    else:
        # Printing error message if something went wrong
        print(f"Failed to fetch embeddings for '{sentence}': {response.text}")

# Save the result into /tmp/index.json
write_index(sentences_list, embeddings_list, "/tmp/index.json")
print("Embeddings saved successfully to /tmp/index.json")
