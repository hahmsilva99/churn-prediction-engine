import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# 1.connecting to Pinecone

PINECONE_API_KEY = "pcsk_j69eJ_MuWr6on4cV2oYeB3n5ejmv3CXugWszPdvU56Rz6AzMaB3m9vcF7dUubxx4dKhpd" 

print("Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("churn-offers")

# 2. Text -> Loading the Vector transform model  (Dimensions 384)
print("Embedding model eka load wenawa (Mekata podi welawak yai)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Offers when we give
offers = [
    {"id": "offer-1", "text": "20% Discount for the next 6 months to stay with us", "type": "discount"},
    {"id": "offer-2", "text": "Free upgrade to Premium High-Speed Internet for 1 year", "type": "upgrade"},
    {"id": "offer-3", "text": "Free Netflix and Spotify subscription bundle for 3 months", "type": "perk"},
    {"id": "offer-4", "text": "Dedicated 24/7 VIP customer support and zero router fees", "type": "service"}
]

# 4. transform the Offers into vectors and put them into database 
print("Uploading Offers into Pinecone...")
vectors_to_upsert = []
for offer in offers:
    # Transform the Text into numbers (vector)
    vector = model.encode(offer['text']).tolist()
    
    vectors_to_upsert.append({
        "id": offer['id'],
        "values": vector,
        "metadata": {"text": offer['text'], "type": offer['type']}
    })

# Pushing data into Pinecone 
index.upsert(vectors=vectors_to_upsert)
print("Task success!")