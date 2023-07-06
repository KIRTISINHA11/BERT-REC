import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel

# Step 1: Collect data
df = pd.read_csv('/content/influencers_data_cleaned2.csv', engine="python", error_bad_lines=False)
df = df.dropna()
df = df.reset_index()
df['content'] = df['content'].astype('string')
df['name'] = df['name'].astype('string')
df = df[:5000]

# Step 2: Preprocess data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def preprocess(text):
    # Remove unnecessary characters
    text = str(text).replace('\n', ' ').replace('\r', '')
    # Tokenize text into sequences of words
    tokens = tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True)
    # Convert tokens to PyTorch tensors
    input_ids = torch.tensor(tokens).unsqueeze(0)
    # Generate embeddings for the text
    with torch.no_grad():
        outputs = model(input_ids)
        last_hidden_states = outputs[0][:, 0, :]
    return last_hidden_states.numpy().reshape(1, -1)

def recommend_posts(user_name, post_embeddings, n=10):
    # Get embeddings for user's posts
    user_posts = df[df['name'] == user_name]['content']
    user_post_embeddings = []
    for post in user_posts:
        user_post_embeddings.append(preprocess(post))
    user_post_embeddings = np.concatenate(user_post_embeddings, axis=0)

    # Calculate mean embedding for user's posts
    user_profile_embedding = np.mean(user_post_embeddings, axis=0)

    # Calculate cosine similarity between user profile and all post embeddings
    similarity_scores = np.dot(post_embeddings, user_profile_embedding.T) / (np.linalg.norm(post_embeddings, axis=1) * np.linalg.norm(user_profile_embedding))

    # Rank posts by similarity score
    rankings = np.argsort(np.ravel(similarity_scores))[::-1]

    # Exclude user's own posts from recommended posts
    user_post_ids = df[df['name'] == user_name].index
    recommended_post_ids = np.delete(rankings, np.where(np.isin(rankings, user_post_ids)))

    # Return top N posts as recommendations
    recommended_posts = df.iloc[recommended_post_ids[:n]][['content']]
    recommended_embeddings = post_embeddings[recommended_post_ids[:n]]
    return recommended_posts, recommended_embeddings

def main():
    st.title("Post Recommender")
    user_name = st.text_input("Enter user name:")
    if st.button("Recommend"):
        if user_name:
            user_profile_embedding = preprocess(' '.join(df[df['name'] == user_name]['content'].values.tolist()))
            recommendations = recommend_posts(user_name, post_embeddings, n=10)
            st.subheader(f"Recommended posts for {user_name}:")
            st.table(recommendations[0])
        else:
            st.warning("Please enter a user name.")

if __name__ == '__main__':
    main()
