import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import argparse

# Load dataset and create embeddings (global for testing)
df = pd.read_csv('movies.csv')
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['plot'].tolist())

def search_movies(query, top_n=5):
    """Search for movies based on semantic similarity to the query."""
    # Encode the search query
    query_embedding = model.encode([query])
    
    # Calculate cosine similarity and scale to [0, 1]
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    similarities = (similarities + 1.0) / 2.0
    
    # Get top N results
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    # Return results as DataFrame
    results = df.iloc[top_indices].copy()
    results['similarity'] = similarities[top_indices]
    results = results.reset_index(drop=True)
    
    return results

def _main():
    parser = argparse.ArgumentParser(description='Semantic search on movie plots')
    parser.add_argument('--query', required=True, help='Search query text')
    parser.add_argument('--top-n', type=int, default=5, help='Number of results to return')
    args = parser.parse_args()
    results = search_movies(args.query, top_n=args.top_n)
    # Print a concise table
    print(results[['title', 'similarity']])

if __name__ == '__main__':
    _main()