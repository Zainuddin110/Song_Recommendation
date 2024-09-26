import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load the CSV file
file_path = 'songs.csv'  # Change this to your file path
songs_df = pd.read_csv(file_path)

# Fill NaN values with empty strings to prevent errors in text processing
songs_df['Genre'] = songs_df['Genre'].fillna('')
songs_df['Singer/Artists'] = songs_df['Singer/Artists'].fillna('')
songs_df['Movie'] = songs_df['Movie'].fillna('')

# Combine relevant features into a single column for comparison
songs_df['combined_features'] = songs_df['Genre'] + " " + songs_df['Singer/Artists'] + " " + songs_df['Movie']

# Vectorize the text data (convert text to numerical form)
count_vectorizer = CountVectorizer(stop_words='english')
count_matrix = count_vectorizer.fit_transform(songs_df['combined_features'])

# Compute cosine similarity between songs
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Function to get song recommendations based on song name, artist, or genre
def get_recommendations(query, cosine_sim=cosine_sim):
    # Filter by song name, artist, genre, or movie
    mask = (songs_df['Song-Name'].str.contains(query, case=False, na=False) |
            songs_df['Singer/Artists'].str.contains(query, case=False, na=False) |
            songs_df['Genre'].str.contains(query, case=False, na=False) |
            songs_df['Movie'].str.contains(query, case=False, na=False))
    
    if mask.any():
        # Get the index of the matching song(s)
        indices = songs_df[mask].index
        
        # Get pairwise similarity scores of all songs with the selected song(s)
        sim_scores = []
        for idx in indices:
            sim_scores += list(enumerate(cosine_sim[idx]))
        
        # Remove duplicates and sort the songs based on similarity scores
        sim_scores = sorted(list(set(sim_scores)), key=lambda x: x[1], reverse=True)
        
        # Get the indices of the top 25 most similar songs
        sim_scores = sim_scores[:25]  # Limit to top 5
        
        # Get the song indices
        song_indices = [i[0] for i in sim_scores]
        
        # Return the top 5 similar songs
        return songs_df[['Song-Name', 'Singer/Artists', 'Movie']].iloc[song_indices]
    else:
        return "Sorry, no song, artist, genre, or movie found matching that query. Please try another."

# Streamlit Chatbot Interface
def chatbot():
    st.title('Song Recommendation Chatbot')

    # Input for song name, artist, genre, or movie
    user_input = st.text_input("Enter a song name, genre, artist, or movie:")

    if user_input:
        # Provide song recommendations based on user input
        st.write(f"Looking for recommendations related to: **{user_input}**")

        # Call the recommendation function
        recommendations = get_recommendations(user_input)

        # Display the recommendations
        if isinstance(recommendations, pd.DataFrame):
            for index, row in recommendations.iterrows():
                st.write(f"**Song Name**: {row['Song-Name']}")
                st.write(f"**Singer/Artists**: {row['Singer/Artists']}")
                st.write(f"**Movie**: {row['Movie']}")
                st.write("---")
        else:
            st.write(recommendations)

# Run the chatbot
if __name__ == "__main__":
    chatbot()
