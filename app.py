pip install streamlit pandas scikit-learn
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the CSV file
file_path = 'songs.csv'  # Change this to your file path
songs_df = pd.read_csv(file_path)

# Combine relevant features into a single column for comparison
songs_df['combined_features'] = songs_df['Genre'] + " " + songs_df['Singer/Artists'] + " " + songs_df['Movie']

# Vectorize the text data (convert text to numerical form)
count_vectorizer = CountVectorizer(stop_words='english')
count_matrix = count_vectorizer.fit_transform(songs_df['combined_features'])

# Compute cosine similarity between songs
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Function to get song recommendations based on song name
def get_recommendations(song_name, cosine_sim=cosine_sim):
    try:
        # Get the index of the song that matches the song name
        idx = songs_df[songs_df['Song-Name'].str.contains(song_name, case=False, na=False)].index[0]
        
        # Get pairwise similarity scores of all songs with the selected song
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the songs based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the indices of the top 5 most similar songs
        sim_scores = sim_scores[1:6]  # Exclude the song itself

        # Get the song indices
        song_indices = [i[0] for i in sim_scores]

        # Return the top 5 similar songs
        return songs_df[['Song-Name', 'Singer/Artists', 'Movie']].iloc[song_indices]
    except IndexError:
        return "Sorry, no song found with that name. Please try another song."

# Streamlit Chatbot Interface
def chatbot():
    st.title('Song Recommendation Chatbot')

    # Input for song name or genre
    user_input = st.text_input("Enter a song name, genre, or artist:")

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
