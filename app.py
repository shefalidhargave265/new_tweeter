import streamlit as st
import pandas as pd
import tweepy
import pickle
import time  # Import the time module

# Function to load sentiment analysis data
def load_sentiment_data():
    with open('data.pkl', 'rb') as f:
        df = pickle.load(f)
    return df

# Function to perform sentiment analysis
def perform_sentiment_analysis(model):  # Pass the model as a parameter
    
    st.title('Twitter Sentiment Analysis')

    tweet = st.text_input('Enter your tweet')

    submit = st.button('Predict')

    if submit:
        start = time.time()
        prediction = model.predict([tweets])
        end = time.time()
        st.write('Prediction time taken: ', round(end-start, 2), 'seconds')

        st.write(prediction[0])  # Display the prediction result
        
# Function to extract data from Twitter
def extract_twitter_data():
    st.title("Twitter Data Extraction")

    # Sidebar options
    st.sidebar.title("Twitter Data Extraction")
    searchTerm = st.sidebar.text_input("Enter Keyword/Tag to search about:")
    NoOfTerms = st.sidebar.number_input("Enter how many tweets to search:", value=10)

    # Authenticate with Twitter API
    auth = tweepy.OAuthHandler("consumerKey", "consumerSecret")
    auth.set_access_token("accessToken", "accessTokenSecret")
    api = tweepy.API(auth)

    # Search for tweets
    if st.sidebar.button("Extract Tweets"):
        tweets = tweepy.Cursor(api.search_tweets, q=searchTerm).items(NoOfTerms)
        st.success("Tweets extracted successfully!")

        # Create a list to store tweet data
        tweet_data = []

        # Extract tweet details and add to tweet_data list
        for tweet in tweets:
            tweet_data.append({
                "User": tweet.user.screen_name,
                "Text": tweet.text,
                "Created At": tweet.created_at
            })

        # Convert tweet_data list to DataFrame
        tweet_df = pd.DataFrame(tweet_data)

        # Display tweet DataFrame
        st.subheader("Extracted Tweets")
        st.write(tweet_df)

        # Load sentiment analysis data
        sentiment_df = load_sentiment_data()

        # Display sentiment-related information from sentiment_df
        st.subheader("Sentiment Analysis Results")
        st.write(sentiment_df[['text', 'polarity', 'sentiment']])

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select a page", ["Sentiment Analysis", "Twitter Data Extraction"])

    # Load the sentiment analysis model
    #with open('data.pkl', 'rb') as f:
    model = pickle.load(open('data.pkl', 'rb'))

    if page == "Sentiment Analysis":
        perform_sentiment_analysis(model)
    elif page == "Twitter Data Extraction":
        extract_twitter_data()

if __name__ == "__main__":
    model = pickle.load(open('data.pkl', 'rb'))

    main()
