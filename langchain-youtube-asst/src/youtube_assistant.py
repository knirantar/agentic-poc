import streamlit as st
from lang_index_own_data_agent import create_vector_store_from_youtube, get_response_for_query

st.title("YouTube Video Query Assistant")

video_url = st.text_input("Enter the YouTube video URL:")
query = st.text_input("Enter your query about the YouTube video:")
if st.button("Get Response"):
    if query:
        # Create a vector store from the YouTube video
        vector_store = create_vector_store_from_youtube(video_url)
        
        # Get response for the query
        response = get_response_for_query(query, vector_store, k=5)
        st.write(f"🔍 **Response:** {response}")
    else:
        st.error("Please enter a query.")
else:
    st.write("Enter your query and click the button to get a response.")

