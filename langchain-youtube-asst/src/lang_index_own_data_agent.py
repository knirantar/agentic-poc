from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

llm = ChatOpenAI(temperature=0.7, model="gpt-4o-mini")  # Use latest model

def create_vector_store_from_youtube(video_url):
    # Load the YouTube video
    loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=False)
    documents = loader.load()

    # Split the documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)

    # Create a vector store from the split documents
    vector_store = FAISS.from_documents(split_docs, embeddings)

    return vector_store

def get_response_for_query(query, vector_store, k):
    # Retrieve top-k docs
    docs = vector_store.similarity_search(query, k=k)
    docs_text = "\n".join([doc.page_content for doc in docs])

    # Define prompt
    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that can answer questions about YouTube videos 
        based on the video's transcript.
        
        Answer the following question: {question}
        By searching the following video transcript: {docs}
        
        Only use factual information from the transcript.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """,
    )

    # Combine prompt → LLM into one runnable
    runnable_sequence = prompt | llm

    # Run it
    response = runnable_sequence.invoke({
        "question": query,
        "docs": docs_text
    })

    return response.content.strip()  # .content for ChatOpenAI outputs
