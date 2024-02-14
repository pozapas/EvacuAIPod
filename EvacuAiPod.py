# langchain version
import streamlit as st
import pandas as pd
import requests
import openai
import tempfile
from typing import List
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
import os

# Function to download the podcast audio file
def download_audio(url):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            return tmp_file.name
    else:
        raise Exception(f"Error downloading file: {response.status_code}")

# Function to transcribe the podcast using OpenAI's Whisper API
def transcribe_podcast(file_path, api_key):
    openai.api_key = api_key
    with open(file_path, 'rb') as audio_file:
        transcript = openai.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1",
            response_format="text"
        )
    
    # Get the path to the system's temporary directory
    temp_dir = tempfile.gettempdir()
    transcript_file_path = os.path.join(temp_dir, "transcript.txt")
    
    # Write the transcript to a file named "transcript.txt" in the temporary directory
    with open(transcript_file_path, 'w') as file:
        file.write(transcript)
    
    return transcript

def add_paragraph_breaks(text, sentences_per_paragraph=7):
    sentences = text.split('. ')
    paragraphs = []
    current_paragraph = []

    for sentence in sentences:
        current_paragraph.append(sentence.strip())
        if len(current_paragraph) >= sentences_per_paragraph:
            paragraph_text = '. '.join(current_paragraph)
            # Check if the last sentence already ends with a period
            if not paragraph_text.endswith('.'):
                paragraph_text += '.'
            paragraphs.append(paragraph_text)
            current_paragraph = []

    # Add any remaining sentences as a paragraph, with the same end-period check
    if current_paragraph:
        paragraph_text = '. '.join(current_paragraph)
        if not paragraph_text.endswith('.'):
            paragraph_text += '.'
        paragraphs.append(paragraph_text)

    return '\n\n'.join(paragraphs)


def chat_with_podcast(openai_key, transcript1=None, podcast_choice=None):
    st.write(f"**Chat with the {podcast_choice} podcast**")
    if openai_key and transcript1:
        temp_dir = tempfile.gettempdir()
        transcript_file_path = os.path.join(temp_dir, "transcript.txt")
        with open(transcript_file_path, 'r') as file_data:
            transcript = file_data.read()
        
        transcript_with_paragraphs = add_paragraph_breaks(transcript)
        st.text_area("Transcript", transcript_with_paragraphs, height=300)
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,  # Consider adjusting chunk size based on your needs
            chunk_overlap=0,
            length_function=len,
        )
        chunks = text_splitter.split_text(transcript)

        # Create embeddings for chunks
        embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        llm = OpenAI(temperature=0, openai_api_key=openai_key)

        # User input area using st.text_input for compatibility
        user_input = st.chat_input(placeholder=f"Talk to 🎙️ {podcast_choice}", key='input')
        if user_input:
            # Find the most relevant chunks
            docs = knowledge_base.similarity_search(user_input)
            chain = load_qa_chain(llm, chain_type="stuff")  # Ensure you have the correct chain_type
            ai_response = chain.run(input_documents=docs, question=user_input)
            if prompt := user_input: # Prompt for user input and save to chat history
                                st.session_state.messages.append({"role": "user", "content": prompt})

            for message in st.session_state.messages: # Display the prior chat messages
                with st.chat_message(message["role"]):
                    st.write(message["content"])
            #st.write(ai_response)
            # If last message is not from assistant, generate a new response
            if st.session_state.messages[-1]["role"] != "assistant":
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        st.write(ai_response)
                        message = {"role": "assistant", "content": ai_response}
                        st.session_state.messages.append(message) # Add response to message history
    else:
        st.error("Please enter your OpenAI API key and ensure a transcript is available.")



def main():
    # Set app title and description
    st.set_page_config(page_title="EvacuAIPod", page_icon="🎙️", layout="wide")
    st.title("EvacuAIPod: AI-bot for Crowd Evacuation Podcasts")
    # Load the CSV file into a DataFrame
    @st.cache_data
    def load_data():
        return pd.read_csv("pod.csv")

    df = load_data()
    # Initialize conversation history in session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if "messages" not in st.session_state.keys(): # Initialize the chat messages history
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me a question about this episode!"}
    ]
    # Display the logo in the sidebar
    st.sidebar.image("evacuaipod.png", use_column_width=True) 
    # Navigation
    page = st.sidebar.selectbox("Navigate", ["Transcript", "Chat with Podcast"])
    # Add an "About" section in the sidebar
    expander = st.sidebar.expander("**About EvacuAIPod**")
    with expander:
                expander.write('''
                        EvacuAIPod is an innovative web application that enables users to search and discover podcasts focusing on crowd evacuation strategies. With the power of targeted keyword searches, listeners can effortlessly find and listen to discussions that pique their interest. Beyond just listening, EvacuAIPod offers an interactive experience by allowing users to engage in conversations with the content of the podcasts themselves. This unique feature opens up deeper insights and a more personalized understanding of the crucial topics of AI technology and safety measures. EvacuAIPod is the go-to platform for anyone looking to delve into the world of emergency preparedness and crowd management through the lens of interactive and informative podcast sessions.
                ''')
    # Create a text input for the OpenAI key
    with st.sidebar.expander("**OpenAI key**"):
        openai_key = st.text_input("Enter your [OpenAI key](https://platform.openai.com/api-keys):", key="openai_input" , type="password")

    expander = st.sidebar.expander("**Contact US**")
    with expander:
                expander.write('''
                        For further information and any enquiries, or if you are interested in collaborating with us, please do not hesitate to get in touch with [Amir Rafe](mailto:amir.rafe@usu.edu).
                ''')
    
    if page == "Transcript":
        # Filter the DataFrame based on the keywords
        # User input for search keywords
        keywords = st.text_input("**Enter keywords to search for podcasts. Separate multiple keywords with spaces.**")
        if keywords:
            search_keywords = keywords.split()
            filtered_dfs = []  # List to hold DataFrames to concatenate
            for keyword in search_keywords:
                matching_df = df[df.apply(lambda x: x.str.contains(keyword, case=False, na=False)).any(axis=1)]
                filtered_dfs.append(matching_df)
            filtered_df = pd.concat(filtered_dfs).drop_duplicates()  # Concatenate all matching DataFrames and drop duplicates

            if not filtered_df.empty:
                # Your existing code for handling the non-empty filtered DataFrame
                st.session_state.filtered_df = filtered_df
                for _, row in filtered_df.iterrows():
                    # Generate Listen Notes player embed code
                    listen_notes_player = f'<iframe src="{row["Link"]}/embed/" height="180px" width="100%" style="width: 1px; min-width: 100%;" frameborder="0" scrolling="no" loading="lazy"></iframe>'
                    # Display podcast information with improved formatting
                    st.markdown(f'**Podcast:** [{row["Podcast"]}]({row["Podcast link"]})')
                    st.markdown(f'**Episode:** {row["Episode"]}')
                    st.markdown(f'**Host:** {row["Host"]} | **Guest(s):** {row["Guest(s)"]}')
                    st.markdown(listen_notes_player, unsafe_allow_html=True)
                    st.markdown("---")
            else:
                st.write("No podcasts found with the given keyword(s).")
        else:
            st.write("")

        if 'filtered_df' in st.session_state and not st.session_state.filtered_df.empty:
            transcript1=[]
            with st.form(key="podcast_form"):
                st.write("**Choose a podcast from the list and request a transcript:**")
                st.write("*Note: The process of creating a transcript can take up to 3 minutes, depending on the duration of the podcast.*")
                # Create a select box for the podcast using the filtered DataFrame from session state
                podcast_options = st.session_state.filtered_df["Episode"].tolist()
                podcast_choice = st.selectbox("Podcast:", podcast_options, key="podcast_select")
                st.session_state['podcast_choice'] = podcast_choice            
                # Create a submit button
                submit_button = st.form_submit_button(label="Submit")
                
                if submit_button:
                    st.session_state.conversation_history = []
                    st.session_state.messages = []
                    if openai_key:  
                        try:
                            podcast_url = st.session_state.filtered_df.loc[st.session_state.filtered_df['Episode'] == podcast_choice, 'Audio'].iloc[0]
                            audio_file_path = download_audio(podcast_url)
                            transcript1 = transcribe_podcast(audio_file_path, openai_key)
                            st.session_state['transcript1'] = transcript1  # Store transcript in session state
                            transcript_with_paragraphs = add_paragraph_breaks(transcript1)
                            st.text_area("Podcast Transcription", transcript_with_paragraphs, height=400)
                            st.write("*Note: To chat with the selected podcast, use the sidebar to navigate to the 'Chat with Podcast' page.*")
                        except Exception as e:
                            st.error(f"An error occurred: {e}")
                    else:
                        st.error("Please enter your OpenAI API key.")
            # If filtered_df isn't in session state or is empty, display a message
            st.write("")
    elif page == "Chat with Podcast":
        # Check if 'transcript1' is in session state before calling the function
        if 'transcript1' in st.session_state:
            chat_with_podcast(openai_key, st.session_state['transcript1'] , st.session_state['podcast_choice'])
        else:
            st.error("Please transcribe a podcast first.")
        

if __name__ == '__main__':
    main()