# langchain version
import os
import streamlit as st
import pandas as pd
import requests
import openai
import tempfile
from typing import List
import smtplib
import re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from gnews import GNews
from newspaper import Article

default_groq_key = st.secrets["groq_key"]
gmail_key = st.secrets["gmail_key"]

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

def read_transcript(episode):
    with open(f"Transcripts/{episode}.txt", 'r', encoding='utf-8') as file_data:
        transcript = file_data.read()
    return transcript


def chat_with_podcast(openai_key, groq_key, chatbot, transcript1=None, podcast_choice=None):
    st.write(f"**Chat with {podcast_choice}**")
    if transcript1:
        st.text_area("Transcript", transcript1, height=300)
        # Split into chunks
        if chatbot in ['Groq (llam3-8B)', 'Groq (gemma-7b)']:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=5000,  
                chunk_overlap=0,
                length_function=len,
            )
        else:
             text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  
            chunk_overlap=0,
            length_function=len,
            ) 
        chunks = text_splitter.split_text(transcript1)

        if chatbot in ['Groq (llam3-8B)', 'Groq (gemma-7b)']:
            if not st.session_state.groq_key_entered and st.session_state.questions_asked > 1:
                st.error("Please enter your own Groq API key to continue using the service.")
                return  # Stop further execution until a new key is provided
            else:
                groq_key_to_use = groq_key if st.session_state.groq_key_entered else default_groq_key
        elif chatbot == 'ChatGPT':
            if not openai_key:
                st.error("Please enter your OpenAI API key to use ChatGPT.")
                return  # Stop further execution until a key is provided
            else:
                groq_key_to_use = openai_key 
        
        if chatbot=='Groq (llam3-8B)':
            embeddings = HuggingFaceEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)
            chat = ChatGroq(temperature=0, groq_api_key=groq_key_to_use, model_name="llama3-8b-8192")
        elif chatbot=='Groq (gemma-7b)':
            embeddings = HuggingFaceEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)
            chat = ChatGroq(temperature=0, groq_api_key=groq_key_to_use, model_name="gemma-7b-it")
        else:
            embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
            knowledge_base = FAISS.from_texts(chunks, embeddings)
            llm = OpenAI(temperature=0, openai_api_key=openai_key)

        # User input area using st.text_input for compatibility
        user_input = st.chat_input(placeholder=f"Talk to 🎙️ {podcast_choice}", key='input')
        if user_input:
            st.session_state.questions_asked += 1
            # Find the most relevant chunks
            docs = knowledge_base.similarity_search(user_input)
            if chatbot=='Groq (llam3-8B)' or chatbot=='Groq (gemma-7b)':
                chain = load_qa_chain(chat, chain_type="stuff")
            else:
                chain = load_qa_chain(llm, chain_type="stuff")
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

def display_podcasts(df,search_engine):
    for _, row in df.iterrows():
        # Generate Listen Notes player embed code for podcasts
        listen_notes_player = f'<iframe src="{row["Link"]}/embed/" height="180px" width="100%" style="width: 1px; min-width: 100%;" frameborder="0" scrolling="no" loading="lazy"></iframe>'

        # Display podcast information with improved formatting
        st.markdown(f'**Podcast:** [{row["Podcast"]}]({row["Podcast link"]})')
        st.markdown(f'**Episode:** {row["Episode"]}')
        st.markdown(f'**Host:** {row["Host"]} | **Guest(s):** {row["Guest(s)"]}')
        st.markdown(listen_notes_player, unsafe_allow_html=True)
        # Keywords handling
        keywords = row["Tags"].split(';')
        if search_engine == 'Google Scholar':
            linked_keywords = [
                f'[**{keyword.strip()}**](https://scholar.google.com/scholar?q={keyword.strip().replace(" ", "+")}%20(crowd%20evacuation%20OR%20fire%20engineering))'
                for keyword in keywords if keyword.strip() != ''
            ]
        elif search_engine == 'Scopus':
            linked_keywords = [
                f'[**{keyword.strip()}**](https://www.scopus.com/results/results.uri?src=s&st1={keyword.strip().replace(" ", "%20")}&sdt=b&sl=58&s=TITLE-ABS-KEY({keyword.strip().replace(" ", "%20")})%20AND%20(TITLE-ABS-KEY(crowd%20AND%20evacuation)%20OR%20TITLE-ABS-KEY(fire%20AND%20engineering)))'
                for keyword in keywords if keyword.strip() != ''
            ]
        
        # Display keywords
        keywords_string = ', '.join(linked_keywords)
        with st.expander("**Keywords**"):
            st.markdown(f'{keywords_string}', unsafe_allow_html=True)
        st.markdown("---")

def display_youtube_videos(df,search_engine):
    for _, row in df.iterrows():
        # Extract YouTube video ID
        youtube_link = row['Link']
        # Split the URL by 'watch?v=' and further split by '&' to isolate the video ID
        youtube_video_id = youtube_link.split('watch?v=')[-1].split('&')[0]

        # Display YouTube video information with improved formatting
        st.markdown(f'**Title:** {row["Episode"]}')
        # Form the correct YouTube embed URL
        youtube_embed_url = f"https://www.youtube.com/watch?v={youtube_video_id}"
        st.video(youtube_embed_url)
        # Keywords handling
        keywords = row["Tags"].split(';')
        if search_engine == 'Google Scholar':
            linked_keywords = [
                f'[**{keyword.strip()}**](https://scholar.google.com/scholar?q={keyword.strip().replace(" ", "+")}%20(crowd%20evacuation%20OR%20fire%20engineering))'
                for keyword in keywords if keyword.strip() != ''
            ]
        elif search_engine == 'Scopus':
            base_focus = "(crowd AND evacuation) OR (fire AND engineering)"
            linked_keywords = [
                f'[**{keyword.strip()}**](https://www.scopus.com/results/results.uri?src=s&st1={keyword.strip().replace(" ", "%20")}&sdt=b&sl=58&s=TITLE-ABS-KEY({keyword.strip().replace(" ", "%20")})%20AND%20(TITLE-ABS-KEY(crowd%20AND%20evacuation)%20OR%20TITLE-ABS-KEY(fire%20AND%20engineering)))'
                for keyword in keywords if keyword.strip() != ''
            ]
        
        # Display keywords
        keywords_string = ', '.join(linked_keywords)
        with st.expander("**Keywords**"):
            st.markdown(f'{keywords_string}', unsafe_allow_html=True)
        st.markdown("---")

def send_email(name, email, position, feedback):
    sender_email = "crwdynamics@gmail.com"
    receiver_email = "crwdynamics@gmail.com"
    password = gmail_key

    # Create the email
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = "New Feedback Received"

    body = f"Name: {name}\nEmail: {email}\nPosition: {position}\n\nFeedback:\n{feedback}"
    msg.attach(MIMEText(body, 'plain'))

    try:
        # Connect to the server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

# Initialize the GNews client
gnews = GNews()

# Search for news articles
def search_news(keywords):
    news = gnews.get_news(keywords)
    st.session_state.news_results = news
    return news

# Get the first image from an article
def get_first_image(url):
    article = Article(url)
    try:
        article.download()
        article.parse()
        images = list(article.images)
        return images[0] if images else None
    except Exception as e:
        return None
    
def display_news(news):
    if news:
        for i, article in enumerate(news[:10]):  # Display top 10 news articles
            col1, col2 = st.columns([1, 4])
            with col1:
                image_url = get_first_image(article['url'])
                if image_url:
                    st.image(image_url, width=150)
                else:
                    st.write("No image available.")
            with col2:
                st.subheader(f"{i + 1}. {article['title']}")
                st.write(article['description'])
                st.write(f"[Read more]({article['url']})")
                st.write(f"Published at: {article['published date']}")
                st.write(f"Source: [{article['publisher']['title']}]({article['publisher']['href']})")
    else:
        st.write('No news articles found.')

def main():
    # Initialize session state variables
    if 'groq_key_entered' not in st.session_state:
        st.session_state.groq_key_entered = False
    if 'questions_asked' not in st.session_state:
        st.session_state.questions_asked = 0
    # Set app title and description
    st.set_page_config(page_title="EvacuAIPod", page_icon="🎙️", layout="wide")
    st.title("EvacuAIPod: AI-bot for Crowd Evacuation and Fire Engineering Podcasts")
    # Load the CSV file into a DataFrame
    @st.cache_data
    def load_data():
        return pd.read_csv("pod.csv", encoding='latin1')

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
    # Sidebar selection for the search engine
    search_engine = st.sidebar.selectbox(
        'Choose scholar search engine:',
        ('Google Scholar', 'Scopus')
    )
    
    with st.sidebar.expander("**Chatbot model**"):
        chatbot = st.selectbox('Choose the chatbot model:', ('Groq (llam3-8B)', 'Groq (gemma-7b)', 'ChatGPT'))
    # Create a text input for the OpenAI key
    with st.sidebar.expander("**API key**"):
        st.write('Openai key')
        openai_key = st.text_input("Enter your [OpenAI key](https://platform.openai.com/api-keys):", key="openai_input", type="password")
        st.write('Groq key')
        groq_key = st.text_input("Enter your [Groq key](https://console.groq.com/keys):", key="groq_input", type="password")
        if groq_key:
            st.session_state.groq_key_entered = True
        if openai_key:
            st.session_state.openai_key_entered = True
        # Add a note about the API key
        st.markdown(":red-background[**Note:**] When you input your API key, it’s stored on the client side (your browser or device). This means the API key is accessible only to you and the app itself. The app owner doesn’t have direct access to your API key.")

    # Add an "About" section in the sidebar
    expander = st.sidebar.expander("**About EvacuAIPod**")
    with expander:
                expander.write('''
                        EvacuAIPod is an innovative web application that enables users to search and discover podcasts focusing on crowd evacuation strategies. With the power of targeted keyword searches, listeners can effortlessly find and listen to discussions that pique their interest. Beyond just listening, EvacuAIPod offers an interactive experience by allowing users to engage in conversations with the content of the podcasts themselves. This unique feature opens up deeper insights and a more personalized understanding of the crucial topics of AI technology and safety measures. EvacuAIPod is the go-to platform for anyone looking to delve into the world of emergency preparedness and crowd management through the lens of interactive and informative podcast sessions.
                ''')
    
    with st.sidebar.expander("**Contact US**"):
        st.write('''
                 Feel free to reach out to us for any further information, enquiries, or if you're interested in collaborating with us. You can contact either [Amir Rafe](mailto:amir.rafe@usu.edu) or [Ruggiero (Rino) Lovreglio](mailto:r.Lovreglio@massey.ac.nz). 
                   ''')
        st.write('''
                 We're always happy to hear from you! Please provide us with your feedback, and don't forget to include your name, email, and position. Your input is invaluable in helping us improve EvacuAIPod.
                   ''')
        with st.form(key='feedback_form'):
            name = st.text_input("Name")
            email = st.text_input("Email")
            position = st.text_input("Position")
            feedback = st.text_area("Feedback")
            submit_feedback = st.form_submit_button(label="Submit")
        
        if submit_feedback:
            if send_email(name, email, position, feedback):
                st.write("Thank you for your feedback!")
            else:
                st.error("There was an error sending your feedback. Please try again later.")

    if page == "Transcript":
        # User input for search keywords
        with st.form(key='Keywords'):
            keywords = st.text_input("**Enter keywords to search for podcasts. Separate multiple keywords with spaces. If you want to search for an exact phrase, enclose it in quotation marks.**")
            submit_button = st.form_submit_button(label='Search')
        st.markdown(":red-background[**Note:**] *The transcripts and keywords for the podcasts are generated automatically using AI. While we strive to ensure their accuracy, there may be some errors or omissions. If you notice any issues, please let us know so we can continue to improve our service.*")
        if submit_button:
            news = search_news(keywords)
            st.session_state.news_results = news
        if 'filtered_df' in st.session_state and not st.session_state.filtered_df.empty and not keywords:
            # Create two tabs for 'Podcasts' and 'YouTube Videos'
                tab1, tab2, tab3 = st.tabs(['Podcasts', 'YouTube Videos', 'News']) 
                filtered_df = st.session_state.filtered_df             
                # 'Podcasts' tab
                with tab1:
                    # Display podcasts
                    display_podcasts(filtered_df[filtered_df['Type'] == 'Pod'],search_engine)

                # 'YouTube Videos' tab
                with tab2:
                    # Display YouTube videos
                    display_youtube_videos(filtered_df[filtered_df['Type'] == 'YouTube'],search_engine)

                with tab3:
                    if 'news_results' in st.session_state:
                        display_news(st.session_state.news_results)
                    else:
                        st.write('No news articles found.')

        else: 
            if keywords:
                # Regular expression to find phrases enclosed in double quotes
                phrases = re.findall(r'"(.*?)"', keywords)
                # Remove the phrases from the original input to get the remaining words
                remaining_keywords = re.sub(r'"(.*?)"', '', keywords).split()
                # List to hold DataFrames for intersection
                filtered_dfs = []
                # Search for each phrase
                for phrase in phrases:
                    matching_df = df[df.apply(lambda x: x.str.contains(phrase, case=False, na=False)).any(axis=1)]
                    filtered_dfs.append(matching_df)
                # Search for each remaining keyword
                for keyword in remaining_keywords:
                    if keyword.strip():  # Ensure it's not an empty string
                        matching_df = df[df.apply(lambda x: x.str.contains(keyword, case=False, na=False)).any(axis=1)]
                        filtered_dfs.append(matching_df)
                # Intersect all DataFrames to ensure all conditions are met
                if filtered_dfs:
                    from functools import reduce
                    filtered_df = reduce(lambda left, right: pd.merge(left, right, how='inner'), filtered_dfs)
                else:
                    filtered_df = df

                if not filtered_df.empty:
                    # Store the filtered DataFrame in session state
                    st.session_state.filtered_df = filtered_df
                    st.session_state.keywords = keywords
                else:
                    # Perform search in News section
                    news = search_news(keywords)
                    st.session_state.news_results = news
                    st.session_state.keywords = keywords
                    st.session_state.filtered_df = pd.DataFrame()  # Empty DataFrame to indicate no podcasts or YouTube videos found

        # Check if there are search results in the session state
        if 'filtered_df' in st.session_state and 'keywords' in st.session_state:
            filtered_df = st.session_state.filtered_df
            keywords = st.session_state.keywords

            # Create tabs for 'Podcasts', 'YouTube Videos', and 'News'
            tab1, tab2, tab3 = st.tabs(['Podcasts', 'YouTube Videos', 'News'])

            # 'Podcasts' tab
            with tab1:
                if not filtered_df.empty and 'Pod' in filtered_df['Type'].values:
                    display_podcasts(filtered_df[filtered_df['Type'] == 'Pod'], search_engine)
                else:
                    st.write("No podcasts found with the given keyword(s).")

            # 'YouTube Videos' tab
            with tab2:
                if not filtered_df.empty and 'YouTube' in filtered_df['Type'].values:
                    display_youtube_videos(filtered_df[filtered_df['Type'] == 'YouTube'], search_engine)
                else:
                    st.write("No YouTube videos found with the given keyword(s).")

            # 'News' tab
            with tab3:
                if 'news_results' in st.session_state and st.session_state.news_results:
                    display_news(st.session_state.news_results)
                else:
                    st.write("No news articles found.")          
        else:
                st.write("")
    
        if 'filtered_df' in st.session_state and not st.session_state.filtered_df.empty:
            transcript1=[]
            with st.form(key="podcast_form"):
                st.subheader("**Transcription**")
                st.write("**Select content from the list and request its transcript:**")
                # Create a select box for the podcast using the filtered DataFrame from session state
                podcast_options = st.session_state.filtered_df["Episode"].tolist()
                podcast_choice = st.selectbox("Title:", podcast_options, key="podcast_select")
                st.session_state['podcast_choice'] = podcast_choice  # Store podcast choice in session state
                # Create a submit button
                submit_button = st.form_submit_button(label="Submit")
                
                if submit_button:
                    st.session_state.conversation_history = []
                    st.session_state.messages = []
                    try:
                        transcript1 = read_transcript(podcast_choice)
                        # Check if the selected podcast is of type 'Pod'
                        podcast_type = filtered_df[filtered_df['Episode'] == podcast_choice]['Type'].iloc[0]
                        if podcast_type == 'Pod':
                            transcript_with_paragraphs = add_paragraph_breaks(transcript1)
                        else:
                            transcript_with_paragraphs = transcript1

                        st.session_state['transcript1'] = transcript_with_paragraphs  # Store transcript in session state
                        st.text_area("Transcription", transcript_with_paragraphs, height=400)
                        st.write(":rainbow-background[*To chat with the selected podcast, use the sidebar to navigate to the 'Chat with Podcast' page.*]")

                    except Exception as e:
                        st.error(f"An error occurred: {e}")

            # If filtered_df isn't in session state or is empty, display a message
            st.write("")
    elif page == "Chat with Podcast":
        if 'filtered_df' in st.session_state and 'podcast_choice' in st.session_state:
            filtered_df = st.session_state.filtered_df
            podcast_choice = st.session_state.podcast_choice
            if 'transcript1' in st.session_state:
                transcript1 = st.session_state['transcript1']
                chat_with_podcast(openai_key, groq_key, chatbot, transcript1, podcast_choice)
            else:
                st.error("Please transcribe a podcast first.")
        else:
            st.error("Please select a podcast first.")
        

if __name__ == '__main__':
    main()
