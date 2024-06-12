[![EvacuAIPod](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://evacuaipod.streamlit.app/)
# EvacuAIPod: AI-bot for Crowd Evacuation Podcasts

<p align="center">
  <img src="https://github.com/pozapas/EvacuAIPod/blob/master/evacuaipod.png?raw=true" alt="Evacuaipod" style="width: 30%;">
</p>

is an innovative web application designed to enhance the podcast listening experience by integrating AI-powered features. It enables users to transcribe podcasts, engage in interactive conversations with the content, and explore podcasts related to crowd evacuation strategies and emergency preparedness.

[![YouTube](https://img.shields.io/badge/YouTube-%23FF0000.svg?style=for-the-badge&logo=YouTube&logoColor=white)](https://www.youtube.com/watch?v=FoXzON4L7d4)

## Features

- **Podcast Transcription**: Utilizes OpenAI's Whisper API to transcribe audio content, making it accessible and searchable.
- **Interactive Chat**: Engages users in conversations with the podcast content, powered by OpenAI's GPT-3 and langchain, providing a unique interactive experience.
- **Keyword-based Podcast Discovery**: Allows users to find podcasts by searching for specific topics related to crowd evacuation and emergency preparedness.
  
## Getting Started

### Installation (local version)

1. Clone the repository:
   
   ```bash
   git clone https://github.com/pozapas/evacuaipod.git
   ```
2. Install the required packages:
  
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your secrets:

    Create a file named `secrets.toml` in the `.streamlit` directory. The `secrets.toml` file should contain the following keys:
    
      ```bash
    groq_key = "****"
    gmail_key = "****"
    ```
    You can get the Groq key from [Groq Console](https://console.groq.com/keys).
    
    To make the `send_email` function work, replace "crwdynamics@gmail.com" with your actual Gmail address. The gmail_key should be an app-specific password generated from your Gmail account. For security reasons, it is recommended to use an app-specific password rather than your main Gmail password.
    
    Set up an app-specific password:
    
    - Go to your Google Account.
    - Navigate to "Security" and find "App passwords" under "Signing in to Google".
    - Generate an app-specific password for the Streamlit app.
   
4. Run the Streamlit app:
     
   ```bash
   streamlit run EvacuAiPod.py
   ```
   
## Future Enhancements
- [x] Whisper Large V3 for Free Podcast Transcriptions
- [x] Add Podcasts from YouTube
- [x] User-Selectable Language Models for Embeddings and Chat
- [x] Add fire engineering content
- [x] Add a subscription form
- [ ] Incorporate a fine-tuned LLM tailored for crowd evacuation and fire engineering scenarios

## Contributing
Contributions, issues, and feature requests are welcome. Feel free to check [issues page](https://github.com/pozapas/EvacuAIPod/issues) for open issues or open a new issue.
