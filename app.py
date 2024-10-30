import os
import yt_dlp
import logging
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

# Load API key from environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize the Groq client
client = Groq(api_key=GROQ_API_KEY)

def download_audio(url, output_path="./"):
    try:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        output_file = os.path.join(output_path, 'audio.mp3')

        if os.path.exists(output_file):
            os.remove(output_file)

        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(output_path, 'audio'),
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)

        return output_file

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise
def transcribe_audio(audio_path):
    """
    Transcribe audio using Groq's Whisper model.

    Args:
        audio_path (str): Path to the audio file.

    Returns:
        str: Transcription text.
    """
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3-turbo",
        )
    return transcription.text

def summarize_text(transcript_text):
    """
    Generate a summary of the transcript using Groq's chat model.

    Args:
        transcript_text (str): Transcript text to summarize.

    Returns:
        str: Summary text.
    """
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are an expert summarizer tasked with creating a clear, concise summary of a YouTube video.\n\n"
                    f"### Video Transcript\n {transcript_text} \n\n"
                    "Aim for a summary that is engaging and accurate, while maintaining a logical flow that would be easy for someone unfamiliar with the content to understand."
                )
            }
        ],
        temperature=1,
        max_tokens=8192,
        top_p=1,
    )
    # Check if completion is a tuple and handle accordingly
    # Check if the completion has choices
    if hasattr(completion, 'choices') and completion.choices:
        # Extract the content from the first choice
        summary_text = completion.choices[0].message['content']
    else:
        summary_text = "Summary generation failed, no valid response received."

    return summary_text
# Streamlit UI
st.title("YouTube Video Summarizer")
url = st.text_input("Enter the YouTube video URL:")

if st.button("Download and Summarize"):
    if url:
        try:
            st.write("Downloading audio...")
            audio_path = download_audio(url)
            st.success("Audio downloaded successfully!")
            st.audio(audio_path)  # Streamlit audio player

            st.write("Transcribing audio...")
            transcript_text = transcribe_audio(audio_path)
            st.success("Transcription completed!")

            st.write("Generating summary...")
            summary = summarize_text(transcript_text)
            st.success("Summary generated successfully!")

            st.subheader("Transcript")
            st.write(transcript_text)

            st.subheader("Summary")
            st.write(summary)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a valid YouTube URL.")
