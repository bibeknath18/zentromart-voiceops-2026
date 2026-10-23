"""Streamlit voice interface for the existing ZentroMart VoiceOps agent.

Run from the project root:
    python -m streamlit run voiceops_interactive.py

Environment: GNANI_API_KEY, Groq/LLM credentials in .env.
"""
from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')

from gnani.stt import GnaniSTTClient
from gnani.tts import GnaniTTSClient
from src.core import Agent

st.set_page_config(page_title='ZentroMart VoiceOps', page_icon='🎙️', layout='centered')
st.title('🎙️ ZentroMart VoiceOps')
st.caption('Gnani STT → ZentroMart agent (Groq + RAG) → Gnani TTS')

if not os.getenv('GNANI_API_KEY'):
    st.error('GNANI_API_KEY is missing. Add it to the project .env file.')
    st.stop()

if 'agent' not in st.session_state:
    st.session_state.agent = Agent()
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.recording_counter = 0


def reset_chat():
    st.session_state.agent = Agent()
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.recording_counter += 1


with st.sidebar:
    st.subheader('Conversation controls')
    language = st.selectbox('Gnani transcription language', ['en-IN', 'hi-IN'], index=0)
    speak_replies = st.checkbox('Generate spoken replies', value=True)
    st.button('Start new conversation', on_click=reset_chat, use_container_width=True)
    st.caption('Ticket creation is disabled in this local demo. No actual handover is performed.')
    st.caption('Sessions are stored only while this browser connection remains active.')


for item in st.session_state.messages:
    with st.chat_message(item['role']):
        st.markdown(item['text'])
        if item.get('decision'):
            st.caption('Decision: ' + item['decision'])
        if item.get('audio'):
            st.audio(item['audio'], format='audio/wav', autoplay=False)


def process_message(message: str):
    message = message.strip()
    if not message:
        st.warning('No speech was detected. Please try again.')
        return

    st.session_state.messages.append({'role': 'user', 'text': message})
    with st.spinner('Getting the support response...'):
        result = st.session_state.agent.respond(
            message,
            session_id=st.session_state.session_id,
            allow_ticket=False,
        )
    answer = result.get('answer') or 'Sorry, I cannot answer that right now.'
    item = {'role': 'assistant', 'text': answer, 'decision': result.get('decision', 'unknown')}

    if speak_replies:
        try:
            with st.spinner('Generating audio with Gnani...'):
                tts = GnaniTTSClient(api_key=os.environ['GNANI_API_KEY'])
                audio = tts.synthesize(answer, voice='Pranav', model='timbre-v2.5')
            item['audio'] = audio
        except Exception as exc:
            st.warning('Speech generation failed; the text answer is still available. '
                       + type(exc).__name__)

    st.session_state.messages.append(item)


st.subheader('Speak a question')
recording = st.audio_input(
    'Record and stop to prepare your message',
    sample_rate=16000,
    key=f'recording_{st.session_state.recording_counter}',
)
if st.button('Send recording', type='primary', disabled=recording is None):
    audio_bytes = recording.getvalue()
    if not audio_bytes:
        st.warning('Recording is empty.')
    else:
        path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as fp:
                fp.write(audio_bytes)
                path = Path(fp.name)
            with st.spinner('Transcribing with Gnani...'):
                stt = GnaniSTTClient(api_key=os.environ['GNANI_API_KEY'])
                transcript = stt.transcribe(str(path), language_code=language).get('transcript', '')
            process_message(transcript)
            st.session_state.recording_counter += 1
            st.rerun()
        except Exception as exc:
            st.error(f'Transcription/processing failed: {type(exc).__name__}: {exc}')
        finally:
            if path is not None:
                path.unlink(missing_ok=True)

st.subheader('Or type a message')
message = st.chat_input('Ask a question or continue this conversation...')
if message:
    try:
        process_message(message)
        st.rerun()
    except Exception as exc:
        st.error(f'Agent error: {type(exc).__name__}: {exc}')

if st.session_state.messages:
    last = st.session_state.messages[-1]
    if last['role'] == 'assistant' and last.get('audio'):
        st.caption('Latest spoken reply')
        st.audio(last['audio'], format='audio/wav', autoplay=True)
