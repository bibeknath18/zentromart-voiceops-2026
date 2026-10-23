
import os
from pathlib import Path
from dotenv import load_dotenv
from gnani.tts import GnaniTTSClient

ROOT=Path(__file__).resolve().parents[1]
load_dotenv(ROOT/".env")

api_key=os.getenv("GNANI_API_KEY")

if not api_key:
    raise RuntimeError("GNANI_API_KEY is missing")

client=GnaniTTSClient(api_key=api_key)

audio=client.synthesize(
    "Hello! Welcome to ZentroMart customer support.",
    voice="Pranav",
    model="timbre-v2.5"
)

output=ROOT/"gnani_test.wav"
output.write_bytes(audio)

print("TTS successful:",output)
print("Audio size:",len(audio),"bytes")
