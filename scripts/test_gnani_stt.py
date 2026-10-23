
import os
from pathlib import Path
from dotenv import load_dotenv
from gnani.stt import GnaniSTTClient

ROOT=Path(__file__).resolve().parents[1]
load_dotenv(ROOT/".env")

client=GnaniSTTClient(
    api_key=os.environ["GNANI_API_KEY"]
)

audio_file=ROOT/"gnani_test.wav"

if not audio_file.exists():
    raise FileNotFoundError("gnani_test.wav not found")

result=client.transcribe(
    str(audio_file),
    language_code="en-IN"
)

print("STT successful!")
print("Transcript:",result["transcript"])
