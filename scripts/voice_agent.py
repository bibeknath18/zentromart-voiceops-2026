
import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
load_dotenv(ROOT/".env")

from gnani.stt import GnaniSTTClient
from gnani.tts import GnaniTTSClient
from src.core import Agent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--audio",
        default="gnani_test.wav",
        help="Input WAV audio file"
    )
    parser.add_argument(
        "--session",
        default="voice-demo",
        help="Conversation session ID"
    )
    args=parser.parse_args()

    api_key=os.getenv("GNANI_API_KEY")
    if not api_key:
        raise RuntimeError("GNANI_API_KEY missing in .env")

    audio_path=Path(args.audio)
    if not audio_path.is_absolute():
        audio_path=ROOT/audio_path

    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)

    stt=GnaniSTTClient(api_key=api_key)
    tts=GnaniTTSClient(api_key=api_key)
    agent=Agent()

    print("Transcribing customer audio...")

    result=stt.transcribe(
        str(audio_path),
        language_code="en-IN"
    )

    transcript=result.get("transcript","").strip()

    if not transcript:
        print("No speech detected.")
        return

    print("Customer:",transcript)

    response=agent.respond(
        transcript,
        session_id=args.session,
        allow_ticket=False
    )

    answer=response["answer"]
    print("Agent:",answer)
    print("Decision:",response["decision"])

    print("Generating speech...")

    audio=tts.synthesize(
        answer,
        voice="Pranav",
        model="timbre-v2.5"
    )

    output=ROOT/"voice_agent_reply.wav"
    output.write_bytes(audio)

    print("Voice response saved:",output)


if __name__=="__main__":
    main()
