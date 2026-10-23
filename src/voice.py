"""Gnani-only speech adapter; no substitute speech service."""
import os, wave, json, struct
from pathlib import Path

class VoiceError(Exception):pass

def validate_audio(path):
    p=Path(path)
    if not p.is_file():raise VoiceError('Audio file not found')
    if p.suffix.lower()!='.wav':raise VoiceError('Only PCM WAV input is supported by local validation')
    if not 44<=p.stat().st_size<=12*1024*1024:raise VoiceError('Invalid audio file size')
    try:
        with wave.open(str(p),'rb') as w:
            duration=w.getnframes()/w.getframerate()
            if w.getcomptype()!='NONE' or w.getnchannels() not in (1,2) or w.getsampwidth()!=2 or not 0<duration<=60:raise VoiceError('Audio must be 16-bit PCM, 1–2 channels, at most 60 seconds')
    except (wave.Error,ZeroDivisionError,EOFError) as exc:raise VoiceError('Invalid PCM WAV') from exc
    return duration

class GnaniVoice:
    def __init__(self,stt=None,tts=None):
        if stt is None or tts is None:
            if not os.getenv('GNANI_API_KEY'):raise VoiceError('GNANI_API_KEY is missing')
            try:
                from gnani.stt import GnaniSTTClient
                from gnani.tts import GnaniTTSClient
            except ImportError as exc:raise VoiceError('Install gnani-vachana') from exc
            stt=stt or GnaniSTTClient();tts=tts or GnaniTTSClient()
        self.stt=stt;self.tts=tts
    def process(self,path,output,agent,session_id):
        validate_audio(path)
        try: result=self.stt.transcribe(str(path),language_code='en-IN')
        except Exception as exc:raise VoiceError('Gnani transcription failed: '+type(exc).__name__) from exc
        transcript=result.get('transcript','') if isinstance(result,dict) else ''
        if not isinstance(transcript,str):raise VoiceError('Gnani returned invalid transcript')
        if not transcript.strip():reply={'answer':"Sorry, I didn't catch that. Could you repeat?",'decision':'clarify','evidence':[]}
        else:reply=agent.respond(transcript,session_id=session_id)
        try:
            self.tts.synthesize(reply['answer'],voice='Kaveri',model='timbre-v2.5',language='en-IN',output_file=str(output))
            if not Path(output).is_file() or Path(output).stat().st_size<44:raise VoiceError('Gnani produced no playable output')
            audio_status='success'
        except Exception as exc:audio_status='failed: '+type(exc).__name__
        return {'status':'success' if audio_status=='success' else 'partial','transcript':transcript,'reply':reply['answer'],'evidence':reply.get('evidence',[]),'audio_status':audio_status,'output_audio':str(output) if audio_status=='success' else None}
