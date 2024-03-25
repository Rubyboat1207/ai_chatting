import os
import threading
from dotenv import load_dotenv

import keyboard
from pvrecorder import PvRecorder
import openai
import wave
import struct

from elevenlabs import play, VoiceSettings
from elevenlabs.client import ElevenLabs
load_dotenv()

eleven_client = ElevenLabs(
    api_key=os.environ.get("ELEVENLABS_API_KEY"),  # Defaults to ELEVEN_API_KEY
)


class AIController:
    def __init__(self, key, system_prompt: str):
        self.transcription_controller = TranscriptionController(key)
        self.client = openai.Client(api_key=key)
        self.messages = [{'role': 'system', 'content': system_prompt}]

    def talk(self):
        self.transcription_controller.begin_recording()
        print('started recording')
        keyboard.wait('f7')

        self.transcription_controller.end_recording()

        prompt = self.transcription_controller.transcribe("./out/temp.wav")

        self.messages.append({"role": 'user', 'content': prompt})

        message = self.client.chat.completions.create(messages=self.messages, model='gpt-4-turbo-preview').choices[
            0].message

        self.messages.append({"role": 'assistant', 'content': message.content})

        settings = VoiceSettings(
            stability=0.71, similarity_boost=0.5, style=0.45, use_speaker_boost=True
        )

        audio = eleven_client.generate(
            text=message.content,
            voice="Rudy",
            model="eleven_multilingual_v2",
            voice_settings=settings
        )
        print(message)
        print(prompt)

        play(audio)


class TranscriptionController:
    """Controls transcriptions"""

    def __init__(self, key: str) -> None:
        self.client = openai.Client(api_key=key)
        self.end_recording_event = threading.Event()
        self.is_recording = False
        self.recorder = PvRecorder(frame_length=250)
        self.audio_cache = []
        self._recording_thread: threading.Thread | None = None

    def transcribe(self, path):
        with open(path, 'rb') as file:
            transcription = self.client.audio.transcriptions.create(
                model='whisper-1', file=file).text

        # print('raw transcription: ' + transcription)

        return transcription

    def _record(self):
        self.recorder.start()
        while not self.end_recording_event.is_set():
            frame = self.recorder.read()
            self.audio_cache.extend(frame)

        self.recorder.stop()
        with wave.open("./out/temp.wav", 'w') as wave_write:
            # pylint: disable=no-member
            wave_write.setparams((1, 2, 16000, 512, "NONE", "NONE"))
            # pylint: disable=no-member
            wave_write.writeframes(struct.pack("h" * len(self.audio_cache), *self.audio_cache))

        self.recorder.delete()
        self.audio_cache = []
        self.recorder = PvRecorder(frame_length=512)

    def begin_recording(self):
        if self.is_recording:
            return

        self._recording_thread = threading.Thread(target=self._record)
        self._recording_thread.start()

    def end_recording(self):
        if self._recording_thread is not None:
            self.end_recording_event.set()
            self._recording_thread.join()
            self.end_recording_event.clear()
