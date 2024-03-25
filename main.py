import keyboard
import os
from Recorder import AIController

ai = AIController(key=os.environ.get("OPENAI_API_KEY"),
                  system_prompt="You are a frat bro. you will speak like one. your responses must be short because you are talking through text to speech")

while True:
    print('waiting...')
    keyboard.wait('f7')
    ai.talk()
