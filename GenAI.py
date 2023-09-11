! apt-get install -y xvfb x11-utils
! pip install xvfbwrapper
! pip install pygobject

# # install dependencies
! pip install SpeechRecognition
! pip install gTTS
! pip install transformers
! pip install tensorflow
! pip install playsound
! pip install noisereduce
! pip install pyttsx3
! pip install simpletransformers

import spacy
import tkinter as tk
from tkinter import Scrollbar, Text, Entry, Button
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
import os
import numpy as np
import noisereduce as nr
import pyttsx3
from xvfbwrapper import Xvfb

vdisplay = Xvfb()
vdisplay.start()
os.system('xvfb: 1 -screen 0 720x720x16 &')
os.environ['display'] = ':1.0'

import json

with open(r"train.json", "r") as read_file: 
  train = json.load(read_file)

with open(r"test.json", "r") as read_file:
    test = json.load(read_file)

import logging

from simpletransformers.question_answering import QuestionAnsweringModel, QuestionAnsweringArgs
     

model_type="bert"
model_name= "bert-base-cased"
if model_type == "bert":
    model_name = "bert-base-cased"

elif model_type == "roberta":
    model_name = "roberta-base"

elif model_type == "distilbert":
    model_name = "distilbert-base-cased"

elif model_type == "distilroberta":
    model_type = "roberta"
    model_name = "distilroberta-base"

elif model_type == "electra-base":
    model_type = "electra"
    model_name = "google/electra-base-discriminator"

elif model_type == "electra-small":
    model_type = "electra"
    model_name = "google/electra-small-discriminator"

elif model_type == "xlnet":
    model_name = "xlnet-base-cased"
     

# Configure the model 
model_args = QuestionAnsweringArgs()
model_args.train_batch_size = 16
model_args.evaluate_during_training = True
model_args.n_best_size=3
model_args.num_train_epochs=5

     

### Advanced Methodology
train_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "use_cached_eval_features": True,
    "output_dir": f"outputs/{model_type}",
    "best_model_dir": f"outputs/{model_type}/best_model",
    "evaluate_during_training": True,
    "max_seq_length": 128,
    "num_train_epochs": 15,
    "evaluate_during_training_steps": 1000,
    "wandb_project": "Question Answer Application",
    "wandb_kwargs": {"name": model_name},
    "save_model_every_epoch": False,
    "save_eval_checkpoints": False,
    "n_best_size":3,
    # "use_early_stopping": True,
    # "early_stopping_metric": "mcc",
    # "n_gpu": 2,
    # "manual_seed": 4,
    # "use_multiprocessing": False,
    "train_batch_size": 128,
    "eval_batch_size": 64,
    # "config": {
    #     "output_hidden_states": True
    # }
}

model = QuestionAnsweringModel(
    model_type,model_name, args=train_args
)

### Remove output folder
!rm -rf outputs
     

# Train the model
model.train_model(train, eval_data=test)

# Evaluate the model
result, texts = model.eval_model(test)

# Make predictions with the model
to_predict = [
    {
        "context": "The transaction was made to buy crypto",
        "qas": [
            {
                "question": "why was the transaction done?",
                "id": "0",
            }
        ],
    }
]
     
answers, probabilities = model.predict(to_predict)

print(answers)


class NLPChatbotUI:
  def __init__(self, root):
    self.root = root
    self.root.title("GENAI Chatbot")

    self.chat_area = Text(root, wrap=tk.WORD, state=tk.DISABLED)
    self.scrollbar = Scrollbar(root, command=self.chat_area.yview)
    self.chat_area.config(yscrollcommand=self.scrollbar.set)

    self.user_input = Entry(root)


    self.voice_button = Button(root, text="Voice", command=self.voice_input)

    self.chat_area.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)
    self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    self.user_input.pack(padx=10, pady=5, expand=True, fill=tk.X)

    self.voice_button.pack(pady=5)

    self.nlp = spacy.load("en_core_web_sm")
    self.add_bot_message("Natwest Agent: Hi! How can I help you?")
    self.recognizer = sr.Recognizer()

  def voice_input(self):
    while True:
        try:
          self.recognizer = sr.Recognizer()

          with sr.Microphone() as source:
            self.chat_area.update_idletasks()
            self.recognizer.adjust_for_ambient_noise(source)
            print("Please speak something...")
            audio = self.recognizer.listen(source)

              # Convert audio to NumPy array
            audio_data = np.frombuffer(audio.frame_data, dtype=np.int16)

            # Reduce noise from audio
            reduced_noise = nr.reduce_noise(y=audio_data, sr=audio.sample_rate)

            # Convert the reduced noise audio back to AudioData
            reduced_noise_audio = sr.AudioData(
                reduced_noise.tobytes(),
                sample_rate=audio.sample_rate,
                sample_width=reduced_noise.dtype.itemsize,
            )

            recognized_text = self.recognizer.recognize_google(reduced_noise_audio)
            self.add_user_message("Customer: " + recognized_text)
            response = self.process_message(recognized_text)
            self.add_bot_message("Natwest Agent: " + response)

            self.text_to_speech(response)


            print("Recognized text:", recognized_text)
        except sr.UnknownValueError:
            print("Google Web Speech API could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Web Speech API; {0}".format(e))

  def process_message(self, user_input):
    user_input = user_input.lower()
    if "hello" in user_input:
      return "Hello! How can I assist you?"
    if "hey" in user_input:
      return "Hello! How can I assist you?"
    elif "how are you" in user_input:
      return "I'm just a chatbot, but I'm here to help."
    elif "what's your name" in user_input:
      return "I'm a chatbot. You can call me GENAI."
    elif "case" in user_input:
      return "Can I have your case reference number "
    elif "what is my case status" in user_input:
      return "Can I have your case reference number "
    elif "123" in user_input:
      return "your case status is under investigation, we will get back to you  "
    elif "no" in user_input:
      return "Can i help with any other thing  "
    else:
      return "I'm sorry, I didn't catch that. Can you please repeat or ask something else?"

  def add_user_message(self, message):
    self.chat_area.config(state=tk.NORMAL)
    self.chat_area.insert(tk.END, message + "\n")
    self.chat_area.config(state=tk.DISABLED)
    self.chat_area.see(tk.END)

  def add_bot_message(self, message):
    self.chat_area.config(state=tk.NORMAL)
    self.chat_area.insert(tk.END, message + "\n", "bot")
    self.chat_area.config(state=tk.DISABLED)
    self.chat_area.see(tk.END)

  def text_to_speech1(self, text, output_file="output.mp3", lang="en"):
    try:
      tts = gTTS(text=text, lang=lang)
      tts.save(output_file)
      print(f"Text saved as '{output_file}'")
      os.system(f"start {output_file}")  # This plays the generated audio on Windows
    except Exception as e:
      print(f"Error: {e}")

  def text_to_speech(self, text):
    try:
        # Initialize the text-to-speech engine
        engine = pyttsx3.init()

        # Convert text to speech
        engine.say(text)
        engine.runAndWait()

        print("Text converted to speech successfully.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
  root = tk.Tk()
  chatbot_ui = NLPChatbotUI(root)
  root.mainloop()
