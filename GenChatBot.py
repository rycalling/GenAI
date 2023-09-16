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
import openai
openai.api_key= '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'
model_id = "gpt-3.5-turbo"

class NLPChatbotUI:
  def __init__(self, root):
    self.root = root
    self.root.title("GEN AI Chatbot")

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
            print("Natwest Agent: Hi, I am GenAI from Natwest Group. How can I help you today")
            self.text_to_speech(self.get_gpt_response("Consider yourself as Gen Ai, who is helping bank customers. Greet the customer who has just called in"))
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
      return "Sorry, I could not recognise the linked account to this number, can you please confirm your bank account number"
    if "hey" in user_input:
      return "Sorry, I could not recognise the linked account to this number, can you please confirm your bank account number"
    elif "how are you" in user_input:
      return "I'm just a chatbot, but I'm here to help."
    elif "one two three four" in user_input:
      return "thank you for confirming the bank account, can you please confirm your name"
    elif "stalin" in user_input:
      return "Great, thank you Stalin. I can see there is a Fraud case created in your account. Would you like to know its status"
    elif "yes" in user_input:
      return "can you please confirm the case number received by email"
    elif "three four five" in user_input:
      return "Thanks for confirming. your case is under progress. we would like a bit more information to progress it further"
    elif "sure" in user_input:
      return "What was the purpose of the transaction"
     elif "Buying Crypto" in user_input:
      return "can you please confirm the transaction amount"
    elif "five pounds" in user_input:
      return "Thanks for confirming. can you please provide the details of the retailer"
    elif "yes it was Binance" in user_input:
      return "thank you for sharing the details, we will progress the case for a refund"
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

  def get_gpt_response(self, input_msg):
    try:
      chat = openai.ChatCompletion.create(
      model=model_id,
      messages=[{"role": "user", "content": input_msg}])
      resp = chat.choices[0].message.content
      return resp
    except Exception as e:
      print(f"Error: {e}")

  
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
