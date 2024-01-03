from transformers import AutoProcessor, SeamlessM4Tv2Model
import torchaudio

from fastapi import FastAPI

App = FastAPI

processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")

# from text
text_inputs = processor(text = "Hi My name is Youngwook Choi ", src_lang="eng", return_tensors="pt")
audio_array_from_text = model.generate(**text_inputs, tgt_lang="kor")[0].cpu().numpy().squeeze()

# from audio
audio, orig_freq =  torchaudio.load("https://www2.cs.uic.edu/~i101/SoundFiles/preamble10.wav")
audio =  torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16_000) # must be a 16 kHz waveform array
audio_inputs = processor(audios=audio, return_tensors="pt")
audio_array_from_audio = model.generate(**audio_inputs, tgt_lang="kor")[0].cpu().numpy().squeeze()

# pip install protobuf
