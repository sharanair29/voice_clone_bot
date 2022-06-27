from encoder.params_model import model_embedding_size as speaker_embedding_size
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import argparse
import torch
import sys
import os
from audioread.exceptions import NoBackendError
import sounddevice as sd





class TTS:

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--enc_model_fpath", type=Path, 
                        default="/Users/hlabs/Desktop/hoop/voicebot/Real-Time-Voice-Cloning/encoder/saved_models/pretrained.pt",
                        help="Path to a saved encoder")
    parser.add_argument("-s", "--syn_model_fpath", type=Path, 
                        default="/Users/hlabs/Desktop/hoop/voicebot/Real-Time-Voice-Cloning/synthesizer/saved_models/pretrained/pretrained.pt",
                        help="Path to a saved synthesizer")
    parser.add_argument("-v", "--voc_model_fpath", type=Path, 
                        default="/Users/hlabs/Desktop/hoop/voicebot/Real-Time-Voice-Cloning/vocoder/saved_models/pretrained/pretrained.pt",
                        help="Path to a saved vocoder")
    parser.add_argument("--cpu", action="store_true", help=\
        "If True, processing is done on CPU, even when a GPU is available.") # This argument is forcing the code to run on my CPU alone, if set to False it would run on GPU
    parser.add_argument("--no_sound", action="store_true", help=\
        "If True, audio won't be played.")
    parser.add_argument("--seed", type=int, default=None, help=\
        "Optional random number seed value to make toolbox deterministic.")
    parser.add_argument("--no_mp3_support", action="store_true", help=\
        "If True, disallows loading mp3 files to prevent audioread errors when ffmpeg is not installed.")
    args = parser.parse_args()


    # ## Load the models one by one.

    encoder.load_model(args.enc_model_fpath)
    synthesizer = Synthesizer(args.syn_model_fpath)
    vocoder.load_model(args.voc_model_fpath)


    # This fixes the audio file path of the voice we want to clone
    in_fpath = Path(("/Users/hlabs/Desktop/worksfemale.flac").replace("\"", "").replace("\'", ""))

    preprocessed_wav = encoder.preprocess_wav(in_fpath)
    # - If the wav is already loaded:
    original_wav, sampling_rate = librosa.load(str(in_fpath))
    preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)

    embed = encoder.embed_utterance(preprocessed_wav)

    # if args.seed is not None:
    #     torch.manual_seed(args.seed)
    #     synthesizer = Synthesizer(args.syn_model_fpath)


    def speech(text):
            texts = [text]
            embeds = [TTS.embed]

            specs = TTS.synthesizer.synthesize_spectrograms(texts, embeds)
            spec = specs[0]

            generated_wav = vocoder.infer_waveform(spec)
            generated_wav = np.pad(generated_wav, (0, TTS.synthesizer.sample_rate), mode="constant")
            generated_wav = encoder.preprocess_wav(generated_wav)

            #This saves each TTS generation into a file called merge3.wav
            from playsound import playsound
            filename = "merge3.wav" 
            sf.write(filename, generated_wav.astype(np.float32), TTS.synthesizer.sample_rate)
            playsound(filename, TTS.synthesizer.sample_rate)


# TTS.speech("While logistic sigmoid neurons are more biologically plausible than hyperbolic tangent neurons, the latter work better for training multi-layer neural networks. This paper shows that rectifying neurons are an even better model of biological neurons and yield equal or better performance than hyperbolic tangent networks in spite of the hard non-linearity and non-differentiability at zero, creating sparse representations with true zeros, which seem remarkably suitable for naturally sparse data")