import numpy as np
from pydub import AudioSegment
import random
import sys
import io
import os
import glob
from td_utils import *
from argparse import ArgumentParser
#from record import *

x = graph_spectrogram("audio_examples/example_train.wav")
_, data = wavfile.read("audio_examples/example_train.wav")
# Tx: The number of time steps input to the model from the spectrogram
# n_freq: Number of frequencies input to the model at each time step of the spectrogramp
n_freq, Tx = x.shape
# the training examples is 10 seconds divided into 1375 step/units
# The number of time steps in the output of our model
Ty = 1375
# Load audio segments using pydub
activates, negatives, backgrounds = load_raw_audio()
#x, y = create_training_example(backgrounds[0], activates, negatives)
#plt.plot(y[0])
# Load preprocessed training examples
X = np.load("./XY_train/X.npy")
Y = np.load("./XY_train/Y.npy")
# Development set
X_dev = np.load("./XY_dev/X_dev.npy")
Y_dev = np.load("./XY_dev/Y_dev.npy")

from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam
import threading
import pyaudio
import wave

FORMAT = pyaudio.paInt16
WAVE_OUTPUT_FILENAME_PREFIX = "output"
WAVE_SUFFIX = ".wav"

model=None

class Listening:
    def __init__(self, dur=10, rate=44100, channels=2, chunk=1024, max=3):
        self.audio = pyaudio.PyAudio()
        self.DURATION=dur
        self.RATE=rate
        self.CHANNELS=channels
        self.CHUNK=chunk
        self.c=0
        self.shut=False
        self.threads = []
        self.MAX=max
    def record(self, sec):
        stream = self.audio.open(format=FORMAT, channels=self.CHANNELS,
                                 rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)
        frames = []
        for i in range(0, int(self.RATE / self.CHUNK * self.DURATION)):
            data = stream.read(self.CHUNK)
            frames.append(data)
        stream.stop_stream()
        stream.close()

        self.out=WAVE_OUTPUT_FILENAME_PREFIX+str(self.c)+WAVE_SUFFIX
        wf = wave.open(self.out, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        self.c=self.c+1
    def background(self, callback):
        while(not self.shut and self.c<=self.MAX):
            self.record(self.DURATION)
            #td = threading.Thread(target=callback, args=(self.out,))
            callback(self.out)
            #...
    def close(self):
        self.shut=True
        for td in  self.threads:
            td.join()
        self.audio.terminate()


def build_model(input_shape):
    """
    Function creating the model's graph in Keras.

    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """
    X_input = Input(shape = input_shape)

    X = Conv1D(filters=196, kernel_size=15, strides=4)(X_input)  # CONV1D
    X = BatchNormalization()(X)                                 # Batch normalization
    X = Activation('relu')(X)                                 # ReLu activation
    X = Dropout(rate=0.8)(X)                                 # dropout (use 0.8)

    X = GRU(units=128, return_sequences=True)(X)   # GRU (use 128 units and return the sequences)
    X = Dropout(rate=0.8)(X)                                 # dropout (use 0.8)
    X = BatchNormalization()(X)                                 # Batch normalization

    X = GRU(units=128, return_sequences=True)(X)   # GRU (use 128 units and return the sequences)
    X = Dropout(rate=0.8)(X)                                 # dropout (use 0.8)
    X = BatchNormalization()(X)                                 # Batch normalization
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)

    X = TimeDistributed(Dense(1, activation="sigmoid"))(X) # time distributed  (sigmoid)

    model = Model(inputs = X_input, outputs = X)

    return model

def detect_triggerword(filename):
    plt.subplot(2, 1, 1)
    x = graph_spectrogram(filename)
    # the spectrogram outputs (freqs, Tx) and we want (Tx, freqs) to input into the model
    x  = x.swapaxes(0,1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)

    plt.subplot(2, 1, 2)
    plt.plot(predictions[0,:,0])
    plt.ylabel('probability')
    #plt.show()
    plt.savefig('./out/detection.png')
    return predictions

chime_file = "audio_examples/chime.wav"
def chime_on_activate(filename, predictions, threshold):
    audio_clip = AudioSegment.from_wav(filename)
    chime = AudioSegment.from_wav(chime_file)
    Ty = predictions.shape[1]
    # Step 1: Initialize the number of consecutive output steps to 0
    consecutive_timesteps = 0
    # Step 2: Loop over the output steps in the y
    for i in range(Ty):
        # Step 3: Increment consecutive output steps
        consecutive_timesteps += 1
        # Step 4: If prediction is higher than the threshold and more than 75 consecutive output steps have passed
        if predictions[0,i,0] > threshold and consecutive_timesteps > 75:
            # Step 5: Superpose audio and background using pydub
            audio_clip = audio_clip.overlay(chime, position = ((i / Ty) * audio_clip.duration_seconds)*1000)
            # Step 6: Reset consecutive output steps to 0
            consecutive_timesteps = 0
    audio_clip.export("chime_output.wav", format='wav')

# Preprocess the audio to the correct format
def preprocess_audio(filename):
    # Trim or pad audio segment to 10000ms
    padding = AudioSegment.silent(duration=10000)
    segment = AudioSegment.from_wav(filename)[:10000]
    segment = padding.overlay(segment)
    # Set frame rate to 44100
    segment = segment.set_frame_rate(44100)
    ## Export as wav
    segment.export(filename, format='wav')
    return filename

def detect_callback(filename):
    chime_threshold = 0.5
    filename = preprocess_audio(filename)
    prediction = detect_triggerword(filename)
    chime_on_activate(filename, prediction, chime_threshold)


parser = ArgumentParser()
parser.add_argument('-t', '--train', type=bool, default=False)
args = parser.parse_args()
do_train = args.train

if do_train:
    # train your model
    print('building...')
    model = build_model(input_shape = (Tx, n_freq))
    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
    model.fit(X, Y, batch_size = 5, epochs=10)
    model.save('./models/tr_model.h5')
else:
    print('loading...')
    #TODO what is the difference, does building take long time?! or there are missing details?
    model = load_model('./models/tr_model.h5')
    #listen to the microphone
# evaluation
loss, acc = model.evaluate(X_dev, Y_dev)
print("Dev set accuracy = ", acc)
listener = Listening()
listener.background(detect_callback)
