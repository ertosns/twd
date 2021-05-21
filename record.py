import threading
import pyaudio
import wave

FORMAT = pyaudio.paInt16
WAVE_OUTPUT_FILENAME_PREFIX = "output"
WAVE_SUFFIX = ".wav"

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
            td = threading.Thread(target=callback, args=(self.out,))
            td.start()
            self.threads.append(td)
    def close(self):
        self.shut=True
        for td in  self.threads:
            td.join()
        self.audio.terminate()


def test_callback(filename):
    print("callback with {}".format(filename))

act = Listening()
act.background(test_callback)
