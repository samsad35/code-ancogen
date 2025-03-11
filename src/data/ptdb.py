import numpy as np
import pandas
import librosa
import os
import pysptk
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt


class PTDB:
    def __init__(self, path: str,
                 sentence=None,
                 sex=None):
        if sex is None:
            sex = ['FEMALE', 'MALE']
        if sentence is None:
            sentence = ['sa', 'sx', 'si']
        self.PATH = path
        self.sex = sex
        self.sentence = sentence
        self.get_table()

    def generator_files(self):
        for sex in self.sex:
            path_mic = f"{self.PATH}/{sex}/MIC"
            path_lar = f"{self.PATH}/{sex}/LAR"
            for dir_mic in os.listdir(path_mic):
                temps_mic = f"{path_mic}/{dir_mic}"
                temps_lar = f"{path_lar}/{dir_mic}"
                files_mic = os.listdir(temps_mic)
                for wav_mic in files_mic:
                    wav_lar = wav_mic.replace("mic", "lar")
                    if wav_mic.split(".")[0].split('_')[-1][:2] in self.sentence:
                        yield f"{temps_mic}/{wav_mic}", f"{temps_lar}/{wav_lar}", '_'.join(
                            wav_mic.split('.')[0].split('_')[1:])

    @staticmethod
    def get_pitch_ref(path: str, hopsize: int = 320):
        y, fs = librosa.load(path, sr=16000)
        y = y / np.max(np.abs(y))
        f_0 = pysptk.swipe(y.astype(np.float64), fs=16000, hopsize=hopsize, max=400)
        flag = np.invert((f_0 == 0.0))
        # plt.plot(f_0)
        # plt.show()
        return f_0, flag

    def get_table(self):
        print(" Table creation : ", end="\r")
        data = dict(mic=[], lar=[], name=[])
        for mic, lar, name in self.generator_files():
            data['mic'].append(mic)
            data['lar'].append(lar)
            data['name'].append(name)
        self.table = pandas.DataFrame.from_dict(data)
        print(" Table creation : ok ", end="\r")


if __name__ == '__main__':
    data = PTDB(path=r"C:\Users\asus9\Documents\data\PTDB_TUG_orig")
    data.get_table()
    print(data.table)
    # data.h5_creation()
    for mic, lar, name in data.generator_files():
        pitch, flag = data.get_pitch_ref(lar)
        print(pitch.shape)