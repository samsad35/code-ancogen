import os
from .librispeech import LibriSpeech
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from pathlib import Path
import pandas
import pickle


class LibriSpeechDataset(Dataset):
    def __init__(self, root: str,
                 h5_path: str,
                 train: bool = True,
                 sequence_length: int = 200,
                 load_table: str = None,
                 save_table: str = None
                 ):
        super().__init__()
        if load_table is not None:
            self.table = pandas.read_csv(load_table, delimiter=",")
        else:
            self.libri = LibriSpeech(root=Path(root))
            self.libri.generate_table()
            self.table: pandas.DataFrame = self.libri.table
        if save_table is not None:
            self.table.to_csv(save_table, sep=',', index=False, encoding='utf-8')
        self.train = train
        assert os.path.isfile(h5_path), "Problem with H5 file !"
        self.h5_path = h5_path
        self.h5_bool = True if h5_path is not None else False
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.table)

    def open(self):
        self.hdf5 = h5py.File(self.h5_path, mode='r')

    def read(self, id, name: str):
        codes = np.array(self.hdf5[f'{id}/{name}/codes'])
        pitch = np.array(self.hdf5[f'{id}/{name}/pitch'])
        content = np.array(self.hdf5[f'{id}/{name}/content'])
        loudness = np.array(self.hdf5[f'{id}/{name}/loudness'])
        identity = np.array(self.hdf5[f'{id}/{name}/id'])
        return codes, pitch, content, loudness, identity

    def __getitem__(self, item):
        if not hasattr(self, 'hdf5') and self.h5_path is not None:
            self.open()
        while True:
            line = self.table.iloc[item]
            id = line['id']
            name = line['name']
            self.indices_mel, self.pitch, self.content, self.loudness, self.identity = self.read(id, name)
            self.number_frames = self.content.shape[0]
            if self.number_frames > self.sequence_length + 1:
                break
            else:
                item += 1
        self.current_frame = np.random.randint(0, self.number_frames - (self.sequence_length + 1))
        self.out_content = self.content[self.current_frame: self.current_frame + self.sequence_length]
        self.current_frame = round(1.6 * self.current_frame)
        self.out_indices = self.indices_mel[self.current_frame: self.current_frame + int(self.sequence_length*1.6)]
        self.out_pitch = self.pitch.T[self.current_frame: self.current_frame + int(self.sequence_length*1.6)][:, 0]
        self.out_loudness = self.loudness[self.current_frame: self.current_frame + int(self.sequence_length*1.6)]
        self.out_identity = torch.from_numpy(self.identity).repeat((int(self.sequence_length*1.6),))
        return torch.from_numpy(self.out_indices).type(torch.LongTensor), \
            torch.from_numpy(self.out_pitch).type(torch.LongTensor), \
            torch.from_numpy(self.out_content).type(torch.LongTensor), \
            torch.from_numpy(self.out_loudness).type(torch.LongTensor), \
            self.out_identity.type(torch.LongTensor)


if __name__ == '__main__':
    data = LibriSpeechDataset(root=Path(r"D:\These\data\Audio\LibriSpeech\test-clean"))
    print(data[0].shape)