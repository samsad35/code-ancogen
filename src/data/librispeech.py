import os
import pandas
import numpy as np
from tqdm import tqdm
import glob
from pathlib import Path


class LibriSpeech:
    def __init__(self, root: Path = None, ext: str = "flac"):
        self.root = root
        self.length = len(glob.glob(f"{root}/**/**/*.{ext}"))
        self.table = None

    @staticmethod
    def __generator__(directory: Path):
        all_dir = os.listdir(directory)
        for d in all_dir:
            yield d, directory / f'{d}'

    def generator(self):
        for id, id_path in self.__generator__(self.root):
            for _, _path in self.__generator__(id_path):
                for name, path in self.__generator__(Path(_path)):
                    if ".txt" in name:
                        continue
                    if ".trans" in name:
                        continue
                    yield id, name, path

    def generate_table(self):
        name_list = []
        path_list = []
        path_ref_list = []
        id_list = []
        with tqdm(total=self.length, desc=f"Create table (LibriSpeech): ") as pbar:
            for id, name, path in self.generator():
                path_list.append(path)
                name_list.append(name)
                path_ref_list.append(path)
                id_list.append(id)
                pbar.update(1)
        self.table = pandas.DataFrame(np.array([id_list, name_list, path_list, path_ref_list]).transpose(),
                                      columns=['id', 'name', 'path', 'path_ref'])


if __name__ == '__main__':
    rvl = LibriSpeech(root=Path(r"C:\Users\asus9\Documents\data\LibriSpeech\test-clean"))
    rvl.generate_table()
    line = rvl.table.iloc[40]
    print(rvl.table)
    print(line['id'])