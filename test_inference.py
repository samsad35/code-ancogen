from pathlib import Path
import numpy as np
from hydra.core.global_hydra import GlobalHydra
import random
from tqdm import tqdm
from src import SpeechVQVAE, HifiganGenerator, AnCoGen, NestAnCoGen, LibriSpeech
from hydra import initialize, compose
import unittest
import torch
import warnings
warnings.filterwarnings("ignore")

LENGTH = 400  # 200, 400
TIME_PATCH = 4
SIZE_PATCH = 4
# ----------------------------------------------------------------------------------------------------------------------
# path = r"./checkpoint/ANCOGEN/2024-4-8/12-15"
path = r"checkpoint/ANCOGEN/21-43"
# path = r"checkpoint/ANCOGEN/improved/19-33"  # Improved
# ----------------------------------------------------------------------------------------------------------------------


class TestAnCoGen(unittest.TestCase):
    def load(self):
        """
        Load the models and the dataset.

        This function loads the VQ-VAE, the HIFI-GAN generator, the AnCoGen model, and the LibriSpeech dataset.
        """
        GlobalHydra.instance().clear()  # Clear previous initialization
        initialize(config_path=f"{path}/config_ancogen")
        cfg = compose(config_name="config")

        """ Device """
        self.device = torch.device('cuda')

        """ Model """
        # VQ-VAE:
        self.vqvae = SpeechVQVAE(**cfg.vqvae, mel_input=True)
        # self.vqvae.load(r"./checkpoint/SPEECH_VQVAE/2024-2-3/18-12/model_checkpoint")
        self.vqvae.load(r"./checkpoint/SPEECH_VQVAE/2024-4-19/15-57/model_checkpoint")
        # self.vqvae.load(r"./checkpoint/SPEECH_VQVAE/2024-11-26/9-53/model_checkpoint")  # Improved
        self.vqvae.to(self.device)

        # HIFI-GAN:
        self.generator = HifiganGenerator()
        self.generator.load_pretrained_model(load_path=r"./checkpoint/HIFI_GAN-librispeech-360/model-best.pt")
        self.generator.to(self.device)
        self.generator.eval()

        # Our model:
        self.model = AnCoGen(*tuple(dict(cfg.model.dimensions).values()),
                             **cfg.model.parameters,
                             alpha=(1., 1.))
        self.model.load(path_model=f"{path}/model_checkpoint")
        self.model.to(self.device)
        # self.model.eval()

        self.ancogen = NestAnCoGen(ancogen=self.model, hifigan=self.generator, vqvae=self.vqvae, improved=False)

        """ Dataset """
        self.librispeech = LibriSpeech(root=Path(r"/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean"))
        # self.librispeech = LibriSpeech(root=Path(r"/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/train-clean-100"))
        self.librispeech.generate_table()

    def test_analyse(self):
        """
        Test the analyse function of NestAnCoGen.
        """
        self.load()
        PATH_AUDIO = self.librispeech.table.iloc[1999]["path"]

        # Preprocess the audio
        audio = self.ancogen.preprocess(PATH_AUDIO)

        # Analyse the audio with the NestAnCoGen
        audio, attributes = self.ancogen.analyse(audio, apply_max=True, plot_bool=True, attribute_name="pitch")

        # Print the results
        print(attributes)

    def test_generation(self):
        """
        Test the generate function of NestAnCoGen.
        """
        self.load()
        ancogen = NestAnCoGen(ancogen=self.model, hifigan=self.generator, vqvae=self.vqvae)
        PATH_AUDIO = r"wavs/audio/7113-86041-0000.flac"
        # Generate output
        generated = ancogen.generate(path=PATH_AUDIO, from_attributes=None, save_dir="wavs", return_metrics=True)
        # Print results
        print(generated)


    def test_ancogen(self):
        """
        Test NestAnCoGen. This test load a model and test the full pipeline and compute metrics.
        """
        self.load()

        # Metrics
        metrics = dict(stoi=[], pesq=[], si_sdr=[], mos=[], similarity=[], bak=[], stoi_v=[])
        length = 200
        lst = list(range(len(self.librispeech.table)))
        random.shuffle(lst)
        lst = lst[:length]
        with tqdm(total=length) as pbar:
            for idx, i in enumerate(lst):
                path = self.librispeech.table.iloc[i]["path"]
                # Generate and compute metrics
                temps = self.ancogen.generate(path=path, from_attributes=None, save_dir="wavs/temps", add_str=str(idx),
                                         return_metrics=True)
                for (m, s) in temps.items():
                    metrics[str(m)].append(s.detach().cpu().numpy() if type(s) == torch.Tensor else s)
                pbar.update(1)
            # Compute mean of metrics
            for (m, s) in metrics.items():
                mean = np.mean(s)
                print(f"{m}: {mean}")