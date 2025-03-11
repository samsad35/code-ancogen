# AnCoGen: Analysis, Control and Generation of Speech with a Masked Autoencoder
[![Generic badge](https://img.shields.io/badge/<STATUS>-<in_progress>-<COLOR>.svg)](https://github.com/samsad35/code-ancogen)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://samsad35.github.io/site-ancogen)
[![PyPI version fury.io](https://badge.fury.io/py/ansicolortags.svg)](https://test.pypi.org/project/ancogen/)

![VQ-VAE](image/overview.svg)

This repository contains the code associated with the following publication:
> **AnCoGen: Analysis, Control and Generation of Speech with a Masked Autoencoder**<br> Samir Sadok, Simon Leglaive, Laurent Girin, Gaël Richard, Xavier Alameda-Pineda<br> International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2025.

If you use this code for your research, please cite the above paper.

Useful links:
- [Abstract](https://arxiv.org/abs/2501.05332)
- [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10887856)
- [Demo website with qualitative results](https://samsad35.github.io/site-ancogen)


## Setup 
- [ ] Pypi:  
  - ```pip install -i https://test.pypi.org/simple/ ancogen --no-deps```
- [x] Install the package locally (for use on your system):  
  - In the current directory: ```pip install -e .```
- [x] Virtual Environment: 
  - ```conda create -n ancogen python=3.9```
  - ```conda activate ancogen```
  - In the current directory: ```pip install -r requirements.txt```

## Usage

### Pretrained models 

After loading the weights of the pre-trained models: speechVQVAE, HIFIGAN and AnCoGen, put them all in the NestAnCoGen class.

```python
from src import NestAnCoGen

ancogen = NestAnCoGen(ancogen=model, hifigan=generator, vqvae=vqvae, improved=False)
```

| Model         	 |        Link    	        | 
|:---------------:|:-----------------------:|
| Speech-VQVAE 	  |       [link](https://huggingface.co/samir-sadok/AnCoGen-VQVAE-LibriSpeech/tree/main) 	        | 
|   HiFi-GAN 	    |       [link](https://huggingface.co/samir-sadok/AnCoGen-HiFiGAN/tree/main) 	        | 	
|    AnCoGen 	    | [link]()  / [link]() (Improved) 	 | 


### Analysis

To do **analysis** with AnCoGen ([link](test_inference.py), test_analyse), which correspond to the estimation of the speech attributes from a Mel-spectrogram.
_Please see the paper for a complete description of the attributes._
```python
"""
Test the analyse function of AnCoGen. 
"""
PATH_AUDIO = "path_wav_signal.wav"

# Preprocess the audio
audio = ancogen.preprocess(PATH_AUDIO)

# Analyse the audio with the AnCoGen
audio, attributes = ancogen.analyse(audio, apply_max=True)

# Pitch estimation + plotting
audio, attributes = ancogen.analyse(audio, apply_max=True, attribute_name="pitch", plot_bool=True)
```


### Generation
To do speech **analysis-resynthesis** mapping wih AnCoGen ([link](test_inference.py), test_generation) which are simply obtained by using AnCoGen to map a Mel-spectrogram to the corresponding speech attributes (analysis stage) and then back to the Mel-spectrogram (generation stage).

```python
"""
Test the generation function of AnCoGen. 
"""
PATH_AUDIO = "path_wav_signal.wav"

# Generate output
generated = ancogen.generate(path=PATH_AUDIO, from_attributes=None, save_dir="wavs", return_metrics=True)
```


### Control
To do **analysis, transformation, and synthesis** with AnCoGen, where the speech attributes are controlled between the analysis and generation stages in order to perform speech denoising (by increasing the SNR attribute), pitch shifting, dereverberation (by increasing the C50 attribute) or voice conversion (by controlling the speaker identity attribute).

```python
"""
Test the control function of AnCoGen. 
"""
PATH_AUDIO = "path_wav_signal.wav"

PATH_AUDIO = "path_wav_signal.wav"

# Preprocess the audio
audio = ancogen.preprocess(PATH_AUDIO)

# Analyse the audio with the AnCoGen
audio, attributes = ancogen.analyse(audio, apply_max=True)

# Control with AnCoGen
ancogen.pitch_control(attributes, target_pitch, **kwargs) 

ancogen.content_control(attributes, target_content_index, **kwargs)

ancogen.snr_control(attributes, target_snr, **kwargs)

ancogen.c50_control(attributes, target_c50, **kwargs)

ancogen.voice_conversion(target_identity: str, source_signal: str, save_dir: str = '')
```

## License
> GNU Affero General Public License (version 3), see LICENSE.txt.