from typing import Union
import librosa
from matplotlib import pyplot as plt
from src.tools import compute_dnsmos, LogMelSpectrogram
from .ancogen import AnCoGen, HifiganGenerator, SpeechVQVAE
from einops import rearrange
import torch
import torchaudio
import soundfile as sf
from pystoi import stoi
import os
import torchaudio.transforms as T
from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE
from resemblyzer import preprocess_wav, VoiceEncoder

class NestAnCoGen:
    """
        AnCoGen: Analysis, Control and Generation of Speech with a Masked Autoencoder
        Samir Sadok, Simon Leglaive, Laurent Girin, Gaël Richard, Xavier Alameda-Pineda
        ICASSP 2025.
    """

    def __init__(self, ancogen: AnCoGen,
                 hifigan: HifiganGenerator,
                 vqvae: SpeechVQVAE,
                 improved: bool = False,):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.ancogen = ancogen.to(self.device)
        self.hifigan = hifigan.to(self.device)
        self.vqvae = vqvae.to(self.device)
        self.log_mel = LogMelSpectrogram()
        self.resampler = T.Resample(orig_freq=22050, new_freq=16000)
        self.objective_model = SQUIM_OBJECTIVE.get_model()
        self.subjective_model = SQUIM_SUBJECTIVE.get_model()
        self.resemblyzer = VoiceEncoder("cpu")
        self.improved = improved
        # ------------------------
        self.TIME_PATCH = 4
        self.LENGTH = 400
        self.ATTRIBUTES = dict(audio=0, pitch=1, content=2, loudness=3, snr=4, c50=5, identity=6)

    def to_token(self, x):
        """
            Convert a time series signal of shape (batch, time * TIME_PATCH)
            into a sequence of tokens of shape (batch, time, TIME_PATCH)
        """
        t = x.shape[1] // self.TIME_PATCH
        x = rearrange(x, 'b (t l) -> b t l', t=t, l=self.TIME_PATCH)
        return x

    def unpatchify(self, x, time_patch: int):
        """
            Convert a sequence of tokens of shape (batch, time, TIME_PATCH)
            into a time series signal of shape (batch, time * TIME_PATCH)

            Args:
                x (torch.Tensor): The input sequence
                time_patch (int): The time patch size

            Returns:
                torch.Tensor: The output signal
        """
        x = rearrange(x, 'b t l -> b (t l)', l=self.TIME_PATCH, t=100)
        return x

    def save_wav(self, indices, save: str = None):
        """
        Decode indices to waveform and optionally save to a file.

        Args:
            indices (torch.Tensor): Indices to be decoded.
            save (str, optional): Path to save the waveform. If None, do not save.

        Returns:
            torch.Tensor: The decoded audio signal.
        """
        # Decode the indices to obtain the audio representation
        audio = self.vqvae.decode(indices)

        # Generate the audio signal using HiFiGAN
        signal = self.hifigan(torch.transpose(audio, 0, 1).unsqueeze(0))

        # Squeeze and move the signal to CPU for further processing
        signal = signal.squeeze(0).cpu().detach()

        # If a save path is provided, save the audio signal
        if save is not None:
            torchaudio.save(save, signal, 16000)

        # Return the generated audio signal
        return signal

    def preprocess(self, path: str):
        """
        Preprocesses the audio file at the given path, converting it to a
        spectrogram and obtaining codebook indices.

        Args:
            path (str): The file path to the audio file.

        Returns:
            torch.Tensor: The codebook indices of the spectrogram.
        """
        # Load the audio file
        audio, sr = torchaudio.load(path)

        # Resample if sample rate is not 16000
        if sr != 16000:
            self.resampler = T.Resample(orig_freq=sr, new_freq=16000)
            audio = self.resampler(audio)
        spectrogram = self.log_mel(audio)[0]

        spectrogram = torch.transpose(spectrogram, 0, 1).unsqueeze(1)
        spectrogram = spectrogram.to(self.device)
        indices = self.vqvae.get_codebook_indices(spectrogram)

        if indices.shape[0] <= self.LENGTH:
            indices = indices.repeat(8, 1)
            # print(f"Repeated tensor ! ")
        return indices

    def post_process(self, input_1, input_2, mask):
        """
        Post-processes the given codebook indices, applying the mask and
        reconstructing the original signal.

        Args:
            input_1 (torch.Tensor): The original codebook indices.
            input_2 (torch.Tensor): The reconstructed codebook indices.
            mask (torch.Tensor): The binary mask to apply to the codebook indices.

        Returns:
            original_signal (torch.Tensor): The original signal.
            reconstructed_signal (torch.Tensor): The reconstructed signal.
            masked_signal (torch.Tensor): The masked signal.
        """
        # Get the index of the max value in the reconstructed signal
        _, input_2 = torch.max(input_2.data, -1)

        # Apply the mask to the reconstructed signal
        input_2 = (input_2 * mask + input_1 * (~mask.to(torch.bool))).type(
            torch.int64)

        # Get the masked signal by applying the inverse of the mask to the original signal
        input_3 = (input_1 * (~mask.to(torch.bool))).type(torch.int64)

        # Save the original, reconstructed and masked signals to the disk
        original_signal = self.save_wav(input_1[0], save=None)
        reconstructed_signal = self.save_wav(input_2[0], save=None)
        masked_signal = self.save_wav(input_3[0], save=None)

        return original_signal, reconstructed_signal, masked_signal

    def analyse(self, indices, apply_max: bool = False, attribute_name: Union[str, None] = None,
                plot_bool: bool = False, verbose: bool = False):
        """
        :param indices:
        :param apply_max:
        :param attribute_name: "pitch", "content", "loudness", "snr", "c50", "identity"
        :param plot_bool:
        :param verbose:
        :return:
        """
        if verbose:
            print("Analysing with AnCoGen")
        current_frame = 0
        # current_frame = np.random.randint(0, indices.shape[0] - self.LENGTH)
        indices = indices[current_frame:current_frame + self.LENGTH]

        audio = indices.unsqueeze(0).type(torch.LongTensor).to(self.device)

        pitch = torch.rand(self.LENGTH).unsqueeze(0).type(torch.LongTensor).to(self.device)
        content = torch.rand(self.LENGTH // 2, 1).unsqueeze(0).type(torch.LongTensor).to(self.device)
        loudness = torch.rand(self.LENGTH).unsqueeze(0).type(torch.LongTensor).to(self.device)
        snr = torch.rand(self.LENGTH).unsqueeze(0).type(torch.LongTensor).to(self.device)
        c50 = torch.rand(self.LENGTH).unsqueeze(0).type(torch.LongTensor).to(self.device)
        if self.improved:
            identity = torch.rand((100, 32)).unsqueeze(0).type(torch.LongTensor).to(self.device)
        else:
            identity = torch.rand(self.LENGTH).unsqueeze(0).type(torch.LongTensor).to(self.device)

        audio_token = audio
        pitch_token = self.to_token(pitch)
        loudness_token = self.to_token(loudness)
        snr_token = self.to_token(snr)
        c50_token = self.to_token(c50)
        identity_token = self.to_token(identity) if not self.improved else identity

        with torch.no_grad():
            if self.improved:
                predicted, mask = self.ancogen(audio_token, pitch_token, content, loudness_token, identity_token,
                                               ratio=(torch.tensor(0.0), torch.tensor(1.0)),
                                               random=True)
            else:
                predicted, mask = self.ancogen(audio_token, pitch_token, content, loudness_token, snr_token, c50_token,
                                               identity_token,
                                               ratio=(torch.tensor(0.0), torch.tensor(1.0)),
                                               random=True)
        # 1 = all mask, 0 = no mask | first element = audio, second element = pitch

        if apply_max:
            _, audio_predicted = torch.max(predicted[0].data, -1)
            _, pitch_predicted = torch.max(predicted[1].data, -1)
            _, content_predicted = torch.max(predicted[2].data, -1)
            _, loudness_predicted = torch.max(predicted[3].data, -1)
            if not self.improved:
                _, snr_predicted = torch.max(predicted[4].data, -1)
                _, c50_predicted = torch.max(predicted[5].data, -1)
            _, identity_predicted = torch.max(predicted[-1].data, -1)
            if not self.improved:
                predicted = (audio_predicted, pitch_predicted, content_predicted, loudness_predicted,
                             snr_predicted, c50_predicted, identity_predicted)
            else:
                predicted = (audio_predicted, pitch_predicted, content_predicted, loudness_predicted, identity_predicted)
        if attribute_name is not None:
            predicted = predicted[self.ATTRIBUTES[attribute_name]]
            if attribute_name is not "content":
                predicted = self.unpatchify(predicted, time_patch=self.TIME_PATCH)
            if plot_bool:
                print(f"Plotting {attribute_name}.png ...")
                plt.plot(predicted.cpu()[0], linewidth=2, color='b')
                plt.title(f"{attribute_name}", fontsize=16)
                plt.xlabel("Frames", fontsize=12)
                plt.grid(True)
                plt.savefig(f"{attribute_name}.png")
        return audio, predicted

    def generate(self, path: Union[str, None] = None,
                 from_attributes: Union[list, tuple, None] = None,
                 save_dir: str = "",
                 add_str: str = "",
                 return_metrics: bool = False,
                 ):
        """
        Generate a signal from a given path or a set of attributes.

        Args:
            path (str or None): The path to the audio file.
            from_attributes (list, tuple or None): The attributes to generate from.
            save_dir (str): The directory to save the generated files.
            add_str (str): The string to add to the generated file names.
            return_metrics (bool): If True, return the metrics.

        Returns:
            None or dict: If return_metrics is True, return a dictionary containing the metrics.
        """
        if path is not None:
            # Preprocess the audio file
            indices = self.preprocess(path)
            indices, predicted = self.analyse(indices, apply_max=True)
        elif from_attributes is not None:
            predicted = from_attributes
            indices = predicted[0]
        else:
            predicted = None
            raise NotImplementedError
        with torch.no_grad():
            # Generate the signal
            predicted, mask = self.ancogen(*predicted,
                                           ratio=(torch.tensor(1.0), torch.tensor(0.0)),
                                           random=True)

        audio_predicted, mask_audio = predicted[0], mask[0]
        original_signal, reconstructed_signal, masked_signal = self.post_process(indices, audio_predicted, mask_audio)

        # Save the generated files
        os.makedirs(save_dir, exist_ok=True)
        torchaudio.save(os.path.join(save_dir, f"{add_str}-original.wav"), original_signal.cpu().detach(), 16000)
        torchaudio.save(os.path.join(save_dir, f"{add_str}-masked.wav"), masked_signal, 16000)
        torchaudio.save(os.path.join(save_dir, f"{add_str}-reconstruction.wav"), reconstructed_signal.cpu().detach(), 16000)

        if return_metrics:
            # Compute the metrics
            stoi_hyp, pesq_hyp, si_sdr_hyp = self.objective_model(reconstructed_signal[0:1, :].cpu())
            mos = self.subjective_model(reconstructed_signal[0:1, :].cpu(), original_signal[0:1, :].cpu())

            y, fs = librosa.load(path=os.path.join(save_dir, f"{add_str}-reconstruction.wav"))
            dns = compute_dnsmos(y)

            wav_rec = preprocess_wav(os.path.join(save_dir, f"{add_str}-reconstruction.wav"))
            wav_ori = preprocess_wav(os.path.join(save_dir, f"{add_str}-original.wav"))
            emb_rec = self.resemblyzer.embed_utterance(wav_rec)
            emb_ori = self.resemblyzer.embed_utterance(wav_ori)
            similarity = emb_rec @ emb_ori

            denoised, fs = sf.read(os.path.join(save_dir, f"{add_str}-reconstruction.wav"))
            clean, fs = sf.read(os.path.join(save_dir, f"{add_str}-original.wav"))
            d = stoi(clean, denoised, fs, extended=False)

            return dict(stoi=stoi_hyp[0],
                        pesq=pesq_hyp[0],
                        si_sdr=si_sdr_hyp[0],
                        mos=mos[0],
                        bak=dns['bak_mos'],
                        stoi_v=d,
                        similarity=similarity)

    def control_pitch(self, attributes,
                      target: Union[int, torch.Tensor],
                      additive: bool = True,
                      **kwargs
                      ):
        """
        Control the pitch of the generated signal.

        Args:
            attributes (list): The attributes of the signal.
            target (int or torch.Tensor): The target pitch. If int, the pitch is added to the current pitch. If torch.Tensor,
                the pitch is set to the target pitch.
            additive (bool): If True, the target pitch is added to the current pitch. If False, the target pitch is set to the
                target pitch.
            **kwargs: Additional arguments to be passed to the generate method.

        Returns:
            None
        """
        attributes = list(attributes)
        if isinstance(target, int):
            if additive:
                pitch = attributes[1] + target
                # Clip the pitch to be between 0 and 350
                pitch = torch.clip(pitch, min=0, max=350)
            else:
                # Generate a pitch that is a cosine function of time
                t = torch.linspace(start=0, end=400, steps=400).type(torch.LongTensor)
                pitch = torch.cos(2 * 3.14 * 100 * t) * 4 + target
                pitch = pitch.type(torch.LongTensor).unsqueeze(0).to(self.device)
                # Clip the pitch to be between 0 and 350
                pitch = torch.clip(pitch, min=0, max=350)
                # Convert the pitch to a token
                pitch = self.to_token(pitch)
        elif isinstance(target, torch.Tensor):
            pitch = target.type(torch.LongTensor).unsqueeze(0).to(self.device)
            # Clip the pitch to be between 0 and 350
            pitch = torch.clip(pitch, min=0, max=350)
            # Convert the pitch to a token
            pitch = self.to_token(pitch)
        else:
            pitch = attributes[1]
        attributes[1] = pitch
        # Generate the signal with the new pitch
        self.generate(from_attributes=tuple(attributes), **kwargs)

    def control_content(self, attributes,
                        target: Union[int, torch.Tensor],
                        **kwargs):
        """
        Control the content of the generated signal.

        Args:
            attributes (list): The attributes of the signal.
            target (int or torch.Tensor): The target content. If int, the content is added to the current content. If torch.Tensor,
                the content is set to the target content.
            **kwargs: Additional arguments to be passed to the generate method.

        Returns:
            None
        """
        attributes = list(attributes)
        if isinstance(target, int):
            # Add the target content to the current content
            content = attributes[2] + target
            # Clip the content to be between 0 and 100
            content = torch.clip(content, min=0, max=100)
        elif isinstance(target, torch.Tensor):
            # Set the content to the target content
            content = target.type(torch.LongTensor).unsqueeze(0).to(self.device)
            # Clip the content to be between 0 and 100
            content = torch.clip(content, min=0, max=100)
        else:
            # Use the current content
            content = attributes[2]
        attributes[2] = content
        # Generate the signal with the new content
        self.generate(from_attributes=tuple(attributes), **kwargs)

    def control_snr(self, attributes,
                    target: Union[int, torch.Tensor],
                    **kwargs):
        """
        Control the SNR of the generated signal.

        Args:
            attributes (list): The attributes of the signal.
            target (int or torch.Tensor): The target SNR. If int, the SNR is set to the target SNR. If torch.Tensor,
                the SNR is set to the target SNR.
            **kwargs: Additional arguments to be passed to the generate method.

        Returns:
            None
        """
        attributes = list(attributes)
        if isinstance(target, int):
            # Set the SNR to the target SNR
            snr = attributes[4] * 0 + target
            # Clip the SNR to be between 0 and 80
            snr = torch.clip(snr, min=0, max=80)
        elif isinstance(target, torch.Tensor):
            # Set the SNR to the target SNR
            snr = target.type(torch.LongTensor).unsqueeze(0).to(self.device)
            # Clip the SNR to be between 0 and 80
            snr = torch.clip(snr, min=0, max=80)
        else:
            # Use the current SNR
            snr = attributes[4]
        attributes[4] = snr
        # Generate the signal with the new SNR
        self.generate(from_attributes=tuple(attributes), **kwargs)

    def control_c50(self, attributes,
                    target: Union[int, torch.Tensor],
                    **kwargs):
        """
        Control the C50 of the generated signal.

        Args:
            attributes (list): The attributes of the signal.
            target (int or torch.Tensor): The target C50. If int, the C50 is set to the target C50. If torch.Tensor,
                the C50 is set to the target C50.
            **kwargs: Additional arguments to be passed to the generate method.

        Returns:
            None
        """
        attributes = list(attributes)
        if isinstance(target, int):
            # Set the C50 to the target C50
            c50 = attributes[5] * 0 + target
            # Clip the C50 to be between 0 and 60
            c50 = torch.clip(c50, min=0, max=60)
        elif isinstance(target, torch.Tensor):
            # Set the C50 to the target C50
            c50 = target.type(torch.LongTensor).unsqueeze(0).to(self.device)
            # Clip the C50 to be between 0 and 60
            c50 = torch.clip(c50, min=0, max=60)
        else:
            # Use the current C50
            c50 = attributes[4]
        attributes[5] = c50
        # Generate the signal with the new C50
        self.generate(from_attributes=tuple(attributes), **kwargs)

    def control_identity(self,
                         target_identity: str,
                         source_signal: str,
                         save_dir: str = ''):
        """
        Control the identity of the generated signal.

        Args:
            target_identity (str): The target identity.
            source_signal (str): The source signal.
            save_dir (str): The directory to save the generated signal.

        Returns:
            tuple: The reconstructed signal, the target signal, and the source signal.
        """
        # Preprocess the target identity and source signal
        indices_target = self.preprocess(target_identity)
        indices_source = self.preprocess(source_signal)

        # Analyze the target identity and source signal
        indices_target, predicted_target = self.analyse(indices_target, apply_max=True)
        predicted_target = list(predicted_target)
        indices_source, predicted_source = self.analyse(indices_source, apply_max=True)
        predicted_source = list(predicted_source)

        # Replace the pitch of the source signal with the target signal
        predicted_source[-1] = predicted_target[-1]

        # Adjust the mean of the pitch of the source signal to be the same as the target signal
        predicted_source[1] = (predicted_source[1] - torch.mean(
            predicted_source[1].type(torch.FloatTensor))) + torch.mean(predicted_target[1].type(torch.FloatTensor))
        predicted_source[1] = predicted_source[1].type(torch.LongTensor).to(self.device)
        predicted_source[1] = torch.clip(predicted_source[1], min=0, max=350)

        # Generate the signal with the new pitch
        with torch.no_grad():
            predicted, mask = self.ancogen(*predicted_source,
                                           ratio=(torch.tensor(1.0), torch.tensor(0.0)),
                                           random=True)

        # Post-process the generated signal
        audio_predicted, mask_audio = predicted[0], mask[0]
        target_signal, reconstructed_signal, _ = self.post_process(indices_target, audio_predicted, mask_audio)
        torchaudio.save(os.path.join(save_dir, "target.wav"), target_signal.cpu().detach(), 16000)
        torchaudio.save(os.path.join(save_dir, "conversion.wav"), reconstructed_signal.cpu().detach(), 16000)
        source_signal, _, _ = self.post_process(indices_source, audio_predicted, mask_audio)
        torchaudio.save(os.path.join(save_dir, "source.wav"), source_signal.cpu().detach(), 16000)
        return reconstructed_signal, target_signal, source_signal
