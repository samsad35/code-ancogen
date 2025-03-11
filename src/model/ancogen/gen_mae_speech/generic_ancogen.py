import torch
from torch.distributions.dirichlet import Dirichlet
import numpy as np
from einops import repeat, rearrange
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
import math

"""
    Functions
"""


def random_indexes(size: int, random: bool = True):
    forward_indexes = np.arange(size)
    if random:
        np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes


def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))


class TokenShuffle(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, tokens: torch.Tensor, ratio, random: bool = True):
        T, B, C = tokens.shape
        remain_T = torch.round(T * (1 - ratio)).to(torch.int)
        indexes = [random_indexes(T, random=random) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(
            tokens.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(
            tokens.device)
        tokens = take_indexes(tokens, forward_indexes)
        tokens = tokens[:remain_T]
        return tokens, forward_indexes, backward_indexes


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        return self.pe[:x.size(0)]


"""
    Encoder AnCoGen
"""


class Encoder(torch.nn.Module):
    """
        Encoder Of Contrastive Multimodal Dynamical Masked AutoEncoder
    """

    def __init__(self,
                 # [seq_length_audio, num_indices_audio, num_embeddings_audio, emb_dim_audio],
                 *args,
                 num_layer: int = None,
                 num_head: int = None,
                 alpha: tuple = None,
                 pos_embedding_trained: bool = True,
                 vqvae_speech_embedding=None,
                 mlp_ratio=4.0
                 ):
        super(Encoder, self).__init__()
        self.dirichlet = Dirichlet(torch.tensor([alpha[0], alpha[1]]))
        self.dirichlet_condition = Dirichlet(torch.tensor([1.0, 1.0]))
        self.shuffle = TokenShuffle()

        self.projections = torch.nn.ModuleList([])
        self.modalities = torch.nn.ParameterList([])
        self.positions = torch.nn.ParameterList([])
        for arg in args:
            self.projections.append(torch.nn.Embedding(num_embeddings=arg[2], embedding_dim=arg[3]))
            self.modalities.append(torch.nn.Parameter(torch.zeros(1, 1, arg[3] * arg[1])))
            if pos_embedding_trained:
                self.positions.append(torch.nn.Parameter(torch.zeros(arg[0], 1, arg[3] * arg[1])))
            else:
                self.positions.append(PositionalEncoding(d_model=arg[3] * arg[1], max_len=arg[0]))
        self.transformer = \
            torch.nn.Sequential(
                *[Block(args[0][3] * args[0][1], num_head, mlp_ratio=mlp_ratio) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(args[0][3] * args[0][1])
        self.pos_embedding_trained = pos_embedding_trained
        self.vqvae_speech_embedding = vqvae_speech_embedding
        self.init_weight()

        """ Parameters """
        self.args = args

    def init_weight(self):
        if self.pos_embedding_trained:
            for position in self.positions:
                trunc_normal_(position, std=.02)
        for modality in self.modalities:
            trunc_normal_(modality, std=.02)
        if self.vqvae_speech_embedding is not None:
            self.projections[0] = torch.nn.Embedding.from_pretrained(embeddings=self.vqvae_speech_embedding,
                                                                     freeze=True)

    def forward(self, *args, ratio=None, random=True):
        data = []
        for i, arg in enumerate(args):
            x = rearrange(arg, 'b t c -> t b c')
            x = self.projections[i](x).reshape(self.args[i][0], -1, self.args[i][1] * self.args[i][3])
            if self.pos_embedding_trained:
                x = x + self.positions[i] + self.modalities[i]
            else:
                x = x + self.positions[i](x) + self.modalities[i]
            data.append(x)

        # Shuffle + Transfomer
        if ratio is not None:
            ratio_audio, ratio_meta = ratio[0], ratio[1]
        else:
            ratio_audio, ratio_meta = self.dirichlet.sample()
        if ratio_audio > ratio_meta + 0.2:
            ratio_id = torch.tensor(0.0)
        else:
            ratio_id = torch.tensor(1.0)

        forward = []
        backward = []
        for i, d in enumerate(data):
            if i == 0:  # audio
                x, forward_, backward_ = self.shuffle(d, ratio=ratio_audio, random=random)
            elif i == len(self.args)-1:  # identity
                x, forward_, backward_ = self.shuffle(d, ratio=ratio_id, random=random)
            else:
                x, forward_, backward_ = self.shuffle(d, ratio=ratio_meta, random=random)
            data[i] = x
            forward.append(forward_)
            backward.append(backward_)

        x_cat = torch.cat(tuple(data), dim=0)
        x_cat = rearrange(x_cat, 't b c -> b t c')
        z_cat = self.layer_norm(self.transformer(x_cat))
        z_cat = rearrange(z_cat, 'b t c -> t b c')
        z = []
        length = 0
        for d in data:
            z_ = z_cat[length:length+d.shape[0]]
            length += d.shape[0]
            z.append(z_)
        return z, forward, backward


"""
    Decoder GenMaeSpeech
"""


class Decoder(torch.nn.Module):
    """
        Decoder Without Cross-Attention
    """

    def __init__(self,
                 # [seq_length_audio, num_indices_audio, num_embeddings_audio, emb_dim_audio],
                 *args,
                 num_layer: int = None,
                 num_head: int = None,
                 dim_tokens: list = None,
                 pos_embedding_trained: bool = True,
                 mlp_ratio=4.0,
                 ):
        super(Decoder, self).__init__()
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, args[0][3] * args[0][1]))

        self.modalities = torch.nn.ParameterList([])
        self.positions = torch.nn.ParameterList([])
        self.heads = torch.nn.ModuleList([])
        for i, arg in enumerate(args):
            self.modalities.append(torch.nn.Parameter(torch.zeros(1, 1, arg[3] * arg[1])))

            if pos_embedding_trained:
                self.positions.append(torch.nn.Parameter(torch.zeros(arg[0], 1, arg[3] * arg[1])))
            else:
                self.positions.append(PositionalEncoding(d_model=arg[3] * arg[1], max_len=arg[0]))
            self.heads.append(torch.nn.Linear(arg[3], dim_tokens[i]))

        self.transformer = \
            torch.nn.Sequential(
                *[Block(args[0][3] * args[0][1], num_head, mlp_ratio=mlp_ratio) for _ in range(num_layer)])

        self.pos_embedding_trained = pos_embedding_trained
        self.init_weight()

        """ Parameters """
        self.args = args

    def init_weight(self):
        if self.pos_embedding_trained:
            for position in self.positions:
                trunc_normal_(position, std=.02)
        for modality in self.modalities:
            trunc_normal_(modality, std=.02)

    def forward(self, z, backward):
        T = []
        for z_ in z:
            T.append(z_.shape[0])

        for i, (z_, backward_) in enumerate(zip(z, backward)):
            tmp = torch.cat([z_, self.mask_token.expand(backward_.shape[0] - z_.shape[0], z_.shape[1], -1)], dim=0)
            tmp = take_indexes(tmp, backward_)
            if self.pos_embedding_trained:
                tmp = tmp + self.positions[i] + self.modalities[i]
            else:
                tmp = tmp + self.positions[i](tmp) + self.modalities[i]
            z[i] = tmp

        # Transformer
        z_cat = torch.cat(tuple(z), dim=0)
        z_cat = rearrange(z_cat, 't b c -> b t c')
        z_cat = self.transformer(z_cat)
        z_cat = rearrange(z_cat, 'b t c -> t b c')

        # Masks
        mask_all = []
        for i, (t, backward_) in enumerate(zip(T, backward)):
            mask = torch.zeros_like(z_cat[:self.args[i][0]])
            mask[t:] = 1
            mask = take_indexes(mask, backward_)
            mask = rearrange(mask, 't b (c d) -> b t c d', c=self.args[i][1])[:, :, :, 0]
            mask_all.append(mask)

        # outputs
        length = 0
        for i, (t, head) in enumerate(zip(T, self.heads)):
            tmp = z_cat[length:length + self.args[i][0]]
            tmp = rearrange(tmp, 't b (c d) -> b t c d', c=self.args[i][1])
            tmp = head(tmp)
            length = length + self.args[i][0]
            z[i] = tmp

        return z, mask_all


"""
    AnCoGen
"""


class AnCoGen(torch.nn.Module):
    """
        AnCoGen Model

        Args:
            *args: Arguments for the Encoder and Decoder
            encoder_num_layer: Number of layers in the Encoder
            encoder_num_head: Number of heads in the Encoder
            decoder_num_layer: Number of layers in the Decoder
            decoder_num_head: Number of heads in the Decoder
            alpha: Parameters for the Dirichlet distribution
            pos_embedding_trained: Whether to use a learned positional embedding
            vqvae_speech_embedding: The embedding matrix for the VQVAE speech model
            dim_tokens: The dimension of the tokens for the Decoder
            mlp_ratio: The ratio of the MLP hidden size to the embedding size

    """

    def __init__(self,
                 *args,
                 encoder_num_layer: int = None,
                 encoder_num_head: int = None,
                 decoder_num_layer: int = None,
                 decoder_num_head: int = None,
                 alpha: tuple = None,
                 pos_embedding_trained: bool = True,
                 vqvae_speech_embedding=None,
                 dim_tokens: list = None,
                 mlp_ratio=4.0
                 ):
        """
            Initialize the AnCoGen model

            Args:
                *args: Arguments for the Encoder and Decoder
                encoder_num_layer: Number of layers in the Encoder
                encoder_num_head: Number of heads in the Encoder
                decoder_num_layer: Number of layers in the Decoder
                decoder_num_head: Number of heads in the Decoder
                alpha: Parameters for the Dirichlet distribution
                pos_embedding_trained: Whether to use a learned positional embedding
                vqvae_speech_embedding: The embedding matrix for the VQVAE speech model
                dim_tokens: The dimension of the tokens for the Decoder
                mlp_ratio: The ratio of the MLP hidden size to the embedding size
        """
        super(AnCoGen, self).__init__()
        self.encoder_num_head = encoder_num_head
        self.mlp_ratio = mlp_ratio

        self.encoder = Encoder(*args,
                               num_layer=encoder_num_layer,
                               num_head=encoder_num_head,
                               alpha=alpha,
                               pos_embedding_trained=pos_embedding_trained,
                               mlp_ratio=mlp_ratio,
                               vqvae_speech_embedding=vqvae_speech_embedding)

        self.decoder = Decoder(*args,
                               num_layer=decoder_num_layer,
                               num_head=decoder_num_head,
                               dim_tokens=dim_tokens,
                               pos_embedding_trained=pos_embedding_trained,
                               mlp_ratio=mlp_ratio)

    def forward(self, *args, ratio=None, random=True):
        """
        Forward pass for the AnCoGen model.

        Args:
            *args: Input arguments for the Encoder.
            ratio: The ratio for shuffling in the Encoder.
            random: Whether to apply random shuffling.

        Returns:
            z: The output from the Decoder after processing.
            mask: The mask applied during decoding.
        """
        # Encode the inputs and perform shuffling
        z, forward, backward = self.encoder(*args, ratio=ratio, random=random)

        # Decode the encoded inputs
        z, mask = self.decoder(z, backward)

        return z, mask

    def extra_repr(self) -> str:

        named_modules = set()
        for p in self.named_modules():
            named_modules.update([p[0]])
        named_modules = list(named_modules)

        string_repr = ''
        for p in self.named_parameters():
            name = p[0].split('.')[0]
            if name not in named_modules:
                string_repr = string_repr + '(' + name + '): ' \
                              + 'tensor(' + str(tuple(p[1].shape)) + ', requires_grad=' + str(
                    p[1].requires_grad) + ')\n'

        return string_repr

    def load(self, path_model: str):
        """
        Load the pre-trained model from a file.

        Args:
            path_model (str): The path to the pre-trained model file.

        Returns:
            None
        """
        # Load the pre-trained model from the file
        checkpoint = torch.load(path_model)
        # Get the state dictionary from the pre-trained model
        state_dict = checkpoint["model"]
        # Load the state dictionary into the current model
        self.load_state_dict(state_dict)
        # Get the loss from the pre-trained model
        loss = checkpoint['loss']
        # Print a message indicating that the model was loaded successfully
        print(f"\t [Model AnCoGen-MAE is loaded successfully with loss = {loss}]")


if __name__ == '__main__':
    a = torch.ones((5, 2, 3))
    shuffle = TokenShuffle()
    a, b, c = shuffle(a, ratio=torch.tensor(0.5))

    print("-"*50)

    audio = torch.rand((5, 640, 20)).type(torch.LongTensor)
    pitch = torch.rand((5, 40, 5)).type(torch.LongTensor)
    content = torch.rand((5, 200, 1)).type(torch.LongTensor)
    # duration = torch.rand((5, 111, 1)).type(torch.LongTensor)
    loudness = torch.rand((5, 40, 5)).type(torch.LongTensor)
    identity = torch.rand((5, 40, 5)).type(torch.LongTensor)

    ancogen = AnCoGen([640, 20, 256, 8],
                      [40, 5, 100, 32],
                      [200, 1, 256, 160],
                      [40, 5, 200, 32],
                      [40, 5, 256, 32],
                      dim_tokens=[256, 256, 256, 256, 256],
                      encoder_num_head=4,
                      encoder_num_layer=6,
                      decoder_num_layer=6,
                      decoder_num_head=4,
                      alpha=(1.0, 1.0),
                      mlp_ratio=2.0)
    print(ancogen)
    output, mask = ancogen(audio, pitch, content, loudness, identity)
    print(output[0].shape)

