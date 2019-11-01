import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


# Number of Linear input connections depends on output of conv2d layers
# and therefore the input image size, so compute it.
def conv2d_size_out(size, kernel_size=5, stride=2):
    return (size - (kernel_size - 1) - 1) // stride + 1


def build_convnet(h, w, c, conv_layers, batch_norm):
    prev_layer_n_channels = c
    convs_and_bns = nn.ModuleList()

    convw, convh = w, h
    for filters, kernel_size, stride in conv_layers:
        convs_and_bns.append(nn.Conv2d(prev_layer_n_channels, filters, kernel_size=kernel_size, stride=stride))
        if batch_norm:
            convs_and_bns.append(nn.BatchNorm2d(filters))
        convs_and_bns.append(nn.ReLU())
        prev_layer_n_channels = filters

        convw, convh = conv2d_size_out(convw, kernel_size, stride), conv2d_size_out(convh, kernel_size, stride)

    # Return network, and size of embedding that it generates
    return nn.Sequential(*convs_and_bns), convw * convh * prev_layer_n_channels


# class Horde(nn.Module):
#
#     def __init__(self, device, h, w, c, heads, outputs,
#                  conv_layers=((16, 5, 2), (32, 5, 2), (32, 5, 2)),
#                  batch_norm=False,
#                  detach_aux_demons=True,
#                  two_streams=False,
#                  phase=1):
#
#         super(Horde, self).__init__()
#         self.device = device
#         self.heads = heads
#         self.n_outputs = outputs
#         self.detach = detach_aux_demons
#         self.two_streams = two_streams
#         self.phase = phase
#
#         # Can't detach if we're learning a separate representation for
#         assert (self.two_streams and not self.detach or not self.two_streams), "Can't detach with two streams."
#
#         self.convs_and_bns, self.linear_input_size = self.build_convnet(h, w, c, conv_layers, batch_norm)
#         if self.two_streams:
#             self.demon_convs_and_bns, _ = self.build_convnet(h, w, c, conv_layers, batch_norm)
#
#         self.main_demon = nn.Linear(self.linear_input_size, outputs)
#         if self.heads > 0:
#             self.prediction_demons = nn.Linear(self.linear_input_size, outputs * heads)
#             self.control_demons = nn.Linear(self.linear_input_size, outputs * heads)
#
#     def forward(self, x):
#         # Main representation
#         z = self.convs_and_bns(x)
#         z = z.view(z.size(0), -1)
#
#         if self.two_streams:
#             # If separate streams, then use the other conv net to get the demon representation
#             demon_z = self.demon_convs_and_bns(x)
#             demon_z = demon_z.view(demon_z.size(0), -1)
#         else:
#             # If same stream, make sure to correctly detach
#             demon_z = z.detach() if self.detach else z
#
#         return SimpleNamespace(embedding=z,
#                                demon_embedding=demon_z,
#                                main_demon=self.main_demon(z),
#                                control_demons=self.control_demons(demon_z).
#                                reshape(demon_z.size(0), self.heads, self.n_outputs) if self.heads > 0 else None,
#                                prediction_demons=self.prediction_demons(demon_z).
#                                reshape(demon_z.size(0), self.heads, self.n_outputs) if self.heads > 0 else None)
#
#     def forward_q(self, x):
#         # Common representation
#         x = self.convs_and_bns(x)
#         x = x.view(x.size(0), -1)
#         return self.main_demon(x)
#
#     def copy_convnet_to_demon_convnet(self):
#         self.demon_convs_and_bns.load_state_dict(self.convs_and_bns.state_dict())


class ACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory

        print (obs_space["image"])
        h, w, c = obs_space["image"]
        conv_layers = ((16, 2, 1), (32, 2, 1))
        self.image_conv, self.image_embedding_size = build_convnet(h, w, c, conv_layers, False)

        # Define image embedding
        # self.image_conv = nn.Sequential(
        #     nn.Conv2d(3, 16, (2, 2)),
        #     nn.ReLU(),
        #     nn.MaxPool2d((2, 2)),
        #     nn.Conv2d(16, 32, (2, 2)),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, (2, 2)),
        #     nn.ReLU()
        # )
        # n = obs_space["image"][0]
        # m = obs_space["image"][1]
        # self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define actor's model
        self.actor = nn.Sequential(
            # nn.Linear(self.embedding_size, 64),
            # nn.Tanh(),
            # nn.Linear(64, action_space.n)
            nn.Linear(self.embedding_size, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            # nn.Linear(self.embedding_size, 64),
            # nn.Tanh(),
            # nn.Linear(64, 1)
            nn.Linear(self.embedding_size, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
