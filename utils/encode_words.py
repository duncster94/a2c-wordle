import torch
from torch import nn
from pathlib import Path

# encodes each word in the vocabulary
with Path("../data/allowable_words.txt").open("r") as f:
    allowable_words = f.read().split("\n")

with Path("../data/answer_words.txt").open("r") as f:
    answer_words = f.read().split("\n")

allowable_words = sorted(allowable_words + answer_words)

alphabet = "abcdefghijklmnopqrstuvwxyz"
char_mapper = {char: idx for idx, char in enumerate(alphabet)}

encoded_words = []
for word in allowable_words:
    encoded_chars = []
    for char in word:
        encoded_char = torch.zeros(26)
        encoded_char[char_mapper[char]] = 1
        encoded_chars.append(encoded_char)
    encoded_word = torch.cat(encoded_chars, dim=-1)
    encoded_words.append(encoded_word)

encoded_words = torch.stack(encoded_words)

torch.save(encoded_words, Path("../data/encoded_words.pt"))

device = "cuda" if torch.cuda.is_available() else "cpu"


class Model(nn.Module):
    """Autoencoder model.
    """

    def __init__(self):
        super(Model, self).__init__()
        self.enc_1 = nn.Linear(130, 64)
        self.enc_2 = nn.Linear(64, 32)
        self.dec_1 = nn.Linear(32, 130)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.enc_1(x)
        x = self.leakyrelu(x)
        emb = self.enc_2(x)
        recon = self.dec_1(emb)

        return emb, recon


model = Model().to(device)
encoded_words = encoded_words.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

n_epochs = 1000
batch_size = 128

criterion = nn.MSELoss()

for epoch in range(n_epochs):

    losses = []

    batches = encoded_words[torch.randperm(len(encoded_words))].split(batch_size)
    for batch in batches:
        optimizer.zero_grad()

        _, reconstructions = model(batch)

        loss = criterion(batch, reconstructions)
        loss.backward()

        optimizer.step()

        losses.append(loss.item())

    print(f"Epoch: {epoch + 1} | Loss: {sum(losses) / len(losses):.3f}")

embeddings, _ = model(encoded_words)
torch.save(embeddings.detach().cpu(), Path("../data/autoencoded_words.pt"))
