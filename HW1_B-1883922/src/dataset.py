import fasttext.util
import shutil
import os
from torch.utils.data import Dataset, DataLoader
import nltk
import json
import torch
from torch.nn.utils.rnn import pad_sequence
import plotly.express as px

nltk.download('punkt')


class HaSpeeDe_Dataset(Dataset):

    def __init__(self, data_path: str, data: list[tuple[list, int]] = None, use_embeddings: bool = False,
                 stopwords_file_path: str = "", device="cpu", encoder=None, direct_embeddings=True) -> None:
        self.device = device
        if not direct_embeddings:
            use_encoder = True
            if encoder is None:
                encoder = {}
                use_encoder = False
                j = 0
                new_embeddings = {}
        if data is not None:
            self.data = data
        else:
            stopwords = []
            if stopwords_file_path != "":
                with open(stopwords_file_path, 'r', encoding="UTF8") as f:
                    stopwords = f.readlines()  # controllare carattere di andare a capo
            if use_embeddings:
                fasttext.util.download_model('it', if_exists='ignore')
                if os.path.exists('cc.it.300.bin.gz'):
                    os.remove('cc.it.300.bin.gz')
                shutil.move('cc.it.300.bin', 'data/cc.it.300.bin')
                embeddings = fasttext.load_model('data/cc.it.300.bin')
            self.data = []
            with open(data_path, 'r', encoding="UTF8") as f:
                for line in f:
                    item = json.loads(line)
                    sentence = nltk.word_tokenize(item['text'], language='italian')
                    filtered_sentence = []
                    i = 0
                    while i < len(sentence):
                        word = sentence[i]
                        if (word == "#" or word == "@") and i + 1 < len(sentence):
                            word = word + sentence[i + 1]
                            i += 1
                        if stopwords_file_path == "" or word not in stopwords:
                            filtered_sentence.append(word)
                        i += 1

                    if use_embeddings:
                        embedded_sentence = []
                        for word in sentence:
                            if word not in encoder and not use_encoder and not direct_embeddings:
                                encoder[word] = j
                                new_embeddings[j] = embeddings.get_word_vector(word)
                                embedded_sentence.append(j)
                                j += 1
                            elif word not in encoder and not direct_embeddings:
                                embedded_sentence.append(encoder["<UNK>"])
                            else:
                                embedded_sentence.append(encoder[word])
                        if len(embedded_sentence) == 0:
                            print("error empty sentence")
                        sentence = embedded_sentence
                    self.data.append((sentence, item['label']))
            if not use_encoder and not direct_embeddings:
                encoder["<UNK>"] = j
                self.encoding = encoder
                new_embeddings[j] = embeddings.get_word_vector("<UNK>")
                self.embeddings = new_embeddings

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[list, int]:
        return self.data[idx]

    def split(self, prc: float) -> list[tuple[list, int]]:
        validation_size = int(prc * len(self.data))
        train_size = len(self.data) - validation_size
        validation_data, self.data = torch.utils.data.random_split(self.data, [validation_size, train_size])
        return validation_data

    def collate(self, batch: list[tuple[list, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        texts, labels = zip(*batch)
        lens = [len(text) for text in texts]
        texts = pad_sequence([torch.tensor(text) for text in texts], batch_first=True)
        return texts.to(self.device), torch.tensor(labels, dtype=torch.float).to(self.device), torch.tensor(lens).to(
            self.device)

    def get_dataloader(self, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate)

    def print_data_analysis(self):
        y = [0, 0]
        for el in self.data:
            if el[1] == 1:
                y[1] += 1
            else:
                y[0] += 1
        self.neutral_count = y[0]
        self.hateful_count = y[1]
        fig = px.bar(x=["neutrale", "odio"], y=y)
        fig.show()


def create_validation_set( data_path: str, prc: float, train_data_path: str, validation_data_path: str):
    with open(data_path, 'r', encoding="UTF8") as f:
        data = f.readlines()
    validation_size = int(prc * len(data))
    train_size = len(data) - validation_size
    validation_data, train_data = torch.utils.data.random_split(data, [validation_size, train_size])
    with open(train_data_path, 'w', encoding="UTF8") as f:
        f.writelines(train_data)
    with open(validation_data_path, 'w', encoding="UTF8") as f:
        f.writelines(validation_data)


def build_dataloaders_fixed_embeddings(device: str):
    train_dataset = HaSpeeDe_Dataset( "data/train-taskA.jsonl", use_embeddings=True,
                                     stopwords_file_path="data/stopwords-it.txt", device=device)
    train_dataset.print_data_analysis()

    val_data = train_dataset.split(0.2)
    val_dataset = HaSpeeDe_Dataset("", data=val_data)

    news_test_dataset = HaSpeeDe_Dataset( "data/test-news-taskA.jsonl", use_embeddings=True,
                                    stopwords_file_path="data/stopwords-it.txt", device=device)
    tweets_test_dataset = HaSpeeDe_Dataset("data//test-tweets-taskA.jsonl", use_embeddings=True,
                                         stopwords_file_path="data/stopwords-it.txt", device=device)

    train_loader = train_dataset.get_dataloader(64, True)
    val_loader = val_dataset.get_dataloader(64, True)
    news_test_loader = news_test_dataset.get_dataloader(64, True)
    tweets_test_loader = tweets_test_dataset.get_dataloader(64, True)


    return train_loader, val_loader, news_test_loader, tweets_test_loader

def build_dataloaders_unfrozen(device: str):

