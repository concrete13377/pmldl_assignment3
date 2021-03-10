# !wget https://www.dropbox.com/s/ve57m5eu9s8a0k4/corpus.en_ru.1m.en
# !wget https://www.dropbox.com/s/a9wzc7hparta7t4/corpus.en_ru.1m.ru

import re
import torch
import contractions
import string
from tqdm import tqdm

pad_token_idx = 0
max_len = 12


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, src_txt, tgt_txt, max_len):

        self.max_len = max_len
        self.start_of_string_token_idx = 1
        self.end_of_string_token_idx = 2
        self.pad_token_idx = pad_token_idx

        self.eng_w2c = {}
        self.eng_i2w = {self.start_of_string_token_idx: "<start_of_string>",
                        self.end_of_string_token_idx: "<end_of_string>", self.pad_token_idx: "<pad>"}
        self.eng_w2i = {v: k for k, v in self.eng_i2w.items()}
        self.eng_n_words = 3

        self.rus_w2c = {}
        self.rus_i2w = {self.start_of_string_token_idx: "<start_of_string>",
                        self.end_of_string_token_idx: "<end_of_string>", self.pad_token_idx: "<pad>"}
        self.rus_w2i = {v: k for k, v in self.rus_i2w.items()}
        self.rus_n_words = 3

        self.pairs = self.parse(src_txt, tgt_txt)

    def norma_string(self, s):
        s = s.lower().strip()
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Zа-яА-Я.!?]+", r" ", s)
        s = s.translate(str.maketrans('', '', string.punctuation))
        s = re.sub(r"[”“,.:;()#%!?+/'@*]", "", s)
        s = re.sub('  +', ' ', s)
        return s

    def add_word_eng(self, word):
        if word not in self.eng_w2i:
            self.eng_w2i[word] = self.eng_n_words
            self.eng_w2c[word] = 1
            self.eng_i2w[self.eng_n_words] = word
            self.eng_n_words += 1
        else:
            self.eng_w2c[word] += 1

    def add_word_rus(self, word):
        if word not in self.rus_w2i:
            self.rus_w2i[word] = self.rus_n_words
            self.rus_w2c[word] = 1
            self.rus_i2w[self.rus_n_words] = word
            self.rus_n_words += 1
        else:
            self.rus_w2c[word] += 1

    def parse(self, src_txt, tgt_txt):
        # src == RUS
        # tgt == ENG
        with open(src_txt, 'r') as src_txt_file:
            src_data = src_txt_file.readlines()

        with open(tgt_txt, 'r') as tgt_txt_file:
            tgt_data = tgt_txt_file.readlines()

        pairs = []

        for src_line, tgt_line in tqdm(zip(src_data, tgt_data), total=len(tgt_data)):
            prep_src_line = [self.start_of_string_token_idx]
            for src_token in self.norma_string(src_line).split()[:self.max_len]:
                self.add_word_rus(src_token)
                prep_src_line.append(self.rus_w2i[src_token])
            prep_src_line.append(self.end_of_string_token_idx)

            prep_tgt_line = [self.start_of_string_token_idx]
            for tgt_token in self.norma_string(contractions.fix(tgt_line)).split()[:self.max_len]:
                self.add_word_eng(tgt_token)
                prep_tgt_line.append(self.eng_w2i[tgt_token])
            prep_tgt_line.append(self.end_of_string_token_idx)

            pairs.append((prep_src_line, prep_tgt_line))

        pairs = sorted(pairs, key=lambda x: len(x[1]), reverse=True)
        return pairs

    def __getitem__(self, index):
        src, tgt = self.pairs[index]
        return {'source': src, 'target': tgt}

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    # wanted to add batching, but have not managed to make it work
    def collate_fn(batch):
        srcs = [torch.tensor(t['source']) for t in batch]
        srcs = torch.nn.utils.rnn.pad_sequence(srcs, batch_first=False, padding_value=pad_token_idx)

        tgt = [torch.tensor(t['target']) for t in batch]
        tgt = torch.nn.utils.rnn.pad_sequence(tgt, batch_first=False, padding_value=pad_token_idx)

        return {'source': srcs, 'target': tgt}

# dataset = CustomDataset('/content/corpus.en_ru.1m.ru', '/content/corpus.en_ru.1m.en', max_len)
