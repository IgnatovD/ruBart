import re
import json
import html
from sklearn.model_selection import train_test_split
import random
import numpy as np
import torch
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader, Dataset

from preprocessing_mlm import get_tokenizer

class GenerateDataloader:
    def __init__(self, path, batch_size, max_length):
        self.path = path
        self.batch_size = batch_size
        self.max_length = max_length
        self.train_test_split = train_test_split
        self.DataLoader = DataLoader
        self.SummaryDataset = SummaryDataset

    def clear_data(self):
        data = []

        with open(self.path, "r", encoding='utf-8') as read_file:
            bar = tqdm(read_file, total=1003869)
            bar.set_description('Data cleaning')
            for line in bar:
                dict_text_title = json.loads(line)
                for key in dict_text_title.keys():
                    text = html.unescape(dict_text_title[key])
                    delete = re.findall(r'<strong>(.*?)</strong>', text)
                    if len(delete) >= 2:
                        text = text.replace(delete[1], '')
                    text = re.sub(r'\n', ' ', text)
                    text = re.sub(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});', '', text)
                    text = ' '.join(text.split())
                    dict_text_title[key] = text
                data.append(dict_text_title)
        return data

    def split(self):
        random.seed(42)
        np.random.seed(42)
        data = self.clear_data()
        train_data, test_data = self.train_test_split(data, test_size=0.019922918, random_state=42)
        train_data, val_data = self.train_test_split(train_data, test_size=0.2, random_state=42)
        return train_data, val_data, test_data

    def collate_fn(self, dataset):
        self.max_length = 512

        new_inputs = torch.zeros((len(dataset), self.max_length), dtype=torch.long)
        new_outputs = torch.zeros((len(dataset), self.max_length), dtype=torch.long)
        for i, sample in enumerate(dataset):
            new_inputs[i, :len(sample['inputs'])] += np.array(sample['inputs'])
            new_inputs[i, 1:] = new_inputs[i, 1:].masked_fill(new_inputs[i, 1:] == 0, 1)
            new_outputs[i, :len(sample['outputs'])] += np.array(sample['outputs'])
            new_outputs[i, 1:] = new_outputs[i, 1:].masked_fill(new_outputs[i, 1:] == 0, 1)
        attention_mask = 1 - new_inputs.masked_fill(new_inputs != 1, 0)
        return {'input_ids': new_inputs, 'outputs': new_outputs, 'attention_mask': attention_mask}

    def get_dataloader(self, data):
        dataset = self.SummaryDataset(data, self.max_length)
        dataloader = self.DataLoader(dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)
        return dataloader

class SummaryDataset(Dataset):
    def __init__(self, data, max_length):
        self.data = data
        self.max_length = max_length
        self.tokenizer = get_tokenizer(max_length=self.max_length)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs = self.tokenizer.encode(self.data[idx]['text']).ids
        outputs = self.tokenizer.encode(self.data[idx]['title']).ids

        return {'inputs': inputs, 'outputs': outputs}
