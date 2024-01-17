import numpy as np
import pickle
import torch
import torch.utils.data as data


class Data(data.Dataset):
    def __init__(self, path, mode = 'train'):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        if mode == 'train':
            dataset = data['train']
        elif mode == 'valid':
            dataset = data['valid']
        else:
            dataset = data['test']

        text = dataset['text'].astype(np.float32)
        text[text == -np.inf] = 0
        self.text = torch.tensor(text)
        audio = dataset['audio'].astype(np.float32)
        audio[audio == -np.inf] = 0
        self.audio = torch.tensor(audio)
        vision = dataset['vision'].astype(np.float32)
        vision[vision == -np.inf] = 0
        self.vision = torch.tensor(vision)
        self.label = dataset['labels'].astype(np.float32)  ##happy, sad, angry, neutral


    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = self.text[index]
        vision = self.vision[index]
        audio = self.audio[index]
        label = torch.argmax(torch.tensor(self.label[index]), -1)
        return text, audio, vision, label









