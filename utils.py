import numpy as np
import random
import torch


class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed = 123):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = []
        random.seed(random_seed)

    def add(self, s, a, r, s2):
        experience = (s, a, r, s2)
        if self.count < self.buffer_size: 
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
            
    def add_batch(self, sars2_list):
#         assert self.count + len(sars2_list) <= self.buffer_size, "ER Buffer Overflowed"
        if self.count + len(sars2_list) < self.buffer_size: 
            self.buffer = self.buffer + sars2_list
            self.count += len(sars2_list)
        else:
            del self.buffer[:len(self.buffer)//2]
            self.buffer = self.buffer + sars2_list
            self.count = len(self.buffer)
        

    def size(self):
        return self.count

    def sample_batch(self, batch_size, length, dim):

        batch = []

        if self.count < batch_size:
            ran_num = np.arange(self.count)
            batch = list(self.buffer)
        else:
            ran_num = np.random.choice(self.count, batch_size, replace = False)
            batch = [self.buffer[i] for i in ran_num]

        s_batch = torch.zeros(size = (batch_size, length*3, dim)).cuda()
        a_batch = torch.zeros(size = (batch_size, length*3, 1)).cuda()
        r_batch = torch.zeros(size = (batch_size,)).cuda()
        s2_batch = torch.zeros(size = (batch_size, length*3, dim)).cuda()
        for idx, b in enumerate(batch):
            s_batch[idx] = b[0]
            a_batch[idx] = b[1]
            r_batch[idx] = b[2]
            s2_batch[idx] = b[3]

        return s_batch, a_batch, r_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0

def seed(s):
    if isinstance(s, int):
        if 0 <= s <= 9999:
            return s
        else:
            raise argparse.ArgumentTypeError(
                "Seed must be between 0 and 2**32 - 1. Received {0}".format(s)
            )
    elif s == "random":
        return random.randint(0, 9999)
    else:
        raise argparse.ArgumentTypeError(
            "Integer value is expected. Recieved {0}".format(s)
        )