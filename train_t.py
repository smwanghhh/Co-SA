import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import data_loader as loader
import argparse
import os
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from torch.utils.data import DataLoader
import network
import sys
from utils import ReplayBuffer, seed
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Co-SA')
parser.add_argument('--emb_size', default=128, type=int)
parser.add_argument('--seqlength', default=20, type=int)
parser.add_argument("--n_epochs", type=int, default=200)
parser.add_argument("--n_head", type=int, default=8)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--atten_dropout", type=float, default=0.0) 
parser.add_argument('--lr', default=0.00022, type=float) ###0.00018 for add; 0.00022 for cat
parser.add_argument("--train_batch_size", type=int, default=128)
parser.add_argument("--dev_batch_size", type=int, default=256)
parser.add_argument("--test_batch_size", type=int, default=256)
parser.add_argument('--dim_t', default=300, type=int)
parser.add_argument('--dim_a', default=74, type=int)
parser.add_argument('--dim_v', default=35, type=int)
parser.add_argument('--seed', default=9324, type=seed)
parser.add_argument('--w1', default=9, type=int) #9
parser.add_argument('--w2', default=10, type=int) #10 
parser.add_argument('--w3', default=9, type=float)##9
parser.add_argument('--path', default='./iemocap_data.pkl', type=str)

args = parser.parse_args()

DEVICE = torch.device("cuda:0")  

def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999

    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):
	set_random_seed(args.seed)

    ###load data
	train_dataset = loader.Data(args.path, 'train')
	dev_dataset = loader.Data(args.path, 'valid')
	test_dataset = loader.Data(args.path, 'test')
	train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, drop_last = True)

	dev_dataloader = DataLoader(
        dev_dataset, batch_size=args.dev_batch_size, shuffle=True)

	test_dataloader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=True)

    ##define model
	model = network.Model(args)
	actor = network.Actor(args)
	critic = network.Critic(args)
	actor_target = network.Actor(args)
	critic_target = network.Critic(args)


	if not os.path.exists('iemocap'):
		os.mkdir('iemocap')

	model_optimizer = optim.Adam(model.parameters(), lr=args.lr)
	actor_optimizer = optim.Adam(actor.parameters(), lr=args.lr)
	critic_optimizer = optim.Adam(critic.parameters(), lr=args.lr)

	model_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, mode='min', patience=20, factor=0.8, verbose=False)  # 30, 0.8 for add
	actor_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(actor_optimizer, mode='min', patience=20, factor=0.8, verbose=False) # 20, 0.8 for cat
	critic_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(critic_optimizer, mode='min', patience=20, factor=0.8, verbose=False)
#	total_params = sum(p.numel() for p in actor.parameters())
#	print(total_params)
#	return 0
	model = model.cuda()
	actor = actor.cuda()
	actor_target = actor_target.cuda()
	critic = critic.cuda()
	critic_target = critic_target.cuda()



	best_ha, best_ha_f, best_sa, best_sa_f, best_an, best_an_f, best_ne, best_ne_f = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	replay_buffer = ReplayBuffer(1000, args.seed)
	last = [None] * 6
      
	for epoch_i in range(args.n_epochs):
		train_loss, actor_loss, critic_loss, last = train_epoch(model, actor, critic, actor_target, critic_target, train_dataloader, model_optimizer, actor_optimizer, critic_optimizer, last, replay_buffer)
		valid_loss = eval_epoch(model, actor, dev_dataloader)
		test_loss, accuracy, f1 = test_score_model(model, actor, test_dataloader)
		model_scheduler.step(valid_loss)
		actor_scheduler.step(valid_loss)
		critic_scheduler.step(valid_loss)

		ne, ha, sa, an = accuracy
		ne_f, ha_f, sa_f, an_f = f1

		print(
			'[Epoch %d] Training Loss: %.4f.    Valid Loss:%.4f.  Test Loss:%.4f.  Happy:%.4f.  Sad:%.4f.   Angry:%.4f.   Neutral:%.4f.    LR:%.4f.'
			% (epoch_i, train_loss, valid_loss, test_loss, ha_f, sa_f, an_f, ne_f,
			   model_optimizer.param_groups[0]["lr"]))
		###record best
		if best_ha <= ha:
			best_ha = ha

		if best_ha_f <= ha_f:
			best_ha_f = ha_f

		if best_sa <= sa:
			best_sa = sa

		if best_sa_f <= sa_f:
			best_sa_f = sa_f

		if best_an <= an:
			best_an = an

		if best_an_f <= an_f:
			best_an_f = an_f

		if best_ne <= ne:
			best_ne = ne

		if best_ne_f <= ne_f:
			best_ne_f = ne_f

	best_mean = (best_ha + best_sa + best_an + best_ne) / 4
	best_mean_f = (best_ha_f + best_sa_f + best_an_f + best_ne_f) / 4

	print('Best_Ha: %.4f.  Best_HaF:%.4f.  Best_Sa:%.4f.  Best_SaF:%.4f.   Best_An:%.4f.  Best_AnF:%.4f.	 Best_Ne:%.4f.  Best_NeF:%.4f.'
		  %(best_ha, best_ha_f, best_sa, best_sa_f, best_an, best_an_f, best_ne, best_ne_f))
	print('Best_Mean: %.4f.      Best_Mean_F: %.4f.' % (best_mean, best_mean_f))



def train_epoch(model, actor, critic, actor_target, critic_target, train_dataloader, model_optimizer, actor_optimizer, critic_optimizer, last, replay_buffer):
    model.train()
    actor.train()
    critic.train()

    criterion = nn.CrossEntropyLoss(reduction='none')

    cls_loss_sum, actor_loss_sum, critic_loss_sum = 0, 0, 0

    data_iter = iter(train_dataloader)
    if last[0] is not None:
        iter_num = len(data_iter)
        input, label, state, corr_loss = last
    else: ###the first batch
        iter_num = len(data_iter) - 1
        batch = next(data_iter)
        batch = tuple(t.to(DEVICE) for t in batch)
        text, acoustic, visual, label_ids = batch
        label_ids = label_ids.squeeze()
        visual = torch.squeeze(visual, 1)
        acoustic = torch.squeeze(acoustic, 1)
        embs, corr_loss = model.forward(text, visual, acoustic)
        input, label, state = embs, label_ids, embs 
    b = state.shape[0]

    for step in tqdm(range(iter_num), desc="Iteration"):

        model_optimizer.zero_grad()
        actor_optimizer.zero_grad()

        a = actor.forward(state)
        rep = actor.fused(input, a)
        pred = actor.temporal(rep)
        pred = pred.view(-1, 2)
        label = label.view(-1)
        loss_e = criterion(pred, label).mean()

        loss = loss_e * args.w1 + corr_loss.mean() * args.w2 
        loss.backward()
        model_optimizer.step()
        actor_optimizer.step()
        cls_loss_sum += loss_e.mean()

        r = torch.gather(nn.Softmax(1)(pred), 1, label.unsqueeze(0).view(-1, 4)).sum(1)

        ##build MDP
        batch = next(data_iter)
        batch = tuple(t.to(DEVICE) for t in batch)
        text, acoustic, visual, label_ids = batch

        label_ids = label_ids.squeeze()
        visual = torch.squeeze(visual, 1)
        acoustic = torch.squeeze(acoustic, 1) 
        embs, corr_loss = model.forward(text, visual, acoustic)
        rep = actor.fused(embs, a)

        s2 = rep

        #critic  
        replay_buffer.add_batch([i for i in zip(state.data, a.data, r.data, s2.data)])

        if replay_buffer.size() > b:
           s1_batch, a1_batch, r_batch, s2_batch = replay_buffer.sample_batch(b, args.seqlength, args.emb_size)
           critic_optimizer.zero_grad()
           predicted_q = critic(s1_batch, a1_batch)
           a2_batch = actor_target.forward(s2_batch)
           target_q = critic_target(s2_batch, a2_batch) * 0.5 + r_batch
           critic_loss = torch.mean(nn.L1Loss()(predicted_q, target_q)) * args.w3
           critic_loss_sum += critic_loss
           critic_loss.backward() 
           critic_optimizer.step()

        ###actor
           actor_optimizer.zero_grad()
           a1 = actor.forward(s1_batch)  ###!!!!!!!very important  otherwise actor gradient will not be updated
           actor_loss = -1 * torch.mean(critic(s1_batch, a1)) * args.w3
           actor_loss_sum += actor_loss
           actor_loss.backward() 
           actor_optimizer.step()

            #update target actor and critic
           for target_param, param in zip(critic_target.parameters(), critic.parameters()):
               target_param.data.copy_(
                   target_param.data * (1.0 - 0.01) + param.data * 0.01
               )
           for target_param, param in zip(actor_target.parameters(), actor.parameters()):
               target_param.data.copy_(
                   target_param.data * (1.0 - 0.01) + param.data * 0.01
               )

        input, label, state = embs, label_ids, s2.data

    last = [input, label, state, corr_loss]
    return cls_loss_sum / (step + 1), actor_loss_sum / (step+1), critic_loss_sum / (step + 1), last


def eval_epoch(model, actor, dev_dataloader):
    model.eval()
    actor.eval()

    dev_loss = 0
    with torch.no_grad():
        for step, batch in enumerate(dev_dataloader):#, desc="Iteration")):
            batch = tuple(t.to(DEVICE) for t in batch)

            text, acoustic, visual, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            embs,  _ = model( text, visual,acoustic)
            a = actor.forward(embs)
            rep = actor.fused(embs, a)
            logits = actor.temporal(rep)
            label_ids = label_ids.squeeze()

            logits = logits.view(-1, 2)
            label_ids = label_ids.view(-1)

            loss = nn.CrossEntropyLoss()(logits, label_ids)
            dev_loss += loss
    return dev_loss / (step + 1)

def test_epoch(model, actor, test_dataloader):
    model.eval()
    actor.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for batch in test_dataloader:
            batch = tuple(t.to(DEVICE) for t in batch)

            text, acoustic, visual, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            label_ids = label_ids.squeeze()
            embs, _ = model.forward(text, visual, acoustic)
            a = actor.forward(embs)
            rep = actor.fused(embs, a)
            logits = actor.temporal(rep)

            preds.extend(logits)
            labels.extend(label_ids)

        preds = torch.cat(preds, 0)
        labels = torch.cat(labels, 0)
        return preds, labels


def test_score_model(model, actor, test_dataloader):

    preds, y_test = test_epoch(model, actor, test_dataloader)
    test_loss = nn.CrossEntropyLoss()(preds.view(-1, 2), y_test.view(-1)).item()

    test_preds = preds.view(-1, 4, 2).cpu().detach().numpy()
    test_truth = y_test.view(-1, 4).cpu().detach().numpy()
    f1, acc = [], []
    for emo_ind in range(4):
        test_preds_i = np.argmax(test_preds[:,emo_ind],axis=1)
        test_truth_i = test_truth[:,emo_ind]
        f1.append(f1_score(test_truth_i, test_preds_i, average='weighted'))
        acc.append(accuracy_score(test_truth_i, test_preds_i))
    
    return test_loss, acc, f1


if __name__ == '__main__':
	main(args)

