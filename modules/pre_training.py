import math
import torch
import os.path
import transformers
from torch.nn import CrossEntropyLoss
from tqdm.notebook import tqdm
from transformers import BartForConditionalGeneration
import matplotlib.pyplot as plt

from preprocessing_mlm import batch_generator

class TrainerMLM:
    def __init__(self, config, **kwargs):
        self.config = config
        self.lr = kwargs['lr']
        self.weight_decay = kwargs['weight_decay']
        self.batch_size = kwargs['batch_size']
        self.device = kwargs['device']
        self.total_steps = kwargs['total_steps']
        self.warmup_steps = kwargs['warmup_steps']
        self.vocab_size = kwargs['vocab_size']
        self.path = kwargs['path_save']
        self.loss_fn = CrossEntropyLoss()
        self.batch_generator = batch_generator

    def train(self, data, model, optimizer, scheduler, history, learning_rate, best_loss):

        model.train()
        epoch_loss = .0
        loss_checkpoint = .0
        total_iter = math.ceil(len(data) / self.batch_size)
        bar = tqdm(enumerate(self.batch_generator(data, batch_size=self.batch_size), start=1), total=total_iter)

        for itter, batch in bar:

            input_ids, attention_mask, labels = self.get_input(**batch)
            optimizer.zero_grad()

            output = model(input_ids, attention_mask=attention_mask)[0]

            loss = self.loss_fn(output.view(-1, self.vocab_size), labels.view(-1))

            loss.backward()

            optimizer.step()
            scheduler.step()

            lr = self.get_lr(optimizer)

            learning_rate.append(lr)
            history.append(round(loss.item(), 3))
            epoch_loss += loss.item()

            bar.set_description(f'lr={lr:.3}, loss={loss.item():.3}; epoch={loss_checkpoint:.3}; best={best_loss:.3}')

            if itter % 6000 == 0:
                self.save(model, optimizer, scheduler, history, learning_rate, best_loss, path=self.path + 'checkpoint.tar')
            if itter == total_iter:
                self.save(model, optimizer, scheduler, history, learning_rate, best_loss, path=self.path + 'checkpoint.tar')

            if itter % 2000 == 0:
                loss_checkpoint = epoch_loss / 2000
                epoch_loss = 0
                if loss_checkpoint < best_loss:
                    best_loss = loss_checkpoint
                    save(model, optimizer, scheduler, history, learning_rate, best_loss, path=self.path + 'best.tar')

        return model, history, learning_rate, best_loss

    def get_input(self, input_ids, attention_mask, masked_lm_labels):
        return input_ids.to(self.device), attention_mask.to(self.device), masked_lm_labels.to(self.device)

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def save(self, model, optimizer, scheduler, history, learning_rate, best_loss, path):
        check_file = os.path.exists(self.path)
        if not check_file:
            os.makedirs(self.path)

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history,
            'learning_rate': learning_rate,
            'best_loss': best_loss
        }, path)

    def load(self):

        history = []
        learning_rate = []
        best_loss = 10.0

        model = BartForConditionalGeneration(self.config)
        model.to(self.device)

        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters()], "weight_decay": self.weight_decay},
        ]

        optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps,
                                                                 num_training_steps=self.total_steps)
        check_file = os.path.exists(self.path + 'checkpoint.tar')

        if check_file:
            checkpoint = torch.load(self.path + 'checkpoint.tar')

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            history = checkpoint['history']
            learning_rate = checkpoint['learning_rate']
            best_loss = checkpoint['best_loss']

        return model, optimizer, scheduler, history, learning_rate, best_loss

    def visualization(self, list, mode='loss'):
        assert mode in ['loss', 'lr']
        plt.figure(figsize=(10, 6))
        plt.plot(list)
        plt.xlabel(r"$steps$", fontsize=12)
        if mode == 'loss':
            plt.ylabel(r"$loss$", fontsize=14)
        if mode == 'lr':
            plt.ylabel(r"$lr$", fontsize=14)
