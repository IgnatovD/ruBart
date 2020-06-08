import torch
import os.path
import transformers
from torch.nn import CrossEntropyLoss
from tqdm.notebook import tqdm
from transformers import BartForConditionalGeneration
import matplotlib.pyplot as plt

class FineTune:
    def __init__(self, config, **kwargs):
        self.config = config
        self.lr = kwargs['lr']
        self.weight_decay = kwargs['weight_decay']
        self.batch_size = kwargs['batch_size']
        self.accum_steps = kwargs['accum_steps']
        self.pad_token_id = config.pad_token_id
        self.device = kwargs['device']
        self.total_steps = kwargs['total_steps']
        self.warmup_steps = kwargs['warmup_steps']
        self.vocab_size = kwargs['vocab_size']
        self.path = kwargs['path_save']
        self.path_pretrained_model = kwargs['path_pretrained_model']
        self.loss_fn = CrossEntropyLoss()

    def train_sum(self, dataloaders, model, optimizer, scheduler, history, learning_rate, best_loss):

        model.train()
        loss_mini_batch = .0
        total_iter = len(dataloaders)

        model.zero_grad()
        bar = tqdm(enumerate(dataloaders, start=1), total=total_iter)
        for itter, batch in bar:

            inputs = self.get_input(batch)
            loss = model(**inputs)[0]

            loss = loss / self.accum_steps
            loss.backward()
            loss_mini_batch += loss.item()
            if itter % self.accum_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                lr = self.get_lr(optimizer)

                learning_rate.append(lr)
                history.append(round(loss_mini_batch, 3))

                if loss_mini_batch < best_loss:
                    best_loss = loss_mini_batch
                    self.save(model, optimizer, scheduler, history, learning_rate, best_loss, path=self.path + 'best.tar')

                bar.set_description(f'lr={lr:.3}; loss={loss_mini_batch:.3}; best loss={best_loss:.3}')

                loss_mini_batch = 0

            if itter % 6000 == 0:
                self.save(model, optimizer, scheduler, history, learning_rate, best_loss, path=self.path + 'checkpoint.tar')
            if itter == total_iter:
                self.save(model, optimizer, scheduler, history, learning_rate, best_loss, path=self.path + 'checkpoint.tar')

        return model, history, learning_rate, best_loss

    def get_input(self, batch):
        source_ids, source_mask, y = batch['input_ids'], batch['attention_mask'], batch['outputs']
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone()
        lm_labels[y[:, 1:] == self.pad_token_id] = -100

        inputs = {
            "input_ids": source_ids.to(self.device),
            "attention_mask": source_mask.to(self.device),
            "decoder_input_ids": y_ids.to(self.device),
            "lm_labels": lm_labels.to(self.device),
        }
        return inputs

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

    def load_pretrained(self):
        history = []
        learning_rate = []
        best_loss = 10.0

        checkpoint = torch.load(self.path_pretrained_model)
        model = BartForConditionalGeneration(self.config)
        model.to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer, scheduler = self.get_optim(model)
        return model, optimizer, scheduler, history, learning_rate, best_loss

    def load(self):

        history = []
        learning_rate = []
        best_loss = .0

        model = BartForConditionalGeneration(self.config)
        model.to(self.device)
        optimizer, scheduler = self.get_optim()

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

    def get_optim(self, model):
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters()], "weight_decay": self.weight_decay},
        ]

        optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps,
                                                                 num_training_steps=self.total_steps)

        return optimizer, scheduler

    def visualization(self, list, mode='loss'):
        assert mode in ['loss', 'lr']
        plt.figure(figsize=(10, 6))
        plt.plot(list)
        plt.xlabel(r"$steps$", fontsize=12)
        if mode == 'loss':
            plt.ylabel(r"$loss$", fontsize=14)
        if mode == 'lr':
            plt.ylabel(r"$lr$", fontsize=14)
