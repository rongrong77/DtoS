import time

import torch
import wandb
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from evaluation import Evaluation
from modelbuilder import ModelBuilder
from model.weight_init import weight_init
from utils.pytorchtools import EarlyStopping
from visualization.wandb_plot import wandb_plot_true_pred
# from torch.utils.tensorboard import SummaryWriter
# from torchinfo import summary


class Train:
    def __init__(self, config, train_dataloader, val_dataloader=None):
        self.config = config
        self.n_epoch = config['n_epoch']
        self.device = config['device']
        self.model_name = config['model_name']
        self.classification = config['classification']
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = []
        self.optimizer = []
        self.criterion = []
        self.trainer = []
        self.validator = []
        self.train_losses = []
        self.val_losses = []

    def run_training(self):
        self.setup_model()
        self.trainer = self.setup_trainer()
        if self.val_dataloader:
            self.validator = self.setup_validator()
        wandb.watch(self.model)
        for epoch in range(self.config['n_epoch']):
            self.epoch = epoch
            if not self.model.training:
                self.model.training = True
            self.model, train_loss = self.trainer(self.model, self.train_dataloader, self.optimizer, self.criterion, self.device)
            self.train_losses.append(train_loss)

            if self.val_dataloader:
                val_pred, val_target, val_loss = self.validator(self.model, self.val_dataloader, self.criterion, self.device, self.epoch)
                self.val_losses.append(val_loss)

        return self.model, self.train_losses, self.val_losses

    def run_training_testing(self):
        self.setup_model()
        self.trainer = self.setup_trainer()
        self.validator = self.setup_validator()
        for epoch in range(self.config['n_epoch']):
            self.epoch = epoch
            if not self.model.training:
                self.model.training = True
            self.model = self.trainer(self.model, self.train_dataloader, self.optimizer, self.criterion, self.device)
            val_pred, val_target, val_loss = self.validator(self.model, self.val_dataloader, self.criterion, self.device, self.epoch)

        return val_pred, val_target

    def setup_model(self):
        modelbuilder_handler = ModelBuilder(self.config)
        model, optimizer, criterion = modelbuilder_handler.run_model_builder()
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion.to(self.device)

    def setup_trainer(self):
        if (self.model_name == 'transformer' and not self.classification) or (self.model_name == 'transformertsai' and not self.classification):
            trainer = self.training_transformer
        else:
            trainer = self.training
        return trainer

    def setup_validator(self):
        if (self.model_name == 'transformer' and not self.classification) or (self.model_name == 'transformertsai' and not self.classification):
            validator = self.validating_transformer
        else:
            validator = self.validating
        return validator

    def training(self, model, train_dataloader, optimizer, criterion, device):
        model.train()
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        start_time = time.time()
        epoch_loss = 0
        for batch_i, (x, y) in enumerate(train_dataloader):
            x = x.to(device).float()
            y = y.to(device)
            y_pred = model(x).double()
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_dataloader)
        wandb.log({"training time": time.time() - start_time})
        wandb.log({"Train Loss": epoch_loss, 'epoch': self.epoch+1})
        wandb.log({"n parameters": params})
        print(f'Epoch [{self.epoch+1}/{self.n_epoch}], Training Loss: {epoch_loss:.4f}')
        return model, epoch_loss

    def training_transformer(self, model, train_dataloader, optimizer, criterion, device):
        model.train()
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        epoch_loss = 0
        start_time = time.time()
        for batch_i, (x, y) in enumerate(train_dataloader):
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x.float())  # just for transformer
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_dataloader)
        wandb.log({"training time": time.time() - start_time})
        wandb.log({"Train Loss": epoch_loss, 'epoch': self.epoch + 1})
        wandb.log({"n parameters": params})
        print(f'Epoch [{self.epoch+1}/{self.n_epoch}], Training Loss: {epoch_loss:.4f}')
        return model, epoch_loss

    def validating(self, model, val_dataloader, criterion, device, epoch):
        model.eval()
        with torch.no_grad():
            test_loss = []
            test_preds = []
            test_trues = []
            for x, y in val_dataloader:
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x.float())
                loss = criterion(y, y_pred)
                test_loss.append(loss.item())
                test_preds.append(y_pred)
                test_trues.append(y)
            test_loss = torch.mean(torch.tensor(test_loss))
            print('Validate Accuracy of the model: {}'.format(test_loss))
        wandb.log({"Validation Loss": test_loss, 'epoch': epoch})
        return torch.cat(test_preds, 0), torch.cat(test_trues, 0), test_loss

    def validating_transformer(self, model, val_dataloader, criterion, device, epoch):
        model.eval()
        with torch.no_grad():
            test_loss = []
            test_preds = []
            test_trues = []
            for x, y in val_dataloader:
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x.float())  # just for transformer
                loss = criterion(y, y_pred)
                test_loss.append(loss.item())
                test_preds.append(y_pred)
                test_trues.append(y)
            test_loss = torch.mean(torch.tensor(test_loss))
            print('Validate Accuracy of the model: {}'.format(test_loss))
        wandb.log({"Validation Loss": test_loss, 'epoch': epoch + 1})
        return torch.cat(test_preds, 0), torch.cat(test_trues, 0), test_loss