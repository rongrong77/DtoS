import time
import numpy as np
import torch
import wandb
from modelbuilder import ModelBuilder


class Test:
    def run_testing(self, config, model, test_dataloader):
        self.config = config
        self.device = config['device']
        self.loss = self.config['loss']
        self.weight = self.config['loss_weight']
        self.model_name = self.config['model_name']
        self.classification = config['classification']
        self.n_output = len(self.config['static_labels'])
        if not self.n_output == len(self.weight):
            self.weight = None
        modelbuilder_handler = ModelBuilder(self.config)
        criterion = modelbuilder_handler.get_criterion(self.weight)
        self.tester = self.setup_tester()
        y_pred, y_true, loss = self.tester(model, test_dataloader, criterion, self.device)
        return y_pred, y_true, loss

    def setup_tester(self):
        if (self.model_name == 'transformer' and not self.classification) or (self.model_name == 'transformertsai' and not self.classification):
            tester = self.testing_transformer
        elif self.classification:
            tester = self.testing_w_classification
        else:
            tester = self.testing
        return tester

    def testing(self, model, test_dataloader, criterion, device):
        model.eval()
        with torch.no_grad():
            test_loss = []
            test_preds = []
            test_trues = []
            inference_times = []
            for x, y in test_dataloader:
                x = x.to(device)
                y = y.to(device)
                start_time = time.time()
                y_pred = model(x.float()).squeeze()
                inference_times.append(time.time() - start_time)
                loss = criterion(y_pred, y)
                test_loss.append(loss.item())
                test_preds.append(y_pred.cpu().numpy())
                test_trues.append(y.cpu().numpy())
            avg_test_loss = torch.tensor(test_loss).mean().item()

            # Convert lists to numpy arrays
            test_preds = np.concatenate(test_preds)
            test_trues = np.concatenate(test_trues)

            # Calculate additional metrics
            absolute_errors = np.abs(test_preds - test_trues)
            mae = np.mean(absolute_errors)
            mape = np.mean(absolute_errors / (test_trues + 1e-10)) * 100  # Adding epsilon to avoid division by zero

            print(f'Test Loss of the model: {avg_test_loss}')
            print(f'Mean Absolute Error (MAE): {mae}')
            print(f'Mean Absolute Percentage Error (MAPE): {mape}%')

            # Optionally, log inference times or other metrics
            avg_inference_time = np.mean(inference_times)
            print(f'Average Inference Time per Batch: {avg_inference_time} seconds')

        return torch.tensor(test_preds), torch.tensor(test_trues), avg_test_loss

    def testing_transformer(self, model, test_dataloader, criterion, device):
        model.eval()
        with torch.no_grad():
            test_loss = []
            test_preds = []
            test_trues = []
            inference_times = []
            for x, y in test_dataloader:
                x = x.to(device)
                y = y.to(device)
                start_time = time.time()
                y_pred = model(x.float())  # just for transformer
                inference_times.append(time.time() - start_time)
                loss = criterion(y, y_pred.to(device))
                test_loss.append(loss.item())
                test_preds.append(y_pred)
                test_trues.append(y)
            avg_test_loss = torch.tensor(test_loss).mean().item()

            # Convert lists to numpy arrays
            test_preds = torch.cat(test_preds, 0)
            test_trues = torch.cat(test_trues, 0)

            print(f'Test Loss of the model: {avg_test_loss}')

        return test_preds, test_trues, avg_test_loss

    def testing_with_classification(self, model, test_dataloader, criterion, device):
        # Implement your classification-specific testing logic here
        pass