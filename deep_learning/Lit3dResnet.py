
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import numpy as np

from deep_learning.ResNetV2 import generate_model




class Lit3dResnet(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        # self.save_hyperparameters()
        self.model = generate_model(params['resnet_model_size'], n_input_channels=params['model_chan_in'], n_classes=params['n_classes'])
        self.params = params
        self.loss_acc_dict = {"train_loss": [], "valid_loss": [], "train_acc": [], "valid_acc": [], "epoch": [], 'valid_preds': [],
                 'valid_targets': [], 'train_preds': [], 'train_targets': []}
        
        
    def forward(self, x):
        return self.model(x)
    
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        _, y_pred = torch.max(y_hat, dim=1)
        # print(f"train_loss: {loss}")
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return {'loss': loss, 'pred': y_pred.cpu(), 'target': y.cpu()}
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        _, y_pred = torch.max(y_hat, dim=1)
        val_loss = F.cross_entropy(y_hat, y)
        val_acc = self.accuracy(y_hat, y)
        self.log('val_loss', val_loss)
        return {'val_loss': val_loss, 'val_pred': y_pred, 'val_target': y}


    def accuracy(self, y_hat, y):
        _, y_pred = torch.max(y_hat, dim=1)
        acc = (y_pred == y).float().mean()
        return acc
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean().item()
        
        valid_preds = [x['val_pred'] for x in outputs]
        valid_targets = [x['val_target'] for x in outputs]
        valid_preds = torch.concat(valid_preds).cpu().numpy()
        valid_targets = torch.concat(valid_targets).cpu().numpy()
        
        avg_acc = (valid_preds == valid_targets).mean()
        self.log("val_loss", avg_loss)
        self.log("val_acc", avg_acc)
        self.loss_acc_dict['valid_loss'].append(avg_loss)
        self.loss_acc_dict['valid_acc'].append(avg_acc)
        self.loss_acc_dict['valid_preds'].append(valid_preds)
        self.loss_acc_dict['valid_targets'].append(valid_targets)
        self.loss_acc_dict['epoch'].append(self.current_epoch)
        
            
    def training_epoch_end(self, outputs):
        train_losses = [x['loss'].item() for x in outputs]
        train_preds = [x['pred'] for x in outputs]
        train_targets = [x['target'] for x in outputs]
        train_preds = torch.concat(train_preds).cpu().detach().numpy()
        train_targets = torch.concat(train_targets).cpu().detach().numpy()
        
        self.loss_acc_dict['train_loss'].append(np.mean(train_losses))
        self.loss_acc_dict['train_acc'].append((train_preds == train_targets).mean())
        self.loss_acc_dict['train_preds'].append(train_preds)
        self.loss_acc_dict['train_targets'].append(train_targets)
        
        print(f"Epoch {self.current_epoch+1}/{self.trainer.max_epochs} complete.")
        print(f"Training loss: {self.loss_acc_dict['train_loss'][-1]:.4f}; Training accuracy: {self.loss_acc_dict['train_acc'][-1]:.2%}")
        print(f"Validation loss: {self.loss_acc_dict['valid_loss'][-1]:.4f}; Validation accuracy: {self.loss_acc_dict['valid_acc'][-1]:.2%}")
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params['lr'], weight_decay=self.params['adam_regularization'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.params['lr_decay'])
        return [optimizer], [scheduler] 
    

    