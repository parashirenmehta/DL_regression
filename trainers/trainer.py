import torch
import wandb
import os


class Trainer:
    def __init__(self, model, optimizer, criterion, train_loader, valid_loader, lr=1e-4, device="cpu"):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.criterion = criterion
        self.lr = lr
        self.optimizer = optimizer

    def train_and_evaluate(self, fold, config):

        wandb.init(project='CNNRegression', group="Folds", name='Fold' + str(fold + 1))
        if fold == 0:
            tab = wandb.Table(data=[["batch_size", str(config['batch_size'])],
                                    ["learning_rate", str(config['learning_rate'])],
                                    ["num_epochs", str(config['epochs'])],
                                    # ["scheduler_params", "step_size=5, gamma=0.1"],
                                    ["optimizer", config['optimizer']],
                                    ["loss_function", config['criterion']],
                                    ["seed", str(config['seed'])],
                                    ["num_folds", str(config['num_folds'])],
                                    ["comment", "10-fold cross validation"],
                                    ], columns=["Parameter", "Value"])

            wandb.log({"Configs": tab})

        train_losses = []
        valid_losses = []

        for epoch in range(config['epochs']):
            # Training Phase
            self.model.train()  # Set the model to training mode
            for i, (images, covers) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                images = images.to(self.device)
                covers = covers.to(self.device)
                covers = covers.to(torch.float32)
                c = self.model(images)
                c = c.reshape((c.shape[0]))
                loss = self.criterion(c, covers)

                loss.backward()
                self.optimizer.step()

            # Validation phase
            train_losses.append(self.evaluate(self.train_loader, self.device, self.criterion))
            valid_losses.append(self.evaluate(self.valid_loader, self.device, self.criterion))

            wandb.log({'Train Loss': train_losses[-1], 'Valid Loss': valid_losses[-1]})
            print('Epoch', epoch, 'Training Loss:', train_losses[-1], 'Validation Loss:', valid_losses[-1])

        model_weights_path = config['save_model_weights']+'CNNRegression_weights/CNNRegression/'
        if not os.path.exists(model_weights_path):
            os.makedirs(model_weights_path)
        torch.save(self.model.state_dict(), model_weights_path+'Fold'+str(fold)+'_weights.pth')

        wandb.finish()

    def evaluate(self, data_loader, device, criterion):
        self.model.eval()
        losses = []
        num_samples = 0

        for i, (images, covers) in enumerate(data_loader):
            images, covers = images.to(device), covers.to(device)
            covers = covers.to(torch.float32)
            num_samples += images.shape[0]
            c = self.model(images)
            c = c.reshape((c.shape[0]))
            loss = criterion(c, covers)
            losses.append(loss.item() * images.shape[0])

        loss = sum(losses) / num_samples
        self.model.train()
        return loss
