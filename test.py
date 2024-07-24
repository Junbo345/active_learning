import numpy as np
import torch
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score


class LightningEstimator():

    def __init__(self, model, max_epochs=-1, batch_size=2 ** 10):
        self.model = model
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.create_trainer()

    def create_trainer(self):
        callbacks = [
            pl.callbacks.EarlyStopping(monitor='train_loss', patience=30),
            # pl.callbacks.ModelCheckpoint(monitor='train_loss', save_top_k=1, mode='min')
        ]
        self.trainer = pl.Trainer(accelerator='cpu', logger=False, callbacks=callbacks, enable_checkpointing=True,
                                  max_epochs=self.max_epochs)

    def fit(self, x, y):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
            y = torch.from_numpy(y).long()
        dataset = torch.utils.data.TensorDataset(x, y.squeeze())
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, drop_last=False)
        self.trainer.fit(self.model, dataloader)
        self.model = self.model.__class__.load_from_checkpoint(self.trainer.checkpoint_callback.best_model_path)
        # return best lightning model checkpoint
        # reset trainer TODO hacky
        self.create_trainer()
        return self  # for sklearn consistency

    def predict(self, x):
        return self.model(torch.from_numpy(x).float()).detach().numpy()

    def predict_proba(self, x):
        return self.model.predict_proba(torch.from_numpy(x).float()).detach().numpy()

    def score(self, x, y):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        y_hat = self.model(x).detach().numpy()

        if isinstance(y, torch.Tensor):
            y = y.detach().numpy()

        return accuracy_score(y.argmax(axis=1), y_hat.argmax(axis=1))


# logistic regression model
# https://machinelearningmastery.com/building-a-logistic-regression-classifier-in-pytorch/
class LogisticRegression(pl.LightningModule):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.save_hyperparameters()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

    def predict_proba(self, x):
        return torch.nn.functional.softmax(self(x), dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        # import time
        # time.sleep(0.01)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.02)
        # return torch.optim.AdamW(self.parameters(), lr=1.)


class MLP(pl.LightningModule):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, 128)
        self.linear2 = torch.nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return self.linear2(x)

    def predict_proba(self, x):
        return torch.nn.functional.softmax(self(x), dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.01)

class Multiregression(pl.LightningModule):
    def __init__(self, input_dim, output_dim):
        super(Multiregression, self).__init__()
        self.save_hyperparameters()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

    def predict_proba(self, x):
        return torch.nn.functional.softmax(self(x), dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_max = torch.argmax(y, dim=1)
        loss = torch.nn.functional.cross_entropy(y_hat, y_max)
        self.log('train_loss', loss)
        # import time
        # time.sleep(0.01)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.02)
        # return torch.optim.AdamW(self.parameters(), lr=1.)


if __name__ == '__main__':
    # simulate data
    # x = torch.randn(1000, 10)
    # y = torch.randint(0, 2, (1000,))
    x = np.random.randn(1000, 10)
    # y = np.random.randint(0, 2, (1000,))
    y = (x.mean(axis=1) > 0.5).astype(int)

    # create estimator
    # model = LogisticRegression(input_dim=10, output_dim=2)
    # model.fit(x, y)
    # y_hat = model.predict(x)
    # print(y_hat)

    model = LogisticRegression(input_dim=10, output_dim=2)
    estimator = LightningEstimator(model, max_epochs=10)
    estimator.fit(x, y)
    print(estimator.score(x, y))