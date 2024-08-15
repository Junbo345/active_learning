import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# LightningEstimator class to train and evaluate models using PyTorch Lightning
class LightningEstimator():

    def __init__(self, model, max_epochs=-1, batch_size=2 ** 10):
        """
        Initialize the LightningEstimator with a given model, maximum epochs, and batch size.

        Args:
            model (nn.Module): The neural network model to be trained.
            max_epochs (int): Maximum number of training epochs (default is -1, no limit).
            batch_size (int): The batch size for training and validation.
        """
        self.model = model
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.create_trainer()  # Create the PyTorch Lightning trainer

    def create_trainer(self):
        """
        Create a PyTorch Lightning trainer with early stopping and checkpointing callbacks.
        """
        callbacks = [
            pl.callbacks.EarlyStopping(monitor='validation_loss', patience=30),
            # Optional checkpointing can be enabled here
            # pl.callbacks.ModelCheckpoint(monitor='train_loss', save_top_k=1, mode='min')
        ]
        self.trainer = pl.Trainer(
            accelerator='cpu',  # Use CPU for training; change to 'gpu' if available
            logger=False,  # Disable logging
            enable_progress_bar=False,  # Disable progress bar
            callbacks=callbacks,  # Add early stopping callback
            enable_checkpointing=True,  # Enable checkpointing to save the best model
            max_epochs=self.max_epochs  # Set the maximum number of epochs
        )

    def fit(self, x, y):
        """
        Train the model using the provided data.

        Args:
            x (np.ndarray or torch.Tensor): Input features.
            y (np.ndarray or torch.Tensor): Target labels.

        Returns:
            self: The trained estimator (for compatibility with scikit-learn).
        """
        # Convert numpy arrays to torch tensors if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
            y = torch.from_numpy(y).long()

        # Split data into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

        # Create DataLoaders for training and validation
        train_dataset = torch.utils.data.TensorDataset(x_train, y_train.squeeze())
        val_dataset = torch.utils.data.TensorDataset(x_val, y_val.squeeze())
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, drop_last=False)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, drop_last=False)

        # Train the model
        self.trainer.fit(self.model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        # Load the best model checkpoint (if checkpointing is enabled)
        self.model = self.model.__class__.load_from_checkpoint(self.trainer.checkpoint_callback.best_model_path)

        # Reset the trainer (useful for subsequent training with the same estimator)
        self.create_trainer()
        return self  # Return the estimator for scikit-learn consistency

    def predict(self, x):
        """
        Make predictions using the trained model.

        Args:
            x (np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted labels.
        """
        return self.model(torch.from_numpy(x).float()).detach().numpy()

    def predict_proba(self, x):
        """
        Predict class probabilities using the trained model.

        Args:
            x (np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted probabilities for each class.
        """
        return self.model.predict_proba(torch.from_numpy(x).float()).detach().numpy()

    def score(self, x, y):
        """
        Evaluate the model's accuracy on the given data.

        Args:
            x (np.ndarray or torch.Tensor): Input features.
            y (np.ndarray or torch.Tensor): True labels.

        Returns:
            float: Accuracy score.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        # Get predictions
        y_hat = self.model(x).detach().numpy()

        # Convert labels to numpy array if they are in torch.Tensor format
        if isinstance(y, torch.Tensor):
            y = y.detach().numpy()

        # Calculate accuracy
        return accuracy_score(y.argmax(axis=1), y_hat.argmax(axis=1))


# LogisticRegression model class using PyTorch Lightning
class LogisticRegression(pl.LightningModule):
    def __init__(self, input_dim, output_dim):
        """
        Initialize a simple Logistic Regression model.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output classes.
        """
        super(LogisticRegression, self).__init__()
        self.save_hyperparameters()  # Save hyperparameters for checkpointing
        self.linear = torch.nn.Linear(input_dim, output_dim)  # Linear layer

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Output logits.
        """
        return self.linear(x)

    def predict_proba(self, x):
        """
        Predict class probabilities using softmax.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Class probabilities.
        """
        return torch.nn.functional.softmax(self(x), dim=1)

    def training_step(self, batch, batch_idx):
        """
        Training step to compute loss on a batch of data.

        Args:
            batch (tuple): A batch of data (features and labels).
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Computed loss.
        """
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss', loss)  # Log the training loss
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step to compute loss on a batch of validation data.

        Args:
            batch (tuple): A batch of validation data (features and labels).
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Computed loss.
        """
        x, y = batch
        y_hat = self(x)
        y_max = torch.argmax(y, dim=1)
        loss = torch.nn.functional.cross_entropy(y_hat, y_max)
        self.log('validation_loss', loss)  # Log the validation loss
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer for the model.

        Returns:
            torch.optim.Optimizer: Optimizer (AdamW) for training.
        """
        return torch.optim.AdamW(self.parameters(), lr=0.02)


# Multi-Layer Perceptron (MLP) model class using PyTorch Lightning
class MLP(pl.LightningModule):
    def __init__(self, input_size, hidden_sizes, output_size):
        """
        Initialize a Multi-Layer Perceptron (MLP) model.

        Args:
            input_size (int): Number of input features.
            hidden_sizes (list of int): List of sizes for hidden layers.
            output_size (int): Number of output classes.
        """
        super(MLP, self).__init__()
        self.save_hyperparameters()  # Save hyperparameters for checkpointing
        layers = []
        sizes = [input_size] + hidden_sizes + [output_size]

        # Construct hidden layers with ReLU activation
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)  # Combine layers into a sequential model

        # Register a hook for the second-to-last layer to capture its output
        self.second_last_output = None
        self.model[-3].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        """
        Hook function to capture the output of the second-to-last layer.

        Args:
            module (torch.nn.Module): The layer that is hooked.
            input (torch.Tensor): Input to the layer.
            output (torch.Tensor): Output from the layer.
        """
        self.second_last_output = output

    def forward(self, x):
        """
        Forward pass through the MLP model.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Output logits.
        """
        return self.model(x)

    def predict_proba(self, x):
        """
        Predict class probabilities using softmax.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Class probabilities.
        """
        return torch.nn.functional.softmax(self.model(x), dim=1)

    def training_step(self, batch, batch_idx):
        """
        Training step to compute loss on a batch of data.

        Args:
            batch (tuple): A batch of data (features and labels).
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Computed loss.
        """
        x, y = batch
        y_hat = self.model(x)
        y_max = torch.argmax(y, dim=1)
        loss = torch.nn.functional.cross_entropy(y_hat, y_max)
        self.log('train_loss', loss)  # Log the training loss
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step to compute loss on a batch of validation data.

        Args:
            batch (tuple): A batch of validation data (features and labels).
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Computed loss.
        """
        x, y = batch
        y_hat = self.model(x)
        y_max = torch.argmax(y, dim=1)
        loss = torch.nn.functional.cross_entropy(y_hat, y_max)
        self.log('validation_loss', loss)  # Log the validation loss
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer for the MLP model.

        Returns:
            torch.optim.Optimizer: Optimizer (AdamW) for training.
        """
        return torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.01)


# Multiregression model class using PyTorch Lightning
class Multiregression(pl.LightningModule):
    def __init__(self, input_dim, output_dim):
        """
        Initialize a Multiregression model.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output features.
        """
        super(Multiregression, self).__init__()
        self.save_hyperparameters()  # Save hyperparameters for checkpointing
        self.linear = torch.nn.Linear(input_dim, output_dim)  # Linear layer

    def forward(self, x):
        """
        Forward pass through the Multiregression model.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Output predictions.
        """
        return self.linear(x)

    def predict_proba(self, x):
        """
        Predict class probabilities using softmax.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Class probabilities.
        """
        return torch.nn.functional.softmax(self(x), dim=1)

    def training_step(self, batch, batch_idx):
        """
        Training step to compute loss on a batch of data.

        Args:
            batch (tuple): A batch of data (features and labels).
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Computed loss.
        """
        x, y = batch
        y_hat = self(x)
        y_max = torch.argmax(y, dim=1)
        loss = torch.nn.functional.cross_entropy(y_hat, y_max)
        self.log('train_loss', loss)  # Log the training loss
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step to compute loss on a batch of validation data.

        Args:
            batch (tuple): A batch of validation data (features and labels).
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Computed loss.
        """
        x, y = batch
        y_hat = self(x)
        y_max = torch.argmax(y, dim=1)
        loss = torch.nn.functional.cross_entropy(y_hat, y_max)
        self.log('validation_loss', loss)  # Log the validation loss
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer for the Multiregression model.

        Returns:
            torch.optim.Optimizer: Optimizer (AdamW) for training.
        """
        return torch.optim.AdamW(self.parameters(), lr=0.02)


# Example usage of the models and LightningEstimator
if __name__ == '__main__':
    # Simulate some data
    x = np.random.randn(1000, 10)  # Generate random input data
    y = (x.mean(axis=1) > 0.5).astype(int)  # Generate binary labels based on the mean of the features

    # Initialize a Logistic Regression model
    model = LogisticRegression(input_dim=10, output_dim=2)
    estimator = LightningEstimator(model, max_epochs=10)  # Create a LightningEstimator for training
    estimator.fit(x, y)  # Train the model
    print(estimator.score(x, y))  # Evaluate and print the model's accuracy
