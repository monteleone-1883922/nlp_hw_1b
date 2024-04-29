from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import torch
import sys


# Function to print a progress bar


class Trainer():

    def __init__(self, model, train_dataloader, validation_dataloader, optimizer, loss_function, device,
                 test_dataloader=None):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device
        self.test_dataloader = test_dataloader

    @staticmethod
    def evaluation_parameters(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred).ravel()
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        return cm, precision, recall, f1, accuracy

    @staticmethod
    def print_progress_bar(percentuale: float, lunghezza_barra: int = 30, text: str = "") -> None:
        blocchi_compilati = int(lunghezza_barra * percentuale)
        barra = "[" + "=" * (blocchi_compilati - 1) + ">" + " " * (lunghezza_barra - blocchi_compilati) + "]"
        sys.stdout.write(f"\r{barra} {percentuale * 100:.2f}% complete " + text)
        sys.stdout.flush()

    def train(self, epochs: int, use_wandb: bool = False, config: dict = {}, name: str = "", target_f1: float = 0.0,
              validate: bool = True):
        best_model = None
        save = False
        if use_wandb:
            pass
            # wandb.init(
            #     # Set the project where this run will be logged
            #     project="nlp-hw-1b",
            #     name=name,
            #     # Track hyperparameters and run metadata
            #     config=config
            # )
        for epoch in range(epochs):
            if epoch == 3:
                self.model.freeze_embeddings(False)
            self.model.train()  # Set the model to training mode
            total_loss = 0
            for i, batch in enumerate(self.train_dataloader):
                self.print_progress_bar(i / len(self.train_dataloader), text=f"| training epoch {epoch}")
                # Get the inputs and targets from the batch
                inputs, targets, lens = batch

                # Zero the gradients
                self.optimizer.zero_grad()
                # Forward pass
                outputs = self.model((inputs, lens))
                # Compute loss
                loss = self.loss_function(outputs, targets)
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                # Accumulate the total loss
                total_loss += loss.item()

            # Print the average loss for this epoch
            if validate:
                validation_loss, precision, recall, f1, accuracy = self.validate(use_wandb)
                if f1 > target_f1:
                    best_model = self.model.state_dict()
                    target_f1 = f1
                    save = True
                if use_wandb:
                    pass
                    # wandb.log({"validation_loss": validation_loss,
                    #       "precision": precision,
                    #       "recall": recall,
                    #       "f1": f1,
                    #       "accuracy": accuracy,
                    #       "train_loss": total_loss / len(self.train_dataloader)})
        if save:
            torch.save(best_model, 'data/' + name +'.pth')
        if use_wandb:
            pass
            # wandb.finish()

    def validate(self, use_wandb: bool = False, test=False):
        dataloader = self.test_dataloader if test else self.validation_dataloader
        if dataloader is None:
            print("empty dataloader!")
            exit(1)
        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0
        all_predictions = torch.tensor([])
        all_targets = torch.tensor([])
        with torch.no_grad():  # Do not calculate gradients
            for i, batch in enumerate(self.validation_dataloader):
                self.print_progress_bar(i / len(dataloader), text="| validation")
                # Get the inputs and targets from the batch
                inputs, targets, lens = batch

                # Forward pass
                outputs = self.model((inputs, lens))
                # Compute loss
                loss = self.loss_function(outputs, targets)
                # Accumulate the total loss
                total_loss += loss.item()
                # Store predictions and targets
                all_predictions = torch.cat((all_predictions, outputs.squeeze().round().cpu()))
                all_targets = torch.cat((all_targets, targets.cpu()))
        validation_loss = total_loss / len(self.validation_dataloader)
        precision = precision_score(all_targets, all_predictions)
        recall = recall_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions)
        accuracy = accuracy_score(all_targets, all_predictions)
        return validation_loss, precision, recall, f1, accuracy
