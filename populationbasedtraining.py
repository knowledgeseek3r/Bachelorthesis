from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.schedulers import PopulationBasedTraining
from transformers import AdamW, get_linear_schedule_with_warmup
import data_preparation
import model as ml

train_data= data_preparation.train_data
test_data = data_preparation.val_data


model = ml.BertClassifier(False, "default")

# Method to start train and validation 
def train_cifar(model, config, checkpoint_dir=None, data_dir=None):
    if checkpoint_dir:
     model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
     model.load_state_dict(model_state)
     optimizer.load_state_dict(optimizer_state)


    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"

    trainloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)
    valloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)

    # Specify loss function
    loss_fn = nn.CrossEntropyLoss()
    epoch = 4
    for epoch_i in range(epoch):
        epoch_steps = 0
        running_loss = 0
        for step, batch in enumerate(trainloader):
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

        # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0


        # Validation loss
            val_loss = 0.0
            val_steps = 0
            total = 0
            correct = 0
            for i, data in enumerate(valloader, 0):
                with torch.no_grad():
                    _input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

                    outputs = model(b_input_ids, b_attn_mask)
                    _, predicted = torch.max(outputs.data, 1)
                    total += b_labels.size(0)
                    correct += (predicted == b_labels).sum().item()

                    loss = loss_fn(logits, b_labels)
                    val_loss += loss.cpu().numpy()
                    val_steps += 1

            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)

            tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
        print("Finished Training")

# Define Parameters and ranges for testing and evaluation of them
# No more parameters were added because while testing population based training (with only varying learning rate and batch size) it came out that this method take already to much time.
config = {
"lr": tune.loguniform(1e-4, 1e-1),
"batch_size": tune.choice([2, 4, 8])
}

scheduler = PopulationBasedTraining(
    time_attr="training_iteration",
    perturbation_interval=5,
    hyperparam_mutations={
        # distribution for resampling
        "lr": lambda: np.random.uniform(0.0001, 1),
        # allow perturbations within this set of categorical values
        "wd": [0.1, 0.2, 0.99],
    })

# Start ray tune run for analyising the impact of different parameters and combinations of them
analysis = tune.run(
    train_cifar,
    resources_per_trial={'gpu': 1},
    name="pbt_test",
    scheduler=scheduler,
    export_formats=None,
    checkpoint_score_attr="mean_accuracy",
    keep_checkpoints_num=4,
    num_samples=4,
    config=config,
    )
