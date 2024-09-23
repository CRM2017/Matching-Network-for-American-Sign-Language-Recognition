import torch
from data_loader import NShotDataset
from DataBuilder import DataBuilder
from matching_networks import MatchingNetwork
from util import setup, plot_graph

# Experiment setup
batch_size = 16
fce = True
classes_per_set = 5
samples_per_class = 1

# data parameters
data_para = setup['Omniglot']
data_path = data_para['path']
channels = data_para['channels']
image_size = data_para['image_size']
train_idx = data_para['train_idx']

# Training setup
total_epochs = 500
total_train_batches = 100
total_val_batches = 32
total_test_batches = 32

data = NShotDataset(data_path=data_path, channels=channels, image_size=image_size, train_idx=train_idx,
                    batch_size=batch_size, classes_per_set=classes_per_set,
                    samples_per_class=samples_per_class, seed=2021, shuffle=True, data_para=data_para)

oneShotBuilder = DataBuilder(data, batch_size=batch_size, num_channels=channels, lr=1e-3, image_size=image_size,
                             classes_per_set=classes_per_set,
                             samples_per_class=samples_per_class, dropout=0.0, fce=True, optim="adam",
                             weight_decay=0,
                             use_cuda=True)

def train(saved_mode):

    best_val_loss = float('inf')
    val_losses = []
    train_losses = []
    val_accs = []
    train_accs = []
    early_stop = 0
    for e in range(total_epochs):
        if early_stop >= 20:
            print('early stop')
            print('The best val accuracy is {}'.format(max(val_accs)))
            break
        total_c_loss, total_accuracy = oneShotBuilder.run_training_epoch(total_train_batches)
        print("Epoch {}: train_loss:{} train_accuracy:{}".format(e, total_c_loss, total_accuracy))
        train_losses.append(total_c_loss)
        train_accs.append(total_accuracy)

        total_val_c_loss, total_val_accuracy = oneShotBuilder.run_val_epoch(total_val_batches)
        val_accs.append(total_val_accuracy)
        val_losses.append(total_val_c_loss)
        print("Epoch {}: val_loss:{} val_accuracy:{}".format(e, total_val_c_loss, total_val_accuracy))
        if total_val_c_loss < best_val_loss:
            early_stop = 0
            best_val_loss = total_val_c_loss
            model = oneShotBuilder.matchNet
            torch.save(model.state_dict(), saved_mode)

        else:
            early_stop += 1
    plot_graph(train_accs, val_accs, train_losses, val_losses)


def test(saved_model):

    model =  MatchingNetwork(dropout=0.0, batch_size=batch_size, num_channels=channels, learning_rate=1e-3,  fce=True,  num_classes_per_set=classes_per_set,
                                        num_samples_per_class=samples_per_class, image_size=image_size, use_cuda=True)
    device = torch.device("cuda")
    model.load_state_dict(torch.load(saved_model))
    model.to(device)
    model.eval()
    test_loss, test_accuracy = oneShotBuilder.run_test_epoch( total_test_batches, model=model)
    print('test loss:', test_loss, 'test acc: ', test_accuracy)
    return  test_loss, test_accuracy


# Train Asl data set
train(saved_mode='omniglot_5w1s')

# test(saved_model='omniglot_5w1s')