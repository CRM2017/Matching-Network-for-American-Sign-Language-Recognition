from matching_networks import MatchingNetwork
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau


class DataBuilder:
    def __init__(self, data, batch_size, num_channels, lr, image_size, classes_per_set, samples_per_class, dropout,
                 fce, optim, weight_decay, use_cuda):
        self.data = data
        self.classes_per_set = classes_per_set
        self.sample_per_class = samples_per_class
        self.dropout = dropout
        self.batch_size = batch_size
        self.lr = lr
        self.image_size = image_size
        self.optim = optim
        self.wd = weight_decay
        self.isCuadAvailable = torch.cuda.is_available()
        self.use_cuda = use_cuda
        self.matchNet = MatchingNetwork(dropout, batch_size, num_channels, self.lr, fce, classes_per_set,
                                        samples_per_class, image_size, self.isCuadAvailable & self.use_cuda)
        self.total_iter = 0
        if self.isCuadAvailable & self.use_cuda:
            cudnn.benchmark = True  # set True to speedup
            torch.cuda.manual_seed_all(2021)
            self.matchNet.cuda()
        self.total_train_iter = 0
        self.optimizer = self.create_optimizer(self.matchNet, self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', verbose=True)

    def create_optimizer(self, model, lr):
        # setup optimizer
        if self.optim == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=self.wd)
        elif self.optim == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, dampening=0.9, weight_decay=self.wd)
        return optimizer

    def get_model (self):
        return self.matchNet

    def fit_model (self, total_batches, train_or_val, model=None):
        total_c_loss = 0.0
        total_accuracy = 0.0
        for i in range(total_batches):
            x_support_set, y_support_set, x_target, y_target = self.data.get_batch(augment=True, dataset_name = train_or_val)
            x_support_set = Variable(torch.from_numpy(x_support_set)).float()
            y_support_set = Variable(torch.from_numpy(y_support_set), requires_grad=False).long()
            x_target = Variable(torch.from_numpy(x_target)).float()
            y_target = Variable(torch.from_numpy(y_target), requires_grad=False).squeeze().long()

            # convert to one hot encoding
            y_support_set = y_support_set.unsqueeze(2)
            sequence_length = y_support_set.size()[1]
            batch_size = y_support_set.size()[0]
            y_support_set_one_hot = Variable(
                torch.zeros(batch_size, sequence_length, self.classes_per_set).scatter_(2,
                                                                                        y_support_set.data,
                                                                                        1), requires_grad=False)
            # reshape channels and change order
            x_support_set = x_support_set.permute(0, 1, 4, 2, 3)
            x_target = x_target.permute(0, 3, 1, 2)

            if train_or_val != 'test':
                acc, c_loss = self.matchNet(x_support_set.cuda(), y_support_set_one_hot.cuda(), x_target.cuda(),
                                            y_target.cuda())
                if train_or_val == 'train':
                    # optimize process
                    self.optimizer.zero_grad()
                    c_loss.backward()
                    self.optimizer.step()

                total_c_loss += c_loss.item()
                total_accuracy += acc.item()

            elif train_or_val == 'test' and model is not None:
                acc, c_loss = model(x_support_set.cuda(), y_support_set_one_hot.cuda(), x_target.cuda(),
                                            y_target.cuda())
                total_c_loss += c_loss.item()
                total_accuracy += acc.item()

            else:
                raise Exception("Fail to run fit_model")
        return total_c_loss, total_accuracy

    def run_training_epoch(self, total_train_batches):
        """
        Run the training epoch
        :param total_train_batches: Number of  batches on training
        :return:
        """
        total_c_loss, total_accuracy = self.fit_model(total_train_batches, train_or_val='train')
        total_c_loss = total_c_loss / total_train_batches
        total_accuracy = total_accuracy / total_train_batches
        return total_c_loss, total_accuracy

    def run_val_epoch(self, total_val_batches):
        """
        Run the val epoch
        :param total_val_batches: Number of batches on validation
        :return:
        """

        total_c_loss, total_accuracy = self.fit_model(total_val_batches, train_or_val='val')
        total_c_loss = total_c_loss / total_val_batches
        total_accuracy = total_accuracy / total_val_batches
        self.scheduler.step(total_c_loss)
        return total_c_loss, total_accuracy

    def run_test_epoch(self, total_val_batches, model):
        """
        Run the val epoch
        :param total_val_batches: Number of batches on validation
        :return:
        """

        self.matchNet.eval()
        total_c_loss, total_accuracy = self.fit_model(total_val_batches, train_or_val='test', model=model)
        total_c_loss = total_c_loss / total_val_batches
        total_accuracy = total_accuracy / total_val_batches
        self.scheduler.step(total_c_loss)
        return total_c_loss, total_accuracy
