from farabio.core.convnettrainer import ConvnetTrainer
from farabio.data.biodatasets import ChestXrayDataset
from torch.utils.data import DataLoader

class VggTrainer(ConvnetTrainer):
    """VGG-Net trainer class. Override with custom methods here.

    Parameters
    ----------
    ConvnetTrainer : BaseTrainer
        Inherits ConvnetTrainer class
    """

    def define_data_attr(self):
        self._data_path = self.config.data_path
        self._batch_size_train = self.config.batch_size_train
        self._batch_size_valid = self.config.batch_size_valid
        self._batch_size_test = self.config.batch_size_test
        self.image_datasets = {x: ChestXrayDataset(root=self._data_path, mode=x, transform=None, download=False)
                               for x in [self._TRAIN, self._VAL, self._TEST]}

    def define_compute_attr(self):
        self._device = self.config.device

    def define_misc_attr(self):
        self._TRAIN = self.config.TRAIN
        self._TEST = self.config.TEST
        self._VAL = self.config.VAL

    def get_trainloader(self):
        self.train_loader = DataLoader(
            image_datasets[self._TRAIN], batch_size=self._batch_size_train, shuffle=True)
        self.valid_loader = DataLoader(
            image_datasets[self._VAL], batch_size=self._batch_size_valid, shuffle=True)

    def get_testloader(self):
        self.test_loader = DataLoader(
            image_datasets[self._TEST], batch_size=self._batch_size_test, shuffle=True)

    def build_model(self):
        model_pre = models.vgg16(pretrained=True)
        class_names = image_datasets[TRAIN].classes

        for param in model_pre.features.parameters():
            param.required_grad = False

        num_features = model_pre.classifier[6].in_features
        features = list(model_pre.classifier.children())[:-1]
        features.extend([nn.Linear(num_features, len(class_names))])
        model_pre.classifier = nn.Sequential(*features)
        self._model_pre = model_pre.to(device)

        self._criterion = nn.CrossEntropyLoss()
        self._exp_lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.1)
        self._optimizer = optim.SGD(
            model_pre.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01)
       # print(model_pre)

    def on_train_start(self):
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

    def on_train_epoch_start(self):
        print(f"Epoch: {epoch+1}/{num_epochs}")
        print("="*10)
        self._exp_lr_scheduler.step()
        self._model_pre.train()

        running_loss = 0.0
        running_corrects = 0
        step = 0

    def training_step(self):
        step += 1
        inputs, labels = data
        inputs = inputs.to(self._device)
        labels = labels.to(self._device)
        self._optimizer.zero_grad()
        with torch.set_grad_enabled(phase == self._TRAIN):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = self._criterion(outputs, labels)
            if phase == 'train':
                loss.backward()
                self._optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        if step % 100 == 0:
            print(f"{step}/{len(self.train_loader)}")

    def on_train_epoch_end(self):
        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))

        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    def on_train_end(self):
        print('Best val Acc: {:4f}'.format(best_acc))
        model.load_state_dict(best_model_wts)
