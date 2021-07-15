import torch


class Trainer:
    def __init__(self, net, dataloader, optimizer):
        self.net = net
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.device = torch.device('cpu')

    def train_epoch(self):
        running_loss = 0
        for iter_id, batch in enumerate(self.dataloader):
            for item in batch:
                batch[item] = batch[item].to(self.device)

            # network forward & computing loss
            loss, _ = self.net(batch)
            running_loss += loss.item()

            # back-propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return running_loss / (iter_id + 1)

    def set_device(self, device_str):
        self.device = torch.device(device_str)
        self.net.to(device=self.device)
