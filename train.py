import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
import wandb

from torchnet import meter
from torch.autograd import Variable

from model import *

class TrainParams(object):
    
    def __init__(self, max_epoch, save_dir, learning_rate = 1e-3, lr_step = 3, lr_factor = 0.5):
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        self.save_dir = save_dir
        self.optimizer = None
        self.lr_scheduler = None
        self.criterion = None
        self.lr_step = lr_step
        self.lr_factor = lr_factor

class Trainer(object):

    def __init__(self, model, dataset, train_params, ckpt = None):
        self.data = dataset
        self.params = train_params
        self.last_epoch = 0
        
        self.criterion = self.params.criterion
        self.optimizer = self.params.optimizer
        self.lr_scheduler = self.params.lr_scheduler

        self.model = model
        self.model.cuda()

        if not os.path.exists(self.params.save_dir):
            os.makedirs(self.params.save_dir)
        
        if ckpt:
            if os.path.exists(ckpt):
                checkpoint = torch.load(ckpt)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.last_epoch = checkpoint["epoch"]
            else:
                raise FileNotFoundError("checkpoint file not found!")
        
        self.loss_meter = meter.AverageValueMeter()

    def train(self):
        
        best_loss = dict(data = np.inf, epoch = -1)

        for epoch in tqdm(range(self.last_epoch, self.params.max_epoch)):
            
            self.loss_meter.reset()
            self.last_epoch += 1
            
            logging.info("Start train epoch {}".format(self.last_epoch))
            
            self._train_one_epoch(epoch)
            
            if best_loss["data"] < self.loss_meter.mean:
                best_loss["data"] = self.loss_meter.mean
                best_loss["epoch"] = epoch

            logger.info("epoch:{epoch}, lr:{lr}, loss:{loss}".format(
                epoch = epoch, 
                loss  = self.loss_meter.mean,
                lr    = self.optimizer.param_groups[0]['lr']
            ))
            
            torch.save({
                'epoch': epoch, 
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.loss_meter.mean
            }, os.path.join(self.params.save_dir, "epoch_{}.pt".format(epoch)))

    def _train_one_epoch(self, epoch):
        enum = tqdm(enumerate(self.data), total = len(self.data))
        for _, data in enum:
            inputs = Variable(data["image"]).cuda()
            target = Variable(data["label"]).cuda()

            # forward
            score = self.model(inputs)
            loss = self.criterion(score, target)
            
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step(None)

            self.loss_meter.add(loss.data.cpu().detach())

            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            s = ('%10s' * 2 + ' loss: %10.6g   batch: %10.4g  img_size: %10.4g') % (
                    '%g/%g' % (epoch, self.params.max_epoch - 1), mem, loss.data, target.shape[0], inputs.shape[-1])
            enum.set_description(s)

        wandb.log({"train_loss": loss.data})
        enum.close()


def set_parser():
    parser = argparse.ArgumentParser(description="train.py")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64, help='total batch size')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--resume-from', type=str, default="runs/project/epoch_70.pt", help='checkpoint file path')
    parser.add_argument('--save-dir', type=str, default="./runs/project", help="save dir")
    return parser
    
if __name__ == "__main__":
    parser = set_parser()
    opt = parser.parse_args()

    train_images_root = "dataset/train_images"
    train_label = "dataset/train.csv"
    train_args = TrainParams(max_epoch=opt.epochs, save_dir=opt.save_dir)

    wandb.init(project="Plant_Pathology", config={
        "epoch": opt.epochs,
        "batch_size": opt.batch_size,
        "init_lr": train_args.learning_rate,
        "step_size": train_args.lr_step
    })

    model = ResNet50(num_outputs = len(class_list))
    model.cuda()

    train_dataset = BasicDataset(train_images_root, train_label, Transformation())
    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                    shuffle=False, 
                                                    batch_size = opt.batch_size, 
                                                    num_workers = opt.workers, 
                                                    pin_memory=True)
    ckpt = None
    if opt.resume_from != "pretrained":
        ckpt = opt.resume_from

    logger.info("Init train_args")

    optimizer = torch.optim.Adam(model.parameters(), lr=train_args.learning_rate)
    train_args.optimizer = optimizer
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(train_args.optimizer, step_size=train_args.lr_step)
    train_args.lr_scheduler = lr_scheduler

    train_args.criterion = MultiLabelCrossEntropyLoss()

    trainer = Trainer(model, train_dataloader, train_args, ckpt = ckpt)

    trainer.train()
    