#!/usr/bin/env python
"""
A demo PySpark job with pytorch distributed data parallel training.

3 simple steps to make a training from single-machine/single-card to multi-machines/multi-cards:

I. In Spark driver, register the job by "ddp.register_job"
II. In Spark executor, convert regular model to distributed by "ddp.model_to_dist"
III. In Spark executor, convert regular data loader to distributed by 
     "ddp.data_loader_to_dist" (optional) 

Run with:
    bazel run //fueling/demo/:gpu_training_with_pytorch_distributed -- --cloud --gpu=2 --workers=2

"""

# Standard packages
import os
import sys
import time

# Third-party packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Apollo-fuel packages
from fueling.common.base_pipeline import BasePipeline
import fueling.common.distributed_data_parallel as ddp
import fueling.common.logging as logging
import fueling.common.socket_utils as socket_utils 
import fueling.common.storage.bos_client as bos_client


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PytorchTraining(BasePipeline):
    """Demo pipeline."""

    def run(self):
        """Run."""
        time_start = time.time()

        # For distributed gpu computing, we need to know how many workers cooperate,
        # and register them corespondingly.
        workers = int(os.environ.get('APOLLO_EXECUTORS', 1))
        job_id = self.FLAGS.get('job_id') 
        ddp.register_job(job_id, workers)

        # Spark distributing as normal
        self.to_rdd(range(workers)).foreach(lambda instance: self.train(instance, workers, job_id))
        logging.info('Training complete in {} seconds.'.format(time.time() - time_start))


    @staticmethod
    def train(instance, world_size, job_id):
        """Run training task"""
        if os.system('nvidia-smi') != 0:
            logging.fatal('Failed to run nvidia-smi.')
            time.sleep(60*3)
            sys.exit(-1)

        logging.info(F'cuda available? {torch.cuda.is_available()}')
        logging.info(F'cuda version: {torch.version.cuda}')
        logging.info(F'gpu device count: {torch.cuda.device_count()}')
        logging.info(F'instance: {instance}, world_size: {world_size}, job_id: {job_id}')

        # Prepare data sets.  The data sets here are put on BOS already so it won't download
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        demo_data_path = os.path.join(bos_client.BOS_MOUNT_PATH, 'test/demo')
        trainset = torchvision.datasets.CIFAR10(root=demo_data_path, train=True,
                                                download=True, transform=transform)
        validset = torchvision.datasets.CIFAR10(root=demo_data_path, train=False,
                                                download=True, transform=transform)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        # Regular model to distributed
        model = Net()
        model, rank = ddp.model_to_dist(model, world_size, job_id) 
        logging.info(F'current worker rank: {rank}, world_size: {world_size}.')

        device_ids = ddp.get_device_ids()

        # Regular data loader to distributed
        trainloader = ddp.data_loader_to_dist(trainset, batch_size=4, num_workers=2,
                                              world_size=world_size, rank=rank)
        validloader = ddp.data_loader_to_dist(validset, batch_size=4, num_workers=2,
                                              world_size=world_size, rank=rank)

        # The rest are just like normal procedures
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        begin_time = time.time()
        logging.info(F'training begin time: {begin_time}')
        
        # loop over the dataset multiple times
        for epoch in range(3):  
            running_loss = 0.0
            last_data_piece, last_label_piece = None, None
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                #inputs, labels = data
                inputs, labels = data[0].to(device_ids[0]), data[1].to(device_ids[0])
                last_data_piece, last_label_piece = inputs, labels
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                logging.info(F'current loss: {loss.item()}')
        end_time = time.time()
        logging.info(F'training end time: {end_time}, duration: {end_time - begin_time}')

        # Validation
        dataiter = iter(validloader)
        images, labels = dataiter.next()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in validloader:
                images, labels = data[0].to(device_ids[0]), data[1].to(device_ids[0])
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        logging.info(F'Accuracy of the network on the 10000 test images: {100 * correct / total}')
        time.sleep(60 * 3)


if __name__ == '__main__':
    PytorchTraining().main()
