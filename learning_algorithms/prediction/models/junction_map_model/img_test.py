#!/usr/bin/env python

import torch

from fueling.learning.train_utils import *
from learning_algorithms.prediction.models.junction_map_model.junction_map_model import *


# Set-up data-loader
# train_dataset = SemanticMapDataset("/data/image-feature/")
valid_dataset = JunctionMapDataset("/data0/image-feature/")

# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=16)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=True, num_workers=16)

model = JunctionMapModel(20, 12)
model.load_state_dict(torch.load("model.pt"))
model.eval()

numerator, denominator = 0, 0
for i, (X, y) in enumerate(valid_loader):
    pred = model(X)
    y_pred = pred.cpu()
    y_true = y.cpu()
    pred_label = y_pred.topk(1)[1]
    true_label = y_true.topk(1)[1]
    accuracy = (pred_label == true_label).type(torch.float).mean().item()
    numerator += np.sum(pred_label == true_label)
    denominator += true_label.shape[0]
    print("step "+str(i)+": "+str(numerator)+" / "+str(denominator)
          + " = "+str(100*numerator/denominator)+" %")

    # max_kappa = 0
    # label = y.detach().numpy()
    # label = np.reshape(label, (10,2,-1))
    # for i in range(1,9):
    #     x1 = label[i,0]-label[i-1,0]
    #     y1 = label[i,1]-label[i-1,1]
    #     x2 = label[i+1,0]+label[i-1,0]-2*label[i,0]
    #     y2 = label[i+1,1]+label[i-1,1]-2*label[i,1]
    #     kappa = np.abs(x1*y2-y1*x2)/np.sqrt(x1**2 + y1**2+1e-9)**3
    #     max_kappa = max(kappa, max_kappa)
    # print(max_kappa, numerator/denominator)
