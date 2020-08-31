#!/usr/bin/env python

import os
import time

from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
import gpytorch
import numpy as np
import torch
torch.cuda.empty_cache()
print(torch.cuda.memory_summary(device=None, abbreviated=False))
from fueling.control.dynamic_model_2_0.conf.model_conf import feature_config
from fueling.control.dynamic_model_2_0.conf.model_conf import smoke_test_training_config
from fueling.control.dynamic_model_2_0.conf.model_conf import training_config
from fueling.control.dynamic_model_2_0.conf.model_conf import toy_test_training_config
from fueling.control.dynamic_model_2_0.gp_regression.dynamic_model_dataset \
    import DynamicModelDataset
from fueling.control.dynamic_model_2_0.gp_regression.encoder import Encoder, TransformerEncoderCNN
from fueling.control.dynamic_model_2_0.gp_regression.train import save_model_state_dict
from fueling.control.dynamic_model_2_0.gp_regression.train import save_model_torch_script
import fueling.common.logging as logging
import fueling.control.dynamic_model_2_0.gp_regression.train_utils as train_utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_model = True
test_type = "full_test"
platform_dir = '/fuel/fueling/control/dynamic_model_2_0/testdata'
econder_type = "Transformer"
# econder_type = "Eoncoder"
# CUDA setup:
if (torch.cuda.is_available()):
    print("Using CUDA to speed up training.")
    use_cuda = True
else:
    print("Not using CUDA.")

if test_type == "full_test":
    config = training_config
    data_dir = '0708_2'
    training_data_path = os.path.join(platform_dir, data_dir, 'train')
    validation_data_path = os.path.join(platform_dir, data_dir, 'test')
elif test_type == "smoke_test":
    config = smoke_test_training_config
    data_dir = '0707_smoke_test'
    training_data_path = os.path.join(platform_dir, data_dir, 'train')
    validation_data_path = os.path.join(platform_dir, data_dir, 'test')
else:
    # default toy test
    config = toy_test_training_config
    training_data_path = "/fuel/fueling/control/dynamic_model_2_0/gp_regression/testdata/train"
    validation_data_path = "/fuel/fueling/control/dynamic_model_2_0/gp_regression/testdata/test"
# time
timestr = time.strftime('%Y%m%d-%H%M')
# save files at
result_folder = os.path.join(validation_data_path, f'{timestr}')
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
offline_model_path = os.path.join(result_folder, 'gp_model.pth')
online_model_path = os.path.join(result_folder, 'gp_model.pt')

print('finished initialization')


# setup data loader
train_dataset = DynamicModelDataset(
    training_data_path, is_train=True, factor_file=training_data_path)
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                          shuffle=True, num_workers=36, drop_last=True)
total_train_number = len(train_loader.dataset)
# train_y = train_dataset[0][1].unsqueeze(0)
# logging.info(train_y.shape)
# for i in range(1, total_train_number):
#     train_y = torch.cat((train_y, train_dataset[i][1].unsqueeze(0)))
# logging.info(train_y.shape)

print('train data set is ready')
# inducing points
step_size = int(max(config["batch_size"] / config["num_inducing_point"], 1))
inducing_point_num = torch.arange(0, config["batch_size"], step=step_size)
for idx, (features, labels) in enumerate(train_loader):
    features = torch.transpose(features, 0, 1).type(torch.FloatTensor)
    inducing_points = features[:, inducing_point_num, :].unsqueeze(0)
    break

inducing_points = torch.cat((inducing_points, inducing_points), 0)
logging.info(inducing_points.shape)


# validate loader
valid_dataset = DynamicModelDataset(
    validation_data_path, is_train=False, factor_file=training_data_path)
# reduce batch size when memory is not enough len(valid_dataset.datasets)
valid_loader = DataLoader(
    valid_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=36, drop_last=True)


# encoder
# bench mark encoder
if econder_type == "Transformer":
    encoder_net_model = TransformerEncoderCNN(
        u_dim=feature_config["input_dim"], kernel_dim=config["kernel_dim"])
else:
    encoder_net_model = Encoder(u_dim=feature_config["input_dim"],
                                kernel_dim=config["kernel_dim"])
model, likelihood, optimizer, loss_fn = train_utils.init_train(
    inducing_points, encoder_net_model, feature_config["output_dim"],
    total_train_number, config["lr"], kernel_dim=config["kernel_dim"])

if use_cuda:
    model = model.cuda()
    likelihood = likelihood.cuda()

# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     model = torch.nn.DataParallel(model)
#     model.to(device)

train_loss_plot = os.path.join(validation_data_path, f'{timestr}', 'train_loss.png')
if train_model:
    model, likelihood = train_utils.train_with_cross_validation(
        config["num_epochs"], train_loader, valid_loader, model, likelihood,
        loss_fn, optimizer, fig_file_path=train_loss_plot, is_transpose=True,
        use_cuda=use_cuda)
    # test save and load model
    save_model_state_dict(model, likelihood, offline_model_path)
    # save model as jit script
    for idx, (test_features, test_labels) in enumerate(valid_loader):
        test_features = torch.transpose(test_features, 0, 1).type(torch.FloatTensor)
        break
    save_model_torch_script(model, likelihood, test_features.cuda(), online_model_path)
    # save inducing points for load model
    np.save(os.path.join(result_folder, 'inducing_points.npy'), inducing_points)
else:
    # load model
    train_utils.load_model(offline_model_path, encoder_net_model, model, likelihood)

# validation
# Set into eval mode
model.eval()
likelihood.eval()

# use all validation data
# Make predictions
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    for i, (test_x, y) in enumerate(valid_loader):
        logging.info(test_x.shape)
        test_x = torch.transpose(test_x, 0, 1)
        predictions = likelihood(model(test_x.cuda()))
        mean = predictions.mean
        lower, upper = predictions.confidence_region()

# visualize
fig, ax = plt.subplots(1)
ax.set_xlabel('$\\Delta$x (m)', fontdict={'size': 12})
ax.set_ylabel('$\\Delta$y (m)', fontdict={'size': 12})
ax.set_title("Result Visualization")
# confidence region
confidence_regions = []
for idx in range(0, y.shape[0]):
    rect = Rectangle((lower[idx, 0], lower[idx, 1]), upper[idx, 0]
                     - lower[idx, 0], upper[idx, 1] - lower[idx, 1])
    confidence_regions.append(rect)
# confident regions
pc = PatchCollection(confidence_regions, facecolor='g', alpha=0.04, edgecolor='b')
ax.add_collection(pc)
# ground truth (dx, dy)
ax.plot(y[:, 0], y[:, 1],
        'o', color='blue', label='Ground truth')
# # # training labels
# ax.plot(train_y[:, 0], train_y[:, 1],
#         'kx', label='Training ground truth')
# predicted mean value
ax.plot(mean[:, 0].cpu(), mean[:, 1].cpu(), 's', color='r', label='Predicted mean')
ax.legend(fontsize=12, frameon=False)
ax.grid(True)
# save validation figures to folder
plt.savefig(os.path.join(validation_data_path, f'{timestr}', 'validation.png'))
plt.show()

#  accuracy
print(y.shape)
print(mean.shape)
mse_x = mean_squared_error(mean[:, 0].cpu(), y[:, 0])
mse_y = mean_squared_error(mean[:, 1].cpu(), y[:, 1])
print(f'mse loss is {mse_x}')
print(f'mse loss is {mse_y}')
