#!/usr/bin/env python

"""Extracting and processing dataset"""

import os
import pickle

from liegroups.torch import SO3
from torch.utils.data.dataset import Dataset
import torch


class KAISTDataset(Dataset):
    """"Transfer the uploaded raw data in the KAIST format into the customized dataset """

    def __init__(self, args):
        """Initialization"""
        self.name = args.dataset_name
        self.path_data_save = os.path.join(args.path_data_save, args.dataset_name)
        self.path_results = os.path.join(args.path_results, args.dataset_name)
        self.path_temp = os.path.join(args.path_temp, args.dataset_name)
        self.test_sequences = args.test_sequences
        self.cross_validation_sequences = args.cross_validation_sequences
        self.get_datasets()
        self.set_normalize_factors(args)

        # Transformation frame for Kaist dataset
        T_vehicle2fog = torch.eye(4)
        T_vehicle2imu = torch.eye(4)
        T_vehicle2fog[:3, 3] = torch.Tensor([-0.335, -0.035, 0.78])
        T_vehicle2imu[:3, 3] = torch.Tensor([-0.07, 0, 1.7])
        self.calibration_parameters = {"Encoder resolution": 4096,
                                       "Encoder left wheel diameter": 0.623803,
                                       "Encoder right wheel diameter": 0.623095,
                                       "Encoder wheel base": 1.52683,
                                       "Vehicle2FOG": T_vehicle2fog,
                                       "Vehicle2IMU": T_vehicle2imu}

    def get_datasets(self):
        """Extract the names of the datasets stored in the saved data path"""
        self.datasets = []
        for dataset in os.listdir(self.path_data_save):
            self.datasets.append(dataset[:-2])  # take just name
        self.divide_datasets()

    def divide_datasets(self):
        """Divide the extraced datasets (name) into traning/validatation/test categories"""
        self.datasets_test = self.test_sequences
        self.datasets_validation = self.cross_validation_sequences
        self.datasets_train = []
        for dataset in self.datasets:
            if ((not dataset in self.datasets_test) and
                    (not dataset in self.datasets_validation)):
                self.datasets_train.append(dataset)

    def dataset_name(self, idx):
        """Inquire the names of the given dataset"""
        return self.datasets[idx]

    def get_filter_data(self, idx):
        """Get the filter data from the dataset"""
        if type(idx) != int:
            idx = self.datasets.index(idx)
        pickle_dict = self[idx]
        t = pickle_dict['t']
        chi0 = pickle_dict['chi'][0]
        Rot0 = chi0[:3, :3]

        angles = SO3.from_matrix(Rot0).to_rpy()
        p0 = chi0[:3, 3]
        u_odo_fog = pickle_dict['u_odo_fog']
        y_imu = pickle_dict['u_imu']
        x0 = torch.zeros(9)
        x0[:3] = p0
        x0[3:6] = angles
        return t, x0, u_odo_fog, y_imu

    def get_ground_truth_data(self, idx):
        """Get the ground truth data from the dataset"""
        pickle_dict = self[self.datasets.index(idx) if type(idx) != int else idx]
        return pickle_dict['t'], pickle_dict['chi']

    def get_test_data(self, idx, gp_name):
        """Get the test data from the dataset"""
        var = "odo_fog" if gp_name == "GpOdoFog" else "imu"
        dataset = self.datasets_test[idx] if type(idx) == int else idx
        pickle_dict = self[self.datasets.index(dataset)]
        u = pickle_dict["u_" + var]
        y = pickle_dict["y_" + var]
        u = self.normalize(u, "u_" + var)
        if u[0].norm() == 0:  # (Urban00-05 and campus00)
            u = torch.zeros(0, u.shape[1], u.shape[2])
            y = torch.zeros(0, y.shape[1])
        return u, y

    def get_validation_data(self, idx, gp_name):
        """Get the cross validation data from the dataset"""
        var = "odo_fog" if gp_name == "GpOdoFog" else "imu"
        dataset = self.datasets_validation[idx] if type(idx) == int else idx
        pickle_dict = self[self.datasets.index(dataset)]
        u = pickle_dict["u_" + var]
        y = pickle_dict["y_" + var]
        u = self.normalize(u, "u_" + var)
        if u[0].norm() == 0:  # (Urban00-05 and campus00)
            u = torch.zeros(0, u.shape[1], u.shape[2])
            y = torch.zeros(0, y.shape[1])
        return u, y

    def get_train_data(self, idx, gp_name):
        """Get the training data from the dataset"""
        var = "odo_fog" if gp_name == "GpOdoFog" else "imu"
        dataset = self.datasets_train[idx] if type(idx) == int else idx
        pickle_dict = self[self.datasets.index(dataset)]
        u = pickle_dict["u_" + var]
        y = pickle_dict["y_" + var]
        u = self.normalize(u, "u_" + var)
        if u[0].norm() == 0:  # (Urban00-05 and campus00)
            u = torch.zeros(0, u.shape[1], u.shape[2])
            y = torch.zeros(0, y.shape[1])
        return u, y

    def __getitem__(self, idx):
        with open(os.path.join(self.path_data_save, self.datasets[idx]) + '.p',
                  "rb") as file_pi:
            mondict = pickle.load(file_pi)
        return mondict

    def __len__(self):
        return len(self.datasets)

    def set_normalize_factors(self, args):
        """Compute mean and variance of input data using only training data"""
        # first mean
        self.num_data = 0
        for i, dataset in enumerate(self.datasets_train):
            with open(os.path.join(self.path_data_save, dataset) + '.p', "rb") as file_pi:
                pickle_dict = pickle.load(file_pi)
            u_odo_fog = pickle_dict['u_odo_fog']
            u_imu = pickle_dict['u_imu']
            if i == 0:
                u_odo_fog_loc = u_odo_fog.mean(dim=0).mean(dim=0)
                u_imu_loc = u_imu.mean(dim=0).mean(dim=0)
            else:
                u_odo_fog_loc += u_odo_fog.mean(dim=0).mean(dim=0)
                u_imu_loc += u_imu.mean(dim=0).mean(dim=0)
            self.num_data += u_imu.shape[0]
        u_odo_fog_loc = u_odo_fog_loc / len(self.datasets_train)
        u_imu_loc = u_imu_loc / len(self.datasets_train)

        # second standard deviation
        u_length = 0
        for i, dataset in enumerate(self.datasets_train):
            with open(os.path.join(self.path_data_save, dataset) + '.p', "rb") as file_pi:
                pickle_dict = pickle.load(file_pi)
            u_odo_fog = pickle_dict['u_odo_fog']
            u_imu = pickle_dict['u_imu']
            if i == 0:
                u_odo_fog_std = ((u_odo_fog - u_odo_fog_loc) ** 2).sum(dim=0).sum(dim=0)
                u_imu_std = ((u_imu - u_imu_loc) ** 2).sum(dim=0).sum(dim=0)
            else:
                u_odo_fog_std += ((u_odo_fog - u_odo_fog_loc) ** 2).sum(dim=0).sum(dim=0)
                u_imu_std += ((u_imu - u_imu_loc) ** 2).sum(dim=0).sum(dim=0)
            u_length += u_odo_fog.shape[0] * u_odo_fog.shape[1]
            u_odo_fog_std = (u_odo_fog_std / u_length).sqrt()
            u_imu_std = (u_imu_std / u_length).sqrt()

        # for constant measurements, set standard deviation to 1
        u_odo_fog_std[u_odo_fog_std == 0] = 1
        u_imu_std[u_imu_std == 0] = 1
        self.normalize_factors = {'u_odo_fog_loc': u_odo_fog_loc,
                                  'u_imu_loc': u_imu_loc,
                                  'u_odo_fog_std': u_odo_fog_std,
                                  'u_imu_std': u_imu_std}

        pickle_dict = {'normalize_factors': self.normalize_factors}
        with open(os.path.join(self.path_temp, "normalize_factors.p"), "wb") as file_pi:
            pickle.dump(pickle_dict, file_pi)

    def normalize(self, x, var="u_odo_fog"):
        """Nomalize the given data"""
        x_loc = self.normalize_factors[var + "_loc"]
        x_std = self.normalize_factors[var + "_std"]
        x_normalized = (x - x_loc) / x_std
        return x_normalized


class NCLTDataset(KAISTDataset):
    """"Transfer the uploaded raw data in the NCLT format into the customized dataset"""

    def __init__(self, args):
        super(NCLTDataset, self).__init__(args)
        # Transformation frame for NCLT dataset
        T_vehicle2fog = torch.eye(4)
        T_vehicle2imu = torch.eye(4)
        T_vehicle2fog[:3, 3] = torch.Tensor([0, -0.25, -0.49])
        T_vehicle2imu[:3, 3] = torch.Tensor([-0.11, -0.18, -0.71])
        self.calibration_parameters = {"Vehicle2FOG": T_vehicle2fog,
                                       "Vehicle2IMU": T_vehicle2imu}
