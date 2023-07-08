#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import logging
import torch
import torch.nn as nn
import numpy as np
import sys
import os

class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass

class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch, ceiling_epoch=15000):
        if epoch<ceiling_epoch:
        	return self.initial * (self.factor ** (epoch / self.interval))
        else:
            return self.initial * (self.factor ** (ceiling_epoch / self.interval))
        
class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, warmed_up, length):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def get_learning_rate(self, epoch):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length
    
class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, lr):
        self.initial = lr

    def get_learning_rate(self, epoch):
        return self.initial
    
    
    
def get_learning_rate_schedules(specs):

    schedule_specs = specs["LearningRateSchedule"]

    schedules = []

    for schedule_specs in schedule_specs:

        if schedule_specs["Type"] == "Step":
            schedules.append(
                StepLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Interval"],
                    schedule_specs["Factor"],
                )
            )
        elif schedule_specs["Type"] == "Warmup":
            schedules.append(
                WarmupLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Final"],
                    schedule_specs["Length"],
                )
            )
        elif schedule_specs["Type"] == "Constant":
            schedules.append(ConstantLearningRateSchedule(schedule_specs["Value"]))

        else:
            raise Exception(
                'no known learning rate schedule of type "{}"'.format(
                    schedule_specs["Type"]
                )
            )

    return schedules

#this is to be compatible with previous exper
def save_model(ws, experiment_directory, filename, decoder, epoch):

    model_params_dir = ws.get_model_params_dir(experiment_directory, True)    #./weights
    torch.save(decoder.state_dict(),os.path.join(model_params_dir,filename+".pt")
        
    )
    
def load_model(ws, experiment_directory, filename, net):

    model_params_dir = ws.get_model_params_dir(experiment_directory, True)
    path=os.path.join(model_params_dir,filename+".pt")
    
    saved_epoch=filename.split("_")[-1]

    if not os.path.isfile(path):
        raise Exception('model state dict "{}" does not exist'.format(path))

    data = torch.load(path)
    net.load_state_dict(data)

    return net, int(saved_epoch)
   
    
def save_optimizer(ws, experiment_directory, filename, optimizer, epoch):

    optimizer_params_dir = ws.get_optimizer_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optimizer_params_dir, filename+".pt"))
    
def load_optimizer(ws, experiment_directory, filename, optimizer):
    
    optimizer_params_dir = ws.get_optimizer_params_dir(experiment_directory, True)
    path = os.path.join(optimizer_params_dir, filename+".pt")

    if not os.path.isfile(path):
        raise Exception(
            'optimizer state dict "{}" does not exist'.format(path)
        )

    data = torch.load(path)

    optimizer.load_state_dict(data["optimizer_state_dict"])

    return optimizer, data["epoch"]

def save_loss(ws, experiment_directory, filename, loss_history, epoch):

    loss_params_dir = ws.get_loss_params_dir(experiment_directory, True)
    np.save(os.path.join(loss_params_dir, filename+".npy"),loss_history)


def load_loss(ws, experiment_directory, filename):

    loss_params_dir = ws.get_loss_params_dir(experiment_directory, True)
    path=os.path.join(loss_params_dir, filename+".npy")
    
    if not os.path.isfile(path):
        raise Exception('log file "{}" does not exist'.format(path))

    loss=np.load(path,allow_pickle=True)
    return loss.item()

def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default

def adjust_learning_rate(lr_schedules, optimizer, epoch):

    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)
