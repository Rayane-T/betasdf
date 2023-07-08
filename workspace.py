#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import json
import os
specs_subdir="specs"

model_params_subdir = "weights"
log_params_subdir = "log"
optimizer_params_subdir = "optimizer"

specifications_filename = "specs.json"



"""
latent_codes_subdir = "LatentCodes"

recon_testset_subdir = "Results_recon_testset"
inter_testset_subdir = "Results_inter_testset"
generation_subdir = "Results_generation"

recon_testset_ttt_subdir = "Results_recon_testset_ttt"
inter_testset_ttt_subdir = "Results_inter_testset_ttt"
generation_ttt_subdir = "Results_generation_ttt"

reconstruction_meshes_subdir = "Meshes"
reconstruction_codes_subdir = "Codes"
reconstruction_models_subdir = "Models"


data_source_map_filename = ".datasources.json"
evaluation_subdir = "Evaluation"
sdf_samples_subdir = "SdfSamples"
surface_samples_subdir = "SurfaceSamples"
normalization_param_subdir = "NormalizationParameters"
training_meshes_subdir = "TrainingMeshes"
"""

def load_experiment_specifications(experiment_directory, exper):

    filename = os.path.join(experiment_directory, specs_subdir, exper+"_"+specifications_filename)
    print(filename)
    if not os.path.isfile(filename):
        raise Exception(
            "The experiment directory ({}) does not include specifications file "+'"specs.json"'.format(experiment_directory)
        )

    return json.load(open(filename))

"""
save and reload

"""

def get_model_params_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, (model_params_subdir))

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_optimizer_params_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, optimizer_params_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir

def get_loss_params_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, log_params_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir