from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from pathlib import Path
from torch import Tensor
from tqdm import tqdm
import numpy as np
import random
import torch
import cv2
import sys
import os

DATASET_PATH = Path(__file__).parent.parent / "data/casia-webface"

class FacesDataset(Dataset):
    """
    This class is for use specifically with a formatted version of the casia-webface dataset
    The dataset_path should point to a directory with folders titled 0, 1, 2, 3, ...
    Inside each folder should be images titled 0.jpg, 1.jpg, 2.jpg, ...
    The max_loaded parameter tells the dataset how many people's images to load into memory at once
    The split parameter is a number between 0 and 1 that indicates how much of the dataset should be reserved for training versus testing
    """

    # folder that the dataset is saved to
    dataset_path: Path

    # paths pointing to folders of each person
    people_folders: list[Path]

    # ---------------------------------------------------- #

    # determines whether to use data before or after the split
    train_mode: bool

    # the index at which the split is made
    split_index: int

    # ---------------------------------------------------- #

    # operation applied to each image as it loads
    transform: T.Compose

    # ---------------------------------------------------- #

    # maximum number of people allowed to be loaded into memory
    max_loaded: int

    # the chunk's actual data
    loaded_data: list[list[Tensor]]

    # the actual number of people loaded into memory (may differ because of dataset size)
    num_loaded: int

    # ---------------------------------------------------- #

    # number of datapoitns that can be loaded before requiring a new chunk
    max_gets_before_reloading: int

    # the number of gets since the last chunk refresh
    num_gets: int


    def __init__(s: "FacesDataset", dataset_path: Path, max_loaded: int, split: float):
        assert isinstance(dataset_path, Path), f"path must be a Path object from pathlib, received {type(dataset_path)}"
        assert max_loaded > 1, f"max_loaded must be greater than 1, received {max_loaded}"
        assert split > 0 and split <= 1, f"split must be a value between 0 and 1, received {split}"

        # save the dataset's path to the state
        s.dataset_path = dataset_path

        # get a list of Path objects that point to the folders of individual 'people'
        s.people_folders = [folder for folder in s.dataset_path.iterdir() if folder.is_dir()]


        # default value for train is True
        s.train_mode = True

        # converts the split floating point to an actual index (person) where the train/test cutoff happens
        s.split_index = round(s.people_folders.__len__() * split)

        # transform that converts the np image array into a normalized Tensor
        s.transform = T.Compose(
            [
                T.ToTensor(),
            ]
        )

        # maximum number of 'people' that the dataset will load into memory
        s.max_loaded = max_loaded

        # loads the initial chunk into memory
        s.load_chunk()


    def __len__(s):
        return s.num_loaded

    # indicates data should come from before the split
    def train (s, val: bool = True):
        
        # data should be refreshed if train_mode changes
        requires_refresh = val != s.train_mode
        s.train_mode = val

        # refresh data if necessary
        if requires_refresh: s.load_chunk()

        return s

    # indicates data should come from after the split
    def eval (s, val: bool = True):

        # data should be refreshed if train_mode changes
        requires_refresh = val == s.train_mode
        s.train_mode = not val

        # refresh data if necessary
        if requires_refresh: s.load_chunk()

        return s

    # loads a fresh selection of data for training
    def load_chunk(s):

        # clear old data
        s.loaded_data = []
        s.num_loaded = 0
        s.num_gets = 0

        # determine the list of people 'allowed' to be loaded based on split
        allowed_people_folders = s.people_folders[:s.split_index] if s.train_mode else s.people_folders[s.split_index:]

        # select a random assortment of folders to load
        selected_people_folders = random.sample(allowed_people_folders, min(allowed_people_folders.__len__(), s.max_loaded))

        # Loop through each person (without going over the max_loaded)
        for person_folder in selected_people_folders:

            # collect each person image in this list
            person_imgs = []

            # load each image individually
            for image in person_folder.iterdir():
                img = cv2.imread(str(image))
                img = s.transform(img)
                person_imgs.append(img)

                # indicate one more is loaded
                s.num_loaded += 1

            # add the new images to the state
            s.loaded_data.append(person_imgs)

    def load_datapoint(s, same: bool):

        # check if a refresh is required
        if s.num_gets >= s.num_loaded:
            s.load_chunk()

        # this function call itself is a get
        s.num_gets += 1

        # indices for the two loaded images
        p1: int
        p2: int
        i1: int
        i2: int

        if same:
            # p1 and p2 are the same random index
            p1 = random.randint(0, s.loaded_data.__len__() - 1)
            p2 = p1
        else:
            # p1 and p2 are two different indices
            p1 = random.randint(0, s.loaded_data.__len__() - 1)
            p2 = random.randint(0, s.loaded_data.__len__() - 2)
            if p2 >= p1:
                p2 += 1

        # i1 and i2 are either unrelated (different) or cannot be the same (same)
        i1 = random.randint(0, s.loaded_data[p1].__len__() - 1)
        i2 = random.randint(0, s.loaded_data[p2].__len__() - (2 if same else 1))
        if same:
            if i2 >= i1:
                i2 += 1

        # return the images from the loaded data at the given indices
        return s.loaded_data[p1][i1], s.loaded_data[p2][i2]


class FacesDataLoader:

    pct_sim: float
    dataset: FacesDataset
    batch_size: int
    idx: int

    def __init__(s, dataset: FacesDataset, batch_size: int, similar_ratio: float):
        assert isinstance(dataset, FacesDataset), f"dataset must be an instance of FacesDataset, received {type(dataset)}"
        assert isinstance(batch_size, int) and batch_size > 0, f"batch size must be an integer greater than zero, received {batch_size} of type {type(batch_size)}"
        assert isinstance(similar_ratio, float) and 0 <= similar_ratio <= 1, "similar ratio must be a float between 0 and 1"

        # store the variables in the state for later use
        s.pct_sim = similar_ratio
        s.dataset = dataset
        s.batch_size = batch_size

        # idx used for iteration
        s.idx = 0

    # wraps the dataset's train method
    def train (s, val: bool = True):
        s.dataset.train(val)
        return s

    # wraps the dataset's eval method
    def eval (s, val: bool = True):
        s.dataset.eval(val)
        return s

    def __len__(s):
        return s.dataset.__len__() // s.batch_size

    def __iter__(s):
        s.idx = 0
        return s

    def __next__(s) -> tuple[Tensor, Tensor, Tensor]:

        # stop the iteration if the end of the epoch is reached
        if s.idx >= s.__len__():
            raise StopIteration()

        # store individual faces in a list (to be stacked into singular batch tensors)
        face1s = []
        face2s = []
        sims = []

        # load n images based on batch size
        for i in range(s.batch_size):

            # randomly determine whether the images will be the same or different people
            similar = True if random.random() < s.pct_sim else False
            sim_val = 1 if similar else 0
            face1, face2 = s.dataset.load_datapoint(similar)

            # add the selected data to the lists
            face1s.append(face1)
            face2s.append(face2)
            sims.append(torch.as_tensor(sim_val).float())

        # stack them into batch tensors
        face1s = torch.stack(face1s)
        face2s = torch.stack(face2s)
        sims = torch.stack(sims).float()

        # increment index
        s.idx += 1

        # return the data
        return face1s, face2s, sims

