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
    path: Path
    people: list[Path]
    train: bool
    split: int
    loaded: list[list[Tensor]]
    max_gets: int
    gets: int
    num_loaded: int

    def __init__(s: "FacesDataset", path: Path, max_people_loaded: int, split: float):
        assert isinstance(path, Path), (
            f"path must be a Path object from pathlib, received {type(path)}"
        )
        assert max_people_loaded > 1, (
            f"max_people_loaded must be greater than 1, received {max_people_loaded}"
        )
        assert split > 0 and split <= 1, (
            f"split must be a value between 0 and 1, received {split}"
        )

        # default value for train is True
        s.train = True

        s.path = path

        # maximum number of 'people' that the dataset will load into memory
        s.max_loaded = max_people_loaded

        # get a list of Path objects that point to the folders of individual 'people'
        s.people = [folder for folder in path.iterdir() if folder.is_dir()]

        # converts the split floating point to an actual index (person) where the train/test cutoff happens
        s.split = round(s.people.__len__() * split)

        # transform that converts the np image array into a normalized Tensor
        s.transform = T.Compose(
            [
                T.ToTensor(),
            ]
        )

        # loads the initial chunk into memory
        s.load_chunk()

        # count the number of datapoints delivered
        s.gets = 0

    def __len__(s):
        res = 0
        for person in s.loaded:
            for image in person:
                res += 1
        return res

    def load_chunk(s):
        # reset loaded and gets
        s.num_loaded = 0
        s.gets = 0

        # list of paths of people that can be loaded (given train/eval)
        to_be_loaded: list[Path] = random.sample(
            s.people[: s.split if s.train else s.split :], s.max_loaded
        )
        random.shuffle(to_be_loaded)

        # clear the currently loaded Tensors
        s.loaded = []

        # Loop through each person (without going over the max_people_loaded)
        for person in to_be_loaded[: s.max_loaded]:
            # collect each person image in this variable
            person_imgs = []

            # load each image individually
            for image in person.iterdir():
                img = cv2.imread(str(image))

                # convert cv2 result into Tensor
                img = s.transform(img)

                person_imgs.append(img)

                s.num_loaded += 1

            # add the new images to the state
            s.loaded.append(person_imgs)

    def load_datapoint(s, same: bool):
        # load a new set of images if it's time
        if s.gets > s.num_loaded:
            s.load_chunk()
        s.gets += 1

        # indices for the two loaded images
        p1: int
        p2: int
        i1: int
        i2: int

        if same:
            # p1 and p2 are the same random index
            p1 = random.randint(0, s.loaded.__len__() - 1)
            p2 = p1
        else:
            # p1 and p2 are two different indices
            p1 = random.randint(0, s.loaded.__len__() - 1)
            p2 = random.randint(0, s.loaded.__len__() - 2)
            if p2 >= p1:
                p2 += 1

        # i1 and i2 are either unrelated (different) or cannot be the same (same)
        i1 = random.randint(0, s.loaded[p1].__len__() - 1)
        i2 = random.randint(0, s.loaded[p2].__len__() - (2 if same else 1))
        if same:
            if i2 >= i1:
                i2 += 1

        # return the images from the loaded data at the given indices
        return s.loaded[p1][i1], s.loaded[p2][i2]


class FacesDataLoader:
    def __init__(s, dataset: FacesDataset, batch_size: int, similar_ratio: float):
        assert isinstance(dataset, FacesDataset), (
            f"dataset must be an instance of FacesDataset, received {type(dataset)}"
        )
        assert isinstance(batch_size, int) and batch_size > 0, (
            f"batch size must be an integer greater than zero, received {batch_size} of type {type(batch_size)}"
        )
        assert isinstance(similar_ratio, float) and 0 <= similar_ratio <= 1, (
            "similar ratio must be a float between 0 and 1"
        )

        # store the variables in the state for later use
        s.sr = similar_ratio
        s.ds = dataset
        s.bs = batch_size
        s.idx = 0

    def __len__(s):
        return s.ds.__len__() // s.bs

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
        for i in range(s.bs):
            # randomly determine whether the images will be the same or different people
            similar = True if random.random() < s.sr else False
            sim = 1 if similar else 0
            face1, face2 = s.ds.load_datapoint(similar)

            # add the selected data to the lists
            face1s.append(face1)
            face2s.append(face2)
            sims.append(torch.as_tensor(sim).float())

        # stack them into batch tensors
        face1s = torch.stack(face1s)
        face2s = torch.stack(face2s)
        sims = torch.stack(sims).float()

        # increment index
        s.idx += 1

        # return the data
        return face1s, face2s, sims
