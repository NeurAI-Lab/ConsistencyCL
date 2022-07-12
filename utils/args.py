# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
from models import get_all_models

def add_arguments(parser):
    parser.add_argument('--pretext_task', type=str, default='mse',
                        help='SSL training algorithm as a pretext task fo DER++')
    parser.add_argument('--barlow_on_weight', type=float, default=0.5,
                        help='weight for barlow twin on_diag')
    parser.add_argument('--barlow_off_weight', type=float, default=0.05,
                        help='weight for barlow twin off_diag')
    parser.add_argument('--dino_weight', type=float, default=1,
                        help='weight for Dino Loss')
    parser.add_argument('--byol_weight', type=float, default=1,
                        help='weight for BYOL Loss')
    parser.add_argument('--simclr_weight', type=float, default=0.05,
                        help='weight for Dino Loss')
    parser.add_argument('--align_weight', type=float, default=0.5,
                        help='multitask weight for alignment loss')
    parser.add_argument('--uni_weight', type=float, default=0.1,
                        help='multitask weight for uniformity loss')
    parser.add_argument('--mi_weight', type=float, default=1,
                        help='multitask weight for mutual information')
    parser.add_argument('--img_size', type=int, required=True,
                        help='Input image size')
    parser.add_argument('--multicrop', action='store_true',
                        help='multicrop augmentation for buffered images')
    parser.add_argument('--size_crops', nargs='+', default=[64, 32],
                        help='size crops for multicrop')

def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())

    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate.')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='Batch size.')
    parser.add_argument('--n_epochs', type=int, required=True,
                        help='The number of epochs for each task.')


def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=None,
                        help='The random seed.')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes for this run.')

    parser.add_argument('--csv_log', action='store_true',
                        help='Enable csv logging')
    parser.add_argument('--tensorboard', action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--validation', action='store_true',
                        help='Test on the validation set')


def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--buffer_size', type=int, required=True,
                        help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int, required=True,
                        help='The batch size of the memory buffer.')
