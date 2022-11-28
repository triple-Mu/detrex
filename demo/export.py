# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import sys

import torch
import torch.nn as nn

sys.path.insert(0, "./")  # noqa
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.utils.logger import setup_logger
from detectron2.structures.image_list import ImageList

# constants
WINDOW_NAME = "Model Export"


def setup(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="detrex model export")
    parser.add_argument(
        "--config-file",
        default="projects/dino/configs/dino_r50_4scale_12ep.py",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--img-size",
        nargs="+",
        default=[640, 640],
        type=int,
        help="Input image shape for height and width",
    )
    parser.add_argument(
        "--work-dir",
        default='./work_dir',
        help="Output dir for save onnx model",
    )
    parser.add_argument(
        "--min_size_test",
        type=int,
        default=800,
        help="Size of the smallest side of the image during testing. Set to zero to disable resize in testing.",
    )
    parser.add_argument(
        "--max_size_test",
        type=float,
        default=1333,
        help="Maximum size of the side of the image during testing.",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def preprocess_image(self, batched_inputs):
    images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
    images = ImageList.from_tensors(images)
    return images




if __name__ == "__main__":
    args = get_parser().parse_args()
    args.img_size *= 2 if len(args.img_size) == 1 else 1

    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup(args)

    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.train.init_checkpoint)

    model.eval()


    height, width = args.img_size
    fake_image = torch.zeros((3, height, width), dtype=torch.float32).to(cfg.train.device)

    inputs = {"image": fake_image, "height": height, "width": width}
    predictions = model([inputs])[0]

    torch.onnx.export(
        model, fake_image, 'dino.onnx', opset_version=16
    )
