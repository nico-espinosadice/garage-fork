# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import collections
import csv
import pathlib
import uuid
from datetime import datetime
from typing import Counter, Dict, List, Mapping, Tuple, Union

import termcolor
import torch
import wandb
from wandb.integration.sb3 import WandbCallback
from omegaconf import OmegaConf

LogFormatType = List[Tuple[str, str, str]]
LogTypes = Union[int, float, torch.Tensor]


class AverageMeter(object):
    def __init__(self):
        self._sum = 0.0
        self._count = 0

    def update(self, value: float, n: int = 1):
        self._sum += value
        self._count += n

    def value(self) -> float:
        return self._sum / max(1, self._count)


class MetersGroup(object):
    def __init__(self, file_name: Union[str, pathlib.Path], formatting: LogFormatType):
        self._csv_file_path = self._prepare_file(file_name, ".csv")
        self._formatting = formatting
        self._meters: Dict[str, AverageMeter] = collections.defaultdict(AverageMeter)
        self._csv_file = open(self._csv_file_path, "w")
        self._csv_writer = None

    @staticmethod
    def _prepare_file(prefix: Union[str, pathlib.Path], suffix: str) -> pathlib.Path:
        file_path = pathlib.Path(prefix).with_suffix(suffix)
        if file_path.exists():
            file_path.unlink()
        return file_path

    def log(self, key: str, value: float):
        self._meters[key].update(value)

    def _dump_to_csv(self, data):
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=sorted(data.keys()), restval=0.0
            )
            self._csv_writer.writeheader()
        self._csv_writer.writerow(data)
        self._csv_file.flush()

    @staticmethod
    def _format(key: str, value: float, format_type: str):
        if format_type == "int":
            value = int(value)
            return f"{key}: {value}"
        elif format_type == "float":
            return f"{key}: {value:.06f}"
        elif format_type == "time":
            return f"{key}: {value:04.1f} s"
        else:
            raise ValueError(f"Invalid format type: {format_type}")

    def _dump_to_console(self, data, prefix: str, color: str = "yellow"):
        prefix = termcolor.colored(prefix, color)
        pieces = [f"| {prefix: <14}"]
        for key, disp_key, ty in self._formatting:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        print(" | ".join(pieces))

    def dump(self, step: int, prefix: str, save: bool = True, color: str = "yellow"):
        if len(self._meters) == 0:
            return
        if save:
            data = {key: meter.value() for key, meter in self._meters.items()}
            data["step"] = step
            self._dump_to_csv(data)
            self._dump_to_console(data, prefix, color)
        self._meters.clear()


class Logger(object):
    """Light-weight csv logger.

    This logger is based on pytorch_sac's
    `logger <https://github.com/denisyarats/pytorch_sac/blob/master/logger.py>`_
    with some modifications and some of its features removed.

    To use this logger you must register logging groups using :meth:`register_group`. Each
    group will save data to a separate csv file, at `log_dir/<group_name>.csv`, and will
    output to console using its own dedicated tabular format.

    Args:
        log_dir (str or pathlib.Path): the directory where to save the logs.
        cfg (dict): the configuration dictionary. It will be saved to the wandb run if provided.
    """

    def __init__(
        self,
        log_dir: Union[str, pathlib.Path],
        cfg: Dict,
    ):
        self._log_dir = pathlib.Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._groups: Dict[str, Tuple[MetersGroup, int, str]] = {}
        self._group_steps: Counter[str] = collections.Counter()
        self.wandb_log = cfg.wandb_log
        if cfg.wandb_log:
            if hasattr(self, 'run'):
                self.run.finish()
            name = f"{cfg.algorithm.name}|{cfg.overrides.env}|{cfg.seed}"
            self.run = wandb.init(
                entity="irl_cpi",
                project=cfg.wandb_project,
                group=f"{cfg.algorithm.name}_{cfg.overrides.env}",
                name=name,
                id=f"{str(uuid.uuid4())[-5:]}",
                config=OmegaConf.to_container(cfg),
                sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                monitor_gym=True,  # auto-upload the videos of agents playing the game
                save_code=True,  # optional
            )
            self.wandb_callback = WandbCallback(
                verbose=2,
            )

    def register_group(
        self,
        group_name: str,
        log_format: LogFormatType,
        dump_frequency: int = 1,
        color: str = "yellow",
    ):
        """Register a logging group.

        Args:
            group_name (str): the name assigned to the logging group.
            log_format (list of 3-tuples): each tuple contains 3 strings, representing
                (variable_name, shortcut, type), for a variable that the logger should keep
                track of in this group. The variable name will be used as a header in the csv file
                for the entries of this variable. The shortcut will be used as a header for
                the console output tabular format. The type should be one of
                "int", "float", "time".
            dump_frequency (int): how often (measured in calls to :meth:`log_data`)
                should the logger dump the data collected since the last call. If
                ``dump_frequency > 1``, then the data collected between calls is averaged.
            color (str): a color to use for this group in the console.

        """
        if group_name in self._groups:
            print(f"Group {group_name} has already been registered.")
            return
        new_group = MetersGroup(self._log_dir / group_name, formatting=log_format)
        self._groups[group_name] = (new_group, dump_frequency, color)
        self._group_steps[group_name] = 0

    def log_histogram(self, *_args):
        pass

    def log_param(self, *_args):
        pass

    def log_data(self, group_name: str, data: Mapping[str, LogTypes], wandb_only=False):
        """Logs the data contained in a given dictionary to the given logging group.

        Args:
            group_name (str): the name of the logging group to use. It must have been registered
                already, otherwise an exception will be thrown.
            data (mapping str->(int/float/torch.Tensor)): the dictionary with the data. Each
                keyword must be a variable name in the log format passed when creating this group.
        """
        if not wandb_only:
            if group_name not in self._groups:
                raise ValueError(f"Group {group_name} has not been registered.")
            meter_group, dump_frequency, color = self._groups[group_name]
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()  # type: ignore
                meter_group.log(key, value)
        
        wandb_metrics = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                value = value.item()  # type: ignore
            wandb_metrics[key] = value
        if self.wandb_log:
            self.run.log(wandb_metrics)

        if not wandb_only:
            self._group_steps[group_name] += 1
            if self._group_steps[group_name] % dump_frequency == 0:
                self._dump(group_name)

    def _dump(self, group_name: str, save: bool = True):
        if group_name not in self._groups:
            raise ValueError(f"Group {group_name} has not been registered.")
        meter_group, dump_frequency, color = self._groups[group_name]
        meter_group.dump(self._group_steps[group_name], group_name, save, color=color)

    def end(self):
        if self.wandb_log:
            self.run.finish()