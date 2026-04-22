import logging
import os
from contextlib import nullcontext
from typing import Optional, Dict

import datasets
import numpy as np
import torch
import torch.distributed as dist
from rich.progress import track
from torch import nn
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from seagull.data_processing.sequence_sampler import SequenceSamplingDataset
from seagull.model.heads.seagull_lm import SeagullLM
from seagull.utils.metrics import compute_loss, compute_perplexity_from_entropy
from seagull.utils.torch_utils import set_pytorch_backends, remove_compiled_model_prefix_from_model_state_dict
from seagull.utils.tracker import Tracker

set_pytorch_backends()


class Trainer(object):
    def __init__(
        self,
        model: SeagullLM,
        optimizer: Optimizer,
        train_data: datasets.Dataset,
        val_data: Optional[datasets.Dataset] = None,
        seq_start_pos: Optional[int] = None,
        labels_ignore_idx: int = -100,
        lr_scheduler: Optional[LRScheduler] = None,
        use_amp: bool = True,
        grad_clip_max_norm: Optional[float] = 1.0,
        class_weights: Optional[torch.Tensor] = None,
        tracker: Optional[Tracker] = None,
        detect_anomaly: bool = False,
        device: torch.device = torch.device("cpu"),
        compile_model: bool = False,
        use_ddp: bool = False,
    ):
        """
        Creates a :py:class:`~seagull.trainers.trainer.Trainer` to train a seagull language model.

        Parameters
        ----------
        model : SeagullLM
            The language model to train: :py:class:`~seagull.model.head.seagull_lm.SeagullLM`.
        optimizer : torch.optim.Optimizer
            The optimizer to be used in training.
        train_data : datasets.Dataset
            The training data to train the model.
        val_data : Optional[Dataset]
            The validation data to evaluate the model.
        seq_start_pos : Optional[int]
            An optional integer specifying the index in the dataset where the sequence begins.
        labels_ignore_idx : int
            Specifies an index to ignore during training such that the gradient computation is unaffected by this value.
        lr_scheduler : Optional[LRScheduler]
            Optional LRScheduler to manipulate the learning rate during training.
        use_amp : bool
            Enables PyTorch automatic mixed precision.
        grad_clip_max_norm : Optional[float]
            The maximum norm to use in gradient clipping.
        tracker : Optional[Tracker]
            Optional Tracker to track model training.
        detect_anomaly : bool
            Enables anomaly detection for autograd.
        device : torch.device
            The device (e.g., cuda) to be used for training and validation.
        compile_model : bool
            Compiles trained model into an optimized form.
        use_ddp : bool
            Enables PyTorch distributed data parallel.
        """
        super().__init__()

        self.use_ddp = use_ddp and device.type == "cuda"
        self._setup(use_ddp=self.use_ddp, device=device)

        self.detect_anomaly = detect_anomaly
        if self.detect_anomaly:
            logging.warning(f"using {detect_anomaly=}; will result in significant speed reductions")

        self.use_amp = use_amp
        if device.type == "mps":
            raise ValueError(f"device {device.type} not supported for kernel fusion; please use cuda or cpu")
        if device.type != "cuda":
            logging.warning(f"ignoring {use_amp=}: device {device.type} not supported")
            self.use_amp = False
        # self.amp_dtype = torch.bfloat16 if (self.use_amp and torch.cuda.is_bf16_supported()) else torch.float16
        self.amp_dtype = torch.float16
        self.grad_scaler = GradScaler(enabled=self.use_amp)

        self.train_data = SequenceSamplingDataset(
            train_data, model_max_positions=model._max_positions, seq_start_pos=seq_start_pos
        )
        self.val_data = (
            SequenceSamplingDataset(val_data, model_max_positions=model._max_positions, seq_start_pos=seq_start_pos)
            if val_data is not None
            else None
        )

        self.optimizer = optimizer
        self.grad_clip_max_norm = grad_clip_max_norm
        self.lr_scheduler = lr_scheduler
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=labels_ignore_idx,
            weight=(class_weights.to(self.device) if class_weights is not None else None),
        )

        self.model = model.to(device=self.device)
        self.compile_model = compile_model and self.device.type == "cuda"  # kernel fusion
        self._model_compiled = False  # for loading from checkpoint
        # Compile, then DDP: https://discuss.pytorch.org/t/combining-torch-compile-and-distributeddataparallel/184591/2.
        self.model = self._get_compiled_model() if self.compile_model else self.model
        if self.use_ddp:
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)

        self.tracker = tracker if self.is_master_process else None
        if self.tracker is not None:
            self.tracker.setup()
        self._epoch = 0
        self._step = 0

    def _setup(self, use_ddp: bool, device: torch.device):
        # DDP training: https://pytorch.org/docs/stable/notes/ddp.html.
        if use_ddp:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.global_rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.device = torch.device(f"cuda:{self.local_rank}")
            self.is_master_process = self.local_rank == 0
        else:
            self.device = device
            self.is_master_process = True

    def _get_compiled_model(self):
        logging.info("compiling the model ... (takes about a minute)")
        self._model_compiled = True
        return torch.compile(self.model)

    def save_checkpoint(self, checkpoint_path: str) -> None:
        """
        Saves the training step, epoch, model, and optimizer, grad scaler, and LR scheduler states to the given path.

        Parameters
        ----------
        checkpoint_path : str
            Path to save the checkpoint to.
        """
        torch.save(
            {
                "step": self._step,
                "epoch": self._epoch,
                "model_state_dict": self.model.module.state_dict() if self.use_ddp else self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "grad_scaler_state_dict": self.grad_scaler.state_dict(),
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            },
            checkpoint_path,
        )

    def from_checkpoint(self, checkpoint_path: str) -> None:
        """
        Loads the training step, epoch, model, and optimizer, grad scaler, and LR scheduler states from the given path.

        Parameters
        ----------
        checkpoint_path : str
            Path to load the checkpoint from.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        model_state_dict = checkpoint["model_state_dict"]
        if not self._model_compiled:
            model_state_dict = remove_compiled_model_prefix_from_model_state_dict(model_state_dict)
        if self.use_ddp:
            self.model.module.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(model_state_dict)

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        self._step = checkpoint["step"] + 1
        self._epoch = checkpoint["epoch"] + 1

    def _get_dataloader(
        self,
        dataset: datasets.Dataset,
        batch_size: int,
        is_training_dataset: bool,
        num_workers: int = 0,
    ):
        sampler = DistributedSampler(dataset) if self.use_ddp else None
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None and is_training_dataset),
            pin_memory=True,
            num_workers=num_workers,
            sampler=sampler,
        )

    def _reduce_on_master(self, tensor: torch.Tensor) -> torch.Tensor:
        # See: https://discuss.pytorch.org/t/way-to-aggregate-loss-in-ddp-training/176929/4.
        tensor.detach_()
        if self.use_ddp:
            if self.world_size > 1:
                dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
                if self.is_master_process:
                    tensor = tensor / self.world_size
        return tensor

    def _train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Trains the model for one epoch on the given dataloader.

        Parameters
        ----------
        dataloader : DataLoader
            The (train) Dataloader used in training the model.

        Returns
        -------
        Dict[str, float]
            A dictionary of metrics, includes loss and perplexity.
        """
        metrics = {"loss": [], "perplexity": []}

        self.model.train()
        for batch in track(dataloader, description=f"train (epoch: {self._epoch})"):
            # Using amp: https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html.
            with torch.autocast(
                device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
            ) if self.use_amp else nullcontext():
                # logits: (batch_size, max_length, vocab_size)
                logits, _ = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    padding_mask=batch["padding_mask"].to(self.device),
                    use_kv_cache=False,
                    return_output_at_all_layers=False,
                    return_attentions=False,
                )
                loss = compute_loss(loss_fn=self.loss_fn, preds=logits, labels=batch["labels"].to(self.device))

            self.optimizer.zero_grad(set_to_none=True)
            self.grad_scaler.scale(loss).backward()  # scale loss
            loss = self._reduce_on_master(loss).item()  # immediately drop buffers
            self.grad_scaler.unscale_(self.optimizer)  # unscale gradients for clipping
            if self.grad_clip_max_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_max_norm)
            self.grad_scaler.step(self.optimizer)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.grad_scaler.update()

            if self.is_master_process:
                metrics["loss"].append(loss)
                metrics["perplexity"].append(compute_perplexity_from_entropy(entropy=loss))
                if self.tracker is not None:
                    self.tracker.log_metrics(
                        epoch_or_step="step",
                        epoch_or_step_num=self._step,
                        split="train",
                        metrics={"loss": metrics["loss"][-1], "perplexity": metrics["perplexity"][-1]},
                        log_to_console=False,
                    )
            self._step = self._step + 1

        if not self.is_master_process:
            return {}
        average_metrics = {metric: float(np.average(score)) for metric, score in metrics.items()}
        return average_metrics

    @torch.no_grad()
    def _eval_epoch(self, dataloader) -> Dict[str, float]:
        """
        Evaluates the model for one epoch on the given dataloader.

        Parameters
        ----------
        dataloader : DataLoader
            The (validation) Dataloader used in evaluating the trained model.

        Returns
        -------
        Dict[str, float]
            A dictionary of metrics, includes loss and perplexity.
        """
        metrics = {"loss": [], "perplexity": []}

        self.model.eval()
        for batch in track(dataloader, description=f"val (epoch: {self._epoch})"):
            with torch.autocast(
                device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
            ) if self.use_amp else nullcontext():
                # logits: (batch_size, max_length, vocab_size)
                logits, _ = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    padding_mask=batch["padding_mask"].to(self.device),
                    use_kv_cache=False,
                    return_output_at_all_layers=False,
                    return_attentions=False,
                )
                loss = self._reduce_on_master(
                    compute_loss(loss_fn=self.loss_fn, preds=logits, labels=batch["labels"].to(self.device))
                ).item()
            if self.is_master_process:
                metrics["loss"].append(loss)
                metrics["perplexity"].append(compute_perplexity_from_entropy(entropy=loss))

        if not self.is_master_process:
            return {}
        average_metrics = {metric: float(np.average(score)) for metric, score in metrics.items()}
        return average_metrics

    def train_and_eval(
        self, batch_size: int = 256, num_epochs: int = 8, checkpoint_every: int = 1, num_workers: int = 0
    ):
        train_dataloader = self._get_dataloader(
            self.train_data,
            is_training_dataset=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        val_dataloader = (
            self._get_dataloader(
                self.val_data,
                is_training_dataset=False,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            if self.val_data is not None
            else None
        )

        for epoch in range(num_epochs):
            if self.use_ddp:
                train_dataloader.sampler.set_epoch(self._epoch)
                val_dataloader.sampler.set_epoch(self._epoch)

            with torch.autograd.set_detect_anomaly(self.detect_anomaly, check_nan=True):
                train_metrics = self._train_epoch(train_dataloader)
                val_metrics = self._eval_epoch(val_dataloader) if val_dataloader is not None else None
            if self.is_master_process and self.tracker is not None:
                self.tracker.log_metrics(
                    epoch_or_step="epoch",
                    epoch_or_step_num=self._epoch,
                    split="train",
                    metrics=train_metrics,
                    log_to_console=True,
                )
                if val_metrics is not None:
                    self.tracker.log_metrics(
                        epoch_or_step="epoch",
                        epoch_or_step_num=self._epoch,
                        split="val",
                        metrics=val_metrics,
                        log_to_console=True,
                    )
                if (epoch + 1) % checkpoint_every == 0:
                    self.tracker.save_checkpoint(self, epoch=self._epoch)
            self._epoch = self._epoch + 1

        if self.is_master_process and self.tracker is not None:
            self.tracker.save_model(self.model.module if self.use_ddp else self.model)
