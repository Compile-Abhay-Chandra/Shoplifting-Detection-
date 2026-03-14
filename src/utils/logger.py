"""
Utility logger: TensorBoard + console logging for STA-MIL training.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False


def setup_logger(name: str, log_dir: str, level=logging.INFO) -> logging.Logger:
    """Configure a Python logger with both console and file handlers."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger  # already configured

    fmt = logging.Formatter('[%(asctime)s] %(levelname)s — %(message)s', datefmt='%H:%M:%S')

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_dir / 'train.log')
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


class TrainingLogger:
    """Combined TensorBoard + console logger for training metrics."""

    def __init__(self, log_dir: str, experiment_name: str = 'sta_mil'):
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logger(experiment_name, str(self.log_dir))
        self.writer: Optional[SummaryWriter] = None

        if HAS_TB:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
            self.logger.info(f"TensorBoard logs: {self.log_dir}")
        else:
            self.logger.warning("TensorBoard not available. Install with: pip install tensorboard")

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = 'train'):
        """Log scalar metrics to TensorBoard and console."""
        for key, val in metrics.items():
            tag = f"{prefix}/{key}"
            if self.writer:
                self.writer.add_scalar(tag, val, global_step=step)

    def log_text(self, msg: str):
        self.logger.info(msg)

    def log_epoch(self, epoch: int, metrics: Dict[str, Any]):
        """Pretty-print epoch summary."""
        parts = [f"Epoch {epoch:03d}"]
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}: {v:.4f}")
            else:
                parts.append(f"{k}: {v}")
        self.logger.info(" | ".join(parts))

        if self.writer:
            for k, v in metrics.items():
                if isinstance(v, float):
                    self.writer.add_scalar(f"epoch/{k}", v, global_step=epoch)

    def close(self):
        if self.writer:
            self.writer.close()
