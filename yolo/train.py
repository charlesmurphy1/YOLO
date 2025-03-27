import os

import yaml
from lightning import Trainer
from omegaconf import DictConfig

from yolo.tools.solver import TrainModel
from yolo.utils.logging_utils import setup


def main():
    with open(os.path.join(os.path.dirname(__file__), "main_config.yaml"), "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = DictConfig(cfg)
    callbacks, loggers, save_path = setup(cfg)

    trainer = Trainer(
        accelerator="auto",
        max_epochs=getattr(cfg.task, "epoch", None),
        precision="16-mixed",
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=1,
        gradient_clip_val=10,
        gradient_clip_algorithm="value",
        deterministic=True,
        enable_progress_bar=not getattr(cfg, "quite", False),
        default_root_dir=save_path,
        accumulate_grad_batches=4,
    )
    model = TrainModel(cfg)
    trainer.fit(model)


if __name__ == "__main__":
    main()
