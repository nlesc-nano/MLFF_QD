import os
import logging
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from mlff_qd.utils.callbacks import StopWhenLRBelow, StopWhenGoodEnough, EarlyStoppingWithLog
import schnetpack as spk

from mlff_qd.utils.helpers import get_optimizer_class, get_scheduler_class 

def setup_task_and_trainer(config, nnpot, outputs, folder):
    optimizer_name = config['settings']['training']['optimizer']['type']
    scheduler_name = config['settings']['training']['scheduler']['type']

    optimizer_cls = get_optimizer_class(optimizer_name)
    scheduler_cls = get_scheduler_class(scheduler_name)
    
    task = spk.task.AtomisticTask(
        model=nnpot,
        outputs=outputs,
        optimizer_cls=optimizer_cls,
        optimizer_args={"lr": config['settings']['training']['optimizer']['lr']},
        scheduler_cls=scheduler_cls,
        scheduler_args={"mode": "min", "factor": config['settings']['training']['scheduler']['factor'],
                        "patience": config['settings']['training']['scheduler']['patience'],
                        "verbose": config['settings']['training']['scheduler']['verbose']},
        scheduler_monitor=config['settings']['logging']['monitor']
    )
    
    tensorboard_logger = pl.loggers.TensorBoardLogger(save_dir=folder)
    callbacks = [
        spk.train.ModelCheckpoint(
            model_path=os.path.join(folder, config['settings']['logging']['checkpoint_dir']),
            save_top_k=1,
            monitor=config['settings']['logging']['monitor']
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]

    # ① EarlyStopping ---------------------------------------------------------
    if 'early_stopping' in config['settings']['training']:
        es = config['settings']['training']['early_stopping']
        callbacks.append(
            EarlyStoppingWithLog(
                monitor = es['monitor'],
                patience = es['patience'],
                min_delta = es.get('min_delta', 0.),
                mode     = es.get('mode', 'min'),
                verbose  = True
            )
        )

    # ② LR threshold ----------------------------------------------------------
    lr_thr = config['settings']['training'].get('lr_stop_threshold')
    if lr_thr is not None:
        callbacks.append(StopWhenLRBelow(threshold=lr_thr))

    # ③ Accuracy target -------------------------------------------------------
    tgt_cfg = config['settings']['training'].get('targets', {}).get('stop_when_good_enough')
    if tgt_cfg:
        callbacks.append(
            StopWhenGoodEnough(
                monitor = tgt_cfg['monitor'],
                target  = tgt_cfg['threshold']
            )
        )

    print("CALLBACKS:", callbacks)
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=tensorboard_logger,
        default_root_dir=folder,
        max_epochs=config['settings']['training']['max_epochs'],
        accelerator=config['settings']['training']['accelerator'],
        precision=config['settings']['training']['precision'],
        devices=config['settings']['training']['devices']
    )
    
    logging.info("Task and trainer set up")
    return task, trainer
