import lightning.pytorch as pl
from utils.train_utils import *
from utils.data_utils import get_dataloaders


def train(configs):
    pl.seed_everything(configs["General"]["seed"])
    logger = get_train_logger(configs)
    print(f"starting experiment: {logger.log_dir}")

    callbacks = get_callbacks(configs)
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=configs["Train"]["epochs"],
        max_steps=configs["Train"]["max_steps"],
        accelerator="gpu",
        devices=configs["General"]["num_devices"],
        limit_train_batches=configs["Train"]["train_dataset_limit"],
        limit_val_batches=configs["Train"]["val_dataset_limit"],
        callbacks=callbacks,
        fast_dev_run=configs["General"]["fast_dev_run"],
        log_every_n_steps=configs["General"]["log_every_n_steps"],
        val_check_interval=configs["General"]["val_check_interval"],
        check_val_every_n_epoch=configs["General"]["check_val_every_n_epoch"],
        accumulate_grad_batches=configs["Train"]["accumulate_grad_batches"],
        num_sanity_val_steps=configs["General"]["num_sanity_val_steps"],
        profiler="simple" if configs["General"]["profiling"] else None,
        inference_mode=configs["General"]["inference_mode"],
    )

    model = get_model(configs, is_test=False)
    train_dataloader, val_dataloader, _ = get_dataloaders(
        configs["Model"], configs["Data"], configs["Train"]
    )
    # key = input("Press Enter to continue or 'q' to quit: ")
    # if key == "q":
    #     return
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    return callbacks[2].state_dict()


if __name__ == "__main__":
    from configs import configs

    if (
        isinstance(configs["Model"]["expert_manipulation"], list)
        and configs["Model"]["expert_task"] == "manipulation"
    ):
        manipulations = configs["Model"]["expert_manipulation"]
        for manipulation in manipulations:
            configs["Model"]["expert_manipulation"] = manipulation
            print(f"Running experiment for manipulation: {manipulation}")
            train(configs)
    else:
        train(configs)
