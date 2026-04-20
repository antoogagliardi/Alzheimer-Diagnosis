## Training Script
from pprint import pprint
import yaml
import os
import torch
import wandb
import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.data.utils_data import MRIDataModule
from src.model.base_3d import UNetAlzheimer3D as BaseUnet
from src.model.advance_3d import UNetAlzheimer3D as AdvanceUnet
from src.model.advance_att_3d import UNetAlzheimer3D as AttAdvanceUnet


# Read configuration file
with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)
print("== Configuration File ==")
pprint(cfg)

# Setup device
device = torch.device(cfg["project"]["device"])
print(f"- Device used: {device}")

# Load training data
load_existing_data = True
BATCH_SIZE = 1              # Hardware limit. It will be set a gradient accumulation procedure during the training
DROP_LAST_BATCH = False

root_path = os.getcwd()                                         ; print("- root path: ", root_path)
root_data = os.path.join(root_path, cfg["paths"]["data"])       ; print("- root data path: ", root_data)
root_db = os.path.join(root_path, cfg["paths"]["data_split"])   ; print("- root db path: ", root_db)
if load_existing_data == True:
    data_manager = MRIDataModule(dataset=None, test_set=None,
                                 saved_db_folder=os.path.join(root_db, f"complete"),
                                 batch_size=BATCH_SIZE,
                                 reload_data=True,
                                 drop_last=DROP_LAST_BATCH)
else:
    data_manager = MRIDataModule(dataset=torch.load(os.path.join(root_data, "complete.db"), weights_only=False),
                                 test_set=[], # torch.load(os.path.join(root_path, "screening.db"), weights_only=False),
                                 saved_db_folder=os.path.join(root_db, f"complete"),
                                 batch_size=BATCH_SIZE,
                                 reload_data=False,
                                 drop_last=DROP_LAST_BATCH)
    

# Prepare the training
config = cfg["training"]

ckpt_path = os.path.join(os.getcwd(), cfg["training"]["ckpt_path"])
print("ckpt path: ", ckpt_path)
model_type = cfg["training"]["model_type"]

PROJECT_NAME = cfg["training"]["wandb_proj"]            # "Alzheimer_Project"
RUN_NAME = cfg["training"]["wandb_run"]                 # "Alzheimer_U-NET_27"
# LAST_RUN_ID is discovered once the project is created

EPOCHS = cfg["training"]["epochs"]                      # 6
ACCUMULATE_BATCH = cfg["training"]["grad_accumulation"] # 64


# Model Instantiation
if cfg["training"]["model_type"] == "base":
    print("Instantiating Unet Base")
    model = BaseUnet(class_weights={"train_weights": data_manager.training_class_weigths})
if cfg["training"]["model_type"] == "advance":
    print("Instantiating Unet Advance")
    model = AdvanceUnet(class_weights={"train_weights": data_manager.training_class_weigths})
if cfg["training"]["model_type"] == "more_advance":
    print("Instantiating Unet More Advance")
    model = AttAdvanceUnet(class_weights={"train_weights": data_manager.training_class_weigths})

if device.type == "mps":    # Note: MPS Backend doesn't support torch.DoubleTensor(=float64)
    model = model.to(device=device, dtype=torch.float32)
else:
    model = model.to(device=device)
pprint(model)


# 
RESUME_TRAIN = cfg["training"]["resume"]
if RESUME_TRAIN == True:
    LAST_RUN_ID = cfg["training"]["wandb_runID"]            # LAST_RUN_ID is discovered once the project is created
    LAST_EPOCH = cfg["training"]["last_epoch"]
    ADD_EPOCHS = cfg["training"]["add_epoch"]
    run = wandb.init(project=PROJECT_NAME, name=RUN_NAME,
                     config=config,
                     resume=True, id=LAST_RUN_ID)
    ckpt = os.path.join(ckpt_path, f"{model_type}/{LAST_RUN_ID}/model-epoch={LAST_EPOCH-1}.ckpt")
    EPOCHS = LAST_EPOCH + ADD_EPOCHS
else:
    run = wandb.init(project=PROJECT_NAME,
                     name=RUN_NAME,config=config)



wandb_logger = WandbLogger(name=RUN_NAME,
                           save_dir=ckpt_path,
                           offline=False,
                           project=PROJECT_NAME, log_model=False) # log_model = "all"/True/False


callback_list = [ModelCheckpoint(dirpath=f"{ckpt_path}/{model_type}/{wandb.run.id}",
                                          filename='model-{epoch}',
                                          save_top_k=-1,
                                          every_n_epochs=1),
                EarlyStopping(monitor="valid_f1", min_delta=0.001, patience=5, mode="max"),
                EarlyStopping(monitor="valid_loss", min_delta=0.001, patience=5, mode="min")]

trainer = pl.Trainer(accelerator=device.type,       # gpu, cpu, mps
                     num_sanity_val_steps=1,
                     limit_train_batches=2,
                     limit_val_batches=5,
                     accumulate_grad_batches=ACCUMULATE_BATCH,
                     logger= wandb_logger,
                     devices=1, max_epochs=EPOCHS,
                     callbacks = callback_list,
                     log_every_n_steps=1)
trainer.wandb_id    = wandb.run.id
trainer.model_type  = model_type
trainer.device      = device
trainer.batch_size  = BATCH_SIZE
# trainer.learning_rate = LEARNING_RATE
# trainer.gradient_accumulation = GRADIENT_ACCUMULATION


training_dataloader = data_manager.train_dataloader()
validation_dataloader = data_manager.val_dataloader()

# Start the training procedure
if RESUME_TRAIN == True:
    trainer.fit(model,
                train_dataloaders=training_dataloader, val_dataloaders=validation_dataloader,
                ckpt_path=ckpt)
else:
    trainer.fit(model,
                train_dataloaders=training_dataloader, val_dataloaders=validation_dataloader)
print("Training Complete")
wandb.finish()
print("End of the training program")