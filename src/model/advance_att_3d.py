import os
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torchmetrics.classification import MulticlassF1Score, MulticlassConfusionMatrix

from src.utils.plots import plot_confusion_matrix

cm_path = os.path.join(os.getcwd(), "confusion_matrices")

class AttentionBlock(pl.LightningModule):
    def __init__(self, dim):
        super(AttentionBlock, self).__init__()
        # Features from down-block convolutions
        self.gate_signal_conv = nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=(1,1,1), stride=1)
        # Features from skip-connections
        self.input_features_conv = nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=(1,1,1), stride=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # 3D Attention map generator
        self.GRID_SAMPLER_SCALE_FACTOR = 1.0
        self.attention_conv = nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=(1,1,1), stride=1)

    def forward(self, gate_signal, skip_features):
        # Features from skip connections
        skip_feat = self.input_features_conv(skip_features)     # ; print("skip features shape: ", skip_feat.shape)
        # Features from up-sampling blocks
        gate_feat = self.gate_signal_conv(gate_signal)          # ; print("gate features shape: ", gate_feat.shape)

        res = torch.add(gate_feat, skip_feat)
        res = self.relu(res)

        res = self.attention_conv(res)                          # ; print("attention features shape: ", res.shape)
        res = self.sigmoid(res)
        res = F.interpolate(res, mode="trilinear", scale_factor=self.GRID_SAMPLER_SCALE_FACTOR) # Grid Resampler

        # Compute the 3D Attention map
        attention_map = torch.mul(res, skip_feat)               # ; print("attention map shape: ", attention_map.shape)

        return attention_map

class DownSampling3D(pl.LightningModule):
    def __init__(self, in_channel:int, out_channels:int, kernel_size:tuple):
        super(DownSampling3D, self).__init__()
        self.down_block = nn.Conv3d(in_channels=in_channel, out_channels=out_channels, kernel_size=kernel_size)
    def forward(self, x):
        return self.down_block(x)
class UpSampling3D(pl.LightningModule):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:tuple):
        super(UpSampling3D, self).__init__()
        self.up_block = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
    def forward(self, x):
        return self.up_block(x)
    
class UNetAlzheimer3D(pl.LightningModule):
    def __init__(self, class_weights:dict):
        super(UNetAlzheimer3D, self).__init__()
        self.sex_classes = {0: "M", 1: "F"}
        self.max_available_age = 100
        self.disease_classes = {0: "CN", 1: "MCI", 2: "AD"}

        # Age and sex embedding layers
        self.age_embedding = nn.Sequential(nn.Embedding(num_embeddings=self.max_available_age, embedding_dim=64),
                                           nn.Linear(in_features=64, out_features=64))
        self.sex_embedding = nn.Sequential(nn.Embedding(num_embeddings=len(self.sex_classes), embedding_dim=32),
                                           nn.Linear(in_features=32, out_features=32))
        
        # U-Net convolution (Image features extraction)
        # Encoder
        self.l_conv1 = nn.Sequential(nn.Conv3d(in_channels=1, out_channels=4, kernel_size=(3,3,3)),
                                     nn.ReLU(),
                                     nn.BatchNorm3d(num_features=4, eps=1e-5, momentum=0.1, affine=True),
                                     DownSampling3D(in_channel=4, out_channels=4, kernel_size=(2,2,2)))
        
        self.l_conv2 = nn.Sequential(nn.Conv3d(in_channels=4, out_channels=16, kernel_size=(3,3,3)),
                                     nn.ReLU(),
                                     nn.BatchNorm3d(num_features=16, eps=1e-5, momentum=0.1, affine=True),
                                     DownSampling3D(in_channel=16, out_channels=16, kernel_size=(2,2,2))) # Skip1
        
        self.l_conv3 = nn.Sequential(nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3,3,3)),
                                     nn.ReLU(),
                                     nn.BatchNorm3d(num_features=32, eps=1e-5, momentum=0.1, affine=True),
                                     DownSampling3D(in_channel=32, out_channels=32, kernel_size=(2,2,2))) # Skip2
        
        self.l_conv4 = nn.Sequential(nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3,3,3)),
                                     nn.ReLU(),
                                     nn.BatchNorm3d(num_features=64, eps=1e-5, momentum=0.1, affine=True)) # Skip3  
        
        # Bottom Convolution
        self.conv5 = nn.Sequential(nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3,3,3)),
                                   nn.ReLU(),
                                   nn.BatchNorm3d(num_features=64, eps=1e-5, momentum=0.1, affine=True),
                                   nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3,3,3)),
                                   nn.ReLU(),
                                   nn.BatchNorm3d(num_features=64, eps=1e-5, momentum=0.1, affine=True))
        
        # Decoder
        self.r_conv4 = nn.Sequential(nn.Conv3d(in_channels=2*64, out_channels=32, kernel_size=(3,3,3)),
                                     nn.ReLU(),
                                     nn.BatchNorm3d(num_features=32, eps=1e-5, momentum=0.1, affine=True),
                                     UpSampling3D(in_channels=32, out_channels=32, kernel_size=(2,2,2))) # Sup3
        
        self.r_conv3 = nn.Sequential(nn.Conv3d(in_channels=2*32, out_channels=16, kernel_size=(3,3,3)),
                                     nn.ReLU(),
                                     nn.BatchNorm3d(num_features=16, eps=1e-5, momentum=0.1, affine=True),
                                     UpSampling3D(in_channels=16, out_channels=16, kernel_size=(2,2,2))) # Sup2
        
        self.r_conv2 = nn.Sequential(nn.Conv3d(in_channels=2*16, out_channels=8, kernel_size=(3,3,3)),
                                     nn.ReLU(),
                                     nn.BatchNorm3d(num_features=8, eps=1e-5, momentum=0.1, affine=True)) # Sup1
        
        # Attention Blocks
        self.attention3 = AttentionBlock(dim=64)
        self.attention2 = AttentionBlock(dim=32)
        self.attention1 = AttentionBlock(dim=16)
        
        # Dropout
        self.dropout = nn.Dropout(p=0.5)
        # Classification layer
        self.disease_detector = nn.Sequential(nn.Conv3d(in_channels=152, out_channels=30, kernel_size=(7,9,7), stride=1, padding=0),
                                              nn.LeakyReLU(negative_slope=0.01),
                                              nn.Conv3d(in_channels=30, out_channels=3, kernel_size=1, stride=1, padding=0))

        # Metrics Definition -----------------------------------------------------------------------------
        self.training_class_weights = class_weights["train_weights"]
        self.loss_function = nn.CrossEntropyLoss(weight=self.training_class_weights.float(),
                                                 reduce=None, reduction='mean', label_smoothing=0.0)
        self.disease_accuracy = MulticlassF1Score(num_classes=len(self.disease_classes),
                                                   top_k=1,
                                                   average='micro',
                                                   multidim_average='global',
                                                   ignore_index=None, validate_args=False)
        self.train_disease_confusion_matrix = MulticlassConfusionMatrix(num_classes=len(self.disease_classes),
                                                                        ignore_index=None, validate_args=True, normalize="all")
        self.valid_disease_confusion_matrix = MulticlassConfusionMatrix(num_classes=len(self.disease_classes),
                                                                        ignore_index=None, validate_args=True, normalize="all")
        # ------------------------------------------------------------------------------------------------
    
    def feature_crop(self, x, shape=None):
        '''
        Function for cropping an image tensor: Given an image tensor and the new shape,
        crops to the center pixels (assumes that the input's size and the new size are
        even numbers).
        Parameters:
            image: image tensor of shape (batch size, channels, height, width)
            new_shape: a torch.Size object with the shape you want x to have
        '''
        _, _, h, w, d = x.shape
        _, _, h_new, w_new, d_new = shape
        
        ch, cw, cd = h//2, w//2, d//2
        ch_new, cw_new, cd_new = h_new//2, w_new//2, d_new//2
        
        x1 = int(cw - cw_new)
        y1 = int(ch - ch_new)
        z1 = int(cd - cd_new)
        x2 = int(x1 + w_new)
        y2 = int(y1 + h_new)
        z2 = int(z1 + d_new)
        
        return x[:, :, y1:y2, x1:x2, z1:z2]
    
    def forward(self, x, sex, age):
        # Encoder part
        features = self.l_conv1(x)                                              # ; print("Conv1: ", features.shape)
        skip_1 = self.l_conv2(features)                                         # ; print("Skip1: ", skip_1.shape) # skip1
        skip_2 = self.l_conv3(skip_1)                                           # ; print("Skip2: ", skip_2.shape) # skip2
        skip_3 = self.l_conv4(skip_2)                                           # ; print("Skip3: ", skip_3.shape) # skip3

        # Bottom convolution
        features = self.conv5(skip_3)                                           # ; print("Bottom Convolution: ", features.shape)
        
        # Skip connection features computation and Decoder part
        skip_3 = self.feature_crop(skip_3, shape=features.shape)                # ; print(skip_3.shape)
        att_skip3 = self.attention3(gate_signal=features, skip_features=skip_3) # ; print("attention1 result shape: ", att_skip3.shape)
        sup_3 = self.r_conv4(torch.cat([features, att_skip3], dim=1))           # ; print("Sup3: ", sup_3.shape) # sup_3

        skip_2 = self.feature_crop(skip_2, shape=sup_3.shape)                   # ; print(skip_2.shape)
        att_skip2 = self.attention2(gate_signal=sup_3, skip_features=skip_2)    # ; print("attention2 result shape: ", att_skip2.shape)
        sup_2 = self.r_conv3(torch.cat([sup_3, att_skip2], dim=1))              # ; print("Sup2: ", sup_2.shape) # sup_2

        skip_1 = self.feature_crop(skip_1, shape=sup_2.shape)                   # ; print(skip_1.shape)
        att_skip1 = self.attention1(gate_signal=sup_2, skip_features=skip_1)    # ; pprint(att_skip1); print("attention3 result shape: ", att_skip1.shape)
        sup_1 = self.r_conv2(torch.cat([sup_2, att_skip1], dim=1))              # ; print("Sup1: ", sup_1.shape) # sup_1

        features_sup_3 = sup_3.clone()
        features_sup_2 = sup_2.clone()
        features_sup_1 = sup_1.clone()
        
        # Make Age and Sex computations
        age = self.age_embedding(age);                  sex = self.sex_embedding(sex.squeeze(0))
        age = age.unsqueeze(dim=-1);                    sex = sex.unsqueeze(dim=-1)
        age = torch.repeat_interleave(age, 7, dim=-1);  sex = torch.repeat_interleave(sex, 7, dim=-1)
        age = age.unsqueeze(dim=-1);                    sex = sex.unsqueeze(dim=-1)
        age = torch.repeat_interleave(age, 9, dim=-1);  sex = torch.repeat_interleave(sex, 9, dim=-1)
        age= age.unsqueeze(dim=-1);                     sex= sex.unsqueeze(dim=-1)
        age = torch.repeat_interleave(age, 7, dim=-1);  sex = torch.repeat_interleave(sex, 7, dim=-1)
        
        # ---- DSP (Deep-Supervision Predictor) ------------------------------------------------------------------------------------
        # Process "sup features" for classification (GAP(global average pooling) + Dropout)
        sup_1 = F.avg_pool3d(sup_1, kernel_size=sup_1.shape[1])
        sup_1 = self.dropout(sup_1) # ; print(sup_1.shape)
        
        sup_2 = F.avg_pool3d(sup_2, kernel_size=sup_2.shape[1])
        sup_2 = F.pad(sup_2, (0, (sup_1.size(4) - sup_2.size(4)),
                              0, (sup_1.size(3) - sup_2.size(3)),
                              0, (sup_1.size(2) - sup_2.size(2))),
                              mode="replicate") # Pad the tensor to match
        sup_2 = self.dropout(sup_2)                                             # ; print(sup_2.shape)

        sup_3 = F.avg_pool3d(sup_3, kernel_size=sup_3.shape[1])
        sup_3 = F.pad(sup_3, (0, (sup_1.size(4) - sup_3.size(4)),
                              0, (sup_1.size(3) - sup_3.size(3)),
                              0, (sup_1.size(2) - sup_3.size(2))),
                              mode="replicate") # Pad the tensor to match
        sup_3 = self.dropout(sup_3)                                             # ; print(sup_3.shape)

        # Classificate the disease
        final = torch.cat([sup_1, sup_2, sup_3], dim=1)                         # ; print("Final Output Shape: ", final.shape)
        res = torch.cat((final, age, sex),dim=1)
        disease_out = self.disease_detector(res)                                # ; print("Out Shape: ", disease_out.shape); # print(disease_out)
        # -----------------------------------------------------------------------------------------------------------------------

        return disease_out, features_sup_3, features_sup_2, features_sup_1
    
    def configure_optimizers(self):
        learning_rate = 0.0001
        return torch.optim.Adam(self.parameters(), lr=learning_rate)
    
    def training_step(self, batch, batch_idx):
        input_img = batch["Image"]
        if len(input_img.shape) < 2: input_img = input_img.unsqueeze(dim=0)
        labels = batch["Disease"]
        if len(labels.shape) < 2: labels = labels.unsqueeze(dim=0)
        labels = torch.argmax(labels, dim=-1)

        sex = batch["Sex"]
        age = batch["Age"]

        output_pred, _, _, _ = self(input_img, sex=sex, age=age)

        # Disease Classification
        output_pred = output_pred.squeeze()
        if len(output_pred.shape) < 2: output_pred = output_pred.unsqueeze(dim=0)
        output_pred = F.softmax(output_pred, dim=1) # Disease Features probability distribution
        if len(output_pred.shape) < 2: output_pred = output_pred.unsqueeze(dim=0)
        # print("- Output probability distribution shape: ", output_pred.shape); print(output_pred)

        train_loss = self.loss_function(output_pred, labels)    # ; print("  - Disease Training Loss: ", train_loss)
        train_f1 = self.disease_accuracy(output_pred, labels)   # ; print("  - Disease Training Acc: ", train_f1)
        self.train_disease_confusion_matrix.update(output_pred, labels) # Training Confusion Matrix
        
        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_f1', train_f1, on_step=True, on_epoch=True, prog_bar=True)
        wandb.log({"train_loss": train_loss, "train_f1": train_f1})

        return {"loss": train_loss, "train_f1": train_f1}
    
    def validation_step(self, batch, batch_idx):
        input_img = batch["Image"]
        if len(input_img.shape) < 2: input_img = input_img.unsqueeze(dim=0)
        labels = batch["Disease"]
        if len(labels.shape) < 2: labels.unsqueeze(dim=0)
        labels = labels = torch.argmax(labels, dim=-1)

        sex = batch["Sex"]
        age = batch["Age"]

        output_pred, _, _, _ = self(input_img, sex=sex, age=age)

        # Disease Classification
        output_pred = output_pred.squeeze()
        if len(output_pred.shape) < 2: output_pred = output_pred.unsqueeze(dim=0)
        output_pred = F.softmax(output_pred, dim=1) # Disease Features probability distribution
        if len(output_pred.shape) < 2: output_pred = output_pred.unsqueeze(dim=0)
        # print("- Output probability distribution shape: ", output_pred.shape); print(output_pred)

        valid_loss = self.loss_function(output_pred, labels);   # print("  - Disease Training Loss: ", valid_loss)
        valid_f1 = self.disease_accuracy(output_pred, labels);  # print("  - Disease Training Acc: ", valid_f1)
        self.valid_disease_confusion_matrix.update(output_pred, labels) # Training Confusion Matrix
        
        self.log('valid_loss', valid_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('valid_f1', valid_f1, on_step=True, on_epoch=True, prog_bar=True)

        return {"valid_loss": valid_loss, "valid_f1": valid_f1}
    
    def on_train_epoch_end(self):
        cm_save_path = os.path.join(cm_path, f"{self.trainer.model_type}/{self.trainer.wandb_id}")
        os.makedirs(cm_save_path, exist_ok=True)

        # Log the Training Confusion Matrix Image into WandB
        print(f"-- Logging Training Confusion Matrix of the epoch {self.current_epoch} --")
        cm = self.train_disease_confusion_matrix.compute()
        cm = cm.cpu().detach().numpy()
        plot_confusion_matrix(cm=cm, out_class=["CN", "MCI", "AD"],
                                cmap="Blues", title=f"Training Confusion Matrix Epoch {self.current_epoch}",
                                save_cm=True, save_dir=cm_save_path, file_name=f"train_cm_{self.current_epoch}")
        wandb.log({"Training Confusion Matrices": wandb.Image(os.path.join(cm_save_path, f"train_cm_{self.current_epoch}.png"))})
        print("-- Confusion Matrix saved --")
        self.train_disease_confusion_matrix.reset()
        
        # Log the Validation Confusion Matrix Image into WandB
        print(f"-- Logging Validation Confusion Matrix of the epoch {self.current_epoch} --")
        cm = self.valid_disease_confusion_matrix.compute()
        cm = cm.cpu().detach().numpy()
        plot_confusion_matrix(cm=cm, out_class=["CN", "MCI", "AD"],
                                cmap="Reds", title=f"Validation Confusion Matrix Epoch {self.current_epoch}",
                                save_cm=True, save_dir=cm_save_path, file_name=f"valid_cm_{self.current_epoch}")
        wandb.log({"Validation Confusion Matrices": wandb.Image(os.path.join(cm_save_path, f"valid_cm_{self.current_epoch}.png"))})
        print("-- Confusion Matrix saved --")
        self.valid_disease_confusion_matrix.reset()
    
    def on_test_epoch_start(self):
        self.test_loss_function = nn.CrossEntropyLoss(reduce=None, reduction='mean', label_smoothing=0.0)
        self.test_disease_accuracy = MulticlassF1Score(num_classes=len(self.disease_classes),
                                                       top_k=1,
                                                       average='micro',
                                                       multidim_average='global',
                                                       ignore_index=None, validate_args=False)
        self.test_disease_confusion_matrix = MulticlassConfusionMatrix(num_classes=len(self.disease_classes),
                                                                       ignore_index=None, validate_args=True, normalize="all")
        
    def test_step(self, batch, batch_idx):
        input_img = batch["Image"]
        if len(input_img.shape) < 2: input_img = input_img.unsqueeze(dim=0)
        labels = batch["Disease"]
        if len(labels.shape) < 2: labels.unsqueeze(dim=0)
        labels = labels = torch.argmax(labels, dim=-1)

        sex = batch["Sex"]
        age = batch["Age"]

        output_pred, _, _, _ = self(input_img, sex=sex, age=age)

        # Disease Classification
        output_pred = output_pred.squeeze()
        if len(output_pred.shape) < 2: output_pred = output_pred.unsqueeze(dim=0)
        output_pred = F.softmax(output_pred, dim=1) # Disease Features probability distribution
        if len(output_pred.shape) < 2: output_pred = output_pred.unsqueeze(dim=0)

        test_loss = self.loss_function(output_pred, labels)
        test_f1 = self.disease_accuracy(output_pred, labels)
        self.test_disease_confusion_matrix.update(output_pred, labels)
        
        self.log('test_loss', test_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('test_f1', test_f1, on_step=True, on_epoch=True, prog_bar=True)

        return {"test_loss": test_loss, "test_f1": test_f1}
    
    def on_test_epoch_end(self):
        cm_save_path = os.path.join(cm_path, f"{self.trainer.model_type}/{self.trainer.wandb_id}")
        os.makedirs(cm_save_path, exist_ok=True)
        cm = self.test_disease_confusion_matrix.compute()
        cm = cm.cpu().detach().numpy()
        plot_confusion_matrix(cm=cm, out_class=["CN", "MCI", "AD"],
                                cmap="Greens", title=f"Test Confusion Matrix Epoch {self.current_epoch}",
                                save_cm=False, save_dir=cm_save_path, file_name=f"test_cm", verbose=True)