import logging
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import monai
from PIL import Image
from monai.data import DataLoader, Dataset, list_data_collate, decollate_batch
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, SqueezeDimd, Activations, AsDiscrete
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
import torch.nn as nn

modellnavn = 'fisk30.pth'
bildenavn = 'F2_SinkabergHansen_Svaberget__2'
datadir = r'C:\Users\ErlendKallelid\datasets\AP1ogVarholmen\AP1'

def hentFarger():
    colormap = {
        0: [100, 100, 100],   # Bakgrunn
        1: [255, 0, 0],       # Klasse 1: Fisk
        2: [0, 255, 0],       # Klasse 2: Ryggrad
        3: [0, 0, 255],       # Klasse 3: Kant/skade
        4: [255, 0, 255]      # Klasse 4: Ikke brukt
    }
    return colormap

def main(tempdir, modellnavn):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    # Filbane for testbildet
    val_files_list = [{'img': os.path.join(tempdir, bildenavn + '.dcm.nii.gz')}]
    val_files = val_files_list

    # Definerer transformasjoner for bildet
    val_transforms = Compose([
        LoadImaged(keys=["img"], reader='PydicomReader'),
        EnsureChannelFirstd(keys=["img"]),
        ScaleIntensityd(keys=["img"]),
        SqueezeDimd(keys=["img"], dim=-1)
    ])
    
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate)
    
    # Definerer post-prosessering
    post_trans = Compose([
        Activations(softmax=True),
        AsDiscrete(argmax=True)
    ])
    
    # Initialiserer modellen
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class SegmentationModel(nn.Module):
        def __init__(self):
            super(SegmentationModel, self).__init__()
            self.unet = UNet(
                spatial_dims=2,
                in_channels=1,  # 1-channel (grayscale input)
                out_channels=2,  # Number of output channels (classes)
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
            )
        
        def forward(self, x):
            return self.unet(x)    
    
    
    model = SegmentationModel().to(device)
    model.load_state_dict(torch.load(modellnavn))
    
    
    # Kjører modellen
    model.eval()
    farger = hentFarger()
    
    with torch.no_grad():
        for val_data in val_loader:
            val_images = val_data["img"].to(device)
            roi_size = (512, 512)
            sw_batch_size = 4
            val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
            
            for val_output in val_outputs:
                val_output_np = val_output.cpu().numpy().squeeze()
                hoyde, bredde = val_output_np.shape
                fargebilde = np.zeros((hoyde, bredde, 3), dtype=np.uint8)
                
                for klasse, farge in farger.items():
                    fargebilde[val_output_np == klasse] = farge
                
                fargebilde_pil = Image.fromarray(fargebilde)
                
                # Visualisering
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(val_images.cpu().numpy().squeeze(), cmap="gray")
                axes[0].set_title(f'Original: {bildenavn}', fontsize=10)
                axes[0].axis('off')
                
                axes[1].imshow(fargebilde)
                axes[1].set_title(f'Segmentation: {modellnavn}', fontsize=10)
                axes[1].axis('off')
                
                plt.show()

if __name__ == "__main__":
    main(datadir, modellnavn)