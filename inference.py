import torch
import os
import cv2
from datasets.LEVIR_CD import LEVIRCDDataset
from torch.utils.data import DataLoader
import numpy as np
import tqdm

def inference(model, image_t0, image_t1):
    predict_rgb = np.zeros([256,256,3])
    image_t0, image_t1 = image_t0.cuda(), image_t1.cuda()
    output = model(image_t0, image_t1)
    predict = torch.argmax(output, 1)
    predict = predict.cpu()
    predict = np.array(predict).astype(int).squeeze(0)
    predict_rgb[predict==1] = [255,255,255]
    
    return predict_rgb

if __name__ == "__main__":
    from model.stnet import STNet

    MODEL_PATH = 'MODEL_PATH'
    CKPT_PATH = 'CKPT_PATH'
    DATASET_PATH = 'DATASET_PATH'
    SAVE_PATH = os.path.join(MODEL_PATH, 'inference')
    
    dataset_name = DATASET_PATH.split('/')[-1]
    if not os.path.exists(os.path.join(SAVE_PATH, dataset_name)):
        os.makedirs(os.path.join(SAVE_PATH, dataset_name))

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    test_dataset = LEVIRCDDataset(dataset_path=DATASET_PATH, mode='test')
    dataloader_test = DataLoader(test_dataset,batch_size=1)

    model = STNet(in_chans=3, num_classes=2, vss_depth=2, vssm_path_size=4).cuda()

    model.eval()
    model.load_state_dict(torch.load(CKPT_PATH))

    for (image_t0, image_t1, label, label_name) in tqdm.tqdm(dataloader_test):
        predict_rgb = inference(model, image_t0, image_t1)
        cv2.imwrite(os.path.join(SAVE_PATH, dataset_name, label_name[0] + '.png'), 
                    predict_rgb)

    