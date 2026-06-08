import torch
import time
import numpy as np
import os
from utils.metrics import Evaluator, fast_hist

def val(args, model,  criterion,num_classes,dataloader_val, epoch, loss_train_mean, iou_train_mean):
    print('[Val] ' + args.model_name)
    start = time.time()
    with torch.no_grad():
        model.eval()
        loss_record = []
        hist = np.zeros((num_classes, num_classes))
        evaluator = Evaluator(num_classes)

        for i, (image_t0, image_t1,  label) in enumerate(dataloader_val):
            image_t0, image_t1, label = image_t0.cuda(), image_t1.cuda(), label.cuda()
            output = model(image_t0,image_t1)
            loss = criterion(output, label)
            predict = torch.argmax(output, 1)
            pre = predict.data.view(-1).cpu().numpy()
            lab = label.data.view(-1).cpu().numpy()
            hist += fast_hist(lab, pre, num_classes)
            loss_record.append(loss.item())
        loss_val_mean = np.mean(loss_record)
        pa = evaluator.pixel_accuracy(hist)
        kappa = evaluator.kappa(hist)
        recall = evaluator.recall(hist)
        precision = evaluator.precision(hist)
        f1 = evaluator.f1_score(hist)
        miou = evaluator.mean_intersection_over_union(hist)
        fwiou = evaluator.frequency_weighted_intersection_over_union(hist)
        str_ = ("%15.5g;" * 11) % (epoch+1, loss_train_mean, loss_val_mean, pa, kappa,
                                   recall, precision, f1, iou_train_mean, miou, fwiou)

        with open(os.path.join(args.summary_path, args.model_name, '')+args.model_name+'_result.txt', 'a') as f:
            f.write(str_ + '\n')

        print('[Val] Loss:        {:}'.format(loss_val_mean))
        print('[Val] PA/OA:       {:}'.format(pa))
        print('[Val] Kappa:       {:}'.format(kappa))
        print('[Val] Recall:      {:}'.format(recall))
        print('[Val] Precision:   {:}'.format(precision))
        print('[Val] F1:          {:}'.format(f1))
        print('[Val] mIoU:        {:}'.format(miou))
        print('[Val] FWIoU:       {:}'.format(fwiou))
        print('[Val] Eval_time:   {:}s'.format(time.time() - start))
        
        return miou,hist