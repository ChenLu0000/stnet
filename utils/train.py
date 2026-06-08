import numpy as np
import os
import tqdm
import torch
from utils.val import val
from utils.metrics import Evaluator, fast_hist

def train(args, model,  optimizer, criterion, dataloader_train, dataloader_val, 
          exp_lr_scheduler
          ):
    print("[Train]")
    miou_max = args.miou_max
    s = ("%15s;" * 11) % ("epoch", "train_loss", "val_loss", "PA/OA","Kappa", "Recall", "Precision",
    "F1", "Train_Miou", "Val_Miou", "FWiou")

    if not os.path.exists(os.path.join(args.summary_path, args.model_name)):
        os.makedirs(os.path.join(args.summary_path, args.model_name))

    with open(os.path.join(args.summary_path, args.model_name, '') + args.model_name + '_result.txt', 'a') as f:
        f.write(s + '\n')

    for epoch in range(args.num_epochs):
        model.train()
        model.cuda()

        lr = optimizer.param_groups[0]['lr']
        tq = tqdm.tqdm(total=len(dataloader_train)*args.batch_size)
        tq.set_description('Epoch %d, lr %f' % (epoch+1, lr))
        loss_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        evaluator = Evaluator(args.num_classes)

        for i, (image_t0, image_t1, label) in enumerate(dataloader_train):
            image_t0, image_t1, label = image_t0.cuda(), image_t1.cuda(), label.cuda()

            if args.warmup == 1 and epoch == 0:
                lr = args.lr / (len(dataloader_train) - i)
                tq.set_description('epoch %d, lr %f' % (epoch + 1, lr))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            output,out5,out4,out3,out2,out1 = model(image_t0, image_t1)

            predict = torch.argmax(output, 1)

            pre = predict.data.view(-1).cpu().numpy()
            lab = label.data.view(-1).cpu().numpy()

            hist += fast_hist(lab, pre, args.num_classes)
            
            loss = criterion(output, label)

            loss_aux1 = criterion(out1, label)
            loss_aux2 = criterion(out2, label)
            loss_aux3 = criterion(out3, label)
            loss_aux4 = criterion(out4, label)
            loss_aux5 = criterion(out5, label)
            loss = loss + loss_aux1 + loss_aux2 + loss_aux3 + loss_aux4 + loss_aux5
            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_record.append(loss.item())

        iou_train_mean = evaluator.mean_intersection_over_union(hist)

        tq.close()
        loss_train_mean = np.mean(loss_record)

        print('[Train] Loss :{:.6f}'.format(loss_train_mean))
        print('[Train] mIoU :{:.6f}'.format(iou_train_mean))
        exp_lr_scheduler.step()
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            save_path = os.path.join(args.summary_path, args.model_name ,args.save_checkpoint_path, 'epoch_{:}.pth'.format(epoch))
            torch.save(model.state_dict(), save_path)
        
        if epoch % args.validation_step == 0:
            miou, val_hist = val(
                                args,
                                model,
                                criterion,
                                args.num_classes,
                                dataloader_val,
                                epoch,
                                loss_train_mean,
                                iou_train_mean,
                                )

            if miou > miou_max:
                if not os.path.exists(os.path.join(args.summary_path, args.model_name, 'checkpoints')):
                    os.makedirs(os.path.join(args.summary_path, args.model_name, 'checkpoints'))
                save_path = os.path.join(args.summary_path, args.model_name, 'checkpoints','miou_{:.6f}.pth'.format(miou))
                torch.save(model.state_dict(), save_path)
                miou_max = miou