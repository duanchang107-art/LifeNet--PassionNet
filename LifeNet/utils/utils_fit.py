import os

import torch
from nets.unet_training import CE_Loss, Dice_loss, Focal_Loss,clss_Loss,loss_builder
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score
import numpy as np
from torch.autograd import Variable



def target_seg2target_cls(labels_batch):
    # labels_batch 是一个形状为 (batch_size, height, width) 的张量

    batch_size, _, _ = labels_batch.shape

    cls_labels_batch = np.zeros(shape=(batch_size, 3), dtype=float)

    for batch_idx in range(batch_size):
        label_set = np.unique(labels_batch[batch_idx])
        

        for i in label_set:
            if i != 0:
                cls_labels_batch[batch_idx, i - 1] += 1

    cls_labels_batch = torch.from_numpy(cls_labels_batch).float()

    return cls_labels_batch
last_iou=0
def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank=0):
    global last_iou
    
    total_loss      = 0
    total_f_score   = 0

    val_loss        = 0
    val_f_score     = 0

    criterion = loss_builder()

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: 
            break
        imgs, pngs, labels,parts1,parts2 = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            # Variable# 将输入和目标转换为 PyTorch 变量并移动到 GPU
            target_seg = pngs.cpu().numpy()
            # target_seg2 = target2.numpy()

            target_cls = target_seg2target_cls(target_seg).cuda()
            
            
            input_var = Variable(imgs).cuda()
            # 分别提取奇偶索引
            target_var_seg = Variable(pngs).cuda()
            target_var_cls = Variable(target_cls).cuda()
            target_var_label=Variable(labels).cuda()
            id_1 = parts2[::2]
            id_2 = parts2[1::2]
            patient_1 = parts1[::2]
            patient_2 = parts1[1::2]
            target_var_seg_1 = target_var_seg[::2]
            target_var_seg_2 = target_var_seg[1::2]
            target_var_cls_1 = target_var_cls[::2]
            target_var_cls_2 = target_var_cls[1::2]
            target_var_label_1 = target_var_label[::2]
            target_var_label_2 = target_var_label[1::2]
            
            # forward

            input_var_1 = input_var[::2]#取偶数
            input_var_2 = input_var[1::2]
            output_seg_1, output_cls_1, cls_logits_1, seg_logits_1 = model(input_var_1)
            output_seg_2, output_cls_2, cls_logits_2, seg_logits_2 = model(input_var_2)
           
           
            output_seg = torch.cat([output_seg_1, output_seg_2], dim=0)
            output_cls = torch.cat([output_cls_1, output_cls_2], dim=0)

           
          


            if focal_loss:

                loss_seg_1 = criterion[0](output_seg_1, target_var_seg_1)
                loss_seg_2 = criterion[0](output_seg_2, target_var_seg_2)
                loss_cls_1 = criterion[2](output_cls_1, target_var_cls_1)
                loss_cls_2 = criterion[2](output_cls_2, target_var_cls_2)

            else:
                loss_seg_1 = CE_Loss(output_seg_1,  target_var_seg_1, weights, num_classes = num_classes)
                loss_seg_2 = CE_Loss(output_seg_2,  target_var_seg_2, weights, num_classes = num_classes)
                # loss_seg_1 = criterion[0](output_seg_1, target_var_seg_1)
                # loss_seg_2 = criterion[0](output_seg_2, target_var_seg_2)
                # loss_cls_1 = criterion[2](output_cls_1, target_var_cls_1)
                # loss_cls_2 = criterion[2](output_cls_2, target_var_cls_2)

            loss_seg = (loss_seg_1 + loss_seg_2) / 2
            #loss_cls=(loss_cls_1 + loss_cls_2) / 2
            c =0
            cls_con_loss = 0
            seg_con_loss = 0
            loss_cls_con_L2 = torch.nn.MSELoss(reduction='mean')
            loss_seg_con_L1 = torch.nn.L1Loss(reduction='mean')
            loss_super = loss_seg 


            #loss_back = torch.nn.KLDivLoss().cuda()

            for j in range(input_var_1.shape[0]):
              for k in  range(input_var_1.shape[0]):
                if int(patient_1[j]) == int(patient_2[k]) and abs(int(id_1[j]) - int(id_2[k])) < 3:
                    c += 1
                    cls_con_loss = cls_con_loss + loss_cls_con_L2(cls_logits_1[j], cls_logits_2[k])

                    seg_con_loss = seg_con_loss + loss_seg_con_L1(seg_logits_1[j], seg_logits_2[k])

                #print('find %s, id:%d, %d' % (patient_1[j], id_1[j], id_2[k]))
                if c > 0:
                    loss_inter = seg_con_loss / c + cls_con_loss / c
                else:
                    loss_inter=0
           

                # loss = loss_super +  loss_background + loss_inter
            loss = loss_super + loss_inter

            # #----------------------#
            # #   损失计算
            # #----------------------#
            # if focal_loss:
            #     loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            # else:
            #     loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)+clss_Loss(outputs, pngs)

            if dice_loss:
                main_dice1 = Dice_loss(seg_logits_1, target_var_label_1)
                main_dice2 = Dice_loss(seg_logits_2, target_var_label_2)
                main_dice = (main_dice1+main_dice2)/2
                loss      = loss + main_dice

            with torch.no_grad():
                #-------------------------------#
                #   计算f_score
                #-------------------------------#
                _f_score = f_score(seg_logits_1, target_var_label_1)+f_score(seg_logits_2, target_var_label_2)

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #   前向传播
                #----------------------#
                outputs = model_train(imgs)
                #----------------------#
                #   损失计算
                #----------------------#
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss      = loss + main_dice

                # with torch.no_grad():
                #     #-------------------------------#
                #     #   计算f_score
                #     #-------------------------------#
                #     _f_score = f_score(outputs, labels)

            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss      += loss.item()
        total_f_score   += _f_score.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                # 'f_score'   : total_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
   

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break

        imgs, pngs, labels, parts1, parts2 = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:
            # ----------------------#
            #   前向传播
            # ----------------------#
            # Variable# 将输入和目标转换为 PyTorch 变量并移动到 GPU
            target_seg = pngs.cpu().numpy()
            # target_seg2 = target2.numpy()

            target_cls = target_seg2target_cls(target_seg).cuda()
            input_var = Variable(imgs).cuda()
            # 分别提取奇偶索引
            target_var_seg = Variable(pngs).cuda()
            target_var_cls = Variable(target_cls).cuda()
            target_var_label = Variable(labels).cuda()
            id_1 = parts2[::2]
            id_2 = parts2[1::2]
            patient_1 = parts1[::2]
            patient_2 = parts1[1::2]
            target_var_seg_1 = target_var_seg[::2]
            target_var_seg_2 = target_var_seg[1::2]
            target_var_cls_1 = target_var_cls[::2]
            target_var_cls_2 = target_var_cls[1::2]
            target_var_label_1 = target_var_label[::2]
            target_var_label_2 = target_var_label[1::2]
            # forward

            input_var_1 = input_var[::2]
            input_var_2 = input_var[1::2]
            output_seg_1, output_cls_1, cls_logits_1, seg_logits_1 = model(input_var_1)
            output_seg_2, output_cls_2, cls_logits_2, seg_logits_2 = model(input_var_2)
            output_seg = torch.cat([output_seg_1, output_seg_2], dim=0)
            output_cls = torch.cat([output_cls_1, output_cls_2], dim=0)

            if focal_loss:

                loss_seg_1 =criterion[0](output_seg_1, target_var_seg_1)
                loss_seg_2 =criterion[0](output_seg_2, target_var_seg_2)
                loss_cls_1 = criterion[2](output_cls_1, target_var_cls_1)
                loss_cls_2 = criterion[2](output_cls_2, target_var_cls_2)

            else:
                loss_seg_1 = CE_Loss(output_seg_1,  target_var_seg_1, weights, num_classes = num_classes)
                loss_seg_2 = CE_Loss(output_seg_2,  target_var_seg_2, weights, num_classes = num_classes)
                # loss_seg_1 = criterion[0](output_seg_1, target_var_seg_1)
                # loss_seg_2 = criterion[0](output_seg_2, target_var_seg_2)
                # loss_cls_1 = criterion[2](output_cls_1, target_var_cls_1)
                # loss_cls_2 = criterion[2](output_cls_2, target_var_cls_2)

            loss_seg = (loss_seg_1 + loss_seg_2) / 2
            #loss_cls = (loss_cls_1 + loss_cls_2) / 2
            c = 0
            cls_con_loss = 0
            seg_con_loss = 0
            loss_cls_con_L2 = torch.nn.MSELoss(reduction='mean')
            loss_seg_con_L1 = torch.nn.L1Loss(reduction='mean')
            loss_super = loss_seg 

            # loss_back = torch.nn.KLDivLoss().cuda()

            for j in range(input_var_1.shape[0]):
                for k in range(input_var_1.shape[0]):
                    if int(patient_1[j]) == int(patient_2[k]) and abs(int(id_1[j]) - int(id_2[k])) < 3:
                        c += 1
                        cls_con_loss = cls_con_loss + loss_cls_con_L2(cls_logits_1[j], cls_logits_2[k])

                        seg_con_loss = seg_con_loss + loss_seg_con_L1(seg_logits_1[j], seg_logits_2[k])

                   
                    if c > 0:
                        loss_inter = seg_con_loss / c + cls_con_loss / c

                    # loss = loss_super +  loss_background + loss_inter
            loss = loss_super + loss_inter
           

            # #----------------------#
            # #   损失计算
            # #----------------------#
            # if focal_loss:
            #     loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            # else:
            #     loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)+clss_Loss(outputs, pngs)

            if dice_loss:
                main_dice1 = Dice_loss(seg_logits_1, target_var_label_1)
                main_dice2 = Dice_loss(seg_logits_2, target_var_label_2)
                main_dice = (main_dice1 + main_dice2) / 2
                loss = loss + main_dice
            #-------------------------------#
            #   计算f_score
            #-------------------------------#
            # f_score(seg_logits_1, target_var_label_1)+f_score(seg_logits_2, target_var_label_2)

            val_loss    += loss.item()
            # val_f_score += _f_score.item()
            
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1),
                                # 'f_score'   : val_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss/ epoch_step, val_loss/ epoch_step_val)
        new_iou = eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth'%((epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val)))

        # if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
        if (new_iou>last_iou):
            print('Save best model to best_epoch_weights.pth')
            last_iou =new_iou
            print(last_iou)
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
            
    # if local_rank == 0:
    #     pbar.close()
    #     print('Finish Validation')
    #     loss_history.append_loss(epoch + 1, total_loss/ epoch_step, val_loss/ epoch_step_val)
    #     eval_callback.on_epoch_end(epoch + 1, model_train)
    #     print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    #     print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        
    #     #-----------------------------------------------#
    #     #   保存权值
    #     #-----------------------------------------------#
    #     if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
    #         torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth'%((epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val)))

    #     if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
    #         print('Save best model to best_epoch_weights.pth')
    #         torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
    #     torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))

def fit_one_epoch_no_val(model_train, model, loss_history, optimizer, epoch, epoch_step, gen, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss      = 0
    total_f_score   = 0
    
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: 
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            outputs = model_train(imgs)
            #----------------------#
            #   损失计算
            #----------------------#
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss      = loss + main_dice

            with torch.no_grad():
                #-------------------------------#
                #   计算f_score
                #-------------------------------#
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #   前向传播
                #----------------------#
                outputs = model_train(imgs)
                #----------------------#
                #   损失计算
                #----------------------#
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss      = loss + main_dice

                with torch.no_grad():
                    #-------------------------------#
                    #   计算f_score
                    #-------------------------------#
                    _f_score = f_score(outputs, labels)

            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss      += loss.item()
        total_f_score   += _f_score.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'f_score'   : total_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        loss_history.append_loss(epoch + 1, total_loss/ epoch_step)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f' % (total_loss / epoch_step))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f.pth'%((epoch + 1), total_loss / epoch_step)))

        if len(loss_history.losses) <= 1 or (total_loss / epoch_step) <= min(loss_history.losses):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))