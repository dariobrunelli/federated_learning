import torch
import numpy as np

from evaluation import decode_preds, compute_nme

# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#     def __init__(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count
        

def train_fn(config, train_loader, model, critertion, optimizer, epoch, writer_dict):

    # losses = AverageMeter()

    model.train()

    #nme_count = 0
    #nme_batch_sum = 0

    nme_count_eyes = 0
    nme_batch_sum_eyes = 0

    nme_count_chin = 0
    nme_batch_sum_chin = 0

    nme_count_eyebrows = 0
    nme_batch_sum_eyebrows = 0

    nme_count_nose = 0
    nme_batch_sum_nose = 0

    nme_count_mouth = 0
    nme_batch_sum_mouth = 0

    nme_count_dbox = 0
    nme_batch_sum_dbox = 0

    for i, (inp, target, meta) in enumerate(train_loader):
        # compute the output
        output = model(inp.cuda())
        
        target = target.cuda(non_blocking=True) #MAX uncomment if cpu
        #Salvare target(= keypoints obiettivo), sull'immagine di input (inp)

        loss = critertion(output, target)

        # NME
        score_map = output.data.cpu()
        preds = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])

        # NME
        nme_computed = compute_nme(preds, meta)
        nme_temp_eyes = [array[0] for array in nme_computed]
        nme_temp_mouth = [array[1] for array in nme_computed]
        nme_temp_nose = [array[2] for array in nme_computed]
        nme_temp_eyebrows = [array[3] for array in nme_computed]
        nme_temp_chin = [array[4] for array in nme_computed]
        nme_temp_dbox = [array[5] for array in nme_computed]

        nme_batch_sum_eyes += np.sum(nme_temp_eyes)
        nme_count_eyes = nme_count_eyes + preds.size(0)

        nme_batch_sum_mouth += np.sum(nme_temp_mouth)
        nme_count_mouth = nme_count_mouth + preds.size(0)

        nme_batch_sum_nose += np.sum(nme_temp_nose)
        nme_count_nose = nme_count_nose + preds.size(0)

        nme_batch_sum_eyebrows += np.sum(nme_temp_eyebrows)
        nme_count_eyebrows = nme_count_eyebrows + preds.size(0)

        nme_batch_sum_chin += np.sum(nme_temp_chin)
        nme_count_chin = nme_count_chin + preds.size(0)

        nme_batch_sum_dbox += np.sum(nme_temp_dbox)
        nme_count_dbox = nme_count_dbox + preds.size(0)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #     losses.update(loss.item(), inp.size(0))

    #     if i % config.PRINT_FREQ == 0:
    #         msg = 'Epoch: [{0}][{1}/{2}]\t' \
    #               'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
    #               'Speed {speed:.1f} samples/s\t' \
    #               'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
    #               'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
    #                   epoch, i, len(train_loader), batch_time=batch_time,
    #                   speed=inp.size(0)/batch_time.val,
    #                   data_time=data_time, loss=losses)

    #         if writer_dict:
    #             writer = writer_dict['writer']
    #             global_steps = writer_dict['train_global_steps']
    #             writer.add_scalar('train_loss', losses.val, global_steps)
    #             writer_dict['train_global_steps'] = global_steps + 1

    # nme_eyes = nme_batch_sum_eyes / nme_count_eyes
    # nme_mouth = nme_batch_sum_mouth / nme_count_mouth
    # nme_nose = nme_batch_sum_nose / nme_count_nose
    # nme_eyebrows = nme_batch_sum_eyebrows / nme_count_eyebrows
    # nme_chin = nme_batch_sum_chin / nme_count_chin
    # nme_dbox = nme_batch_sum_dbox / nme_count_dbox

    # msg = 'Train Epoch {} time:{:.4f} loss:{:.4f}'.format(epoch, batch_time.avg, losses.avg)
    
    # msg_eyes = 'Train Results Eyes: nme_eyes:{:.4f}'.format(nme_eyes)
    
    # msg_mouth = 'Train Results Mouth: nme_mouth:{:.4f}'.format(nme_mouth)
    
    # msg_nose = 'Train Results Nose: nme_nose:{:.4f}'.format(nme_nose)
    
    # msg_eyebrows = 'Train Results Eyebrows: nme_eyebrows:{:.4f}'.format(nme_eyebrows)
    
    # msg_chin = 'Train Results Chin: nme_chin:{:.4f}'.format(nme_chin)

    # msg_dbox = 'Train Results Diagnoal Box: nme_dbox:{:.4f}'.format(nme_dbox)


def validate(config, val_loader, model, criterion):
    # batch_time = AverageMeter()
    # data_time = AverageMeter()

    # losses = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
    # predictions = torch.zeros((len(val_loader), num_classes, 2))     # val_loader.dataset solo se più val_loaders

    model.eval()

    nme_count_eyes = 0
    nme_batch_sum_eyes = 0

    nme_count_chin = 0
    nme_batch_sum_chin = 0

    nme_count_eyebrows = 0
    nme_batch_sum_eyebrows = 0

    nme_count_nose = 0
    nme_batch_sum_nose = 0

    nme_count_mouth = 0
    nme_batch_sum_mouth = 0

    nme_count_dbox = 0
    nme_batch_sum_dbox = 0

    # end = time.time()

    with torch.no_grad():
        for i, (inp, target, meta) in enumerate(val_loader):
            # data_time.update(time.time() - end)
            output = model(inp.cuda())
            target = target.cuda(non_blocking=True) #MAX uncomment if cpu

            score_map = output.data.cpu()
            # loss
            loss = criterion(output, target)

            preds = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])

            # NME
            nme_computed = compute_nme(preds, meta)
            nme_temp_eyes = [array[0] for array in nme_computed]
            nme_temp_mouth = [array[1] for array in nme_computed]
            nme_temp_nose = [array[2] for array in nme_computed]
            nme_temp_eyebrows = [array[3] for array in nme_computed]
            nme_temp_chin = [array[4] for array in nme_computed]
            nme_temp_dbox = [array[5] for array in nme_computed]

            # Failure Rate under different threshold
            #failure_008 = (nme_temp > 0.08).sum()
            #failure_010 = (nme_temp > 0.10).sum()
            #count_failure_008 += failure_008
            #count_failure_010 += failure_010

            nme_batch_sum_eyes += np.sum(nme_temp_eyes)
            nme_count_eyes = nme_count_eyes + preds.size(0)

            nme_batch_sum_mouth += np.sum(nme_temp_mouth)
            nme_count_mouth = nme_count_mouth + preds.size(0)

            nme_batch_sum_nose += np.sum(nme_temp_nose)
            nme_count_nose = nme_count_nose + preds.size(0)

            nme_batch_sum_eyebrows += np.sum(nme_temp_eyebrows)
            nme_count_eyebrows = nme_count_eyebrows + preds.size(0)

            nme_batch_sum_chin += np.sum(nme_temp_chin)
            nme_count_chin = nme_count_chin + preds.size(0)

            nme_batch_sum_dbox += np.sum(nme_temp_dbox)
            nme_count_dbox = nme_count_dbox + preds.size(0)

            # for n in range(score_map.size(0)):
            #     predictions[meta['index'][n], :, :] = preds[n, :, :]

            # losses.update(loss.item(), inp.size(0))

            # measure elapsed time
            # batch_time.update(time.time() - end)
            # end = time.time()

    # #nme = nme_batch_sum / nme_count
    # #failure_008_rate = count_failure_008 / nme_count
    # #failure_010_rate = count_failure_010 / nme_count

    nme_eyes = nme_batch_sum_eyes / nme_count_eyes
    nme_mouth = nme_batch_sum_mouth / nme_count_mouth
    nme_nose = nme_batch_sum_nose / nme_count_nose
    nme_eyebrows = nme_batch_sum_eyebrows / nme_count_eyebrows
    nme_chin = nme_batch_sum_chin / nme_count_chin
    nme_dbox = nme_batch_sum_dbox / nme_count_dbox

    nme = {
        "nme_eyes": nme_eyes,
        "nme_mouth": nme_mouth,
        "nme_nose": nme_nose,
        "nme_eyebrows": nme_eyebrows,
        "nme_chin": nme_chin,
        "nme_dbox": nme_dbox
    }

    # msg = 'Validation Results time:{:.4f} loss:{:.4f}'.format(batch_time.avg, losses.avg)
    
    # msg_eyes = 'Validation Results Eyes: nme_eyes:{:.4f}'.format(nme_eyes)
    
    # msg_mouth = 'Validation Results Mouth: nme_mouth:{:.4f}'.format(nme_mouth)
    
    # msg_nose = 'Validation Results Nose: nme_nose:{:.4f}'.format(nme_nose)
    
    # msg_eyebrows = 'Validation Results Eyebrows: nme_eyebrows:{:.4f}'.format(nme_eyebrows)
    
    # msg_chin = 'Validation Results Chin: nme_chin:{:.4f}'.format(nme_chin)

    # msg_dbox = 'Validation Results Diagnoal Box: nme_dbox:{:.4f}'.format(nme_dbox)
    
    # logger.info(msg)
    # logger.info(msg_eyes)
    # logger.info(msg_mouth)
    # logger.info(msg_nose)
    # logger.info(msg_eyebrows)
    # logger.info(msg_chin)
    # logger.info(msg_dbox)

    # if writer_dict:
    #     writer = writer_dict['writer']
    #     global_steps = writer_dict['valid_global_steps']
    #     writer.add_scalar('valid_loss', losses.avg, global_steps)
    #     #writer.add_scalar('valid_nme', nme, global_steps)
    #     writer.add_scalar('valid_nme_eyes', nme_eyes, global_steps)
    #     writer.add_scalar('valid_nme_mouth', nme_mouth, global_steps)
    #     writer.add_scalar('valid_nme_nose', nme_nose, global_steps)
    #     writer.add_scalar('valid_nme_eyebrows', nme_eyebrows, global_steps)
    #     writer.add_scalar('valid_nme_chin', nme_chin, global_steps)
    #     writer_dict['valid_global_steps'] = global_steps + 1
    return loss.item(), nme
