import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import copy
import time
import random
import numpy as np
from asyncdrop_utils import DatasetSplit
#from rnn_data_prepare import collate
#from multiserver import parameter_receiver, send_model



def generate_resnet_drop_smart(rank,args,device,local_model,model,model_bac,epoch):
    slow_index=rank
    total_num_worker=args.num_processes
    if args.baseline:
        pass
    else:
        if args.random_mask:
            for block1,block2,block3 in zip(model.conv3_x,model_bac.conv3_x,local_model.conv3_x):
                num_filter=block1.conv1.weight.data.shape[0]
                block3.mask1=np.random.choice(num_filter,int(num_filter*args.hidden_dim_prob),replace=False)
            for block1,block2,block3 in zip(model.conv4_x,model_bac.conv4_x,local_model.conv4_x):
                num_filter=block1.conv1.weight.data.shape[0]
                block3.mask1=np.random.choice(num_filter,int(num_filter*args.hidden_dim_prob),replace=False)
            for block1,block2,block3 in zip(model.conv5_x,model_bac.conv5_x,local_model.conv5_x):
                num_filter=block1.conv1.weight.data.shape[0]
                block3.mask1=np.random.choice(num_filter,int(num_filter*args.hidden_dim_prob),replace=False)
        else:
            layer2_score=[]

            for block1,block2,block3 in zip(model.conv3_x,model_bac.conv3_x,local_model.conv3_x):

                
                score=torch.abs(block1.conv1.weight.data-block2.conv1.weight.data).sum([1,2,3])+torch.abs(block1.conv2.weight.data-block2.conv2.weight.data).sum([0,2,3])
                layer2_score.append(score)
                


                num_filter=score.shape[0]
                filter_each=int(num_filter*args.hidden_dim_prob)
                start=slow_index*int(num_filter/total_num_worker)
                if start+filter_each>num_filter:
                    block3.mask1=torch.argsort(score,descending=args.descending)[(num_filter-filter_each):]
                else:
                    block3.mask1=torch.argsort(score,descending=args.descending)[start:start+filter_each]

                    


            layer3_score=[]

            for block1,block2,block3 in zip(model.conv4_x,model_bac.conv4_x,local_model.conv4_x):
                score=torch.abs(block1.conv1.weight.data-block2.conv1.weight.data).sum([1,2,3])+torch.abs(block1.conv2.weight.data-block2.conv2.weight.data).sum([0,2,3])
                layer3_score.append(score)

                num_filter=score.shape[0]
                filter_each=int(num_filter*args.hidden_dim_prob)
                start=slow_index*int(num_filter/total_num_worker)
                if start+filter_each>num_filter:
                    block3.mask1=torch.argsort(score,descending=args.descending)[(num_filter-filter_each):]
                else:
                    block3.mask1=torch.argsort(score,descending=args.descending)[start:start+filter_each]

            layer4_score=[]

            for block1,block2,block3 in zip(model.conv5_x,model_bac.conv5_x,local_model.conv5_x):
                score=torch.abs(block1.conv1.weight.data-block2.conv1.weight.data).sum([1,2,3])+torch.abs(block1.conv2.weight.data-block2.conv2.weight.data).sum([0,2,3])
                layer4_score.append(score)

                num_filter=score.shape[0]
                filter_each=int(num_filter*args.hidden_dim_prob)
                start=slow_index*int(num_filter/total_num_worker)
                if start+filter_each>num_filter:
                    block3.mask1=torch.argsort(score,descending=args.descending)[(num_filter-filter_each):]
                else:
                    block3.mask1=torch.argsort(score,descending=args.descending)[start:start+filter_each]




def train_update_gradient_only(rank, args, model, model_bac, device, global_lr, global_iter,dataset, dataset_2, start_epoch, dataloader_kwargs,non_iid_idx=None):
    torch.manual_seed(args.seed*args.num_processes + rank )
    random.seed(args.seed*args.num_processes + rank )
    np.random.seed(args.seed*args.num_processes+rank )

    local_model=copy.deepcopy(model).to(device)
    local_model_bac=copy.deepcopy(model)
    local_model_bac_gpu=None
    if args.model_type=='resnet':
        generate_resnet_drop_smart(rank,args,device,local_model,model,model_bac,epoch=start_epoch)
    else:
        raise ValueError
    
    #print(model.conv1.weight)
    optimizer = optim.SGD(local_model.parameters(), lr=global_lr.data[0], momentum=args.momentum)
    if args.lr_type=='cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    test_acc=[]
    total_communication_round=[]
    for epoch in range(start_epoch+1, args.epochs + 1):
        if args.non_iid:    
            current_user=random.randint(0,len(non_iid_idx)-1)
            train_loader = torch.utils.data.DataLoader(DatasetSplit(dataset, non_iid_idx[current_user]),**dataloader_kwargs)
        else:
            raise ValueError

        if args.lr_type=='step_wise':
            lr_t=args.lr
            if (epoch-start_epoch-1) > int(args.epochs*0.5):
                lr_t /= 10
            if (epoch-start_epoch-1) > int(args.epochs*0.75):
                lr_t /= 10
            if lr_t<global_lr:
                global_lr.data.copy_(torch.tensor([lr_t]))
            for pg in optimizer.param_groups:
                pg['lr'] = global_lr.data[0]

        train_epoch_update_gradient_only(rank, epoch, args,  model, model_bac, global_lr, global_iter, local_model, local_model_bac, device, train_loader, optimizer,local_model_bac_gpu=local_model_bac_gpu)
        if (epoch-start_epoch-1)%args.log_epoch==0:
            acc=test(rank,args,copy.deepcopy(model).to(device),device,dataset_2,dataloader_kwargs)
            test_acc.append(acc)
            total_communication_round.append(copy.deepcopy(global_iter.data[0]).numpy())
            print(global_iter.data[0])
        
        if epoch==args.epochs:
            os.system('killall python3')
        
        if rank==0:
            if args.non_iid:
                if args.baseline:
                    np.savetxt('Baseline_'+args.dataset+'.txt',np.array(test_acc))
                else:
                    if args.random_mask:
                        np.savetxt('AsyncDrop_'+args.dataset+'.txt',np.array(test_acc))
                    else:
                        np.savetxt('HeteroAsyncDrop_'+args.dataset+'.txt',np.array(test_acc))
            else:
                raise ValueError

def test(rank,args, model, device, dataset, dataloader_kwargs):
    #torch.manual_seed(args.seed)
    test_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    acc=test_epoch(rank,args,model, device, test_loader)
    return(acc)



def train_epoch_update_gradient_only(rank, epoch, args, model, model_bac, global_lr, global_iter, local_model,local_model_bac, device, data_loader, optimizer,local_model_bac_gpu=None):
    model.train()
    pid = os.getpid()
    if args.non_iid:
        num_iter=0
        sub_epoch_num=args.num_processes
        for sub_epoch in range(sub_epoch_num):

            for batch_idx, batch in enumerate(data_loader):
                

                optimizer.zero_grad()

                data=batch[0]
                target=batch[1]
                output = local_model(data.to(device))
                loss = F.cross_entropy(output, target.to(device))

                loss.backward()


                optimizer.step()

                
                if num_iter % args.log_interval == 0:
                    print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        pid, epoch, batch_idx * args.batch_size, len(data_loader.dataset),
                        100. * batch_idx / len(data_loader), loss.item()))
                    if args.dry_run:
                        break

                if args.baseline:
                    time.sleep((rank+1)/args.delay)
                else:
                    time.sleep((rank+1)/args.delay*(1-args.hidden_dim_prob))


                if num_iter%args.local_iterations==0:
                    

                    model,model_bac=model_average_update_gradient_only(rank, args, model,model_bac, (copy.deepcopy(local_model)).cpu(),local_model_bac)

                    global_iter.data.copy_(global_iter.data+1)


                    local_model_bac=copy.deepcopy(model)

                    local_model.load_state_dict(copy.deepcopy(local_model_bac).state_dict())
                    

                    optimizer = optim.SGD(local_model.parameters(), lr=global_lr.data[0], momentum=args.momentum)

                    
                    if args.model_type=='resnet':
                        generate_resnet_drop_smart(rank,args,device,local_model,model,model_bac, epoch)
                    else:
                        raise ValueError


                num_iter=num_iter+1
    else:
        
        raise ValueError

                #raise


def model_average_update_gradient_only(rank, args, global_model, global_model_bac, local_model,local_model_bac):
    alpha=args.alpha


    global_model_para = global_model.state_dict()
    local_model_para = local_model.state_dict()
    local_model_bac_para = local_model_bac.state_dict()
    
    if not(args.smart_long_memory):
        global_model_bac.load_state_dict(global_model_para)


    for key in global_model_para:
        if args.baseline:
            global_model_para[key].data.copy_((1-alpha)*global_model_para[key].data+alpha*local_model_para[key].data)
        else:
            diff=local_model_para[key] -local_model_bac_para[key]
            mask=(torch.abs(diff)>1e-9).type(torch.FloatTensor)
            global_model_para[key].data.copy_(global_model_para[key].data*(1-mask)+(1-alpha)*global_model_para[key].data*mask+alpha*local_model_para[key].data*mask)
        
    global_model.load_state_dict(global_model_para)
    return(global_model,global_model_bac)


def test_epoch(rank,args, model, device, data_loader):
    model.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data.to(device))
            if rank==0:
                test_loss += F.cross_entropy(output, target.to(device), reduction='sum').item() # sum up batch loss
                pred = output.max(1)[1] # get the index of the max log-probability
                correct += pred.eq(target.to(device)).sum().item()

    if rank==0:
        test_loss /= len(data_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset),
            100. * correct / len(data_loader.dataset)))

    return (100. * correct / len(data_loader.dataset))
