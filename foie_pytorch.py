"""
Foie challenge 2018
"""
import torch
import torch.nn as nn
from argparse import ArgumentParser
import warnings
from model_pytorch import Model1
import  dataloaders_pytorch as foieDataLoaders
from tqdm import tqdm
import utils_pytorch as utils
import numpy as np

def validate(valloader, model, criterion,device,lossWeights):
    model.eval()
    val_loss = 0
    val_acc1,val_acc2,val_acc3=0.0,0.0,0.0
    count=0

    with torch.no_grad():
        for inputs, label1,label2,label3 in valloader:
            inputs, label1,label2,label3 = inputs.to(device), label1.to(device), label2.to(device), label3.to(device)
            output1,output2,output3 = model(inputs)
            count+=inputs.size(0)
            val_loss += lossWeights['sain_output'] * criterion(output1, label1) + lossWeights['malin_output'] * criterion(output2, label2) + lossWeights['anomaly_output'] * criterion(output3, label3)
            _, pred1 = torch.max(output1.data, 1)
            _, pred2 = torch.max(output2.data, 1)
            _, pred3 = torch.max(output3.data, 1)    
            val_acc1 += (pred1 == label1).sum()
            val_acc2 += (pred2 == label2).sum()
            val_acc3 += (pred3 == label3).sum()
        #print("TEST Lenght: {}, ACC1 {} ACC2 {} ACC3 {}".format(count,val_acc1,val_acc2,val_acc3))
        val_loss = float(val_loss)  /  count
        val_acc1f = float(val_acc1) /  count
        val_acc2f = float(val_acc2) /  count
        val_acc3f = float(val_acc3) /  count
        
    return val_loss, val_acc1f, val_acc2f, val_acc3f



def train(trainloader, model, criterion, optimizer,device,lossWeights):
    model.train()
    
    train_loss = 0.0
    train_acc1,train_acc2,train_acc3 = 0.0,0.0,0.0
    count=0
    for i, (inputs, label1,label2,label3) in enumerate(trainloader):
        if len(inputs)>1:
            inputs, label1,label2,label3 = inputs.to(device), label1.to(device), label2.to(device), label3.to(device)
            optimizer.zero_grad()

            output1,output2,output3 = model(inputs)
    
            loss = lossWeights['sain_output'] * criterion(output1, label1) + lossWeights['malin_output'] * criterion(output2, label2) + lossWeights['anomaly_output'] * criterion(output3, label3)
            loss.backward()
            optimizer.step()

            #Check how to see Accuracy
            count+=inputs.size(0)
            _, pred1 = torch.max(output1.data, 1)
            _, pred2 = torch.max(output2.data, 1)
            _, pred3 = torch.max(output3.data, 1)

            train_loss += loss.data[0]
            train_acc1 += (pred1 == label1).sum()
            train_acc2 += (pred2 == label2).sum()
            train_acc3 += (pred3 == label3).sum()


    #print("TRAIN Lenght: {}, ACC1 {} ACC2 {} ACC3 {}".format(count,train_acc1,train_acc2,train_acc3))
    train_loss = float(train_loss) / count
    train_acc1 = float(train_acc1) / count
    train_acc2 = float(train_acc2) / count
    train_acc3 = float(train_acc3) / count
    return train_loss, train_acc1, train_acc2, train_acc3
    #return 0,0,0,0


def run(imgf,imgfval,csv,csvval,train_batch_size, val_batch_size, epochs, lr,  log_interval, log_dir,checkpoint_model_dir,checkpoint_interval,gpu):
    if gpu==1:
        use_cuda =  torch.cuda.is_available()
    else:
        use_cuda= False
    device = torch.device("cuda" if use_cuda else "cpu")
    
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    print("Starting training for   epochs: {}, batch size: {}, lr: {:.4f}, using: {}"
			  .format( epochs, train_batch_size,lr,device ))

    model1=Model1(2,3,6)
    
    if use_cuda:
	    model1.cuda()
	    model1 = nn.DataParallel(model1, device_ids=range(torch.cuda.device_count()))

    optimizer = torch.optim.Adam(model1.parameters(), lr=lr)

    train_loader, val_loader = foieDataLoaders.get_data_loaders(imgf,imgfval,csv,csvval,train_batch_size,val_batch_size,kwargs)
    
    lossWeights = {"sain_output": 1.0, "malin_output": 1.0, "anomaly_output": 2.0}
    
    criterion = nn.CrossEntropyLoss()

    writer = utils.create_summary_writer(model1, train_loader, log_dir)

    #media(val_loader,device)

    for i in tqdm(range(epochs)):
        train_loss, train_acc1, train_acc2, train_acc3= train(train_loader,model1,criterion,optimizer,device,lossWeights)
        print("Epoch {} => Train Loss: {:.4f}, Accuracy Sain: {:.4f}, Accuracy Malin: {:.4f}, Accuracy Anomaly: {:.4f}".format(i,train_loss, train_acc1, train_acc2, train_acc3))
        writer.add_scalar("training/avg_loss", train_loss,i)
        val_loss, val_acc1, val_acc2, val_acc3 = validate(val_loader, model1, criterion,device,lossWeights)
        print("Epoch {} => Test Loss: {:.4f}, Accuracy Sain: {:.4f}, Accuracy Malin: {:.4f}, Accuracy Anomaly: {:.4f}".format(i,val_loss, val_acc1, val_acc2, val_acc3))     
        writer.add_scalar("validation/avg_loss", val_loss,i)
        writer.add_scalar("validation/accuracy Sain", val_acc1, i)
        writer.add_scalar("validation/accuracy Malin", val_acc2, i)
        writer.add_scalar("validation/accuracy Anomaly", val_acc3, i)

if __name__ == "__main__":
	
    warnings.filterwarnings("ignore")
	
    parser = ArgumentParser()

    parser.add_argument('--imgf', default="../../JFR/foie_train_set", help="Training set image folder")
    parser.add_argument('--imgfval', default="../../JFR/foie_validation_set",
                        help='Validation set image folder')
    parser.add_argument('--csv',
                    default="../../JFR/foie_train_set_arnEqui.csv",
                    help="csv file")
    parser.add_argument('--csvval',
                        default="../../JFR/foie_validation_setEqui.csv",
                        help="csv file")
    parser.add_argument('--batch_size', type=int, default=5,
                        help='input batch size for training (default: 4)')
    parser.add_argument('--val_batch_size', type=int, default=5,
                        help='input batch size for validation (default: 4)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs to train (default: 10000)')
    parser.add_argument('--lr', type=float, default=0.00001,
                        help='learning rate (default: 0.00001)')
    parser.add_argument('--log_interval', type=int, default=1000,
                        help='how many batches to wait before logging training status')
    parser.add_argument("--log_dir", type=str, default="../logs/try1/",
                        help="log directory for Tensorboard log output")
	
    parser.add_argument("--checkpoint_model_dir", type=str, default='../tmp/checkpoints',
                                  help="path to folder where checkpoints of trained models will be saved")
    parser.add_argument("--checkpoint_interval", type=int, default=1,
                                  help="number of batches after which a checkpoint of trained model will be created")
    parser.add_argument("--gpu", type=int, default=1,
                        help="with gpu=1, no gpu=0")
	

    args = parser.parse_args()

    run(args.imgf,args.imgfval,args.csv,args.csvval,args.batch_size, args.val_batch_size, args.epochs, args.lr,
        args.log_interval, args.log_dir,args.checkpoint_model_dir,args.checkpoint_interval,args.gpu)