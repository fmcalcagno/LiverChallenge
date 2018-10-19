
import numpy as np
import torch 
try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")


def create_summary_writer(model, data_loader, log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    data_loader_iter = iter(data_loader)
    x, _, _,  _ = next(data_loader_iter)
    try:
        writer.add_graph(model, x)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer


def media(trainloader, device):
    g_mean_channel1=0;g_mean_channel2=0;g_mean_channel3=0
    g_std_channel1=0;g_std_channel2=0;g_std_channel3=0   
    nobs=0   
    for i, (inputs, label1,label2,label3) in enumerate(trainloader):
        inputs=inputs.to(device)
        s=len(inputs)
        mean_channel1=torch.mean(inputs[0,:,:]).to("cpu")
        mean_channel2=torch.mean(inputs[1,:,:]).to("cpu")
        mean_channel3=torch.mean(inputs[2,:,:]).to("cpu")

        std_channel1=torch.std(inputs[0,:,:]).to("cpu")
        std_channel2=torch.std(inputs[1,:,:]).to("cpu")
        std_channel3=torch.std(inputs[2,:,:]).to("cpu")
        m = nobs * 1.0
        n = s
        tmp = g_mean_channel1
        g_mean_channel1 = m/(m+n)*tmp + n/(m+n)*mean_channel1
        g_std_channel1  = np.sqrt(m/(m+n)*g_std_channel1**2 + n/(m+n)*std_channel1**2 +  m*n/(m+n)**2 * (tmp - mean_channel1)**2)
        tmp = g_mean_channel2
        g_mean_channel2 = m/(m+n)*tmp + n/(m+n)*mean_channel2
        g_std_channel2  = np.sqrt(m/(m+n)*g_std_channel2**2 + n/(m+n)*std_channel2**2 +  m*n/(m+n)**2 * (tmp - mean_channel2)**2)
        tmp = g_mean_channel3
        g_mean_channel3 = m/(m+n)*tmp + n/(m+n)*mean_channel3
        g_std_channel3  = np.sqrt(m/(m+n)*g_std_channel3**2 + n/(m+n)*std_channel3**2 +  m*n/(m+n)**2 * (tmp - mean_channel3)**2)
        nobs+=n
    print("Mean: ",g_mean_channel1,g_mean_channel2,g_mean_channel3)
    print("STDdev: ",g_std_channel1,g_std_channel2,g_std_channel3)

