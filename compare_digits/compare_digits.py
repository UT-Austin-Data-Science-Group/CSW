import numpy as np
import torch
import torchvision.datasets as datasets
from torchvision import transforms
import tqdm
from utils import *
from slicers import *
np.random.seed(1)
torch.manual_seed(1)
train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                './data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])
            ),
            batch_size=100,
            shuffle=True,
            num_workers=16,
        )
digits_0=[]
digits_1=[]
digits_2=[]
digits_3 =[]
digits_4 =[]
digits_5=[]
digits_6=[]
digits_7=[]
digits_8 =[]
digits_9 =[]
for batch_idx, (data, y) in tqdm.tqdm(enumerate(train_loader, start=0)):
    digits_0.append(data[np.where(y==0)])
    digits_1.append(data[np.where(y == 1)])
    digits_2.append(data[np.where(y == 2)])
    digits_3.append(data[np.where(y == 3)])
    digits_4.append(data[np.where(y == 4)])
    digits_5.append(data[np.where(y == 5)])
    digits_6.append(data[np.where(y == 6)])
    digits_7.append(data[np.where(y == 7)])
    digits_8.append(data[np.where(y == 8)])
    digits_9.append(data[np.where(y == 9)])
digits_0 = torch.cat(digits_0,dim=0)
digits_1 = torch.cat(digits_1,dim=0)
digits_2 = torch.cat(digits_2,dim=0)
digits_3 = torch.cat(digits_3,dim=0)
digits_4 = torch.cat(digits_4,dim=0)
digits_5 = torch.cat(digits_5,dim=0)
digits_6 = torch.cat(digits_6,dim=0)
digits_7 = torch.cat(digits_7,dim=0)
digits_8 = torch.cat(digits_8,dim=0)
digits_9 = torch.cat(digits_9,dim=0)
digits=[digits_0,digits_1,digits_2,digits_3,digits_4,digits_5,digits_6,digits_7,digits_8,digits_9]
# SW = np.zeros((10,10))
# CSW = np.zeros((10,10))
# SW_var = np.zeros((10,10))
# CSW_var = np.zeros((10,10))
# CSWnosp = np.zeros((10,10))
# CSWnop = np.zeros((10,10))
# CSWnosp_var = np.zeros((10,10))
# CSWnop_var = np.zeros((10,10))
CSWd = np.zeros((10,10))
CSWd_var = np.zeros((10,10))
L=100
# SWslicer = Base_Slicer(d=784,L=L)
# CSWslicer = Conv_MNIST_Slicer(L=L)
# CSWnospslicer = Conv_MNIST_Slicer_no_sp(L=L)
# CSWnopslicer = Conv_MNIST_Slicer_no_p(L=L)
CSWdslicer = Conv_MNIST_Slicer_d(L=L)

for i in range(10):
    for j in range(10):
        print(i*10+j)
        if(i==j):
            n = digits[i].shape[0]
            index = int(n/2)
            X= digits[i][:index]
            Y= digits[i][index:]
        else:
            X = digits[i]
            Y = digits[j]
        n = np.min([X.shape[0],Y.shape[0]])
        X=X[:n]
        Y=Y[:n]
        # dis_sw=[]
        # dis_csw=[]
        # dis_cswnosp=[]
        # dis_cswnop=[]
        dis_cswd=[]
        for _ in tqdm.tqdm(range(5)):
            CSWdslicer.reset()
            dis_cswd.append(one_dimensional_Wasserstein(CSWdslicer(X),CSWdslicer(Y),p=2).data)
            # CSWnospslicer.reset()
            # CSWnopslicer.reset()
            # dis_cswnosp.append(one_dimensional_Wasserstein(CSWnospslicer(X),CSWnospslicer(Y),p=2).data)
            # dis_cswnop.append(one_dimensional_Wasserstein(CSWnopslicer(X), CSWnopslicer(Y), p=2).data)
            # SWslicer.reset()
            # CSWslicer.reset()
            # dis_sw.append(one_dimensional_Wasserstein(SWslicer(X),SWslicer(Y),p=2).data)
            # dis_csw.append(one_dimensional_Wasserstein(CSWslicer(X), CSWslicer(Y), p=2).data)
        # SW[i, j] = np.mean(dis_sw)
        # CSW[i, j] =np.mean(dis_csw)
        # SW_var[i,j] = np.std(dis_sw)
        # CSW_var[i, j] = np.std(dis_csw)
        # CSWnosp[i, j] = np.mean(dis_cswnosp)
        # CSWnop[i, j] =np.mean(dis_cswnop)
        # CSWnosp_var[i,j] = np.std(dis_cswnosp)
        # CSWnop_var[i, j] = np.std(dis_cswnop)
        CSWd[i, j] = np.mean(dis_cswd)
        CSWd_var[i, j] = np.std(dis_cswd)
np.savetxt("digits/CSWd_mean_{}.csv".format(L), CSWd, delimiter=",")
np.savetxt("digits/CSWd_var_{}.csv".format(L), CSWd_var, delimiter=",")
# np.savetxt("digits/SW_mean_{}.csv".format(L), SW, delimiter=",")
# np.savetxt("digits/CSW_mean_{}.csv".format(L), CSW, delimiter=",")
# np.savetxt("digits/SW_var_{}.csv".format(L), SW_var, delimiter=",")
# np.savetxt("digits/CSW_var_{}.csv".format(L), CSW_var, delimiter=",")

# np.savetxt("digits/CSWnosp_mean_{}.csv".format(L), CSWnosp, delimiter=",")
# np.savetxt("digits/CSWnop_mean_{}.csv".format(L), CSWnop, delimiter=",")
# np.savetxt("digits/CSWnosp_var_{}.csv".format(L), CSWnosp_var, delimiter=",")
# np.savetxt("digits/CSWnop_var_{}.csv".format(L), CSWnop_var, delimiter=",")