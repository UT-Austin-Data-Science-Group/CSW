import torch
import torch.nn as nn
from power_spherical import PowerSpherical
class PS_Slicer(nn.Module):
    def __init__(self,d,L,k,kappa):
        super(PS_Slicer, self).__init__()
        self.d = d
        self.L = L
        self.k = k
        self.kappa =kappa
        self.U = nn.Linear(self.d, self.k, bias=False)
    def forward(self,x):
        x = x.view(x.shape[0],-1)
        return torch.matmul(x,self.thetas.T)
    def project_parameters(self):
        self.U.weight.data = self.U.weight/torch.sqrt(torch.sum(self.U.weight**2,dim=1,keepdim=True))
        distribution = PowerSpherical(self.U.weight.data, torch.full((self.k,), self.kappa))
        self.thetas = distribution.sample((int(self.L / self.k),)).view(self.L, -1)  # Lxd
    def reset(self):
        self.U.weight.data = torch.randn(self.U.weight.shape)
        self.project_parameters()
class PRW_Slicer(nn.Module):
    def __init__(self,d,k):
        super(PRW_Slicer, self).__init__()
        self.d=d
        self.k=k
        self.U = nn.Linear(self.d,self.k,bias=False)
        self.U.weight.data = torch.randn(self.U.weight.shape)
        self.project_parameters()
    def forward(self,x):
        x = x.view(x.shape[0], -1)
        self.project_parameters()
        return self.U(x)
    def project_parameters(self):
        self.U.weight.data = torch.linalg.qr(self.U.weight.T)[0].T
    def reset(self):
        self.U.weight.data = torch.randn(self.U.weight.shape)
        self.project_parameters()
class Base_Slicer(nn.Module):
    def __init__(self,d,L):
        super(Base_Slicer, self).__init__()
        self.d=d
        self.L=L
        self.U = nn.Linear(self.d,self.L,bias=False)
        self.U.weight.data = torch.randn(self.U.weight.shape)
        self.project_parameters()
    def forward(self,x):
        x = x.view(x.shape[0],-1)
        self.project_parameters()
        return self.U(x)
    def project_parameters(self):
        self.U.weight.data = self.U.weight/torch.sqrt(torch.sum(self.U.weight**2,dim=1,keepdim=True))
    def reset(self):
        self.U.weight.data = torch.randn(self.U.weight.shape)
        self.project_parameters()
# class Multilayer_PRW_Slicer(nn.Module):
#     def __init__(self,sizes):
#         super(Multilayer_PRW_Slicer, self).__init__()
#         self.sizes=sizes
#         self.d=sizes[0]
#         self.k=sizes[-1]
#         self.U_list=[]
#         for i in range(0,len(sizes)-1):
#             self.U_list.append(nn.Linear(self.sizes[i],self.sizes[i+1],bias=False))
#         self.project_parameters()
#     def forward(self,x,activation=None):
#         self.project_parameters()
#         for i in range(len(self.U_list)):
#             x = self.U_list[i](x)
#             if(i!=len(self.U_list)-1 and activation is not None):
#                 x=activation(x)
#         return x
#     def project_parameters(self):
#         for i in range(len(self.U_list)):
#             self.U_list[i].weight.data = torch.linalg.qr(self.U_list[i].weight.T)[0].T
#
# class Multilayer_Base_Slicer(nn.Module):
#     def __init__(self,sizes):
#         super(Multilayer_Base_Slicer, self).__init__()
#         self.sizes=sizes
#         self.d=sizes[0]
#         self.k=sizes[-1]
#         self.U_list=[]
#         for i in range(0,len(sizes)-1):
#             self.U_list.append(nn.Linear(self.sizes[i],self.sizes[i+1],bias=False))
#         self.project_parameters()
#     def forward(self,x,activation=None):
#         self.project_parameters()
#         for i in range(len(self.U_list)):
#             x = self.U_list[i](x)
#             if(i!=len(self.U_list)-1 and activation is not None):
#                 x=activation(x)
#         return x
#     def project_parameters(self):
#         for i in range(len(self.U_list)):
#             self.U_list[i].weight.data = self.U_list[i].weight/torch.sqrt(torch.sum(self.U_list[i].weight**2,dim=1,keepdim=True))

# class Multilayer_Conv_MNIST_Slicer(nn.Module):
#     def __init__(self,k,num_layer):
#         super(Multilayer_Conv_MNIST_Slicer, self).__init__()
#         image_sizes=(1,28,28)
#         self.k=k
#         self.num_layer=num_layer
#         self.U_list=[]
#         if(k==14*14):
#             if(num_layer==2):
#                 self.U=nn.Conv2d(image_sizes[0], image_sizes[0], 4, 2, 1, bias=False)
#             elif(num_layer==3):
#                 self.U=nn.Conv2d(image_sizes[0], image_sizes[0], 4, 2, 1, bias=False)
#
#         elif(k==7*7):
#             self.U = nn.Conv2d(image_sizes[0], image_sizes[0], 18, 2, 1, bias=False)
#         elif (k == 2*2):
#             self.U = nn.Conv2d(image_sizes[0], image_sizes[0], 28, 2, 1, bias=False)
#         elif (k == 2):
#             self.U = nn.Conv2d(image_sizes[0], image_sizes[0], (28,28), 2, (0,1), bias=False)
#
#     def forward(self, x):
#         self.project_parameters()
#         return self.U(x)
#
#     def project_parameters(self):
#         self.U.weight.data = self.U.weight / torch.sqrt(torch.sum(self.U.weight ** 2))

class Conv_MNIST_Slicer(nn.Module):
    def __init__(self,L):
        super(Conv_MNIST_Slicer, self).__init__()
        image_sizes=(1,28,28)
        self.U1 = nn.Conv2d(image_sizes[0], 1*L, kernel_size=4, stride=2, padding=1, bias=False)
        self.U2 = nn.Conv2d(1*L, 1 * L, kernel_size=4, stride=2, padding=1, bias=False,groups=L)
        self.U3 = nn.Conv2d(1*L, 1 * L, kernel_size=7, stride=1, padding=0, bias=False, groups=L)
        self.U1.weight.data = torch.randn(self.U1.weight.shape)
        self.U2.weight.data = torch.randn(self.U2.weight.shape)
        self.U3.weight.data = torch.randn(self.U3.weight.shape)
        self.project_parameters()

    def forward(self, x):
        self.project_parameters()
        return self.U3(self.U2(self.U1(x)))

    def project_parameters(self):
        self.U1.weight.data = self.U1.weight / torch.sqrt(torch.sum(self.U1.weight ** 2,dim=[1,2,3],keepdim=True))
        self.U2.weight.data = self.U2.weight / torch.sqrt(torch.sum(self.U2.weight ** 2,dim=[1,2,3],keepdim=True))
        self.U3.weight.data = self.U3.weight / torch.sqrt(torch.sum(self.U3.weight ** 2,dim=[1,2,3],keepdim=True))
    def reset(self):
        self.U1.weight.data = torch.randn(self.U1.weight.shape)
        self.U2.weight.data = torch.randn(self.U2.weight.shape)
        self.U3.weight.data = torch.randn(self.U3.weight.shape)
        self.project_parameters()


class Conv_MNIST_Slicer_no_sp(nn.Module):
    def __init__(self,L):
        super(Conv_MNIST_Slicer_no_sp, self).__init__()
        image_sizes=(1,28,28)
        self.U1 = nn.Conv2d(image_sizes[0], 1*L, kernel_size=14, stride=1, padding=0, bias=False)
        self.U2 = nn.Conv2d(1*L, 1 * L, kernel_size=7, stride=1, padding=0, bias=False,groups=L)
        self.U3 = nn.Conv2d(1*L, 1 * L, kernel_size=7, stride=1, padding=0, bias=False, groups=L)
        self.U1.weight.data = torch.randn(self.U1.weight.shape)
        self.U2.weight.data = torch.randn(self.U2.weight.shape)
        self.U3.weight.data = torch.randn(self.U3.weight.shape)
        self.project_parameters()

    def forward(self, x):
        self.project_parameters()
        return self.U3(self.U2(self.U1(x)))

    def project_parameters(self):
        self.U1.weight.data = self.U1.weight / torch.sqrt(torch.sum(self.U1.weight ** 2,dim=[1,2,3],keepdim=True))
        self.U2.weight.data = self.U2.weight / torch.sqrt(torch.sum(self.U2.weight ** 2,dim=[1,2,3],keepdim=True))
        self.U3.weight.data = self.U3.weight / torch.sqrt(torch.sum(self.U3.weight ** 2,dim=[1,2,3],keepdim=True))
    def reset(self):
        self.U1.weight.data = torch.randn(self.U1.weight.shape)
        self.U2.weight.data = torch.randn(self.U2.weight.shape)
        self.U3.weight.data = torch.randn(self.U3.weight.shape)
        self.project_parameters()

class Conv_MNIST_Slicer_no_p(nn.Module):
    def __init__(self,L):
        super(Conv_MNIST_Slicer_no_p, self).__init__()
        image_sizes=(1,28,28)
        self.U1 = nn.Conv2d(image_sizes[0], 1*L, kernel_size=2, stride=2, padding=0, bias=False)
        self.U2 = nn.Conv2d(1*L, 1 * L, kernel_size=2, stride=2, padding=0, bias=False,groups=L)
        self.U3 = nn.Conv2d(1*L, 1 * L, kernel_size=7, stride=1, padding=0, bias=False, groups=L)
        self.U1.weight.data = torch.randn(self.U1.weight.shape)
        self.U2.weight.data = torch.randn(self.U2.weight.shape)
        self.U3.weight.data = torch.randn(self.U3.weight.shape)
        self.project_parameters()

    def forward(self, x):
        self.project_parameters()
        return self.U3(self.U2(self.U1(x)))

    def project_parameters(self):
        self.U1.weight.data = self.U1.weight / torch.sqrt(torch.sum(self.U1.weight ** 2,dim=[1,2,3],keepdim=True))
        self.U2.weight.data = self.U2.weight / torch.sqrt(torch.sum(self.U2.weight ** 2,dim=[1,2,3],keepdim=True))
        self.U3.weight.data = self.U3.weight / torch.sqrt(torch.sum(self.U3.weight ** 2,dim=[1,2,3],keepdim=True))
    def reset(self):
        self.U1.weight.data = torch.randn(self.U1.weight.shape)
        self.U2.weight.data = torch.randn(self.U2.weight.shape)
        self.U3.weight.data = torch.randn(self.U3.weight.shape)
        self.project_parameters()


class Conv_MNIST_Slicer_d(nn.Module):
    def __init__(self,L):
        super(Conv_MNIST_Slicer_d, self).__init__()
        image_sizes=(1,28,28)
        self.U1 = nn.Conv2d(image_sizes[0], 1*L, kernel_size=2, stride=1,dilation=14, padding=0, bias=False)
        self.U2 = nn.Conv2d(1*L, 1 * L, kernel_size=2, stride=1,dilation=7, padding=0, bias=False,groups=L)
        self.U3 = nn.Conv2d(1*L, 1 * L, kernel_size=7, stride=1, padding=0, bias=False, groups=L)
        self.U1.weight.data = torch.randn(self.U1.weight.shape)
        self.U2.weight.data = torch.randn(self.U2.weight.shape)
        self.U3.weight.data = torch.randn(self.U3.weight.shape)
        self.project_parameters()

    def forward(self, x):
        self.project_parameters()
        return self.U3(self.U2(self.U1(x)))

    def project_parameters(self):
        self.U1.weight.data = self.U1.weight / torch.sqrt(torch.sum(self.U1.weight ** 2,dim=[1,2,3],keepdim=True))
        self.U2.weight.data = self.U2.weight / torch.sqrt(torch.sum(self.U2.weight ** 2,dim=[1,2,3],keepdim=True))
        self.U3.weight.data = self.U3.weight / torch.sqrt(torch.sum(self.U3.weight ** 2,dim=[1,2,3],keepdim=True))
    def reset(self):
        self.U1.weight.data = torch.randn(self.U1.weight.shape)
        self.U2.weight.data = torch.randn(self.U2.weight.shape)
        self.U3.weight.data = torch.randn(self.U3.weight.shape)
        self.project_parameters()