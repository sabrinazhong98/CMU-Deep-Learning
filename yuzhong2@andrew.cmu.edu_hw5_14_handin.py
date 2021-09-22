import torch
import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    """
    Receives z and outputs x
    """
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1, 10)
        self.linear2 = nn.Linear(10, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class Discriminator(nn.Module):
    """
    Receives x from real and from generated and outputs either 1 or 0
    """

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1, 10)
        self.linear2 = nn.Linear(10, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

def gan_loss_discriminator_assignment(Discriminator, Generator, x_real, z):
    """
    Implement original GAN's discriminator loss
    """

    real = Discriminator.forward(x_real)
 
    gen_imgs = Generator.forward(z)
    dis_imgs = Discriminator.forward(gen_imgs)

    logs_real = torch.log(real)
    logs_fake = torch.log(1-dis_imgs)

    returned = -logs_real.mean() + logs_fake.mean()
    

    return returned


def gan_loss_generator_assignment(Discriminator, Generator, z):
    """
    Implement original GAN's generator loss
    """

    gen_imgs = Generator.forward(z)
    dis_imgs = Discriminator.forward(gen_imgs)

    loss = torch.log(1-dis_imgs).mean()
    return loss

def wgan_loss_discriminator_assignment(Discriminator, Generator, x_real, z):
    """
    Imeplement wgan discriminator loss
    """
    
    real = Discriminator.forward(x_real)
 
    gen_imgs = Generator.forward(z)
    dis_imgs = Discriminator.forward(gen_imgs)

    returned = abs(real.sum() + dis_imgs.sum())/len(real)
    
    return returned

from torch.autograd import Variable
import torch.autograd as autograd
def wgan_gradient_penalty_assignment(Discriminator, real_data, fake_data):
    """
    Implement gradient penalty term
    """
    print(real_data, fake_data)
    H, W = real_data.shape
    
    lambda_val = 10
    epsilon = torch.rand(1)
    interpolated = real_data * epsilon + fake_data * (1-epsilon)
    interpolated = interpolated.requires_grad_(True)
    
    dis_interpolated = Discriminator.forward(interpolated)
    dis_fake = Discriminator.forward(fake_data)
    dis_real = Discriminator.forward(real_data)
    
    fake_one = torch.ones(real_data.shape[0],1)
    gradients = autograd.grad(outputs = dis_interpolated,
                              inputs = interpolated,
                              grad_outputs = fake_one,
                              )[0]
    gp = dis_fake - dis_real + lambda_val*(gradients -1)**2

    return gp






