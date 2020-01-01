import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from Net import Encoder, Decoder
from torchvision.utils import save_image
import matplotlib.pyplot as plt

if not os.path.exists('./img'):
    os.mkdir('./img')
if not os.path.exists('./params'):
    os.mkdir('./params')

mnist_data = datasets.MNIST(root='E:\AI\Seq2Seq\data', train=True, transform=transforms.ToTensor(), download=False)
dataloader = DataLoader(mnist_data, batch_size=100, shuffle=True)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

en_net = Encoder().to(device)
de_net = Decoder().to(device)

en_net.train()
# en_net.eval()
de_net.train()

if os.path.exists('./params/en_params.pth'):
    print('编码器网络已存在，继续训练！')
    en_net.load_state_dict(torch.load('./params/en_params.pth'))
if os.path.exists('./params/de_params.pth'):
    print('解码器网络已存在，继续训练！')
    de_net.load_state_dict(torch.load('./params/de_params.pth'))

# 创建重构损失
de_lossfn = nn.MSELoss(reduction='sum')  # reduction默认为mean，改为sum
en_opt = torch.optim.Adam(en_net.parameters())
de_opt = torch.optim.Adam(de_net.parameters())

for epoch in range(10):
    for i, (img, _) in enumerate(dataloader):
        img = img.to(device)
        miu, logsigma = en_net(img)

        # 计算KL损失
        en_loss = torch.mean(0.5 * (-torch.log(logsigma ** 2) + miu ** 2 + logsigma ** 2 - 1))

        z = torch.randn(128).to(device)
        img_out = de_net(z, logsigma, miu)

        de_loss = de_lossfn(img_out, img)
        loss = en_loss + de_loss

        en_opt.zero_grad()
        de_opt.zero_grad()
        loss.backward()
        en_opt.step()
        de_opt.step()

        if i % 10 == 0:
            print('Epoch:{}/{} En_loss:{} De_loss:{} loss:{}'.format(i, epoch, en_loss, de_loss, loss))

    imgs = img_out.cpu().data
    show_imgs = imgs.permute([0, 2, 3, 1])
    show_img = show_imgs[0].reshape(28, 28)
    plt.imshow(show_img)
    plt.pause(1)

    fake_img = img_out.cpu().data
    real_img = img.cpu().data

    save_image(fake_img, './img/{}-fake_img.jpg'.format(epoch + 1), nrow=10, normalize=True, scale_each=True)
    save_image(real_img, './img/{}-real_img.jpg'.format(epoch + 1), nrow=10, normalize=True, scale_each=True)

    torch.save(en_net.state_dict(), './params/en_params.pth')
    torch.save(de_net.state_dict(), './params/de_params.pth')