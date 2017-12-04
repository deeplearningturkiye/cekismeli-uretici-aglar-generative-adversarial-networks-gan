import visdom
import numpy as np
import torch
import os
from models_losses import *


# Kullandigim gorsellestirme araci
vis = visdom.Visdom()


def train(epochs, D, G, train_loader, D_optimizer, G_optimizer, batch_size, dtype):

    num_iter = 0
    print('model egitilmeye basladi!')
    if not os.path.isdir('modeller'):
        os.mkdir('modeller')

    for epoch in range(epochs):
        D_losses = []
        G_losses = []

        for x, _ in train_loader: 
            if len(x) != batch_size: # Eger aldigimiz mini batch, batch_size'a esit degilse modele uymadigi icin bunu pas geciyoruz(Aslinda uyuyuyor da karisabiliyor, garanti olsun diyelim)
                continue

            # Ayristirici agin egitimi
            D.zero_grad()  # Onceki backpropagationdan kalan gradientleri sıfırlayarak basliyoruz.

            mini_batch = x.size()[0]
            x = Variable(x).type(dtype)    # Siradaki mini_batch'i Variable icine alıyoruz ki türevlenebilir hale gelsin
            D_real = D(x).squeeze()    # Gercek fotograflari ayristiricidan geciriyoruz

            z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)     # Noise uretimi
            z_ = Variable(z_).type(dtype)
            fake_images = G(z_)

            D_fake = D(fake_images).squeeze()

            D_train_loss = discriminator_loss(D_real, D_fake)   # Urettigimiz gercek ve sahte fotograflarin sınıflarini göre loss hesaplıyoruz

            D_train_loss.backward()     # Ayristirici agin yeni gradientlerini bulduk
            D_optimizer.step()      # Ayristici agin parametrelerini guncelledik

            D_losses.append(D_train_loss.data[0])   # Sonradan grafik cizmek icin kaydediyoruz

            # Uretici agin egitimi
            G.zero_grad()   # Yine onceki backpropagationdan kalan gradientleri sıfırlayarak basliyoruz.

            z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)     # Yine noise uretiyoruz
            z_ = Variable(z_).type(dtype)    # Siradaki mini_batch'i Variable icine alıyoruz ki türevlenebilir hale gelsin

            fake_images = G(z_)     # Tekrardan sahte fotograflar uretiyoruz
            D_result = D(fake_images).squeeze()
            G_train_loss = generator_loss(D_result)     # Urettigimiz fotograflarin basarimini olcuyoruz
            G_train_loss.backward()     # Sonuclara gore parametrelerin turevlerini hesaplıyoruz
            G_optimizer.step()  # Parametreleri guncelledik

            G_losses.append(G_train_loss.data[0])

            num_iter += 1
            if num_iter % 250 == 249:
                    img_np = fake_images.data.cpu().numpy()
                    vis.images(img_np[0:15])
                    print('[%d/%d], loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), epochs,  torch.mean(torch.FloatTensor(D_losses)),
                                                                   torch.mean(torch.FloatTensor(G_losses))))

        # Hesaplanan yeni parametreleri epoch basina kayediyoruz
        torch.save(G.state_dict(), './modeller/' + str(epoch) + ' G')
        torch.save(D.state_dict(), './modeller/' + str(epoch) + ' D')


def show_images(images):
    images = np.reshape(images, [images.shape[0], -1])
    vis.images(images.reshape([16, 1, 64, 64]))


def preprocess_img(x):
    return 2 * x - 1.0


def deprocess_img(x):
    return (x + 1.0) / 2
