import torch
from torchvision import datasets, transforms

# load the dataset

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

train_dataset = datasets.MNIST(
    root='data',
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.MNIST(
    root='data',
    train=False,
    transform=transform,
    download=True
)

import matplotlib.pyplot as plt

img = train_dataset[0][0].numpy().squeeze()
print(img.shape)
plt.imshow(img, cmap='gray')


import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.fc = nn.Linear(latent_dim, 128 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ReLU(),
            nn.Unflatten(1, (128, 7, 7)),  # Reshape to (128, 7, 7)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.Tanh()  # Output in range [-1, 1]
        )

    def forward(self, z):
        x = self.fc(z)
        img = self.deconv(x)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 14x14 -> 7x7
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1)
        )

    def forward(self, img):
        x = self.conv(img)
        return self.fc(x), x

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, SubsetRandomSampler, RandomSampler
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import grad
import os

random.seed(42)
torch.manual_seed(42)

class Trainer:
    def __init__(self, exp, k, m, latent_size, train_dataset, debug=False):
        self.exp = exp
        self.k = k
        self.m = m
        self.debug = debug
        self.latent_size = latent_size
        sampler = RandomSampler(train_dataset, replacement=False)
        self.train_dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=m)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = Generator(latent_dim=latent_size).to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.generator_optim = Adam(self.generator.parameters(), lr=1e-4, betas=(0, 0.9))
        self.discriminator_optim = Adam(self.discriminator.parameters(), lr=1e-4, betas=(0, 0.9))
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Tracking losses and gradients
        self.discriminator_losses = []
        self.generator_losses = []
        self.generator_gradients = []
        self.discriminator_gradients = []
        self.gradient_norms = []
    
    def train(self, iterations):
        data_iterator = iter(self.train_dataloader)
        for itr in tqdm(range(iterations)):
            torch.cuda.empty_cache()

            # Train the discriminator
            total_disc_loss = 0
            for _ in range(self.k):
                try:
                    sample = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(self.train_dataloader)
                    sample = next(data_iterator)
                real_data = sample[0].to(self.device)
                # real_data = real_data + 0.1 * torch.randn_like(real_data)  # Add noise to real data
                noise_samples = torch.randn(real_data.size(0), self.latent_size).to(self.device)
                d_loss = self.train_discriminator(real_data, noise_samples)
                if self.debug:
                    total_disc_loss += d_loss.item()
            
            if self.debug:
                self.discriminator_losses.append(float(total_disc_loss) / self.k)

            # Train the generator
            noise_samples = torch.randn(self.m, self.latent_size).to(self.device)
            g_loss = self.train_generator(real_data, noise_samples)
            if self.debug:
                self.generator_losses.append(g_loss.item())

            if self.debug:
                if (itr+1) % 500 == 0:
                    self.save_loss_graphs(directory=f'iteration_{itr}')
                if (itr+1) % 5000 == 0:
                    os.makedirs(f'/home/ubuntu/generative-adversarial-networks/notebooks/{self.exp}/{itr}', exist_ok=True)
                    torch.save(self.generator.state_dict(), f'/home/ubuntu/generative-adversarial-networks/notebooks/{self.exp}/{itr}/generator.pt')
                    torch.save(self.discriminator.state_dict(), f'/home/ubuntu/generative-adversarial-networks/notebooks/{self.exp}/{itr}/discriminator.pt')
                    torch.save(self.generator_optim.state_dict(), f'/home/ubuntu/generative-adversarial-networks/notebooks/{self.exp}/{itr}/generator_optim.pt')
                    torch.save(self.discriminator_optim.state_dict(), f'/home/ubuntu/generative-adversarial-networks/notebooks/{self.exp}/{itr}/discriminator_optim.pt')

        if self.debug:
            self.save_loss_graphs('final')
            os.makedirs(f'/home/ubuntu/generative-adversarial-networks/notebooks/{self.exp}/{iterations-1}', exist_ok=True)
            torch.save(self.generator.state_dict(), f'/home/ubuntu/generative-adversarial-networks/notebooks/{self.exp}/{iterations-1}/generator.pt')
            torch.save(self.discriminator.state_dict(), f'/home/ubuntu/generative-adversarial-networks/notebooks/{self.exp}/{iterations-1}/discriminator.pt')
            torch.save(self.generator_optim.state_dict(), f'/home/ubuntu/generative-adversarial-networks/notebooks/{self.exp}/{iterations-1}/generator_optim.pt')
            torch.save(self.discriminator_optim.state_dict(), f'/home/ubuntu/generative-adversarial-networks/notebooks/{self.exp}/{iterations-1}/discriminator_optim.pt')

    def train_discriminator(self, real_data, noise_samples):
        fake_data = self.generator(noise_samples)
        # fake_data = fake_data + 0.1 * torch.randn_like(fake_data)  # Add noise to fake data
        real_labels = torch.ones(real_data.size(0), 1).to(self.device)
        fake_labels = torch.zeros(fake_data.size(0), 1).to(self.device)

        real_output, _ = self.discriminator(real_data)
        fake_output, _ = self.discriminator(fake_data)

        # real_loss = self.loss_fn(real_output, real_labels)
        # fake_loss = self.loss_fn(fake_output, fake_labels)

        d_loss = -torch.mean(real_output) + torch.mean(fake_output)

        # Gradient penalty calculation
        alpha = torch.rand(real_data.size(0), 1, 1, 1).to(self.device)
        interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
        disc_interpolates, _ = self.discriminator(interpolates)

        gradients = grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients_norm = gradients.view(gradients.size(0), -1).norm(2, dim=1)
        self.gradient_norms.append(gradients_norm.mean().item())  # Track gradient norm

        lambda_gp = 10
        gp = lambda_gp * ((gradients_norm - 1) ** 2).mean()

        # loss = real_loss + fake_loss + gp
        loss = d_loss + gp

        self.discriminator_optim.zero_grad()
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=10)

        # Track discriminator gradients
        total_grad_norm = 0
        for param in self.discriminator.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm(2).item()
        self.discriminator_gradients.append(total_grad_norm)

        self.discriminator_optim.step()
        
        return loss

    def train_generator(self, real_data, noise_samples):
        fake_data = self.generator(noise_samples)
        fake_labels = torch.ones(fake_data.size(0), 1).to(self.device)
        fake_output, fake_features = self.discriminator(fake_data)
        real_output, real_features = self.discriminator(real_data)

        # loss = self.loss_fn(fake_output, fake_labels)
        # loss = -torch.mean(fake_output)
        loss = torch.mean((real_features.mean(dim=0) - fake_features.mean(dim=0)) ** 2)
        self.generator_optim.zero_grad()
        loss.backward()

        # Track generator gradients
        total_grad_norm = 0
        for param in self.generator.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm(2).item()
        self.generator_gradients.append(total_grad_norm)

        self.generator_optim.step()
        
        return loss
    
    def save_loss_graphs(self, directory):
        directory = f'/home/ubuntu/generative-adversarial-networks/notebooks/{self.exp}/{directory}'
        os.makedirs(directory, exist_ok=True)

        # Save Discriminator Loss
        plt.figure(figsize=(10, 5))
        plt.plot(self.discriminator_losses, label='Discriminator Loss')
        plt.title('Discriminator Loss Over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(directory, 'discriminator_loss.png'))
        plt.close()

        # Save Generator Loss
        plt.figure(figsize=(10, 5))
        plt.plot(self.generator_losses, label='Generator Loss')
        plt.title('Generator Loss Over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(directory, 'generator_loss.png'))
        plt.close()

        # Save Discriminator Gradients
        plt.figure(figsize=(10, 5))
        plt.plot(self.discriminator_gradients, label='Discriminator Gradients')
        plt.title('Discriminator Gradient Norm Over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Gradient Norm')
        plt.legend()
        plt.savefig(os.path.join(directory, 'discriminator_gradients.png'))
        plt.close()

        # Save Generator Gradients
        plt.figure(figsize=(10, 5))
        plt.plot(self.generator_gradients, label='Generator Gradients')
        plt.title('Generator Gradient Norm Over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Gradient Norm')
        plt.legend()
        plt.savefig(os.path.join(directory, 'generator_gradients.png'))
        plt.close()

        # Save Gradient Norms
        plt.figure(figsize=(10, 5))
        plt.plot(self.gradient_norms, label='Gradient Norm (Gradient Penalty)')
        plt.title('Gradient Norm Over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Gradient Norm')
        plt.legend()
        plt.savefig(os.path.join(directory, 'gradient_norm.png'))
        plt.close()

trainer = Trainer('baseline_wasserstein_k5_50k_feature_matching_loss', 5, 128, 100, train_dataset, debug=True)
trainer.train(50000)