import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, SubsetRandomSampler, RandomSampler
from tqdm import tqdm
import random
from torch.autograd import grad
import os
import shutil

random.seed(42)
torch.manual_seed(42)

class Generator(nn.Module):
    """
    A PyTorch implementation of a Generator model for a Generative Adversarial Network (GAN).

    Args:
        latent_dim (int): Dimensionality of the latent space (default: 100).
        fc_layer_output_size (tuple): Output size of the fully connected layer before deconvolution layers (default: (128, 7, 7)).
        deconv_layer_params (list): List of tuples containing parameters for deconvolution layers. Each tuple should contain
                                    (in_channels, out_channels, kernel_size, stride, padding) (default: [(128, 64, 4, 2, 1), (64, 1, 4, 2, 1)]).

    Attributes:
        latent_dim (int): Dimensionality of the latent space.
        fc (nn.Linear): Fully connected layer that maps the latent space to the intermediate feature space.
        deconv (nn.Sequential): Sequential container of deconvolution layers and activation functions.

    Methods:
        forward(z):
            Forward pass through the generator network.
            Args:
                z (torch.Tensor): Input tensor from the latent space.
            Returns:
                torch.Tensor: Generated image tensor.
    """
    def __init__(self, latent_dim=100, fc_layer_output_size=(128, 7, 7), deconv_layer_params=[(128, 64, 4, 2, 1), (64, 1, 4, 2, 1)]):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Linear(latent_dim, fc_layer_output_size[0] * fc_layer_output_size[1] * fc_layer_output_size[2])
        deconv_layers = []
        for i, (in_channels, out_channels, kernel_size, stride, padding) in enumerate(deconv_layer_params):
            deconv_layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
            if i != len(deconv_layer_params) - 1:
                deconv_layers.append(nn.ReLU())
        self.deconv = nn.Sequential(
            nn.ReLU(),
            nn.Unflatten(1, fc_layer_output_size),
            *deconv_layers,
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        img = self.deconv(x)
        return img

class Discriminator(nn.Module):
    """
    A PyTorch implementation of a Discriminator network for a Generative Adversarial Network (GAN).

    Args:
        conv_layer_params (list of tuples): A list where each tuple contains parameters for a convolutional layer.
            Each tuple should have the format (in_channels, out_channels, kernel_size, stride, padding).
        fc_layer_input_channels (int): The number of input channels for the fully connected layer.

    Attributes:
        conv (nn.Sequential): A sequential container of convolutional layers followed by ReLU activations.
        fc (nn.Sequential): A sequential container of a flatten layer followed by a fully connected layer.

    Methods:
        forward(img):
            Defines the computation performed at every call.
            Args:
                img (torch.Tensor): Input image tensor.
            Returns:
                torch.Tensor: Output tensor after passing through the network.
    """
    def __init__(self, conv_layer_params=[(1, 64, 4, 2, 1), (64, 128, 4, 2, 1)], fc_layer_input_channels=128*7*7):
        super(Discriminator, self).__init__()
        conv_layers = []
        for i, (in_channels, out_channels, kernel_size, stride, padding) in enumerate(conv_layer_params):
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
            conv_layers.append(nn.ReLU())
        self.conv = nn.Sequential(
            *conv_layers
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_layer_input_channels, 1)
        )

    def forward(self, img):
        x = self.conv(img)
        return self.fc(x)

class Trainer:
    """
    Trainer class for training a Generative Adversarial Network (GAN).
    Args:
        output_dir (str): Directory to save outputs.
        log_steps (int): Number of steps between logging.
        ckpt_steps (int): Number of steps between saving checkpoints.
        exp (str): Experiment name.
        k (int): Number of discriminator updates per generator update.
        m (int): Batch size.
        generator (nn.Module): Generator model.
        discriminator (nn.Module): Discriminator model.
        train_dataset (Dataset): Training dataset.
        lr_gen (float, optional): Learning rate for the generator. Default is 1e-4.
        lr_disc (float, optional): Learning rate for the discriminator. Default is 1e-4.
        betas_gen (list, optional): Betas for the Adam optimizer of the generator. Default is [0, 0.9].
        betas_disc (list, optional): Betas for the Adam optimizer of the discriminator. Default is [0, 0.9].
        debug (bool, optional): If True, enables debug mode. Default is False.
    Methods:
        train(iterations):
            Trains the GAN for a specified number of iterations.
        train_discriminator(real_data, noise_samples):
            Trains the discriminator with real and fake data.
        train_generator(noise_samples):
            Trains the generator with noise samples.
        save_loss_graphs(directory):
            Saves loss and gradient graphs to the specified directory.
    """
    def __init__(self, output_dir, log_steps, ckpt_steps, exp, k, m, generator, discriminator, train_dataset, lr_gen=1e-4, lr_disc=1e-4, betas_gen=[0, 0.9], betas_disc=[0, 0.9], debug=False):
        self.output_dir = output_dir
        self.log_steps = log_steps
        self.ckpt_steps = ckpt_steps
        self.exp = exp
        self.k = k
        self.m = m
        self.debug = debug
        sampler = RandomSampler(train_dataset, replacement=False)
        self.train_dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=m)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.generator_optim = Adam(self.generator.parameters(), lr=lr_gen, betas=betas_gen)
        self.discriminator_optim = Adam(self.discriminator.parameters(), lr=lr_disc, betas=betas_disc)
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.discriminator_losses = []
        self.generator_losses = []
        self.generator_gradients = []
        self.discriminator_gradients = []
        self.gradient_norms = []
    
    def train(self, iterations):
        data_iterator = iter(self.train_dataloader)
        for itr in tqdm(range(iterations)):
            torch.cuda.empty_cache()

            total_disc_loss = 0
            for _ in range(self.k):
                try:
                    sample = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(self.train_dataloader)
                    sample = next(data_iterator)
                real_data = sample[0].to(self.device)
                noise_samples = torch.randn(real_data.size(0), self.generator.latent_dim).to(self.device)
                d_loss = self.train_discriminator(real_data, noise_samples)
                if self.debug:
                    total_disc_loss += d_loss.item()
            
            if self.debug:
                self.discriminator_losses.append(float(total_disc_loss) / self.k)

            noise_samples = torch.randn(self.m, self.generator.latent_dim).to(self.device)
            g_loss = self.train_generator(noise_samples)
            if self.debug:
                self.generator_losses.append(g_loss.item())

            if self.debug:
                if (itr+1) % self.log_steps == 0:
                    self.save_loss_graphs(directory=f'iteration_{itr}')
                if (itr+1) % self.ckpt_steps == 0:
                    os.makedirs(os.path.join(self.output_dir, self.exp, str(itr)), exist_ok=True)
                    torch.save(self.generator.state_dict(), os.path.join(self.output_dir, self.exp, str(itr), 'generator.pt'))
                    torch.save(self.discriminator.state_dict(), os.path.join(self.output_dir, self.exp, str(itr), 'discriminator.pt'))
                    torch.save(self.generator_optim.state_dict(), os.path.join(self.output_dir, self.exp, str(itr), 'generator_optim.pt'))
                    torch.save(self.discriminator_optim.state_dict(), os.path.join(self.output_dir, self.exp, str(itr), 'discriminator_optim.pt'))

        if self.debug:
            self.save_loss_graphs('final')
            os.makedirs(os.path.join(self.output_dir, self.exp, str(iterations-1)), exist_ok=True)
            torch.save(self.generator.state_dict(), os.path.join(self.output_dir, self.exp, str(iterations-1), 'generator.pt'))
            torch.save(self.discriminator.state_dict(), os.path.join(self.output_dir, self.exp, str(iterations-1), 'discriminator.pt'))
            torch.save(self.generator_optim.state_dict(), os.path.join(self.output_dir, self.exp, str(iterations-1), 'generator_optim.pt'))
            torch.save(self.discriminator_optim.state_dict(), os.path.join(self.output_dir, self.exp, str(iterations-1), 'discriminator_optim.pt'))

    def train_discriminator(self, real_data, noise_samples):
        fake_data = self.generator(noise_samples)
        real_labels = torch.ones(real_data.size(0), 1).to(self.device)
        fake_labels = torch.zeros(fake_data.size(0), 1).to(self.device)

        real_output = self.discriminator(real_data)
        fake_output = self.discriminator(fake_data)

        d_loss = -torch.mean(real_output) + torch.mean(fake_output)

        alpha = torch.rand(real_data.size(0), 1, 1, 1).to(self.device)
        interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
        disc_interpolates = self.discriminator(interpolates)

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

        loss = d_loss + gp

        self.discriminator_optim.zero_grad()
        loss.backward()

        # Track discriminator gradients
        total_grad_norm = 0
        for param in self.discriminator.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm(2).item()
        self.discriminator_gradients.append(total_grad_norm)

        self.discriminator_optim.step()
        
        return d_loss

    def train_generator(self, noise_samples):
        fake_data = self.generator(noise_samples)
        fake_labels = torch.ones(fake_data.size(0), 1).to(self.device)
        fake_output = self.discriminator(fake_data)

        loss = -torch.mean(fake_output)
        self.generator_optim.zero_grad()
        loss.backward()

        total_grad_norm = 0
        for param in self.generator.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm(2).item()
        self.generator_gradients.append(total_grad_norm)

        self.generator_optim.step()
        
        return loss
    
    def save_loss_graphs(self, directory):
        directory = os.path.join(self.output_dir, self.exp, directory)
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

def train(config, train_dataset):
    """
    Trains a Generative Adversarial Network (GAN) using the provided configuration and training dataset.
    Args:
        config (dict): Configuration dictionary containing parameters for the generator, discriminator, and trainer.
        train_dataset (Dataset): The dataset to be used for training the GAN.
    The configuration dictionary should have the following structure:
    {
        'generator': {
            'latent_dim': int,
            'fc_layer': {
                'output_channels': int,
                'output_height': int,
                'output_width': int
            },
            'deconv_layer_params': {
                'layer_name': {
                    'input_channels': int,
                    'output_channels': int,
                    'kernel_size': int,
                    'stride': int,
                    'padding': int
                },
                ...
            }
        },
        'discriminator': {
            'conv_layer_params': {
                'layer_name': {
                    'input_channels': int,
                    'output_channels': int,
                    'kernel_size': int,
                    'stride': int,
                    'padding': int
                },
                ...
            },
            'fc_layer': {
                'input_channels': int
            }
        },
        'trainer': {
            'output_dir': str,
            'log_steps': int,
            'ckpt_steps': int,
            'experiment': str,
            'k': int,
            'batch_size': int,
            'lrs': {
                'lr_gen': float,
                'lr_disc': float
            },
            'betas': {
                'betas_gen': tuple,
                'betas_disc': tuple
            },
            'ckpt_dir': str,
            'iterations': int
        }
    }
    The function performs the following steps:
    1. Creates the generator and discriminator models based on the provided configuration.
    2. Initializes the trainer with the models, training dataset, and training parameters.
    3. Loads the latest checkpoint if available.
    4. Deletes the experiment directory if it exists.
    5. Trains the GAN for the specified number of iterations.
    """
    # Create the generator
    gen_config = config['generator']
    fc_layer_output_size = (gen_config['fc_layer']['output_channels'], gen_config['fc_layer']['output_height'], gen_config['fc_layer']['output_width'])
    deconv_layer_params = [(deconv_layer['input_channels'], deconv_layer['output_channels'], deconv_layer['kernel_size'], deconv_layer['stride'], deconv_layer['padding']) for deconv_layer in gen_config['deconv_layer_params'].values()]
    generator = Generator(latent_dim=gen_config['latent_dim'], fc_layer_output_size=fc_layer_output_size, deconv_layer_params=deconv_layer_params)

    # Create the discriminator
    disc_config = config['discriminator']
    conv_layer_params = [(conv_layer['input_channels'], conv_layer['output_channels'], conv_layer['kernel_size'], conv_layer['stride'], conv_layer['padding']) for conv_layer in disc_config['conv_layer_params'].values()]
    fc_layer_input_channels = disc_config['fc_layer']['input_channels']
    discriminator = Discriminator(conv_layer_params=conv_layer_params, fc_layer_input_channels=fc_layer_input_channels)

    # Create the trainer
    trainer_config = config['trainer']
    trainer = Trainer(output_dir=trainer_config['output_dir'], log_steps=trainer_config['log_steps'], ckpt_steps=trainer_config['ckpt_steps'], exp=trainer_config['experiment'], k=trainer_config['k'], m=trainer_config['batch_size'], generator=generator, discriminator=discriminator, train_dataset=train_dataset, lr_gen=trainer_config['lrs']['lr_gen'], lr_disc=trainer_config['lrs']['lr_disc'], betas_gen=trainer_config['betas']['betas_gen'], betas_disc=trainer_config['betas']['betas_disc'], debug=True)

    # Load checkpoint if available
    ckpt_dir = trainer_config['ckpt_dir']
    if os.path.exists(ckpt_dir):
        print('Loading checkpoint...')
        trainer.generator.load_state_dict(torch.load(os.path.join(ckpt_dir, 'generator.pt')))
        trainer.discriminator.load_state_dict(torch.load(os.path.join(ckpt_dir, 'discriminator.pt')))
        trainer.generator_optim.load_state_dict(torch.load(os.path.join(ckpt_dir, 'generator_optim.pt')))
        trainer.discriminator_optim.load_state_dict(torch.load(os.path.join(ckpt_dir, 'discriminator_optim.pt')))

    # Delete the experiment directory if it exists
    experiment_dir = os.path.join(trainer.output_dir, trainer.exp)
    if os.path.exists(experiment_dir):
        shutil.rmtree(experiment_dir)
    
    # Train the GAN
    trainer.train(trainer_config['iterations'])