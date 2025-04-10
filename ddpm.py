import torch
import torch.utils
import torch.utils.data
from tqdm.auto import tqdm
from torch import nn
import argparse
import torch.nn.functional as F
import utils
import dataset
import os
import matplotlib.pyplot as plt
import numpy as np

class NoiseScheduler():
    """
    Noise scheduler for the DDPM model

    Args:
        num_timesteps: int, the number of timesteps
        type: str, the type of scheduler to use
        **kwargs: additional arguments for the scheduler

    This object sets up all the constants like alpha, beta, sigma, etc. required for the DDPM model
    """
    def __init__(self, num_timesteps=50, type="linear", **kwargs):
        self.num_timesteps = num_timesteps
        self.type = type

        if type == "linear":
            self.init_linear_schedule(**kwargs)
        elif type == "cosine":
            print("cosine")
            self.init_cosine_schedule(**kwargs)
            print("cosine done")
        elif type == "sigmoid":
            self.init_sigmoid_schedule(**kwargs)
        else:
            raise NotImplementedError(f"{type} scheduler is not implemented")

    def init_linear_schedule(self, beta_start, beta_end):
        """
        Precompute whatever quantities are required for training and sampling
        """
        self.betas = torch.linspace(beta_start, beta_end, self.num_timesteps, dtype=torch.float32)
        # Compute alphas and the cumulative product (alpha_bar) for each timestep
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
    
    def init_cosine_schedule(self, beta_start, beta_end):
        print("cosine started")
        self.betas = beta_start + 0.5 * (1 + torch.cos(torch.linspace(0, 1, self.num_timesteps) * np.pi)) * (beta_end - beta_start)
        print("cosine betas")
        # Compute alphas and the cumulative product (alpha_bar) for each timestep
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0) 
        print("cosine alphas")


    def init_sigmoid_schedule(self, beta_start, beta_end, k=10):
        t = torch.linspace(0, 1, self.num_timesteps)
        sigmoid = 1 / (1 + torch.exp(-k * (t - 0.5)))
        self.betas = beta_start + sigmoid * (beta_end - beta_start)
        # Compute alphas and the cumulative product (alpha_bar) for each timestep
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def __len__(self):
        return self.num_timesteps
    
class DDPM(nn.Module):
    def __init__(self, n_dim=3, n_steps=200):
        """
        Noise prediction network for the DDPM

        Args:
            n_dim: int, the dimensionality of the data
            n_steps: int, the number of steps in the diffusion process
        We have separate learnable modules for `time_embed` and `model`. `time_embed` can be learned or a fixed function as well

        """
        super(DDPM, self).__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        self.model = nn.Sequential(
            nn.Linear(n_dim + 128, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_dim)
        )
        self.n_dim = n_dim 

    def forward(self, x, t):
        """
        Args:
            x: torch.Tensor, the input data tensor [batch_size, n_dim]
            t: torch.Tensor, the timestep tensor [batch_size]

        Returns:
            torch.Tensor, the predicted noise tensor [batch_size, n_dim]
        """
        t = t.unsqueeze(-1)
        t = t.float()
        time_emb = self.time_embed(t)
        x_input = torch.cat([x, time_emb], dim=1)
        noise_pred = self.model(x_input)
        return noise_pred

    
class ConditionalDDPM(nn.Module):
    def __init__(self, n_classes = 2, n_dim=3, n_steps=200):
        """
        Class dependernt noise prediction network for the DDPM

        Args:
            n_classes: number of classes in the dataset
            n_dim: int, the dimensionality of the data
            n_steps: int, the number of steps in the diffusion process
        We have separate learnable modules for `time_embed` and `model`. `time_embed` can be learned or a fixed function as well

        """
        super(ConditionalDDPM, self).__init__()
        self.n_dim = n_dim
        self.n_classes = n_classes
        self.time_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.label_embed = nn.Embedding(n_classes + 1, 32)
        self.model = nn.Sequential(
            nn.Linear(n_dim + 32 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, n_dim)
        )
        self.class_embed = None

    def forward(self, x, t, y):
        """
        Args:
            x: torch.Tensor, the input data tensor [batch_size, n_dim]
            t: torch.Tensor, the timestep tensor [batch_size]
            y: torch.Tensor, the class label tensor [batch_size]
        Returns:
            torch.Tensor, the predicted noise tensor [batch_size, n_dim]
        """
        t = t.unsqueeze(-1).float()
        time_emb = self.time_embed(t)
        y = y.long()
        label_emb = self.label_embed(y)
        x_input = torch.cat([x, time_emb, label_emb], dim=1)
        noise_pred = self.model(x_input)
        return noise_pred
    
def trainConditional(model, noise_scheduler, dataloader, optimizer, epochs, run_name, drop_prob=0.1):

    device = next(model.parameters()).device
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            x = batch[0]
            y = batch[-1]
            x = x.to(device)
            y = y.to(device)
            batch_size = x.shape[0]
            t = torch.randint(0, len(noise_scheduler), (batch_size,), device=device)
            alpha_bars = noise_scheduler.alpha_bars.to(device)[t].unsqueeze(1)
            noise = torch.randn_like(x)
            x_noisy = x * torch.sqrt(alpha_bars) + noise * torch.sqrt(1.0 - alpha_bars)

            # Randomly drop the condition: with probability drop_prob, set condition to the null index (num_classes)
            y_input = y.clone()
            drop_mask = torch.rand(batch_size, device=device) < drop_prob
            y_input[drop_mask] = model.num_classes  # use the reserved null index

            noise_pred = model(x_noisy, t, y_input)
            loss = F.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), f'{run_name}/conditional_model.pth')

@torch.no_grad()
def sampleConditional(model, n_samples, noise_scheduler, guidance_scale, class_label):

    device = next(model.parameters()).device
    n_dim = model.n_dim
    x = torch.randn(n_samples, n_dim, device=device)
    # Create constant condition tensors:
    y_cond = torch.full((n_samples,), class_label, device=device, dtype=torch.long)
    # For unconditional, use the reserved index
    y_uncond = torch.full((n_samples,), model.num_classes, device=device, dtype=torch.long)

    for t in reversed(range(len(noise_scheduler))):
        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
        noise_pred_cond = model(x, t_batch, y_cond)
        noise_pred_uncond = model(x, t_batch, y_uncond)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        beta_t = noise_scheduler.betas[t].to(device)
        alpha_t = noise_scheduler.alphas[t].to(device)
        alpha_bar_t = noise_scheduler.alpha_bars[t].to(device)
        coef = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
        mean = (1 / torch.sqrt(alpha_t)) * (x - coef * noise_pred)

        if t > 0:
            sigma_t = torch.sqrt(beta_t)
            noise = torch.randn_like(x)
            x = mean + sigma_t * noise
        else:
            x = mean

    return x

class ClassifierDDPM():
    """
    ClassifierDDPM implements a classification algorithm using the DDPM model
    """
    
    def __init__(self, model: ConditionalDDPM, noise_scheduler: NoiseScheduler):
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.device = next(model.parameters()).device
        self.n_dim = model.n_dim

    def __call__(self, x):
        pass

    def predict(self, x):
        """
        Args:
            x: torch.Tensor, the input data tensor [batch_size, n_dim]
        Returns:
            torch.Tensor, the predicted class tensor [batch_size]
        """
        scores = []
        for cls in range(self.model.num_classes):
            y = torch.full((x.size(0),), cls, device=self.device, dtype=torch.long)
            t = torch.full((x.size(0),), len(self.noise_scheduler)-1, device=self.device, dtype=torch.long)
            noise_pred = self.model(x, t, y)
            score = torch.mean(torch.abs(noise_pred)).item()
            scores.append(score)
        pred_class = np.argmin(scores)
        return pred_class


    def predict_proba(self, x):
        """
        Args:
            x: torch.Tensor, the input data tensor [batch_size, n_dim]
        Returns:
            torch.Tensor, the predicted probabilites for each class  [batch_size, n_classes]
        """

        scores = []
        for cls in range(self.model.num_classes):
            y = torch.full((x.size(0),), cls, device=self.device, dtype=torch.long)
            t = torch.full((x.size(0),), len(self.noise_scheduler)-1, device=self.device, dtype=torch.long)
            noise_pred = self.model(x, t, y)
            score = torch.mean(torch.abs(noise_pred)).item()
            scores.append(score)
        inv_scores = np.reciprocal(np.array(scores) + 1e-8)
        proba = inv_scores / np.sum(inv_scores)
        return proba
    

def train(model, noise_scheduler, dataloader, optimizer, epochs, run_name):
    """
    Train the model and save the model and necessary plots

    Args:
        model: DDPM, model to train
        noise_scheduler: NoiseScheduler, scheduler for the noise
        dataloader: torch.utils.data.DataLoader, dataloader for the dataset
        optimizer: torch.optim.Optimizer, optimizer to use
        epochs: int, number of epochs to train the model
        run_name: str, path to save the model
    """
    print("Training the model")
    device = next(model.parameters()).device
    losses = []


    for epoch in range(epochs):
        epoch_loss = 0
        # Wrap the dataloader with tqdm for a progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            x = batch[0]
            x = x.to(device)
            batch_size = x.shape[0]
            t = torch.randint(0, len(noise_scheduler), (batch_size,)).to(device)
            alpha_bars = noise_scheduler.alpha_bars.to(device)[t].unsqueeze(1)
            noise = torch.randn_like(x)
            x_noisy = x * torch.sqrt(alpha_bars) + noise * torch.sqrt(1.0 - alpha_bars)
            noise_pred = model(x_noisy, t)
            loss = F.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # Update tqdm progress bar with current loss
            pbar.set_postfix(loss=loss.item())
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")
        
    torch.save(model.state_dict(), f'{run_name}/model.pth')


@torch.no_grad()
def sample(model, n_samples, noise_scheduler, return_intermediate=False): 
    """
    Sample from the model
    
    Args:
        model: DDPM
        n_samples: int
        noise_scheduler: NoiseScheduler
        return_intermediate: bool
    Returns:
        torch.Tensor, samples from the model [n_samples, n_dim]

    If `return_intermediate` is `False`,
            torch.Tensor, samples from the model [n_samples, n_dim]
    Else
        the function returns all the intermediate steps in the diffusion process as well 
        Return: [[n_samples, n_dim]] x n_steps
        Optionally implement return_intermediate=True, will aid in visualizing the intermediate steps
    """   
    device = next(model.parameters()).device
    n_dim = model.n_dim

    # Start from pure noise
    x = torch.randn(n_samples, n_dim, device=device)
    intermediates = [x.cpu().detach()] if return_intermediate else None

    # Reverse diffusion process
    for t in reversed(range(len(noise_scheduler))):
        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
        # Predict noise for current timestep
        noise_pred = model(x, t_batch)
        beta_t = noise_scheduler.betas[t].to(device)
        alpha_t = noise_scheduler.alphas[t].to(device)
        alpha_bar_t = noise_scheduler.alpha_bars[t].to(device)

        # Compute the mean (mu) of the reverse process
        # According to the DDPM paper, one formulation is:
        #   x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * noise_pred)
        coef = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
        mean = (1 / torch.sqrt(alpha_t)) * (x - coef * noise_pred)

        if t > 0:
            # Add noise scaled by sigma_t = sqrt(beta_t)
            sigma_t = torch.sqrt(beta_t)
            noise = torch.randn_like(x)
            x = mean + sigma_t * noise
        else:
            x = mean

        if return_intermediate:
            intermediates.append(x.cpu().detach())

    if return_intermediate:
        return x, intermediates
    return x

def sampleCFG(model, n_samples, noise_scheduler, guidance_scale, class_label):
    """
    Sample from the conditional model
    
    Args:
        model: ConditionalDDPM
        n_samples: int
        noise_scheduler: NoiseScheduler
        guidance_scale: float
        class_label: int

    Returns:
        torch.Tensor, samples from the model [n_samples, n_dim]
    """
    pass

def sampleSVDD(model, n_samples, noise_scheduler, reward_scale, reward_fn):
    """
    Sample from the SVDD model

    Args:
        model: DDPM
        n_samples: int
        noise_scheduler: NoiseScheduler
        reward_scale: float
        reward_fn: callable, takes in a batch of samples torch.Tensor:[n_samples, n_dim] and returns torch.Tensor[n_samples]

    Returns:
        torch.Tensor, samples from the model [n_samples, n_dim]
    """
    pass
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['train', 'sample'], default='sample')
    parser.add_argument("--n_steps", type=int, default=None)
    parser.add_argument("--lbeta", type=float, default=None)
    parser.add_argument("--ubeta", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dataset", type=str, default = None)
    parser.add_argument("--seed", type=int, default = 42)
    parser.add_argument("--n_dim", type=int, default = None)

    args = parser.parse_args()
    utils.seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_name = f'exps/ddpm_{args.n_dim}_{args.n_steps}_{args.lbeta}_{args.ubeta}_{args.dataset}' # can include more hyperparams
    os.makedirs(run_name, exist_ok=True)

    model = DDPM(n_dim=args.n_dim, n_steps=args.n_steps)
    noise_scheduler = NoiseScheduler(num_timesteps=args.n_steps, beta_start=args.lbeta, beta_end=args.ubeta)
    model = model.to(device)

    if args.mode == 'train':
        epochs = args.epochs
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        data_X, data_y = dataset.load_dataset(args.dataset)
        # can split the data into train and test -- for evaluation later
        data_X = data_X.to(device)
        data_y = data_y.to(device)
        dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data_X, data_y), batch_size=args.batch_size, shuffle=True)
        train(model, noise_scheduler, dataloader, optimizer, epochs, run_name)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(f'{run_name}/model.pth'))
        samples = sample(model, args.n_samples, noise_scheduler)
        torch.save(samples, f'{run_name}/samples_{args.seed}_{args.n_samples}.pth')
    else:
        raise ValueError(f"Invalid mode {args.mode}")