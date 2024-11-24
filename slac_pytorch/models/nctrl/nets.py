import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from functorch import grad
from torch.autograd.functional import jacobian
import torch.distributions as tD

from slac_pytorch.models.slac.initializer import initialize_weight


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()

def reparametrize(mu, logvar):
    print('logvar: ', logvar.shape)
    std = logvar.div(2).exp()
    eps = std.data.new(std.size()).normal_()

    print('mu', mu.shape)
    print('std: ', std.shape)
    return mu + std*eps


class MLP(nn.Module):
    """A simple MLP with ReLU activations"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, leaky_relu_slope=0.2):
        super().__init__()
        layers = []
        for l in range(num_layers):
            if l == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.LeakyReLU(leaky_relu_slope))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.LeakyReLU(leaky_relu_slope))
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class BetaVAE_CNN(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self,input_dim=3, z_dim=10, nc=3, hidden_dim=256):
        super(BetaVAE_CNN, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            # (1, 1, 17) -> (32, 32, 32)
            nn.Conv2d(input_dim, 32, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (32, 32, 32) -> (64, 16, 16)
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (64, 16, 16) -> (128, 8, 8)
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (128, 8, 8) -> (256, 4, 4)
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (256, 4, 4) -> (256, 1, 1)
            nn.Conv2d(256, hidden_dim, 4),
            nn.LeakyReLU(0.2, inplace=True),
            View((-1, hidden_dim*1*1)),                 # B, hidden_dim
            nn.Linear(hidden_dim, z_dim*2),             # B, z_dim*2

        )


        
        # nn.Sequential(
        #     nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 64, 64
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 32, 32
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(True),
        #     nn.Conv2d(64, hidden_dim, 4, 1),            # B, hidden_dim,  1,  1
        #     nn.BatchNorm2d(hidden_dim),
        #     nn.ReLU(True),
        #     View((-1, hidden_dim*1*1)),                 # B, hidden_dim
        #     nn.Linear(hidden_dim, z_dim*2),             # B, z_dim*2
        # )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),               # B, hidden_dim
            View((-1, hidden_dim, 1, 1)),               # B, hidden_dim,  1,  1
            nn.ReLU(True),
            # (32+256, 1, 1) -> (256, 4, 4)
            nn.ConvTranspose2d(hidden_dim, 256, 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (256, 4, 4) -> (128, 8, 8)
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (128, 8, 8) -> (64, 16, 16)
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (64, 16, 16) -> (32, 32, 32)
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (32, 32, 32) -> (1, 1, 17)
            nn.ConvTranspose2d(32, input_dim, 5, 2, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ).apply(initialize_weight)
        # nn.Sequential(
        #     nn.Linear(z_dim, hidden_dim),               # B, hidden_dim
        #     View((-1, hidden_dim, 1, 1)),               # B, hidden_dim,  1,  1
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(hidden_dim, 64, 4),      # B,  64,  4,  4
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 64, 64
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 128, 128
        # )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, return_z=True):
        B, S, C, H, W = x.size()
        self.B = B
        self.S = S
        x = x.view(B * S, C, H, W)
        x = self.encoder(x)
        x = x.view(B, S, -1)
        print('self.z_dim', self.z_dim)
        print('x', x.shape)
        mu = x[..., :self.z_dim]
        print('x', mu.shape)

        logvar = x[..., self.z_dim:]

        logvar = logvar - 2
        print('x', logvar.shape)

        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        if return_z:
            return x_recon, mu, logvar, z
        else:
            return x_recon, mu, logvar

    def encode(self, x):
        B, S, C, H, W = x.size()
        self.B = B
        self.S = S
        self.C = C
        self.H = H
        self.W = W
        x = x.view(B * S, C, H, W)
        x = self.encoder(x)
        x = x.view(B, S, -1)

        mu = x[..., :self.z_dim]
        logvar = x[..., self.z_dim:]
        logvar = logvar - 2
        z = reparametrize(mu, logvar)

        return z, mu, logvar

    def decode(self, z):   
        print('decoder z shape: ', z.shape)   
        x = self.decoder(z)
        x = x.view(self.B, self.S, self.C, self.H, self.W)
        print('decoder x.shape: ', x.shape)  

        return x


class BetaVAE_MLP(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, input_dim=3, z_dim=10, hidden_dim=128, leaky_relu_slope=0.2):
        super(BetaVAE_MLP, self).__init__()
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(hidden_dim, 2*z_dim)
        )
        # Fix the functional form to ground-truth mixing function
        self.decoder = nn.Sequential(
            nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(z_dim, hidden_dim),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(hidden_dim, input_dim)
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, return_z=True):

        distributions = self._encode(x)
        mu = distributions[..., :self.z_dim]
        logvar = distributions[..., self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        if return_z:
            return x_recon, mu, logvar, z
        else:
            return x_recon, mu, logvar

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


class NPTransitionPrior(nn.Module):

    def __init__(
            self,
            lags,
            latent_size,
            num_layers=3,
            hidden_dim=64):
        super().__init__()
        self.lags = lags
        self.latent_size = latent_size
        self.gs = nn.ModuleList([MLP(input_dim=lags*latent_size + 1, hidden_dim=hidden_dim,
                                output_dim=1,  num_layers=num_layers) for _ in range(latent_size)])
        # self.fc = MLP(input_dim=embedding_dim,hidden_dim=hidden_dim, output_dim=hidden_dim, num_layers=2)

    def forward(self, x, mask=None):
        batch_size, lags_and_length, x_dim = x.shape
        length = lags_and_length - self.lags
        # batch_x: (batch_size, lags+length, x_dim) -> (batch_size, length, lags+1, x_dim)
        batch_x = x.unfold(dimension=1, size=self.lags +
                           1, step=1).transpose(2, 3)
        batch_x = batch_x.reshape(-1, self.lags+1, x_dim)
        batch_x_lags = batch_x[:, :-1]  # (batch_size x length, lags, x_dim)
        batch_x_t = batch_x[:, -1]  # (batch_size*length, x_dim)
        # (batch_size*length, lags*x_dim)
        batch_x_lags = batch_x_lags.reshape(-1, self.lags * x_dim)
        sum_log_abs_det_jacobian = 0
        residuals = []
        for i in range(self.latent_size):
            # (batch_size x length, hidden_dim + lags*x_dim + 1)
            
            if mask is not None:
                batch_inputs = torch.cat(
                    (batch_x_lags*mask[i], batch_x_t[:, i:i+1]), dim=-1)
            else:
                batch_inputs = torch.cat(
                (batch_x_lags, batch_x_t[:, i:i+1]), dim=-1)
            residual = self.gs[i](batch_inputs)  # (batch_size x length, 1)

            J = torch.funcjacfwd(self.gs[i])
            data_J = torch,vmap(J)(batch_inputs).squeeze()
            logabsdet = torch.log(torch.abs(data_J[:, -1]))

            sum_log_abs_det_jacobian += logabsdet
            residuals.append(residual)
        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, length, x_dim)
        log_abs_det_jacobian = sum_log_abs_det_jacobian.reshape(batch_size, length)
        return residuals, log_abs_det_jacobian


class NPChangeTransitionPrior(nn.Module):

    def __init__(
            self,
            lags,
            latent_size,
            embedding_dim,
            num_layers=3,
            hidden_dim=64):
        super().__init__()
        self.latent_size = latent_size
        self.lags = lags
        self.gs = nn.ModuleList([MLP(input_dim=hidden_dim+lags*latent_size + 1, hidden_dim=hidden_dim,
                                output_dim=1,  num_layers=num_layers) for _ in range(latent_size)])
        self.fc = MLP(input_dim=embedding_dim, hidden_dim=hidden_dim,
                      output_dim=hidden_dim, num_layers=num_layers)

    def forward(self, x, embeddings):
        batch_size, lags_and_length, x_dim = x.shape
        length = lags_and_length - self.lags
        # batch_x: (batch_size, lags+length, x_dim) -> (batch_size, length, lags+1, x_dim)
        batch_x = x.unfold(dimension=1, size=self.lags +
                           1, step=1).transpose(2, 3)
        # (batch_size, lags+length, hidden_dim)
        embeddings = self.fc(embeddings)
        # batch_embeddings: (batch_size, lags+length, hidden_dim) -> (batch_size, length, lags+1, hidden_dim) -> (batch_size*length, hidden_dim)
        # batch_embeddings = embeddings.unfold(
        #     dimension=1, size=self.lags+1, step=1).transpose(2, 3)[:, :, -1].reshape(batch_size * length, -1)
        batch_embeddings = embeddings[:, -length:].expand(batch_size,length,-1).reshape(batch_size*length,-1)
        batch_x = batch_x.reshape(-1, self.lags+1, x_dim)
        batch_x_lags = batch_x[:, :-1]  # (batch_size x length, lags, x_dim)
        batch_x_t = batch_x[:, -1:]  # (batch_size*length, x_dim)
        # (batch_size*length, lags*x_dim)
        batch_x_lags = batch_x_lags.reshape(-1, self.lags * x_dim)
        sum_log_abs_det_jacobian = 0
        residuals = []
        print('self.latent_size', self.latent_size)
        for i in range(self.latent_size):
            # (batch_size x length, hidden_dim + lags*x_dim + 1)
            batch_inputs = torch.cat(
                (batch_embeddings, batch_x_lags, batch_x_t[:, :, i]), dim=-1)
            residual = self.gs[i](batch_inputs)  # (batch_size x length, 1)

            J = torch.func.jacfwd(self.gs[i])
            data_J = torch.vmap(J)(batch_inputs).squeeze()
            logabsdet = torch.log(torch.abs(data_J[:, -1]))

            sum_log_abs_det_jacobian += logabsdet
            residuals.append(residual)
        print('finishid')
        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, length, x_dim)
        print('resid; ', residuals.shape)

        log_abs_det_jacobian = sum_log_abs_det_jacobian.reshape(batch_size, length)
        return residuals, log_abs_det_jacobian

