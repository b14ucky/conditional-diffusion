import torch
from tqdm import tqdm
import torch.nn as nn
from torch import Tensor
from typing import Iterable
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


class SinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps: int, embed_dim: int) -> None:
        super(SinusoidalEmbeddings, self).__init__()
        position = torch.arange(time_steps, dtype=torch.float32).unsqueeze(1)
        divisor = torch.tensor(10_000) ** (
            torch.arange(0, embed_dim, 2, dtype=torch.float32) / embed_dim
        )
        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(position / divisor)
        embeddings[:, 1::2] = torch.cos(position / divisor)

        self.embeddings = embeddings

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.embeddings[t.cpu()].to(x.device)[:, :, None, None]


class ResBlock(nn.Module):
    def __init__(self, C: int, n_groups: int, dropout: float, n_labels: int) -> None:
        super(ResBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.norm1 = nn.GroupNorm(n_groups, C)
        self.norm2 = nn.GroupNorm(n_groups, C)
        self.conv1 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.label_embed = nn.Embedding(n_labels, C)

    def forward(self, x: Tensor, embeddings: Tensor, label: Tensor) -> Tensor:
        _, C, _, _ = x.shape

        x = x + embeddings[:, :C, :, :] + self.label_embed(label)[:, :, None, None]
        r = self.conv1(self.relu(self.norm1(x)))
        r = self.dropout(r)
        r = self.conv2(self.relu(self.norm2(x)))

        return r + x


class Attention(nn.Module):
    def __init__(self, C: int, n_heads: int, dropout: float) -> None:
        super(Attention, self).__init__()
        self.proj1 = nn.Linear(C, C * 3)
        self.proj2 = nn.Linear(C, C)
        self.n_heads = n_heads
        self.dropout = dropout

    def forward(self, x: Tensor):
        B, C, H_img, W_img = x.shape
        L = H_img * W_img
        K = 3
        head_dim = C // self.n_heads

        # [B,C,H,W] -> [B,L,C]
        x = x.reshape(B, C, L).permute(0, 2, 1)

        x = self.proj1(x)  # [B,L,C*3]
        x = x.reshape(B, L, K, self.n_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = x[0], x[1], x[2]  # [B, n_heads, L, head_dim]

        x = F.scaled_dot_product_attention(
            q, k, v, is_causal=False, dropout_p=self.dropout
        )
        x = x.permute(0, 2, 1, 3).reshape(B, H_img, W_img, C)  # [B,H,W,C]

        x = self.proj2(x)
        x = x.permute(0, 3, 1, 2)  # [B,C,H,W]

        return x


class UNetLayer(nn.Module):
    def __init__(
        self,
        upscale: bool,
        attention: bool,
        n_groups: int,
        dropout: float,
        n_heads: int,
        C: int,
        n_labels: int,
    ) -> None:
        super(UNetLayer, self).__init__()
        self.ResBlock1 = ResBlock(
            C=C, n_groups=n_groups, dropout=dropout, n_labels=n_labels
        )
        self.ResBlock2 = ResBlock(
            C=C, n_groups=n_groups, dropout=dropout, n_labels=n_labels
        )

        if upscale:
            self.conv = nn.ConvTranspose2d(
                C, C // 2, kernel_size=4, stride=2, padding=1
            )
        else:
            self.conv = nn.Conv2d(C, C * 2, kernel_size=3, stride=2, padding=1)

        if attention:
            self.attention_layer = Attention(C=C, n_heads=n_heads, dropout=dropout)

        self.attention = attention

    def forward(
        self, x: Tensor, embeddings: Tensor, label: Tensor
    ) -> tuple[Tensor, Tensor]:
        x = self.ResBlock1(x, embeddings, label)
        if self.attention:
            x = self.attention_layer(x)
        x = self.ResBlock2(x, embeddings, label)
        return self.conv(x), x


class UNet(nn.Module):
    def __init__(
        self,
        channels: list[int] = [64, 128, 256, 512, 512, 384],
        upscales: list[bool] = [False, False, False, True, True, True],
        attentions: list[bool] = [False, True, False, False, False, True],
        n_groups: int = 32,
        dropout: float = 0.1,
        n_heads: int = 8,
        n_labels: int = 10,
        input_channels: int = 3,
        output_channels: int = 3,
        time_steps: int = 1_000,
    ) -> None:
        super(UNet, self).__init__()

        self.n_layers = len(channels)
        self.conv_first = nn.Conv2d(
            input_channels, channels[0], kernel_size=3, padding=1
        )
        out_channels = channels[-1] // 2 + channels[0]
        self.conv_second_last = nn.Conv2d(
            out_channels, out_channels // 2, kernel_size=3, padding=1
        )
        self.conv_last = nn.Conv2d(out_channels // 2, output_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.embeddings = SinusoidalEmbeddings(
            time_steps=time_steps, embed_dim=max(channels)
        )

        for i in range(self.n_layers):
            layer = UNetLayer(
                upscale=upscales[i],
                attention=attentions[i],
                n_groups=n_groups,
                dropout=dropout,
                C=channels[i],
                n_heads=n_heads,
                n_labels=n_labels,
            )
            setattr(self, f"layer{i}", layer)

    def forward(self, x: Tensor, t: Tensor, label: Tensor) -> Tensor:
        x = self.conv_first(x)
        residuals = []

        for i in range(self.n_layers // 2):
            layer = getattr(self, f"layer{i}")
            embeddings = self.embeddings(x, t)
            x, r = layer(x, embeddings, label)
            residuals.append(r)

        for i in range(self.n_layers // 2, self.n_layers):
            layer = getattr(self, f"layer{i}")
            embeddings = self.embeddings(x, t)
            x = torch.cat(
                (layer(x, embeddings, label)[0], residuals[self.n_layers - i - 1]),
                dim=1,
            )

        return self.conv_last(self.relu(self.conv_second_last(x)))


class DDPMScheduler(nn.Module):
    def __init__(
        self,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        time_steps: int = 1_000,
    ) -> None:
        super(DDPMScheduler, self).__init__()
        self.beta = torch.linspace(
            beta_start, beta_end, time_steps, requires_grad=False
        )
        self.alpha = torch.cumprod(1 - self.beta, dim=0)

    def forward(self, x, t) -> tuple[Tensor, Tensor]:
        return self.beta[t].to(x.device), self.alpha[t].to(x.device)


class EMA(nn.Module):
    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        super(EMA, self).__init__()

        self.decay = decay
        self.ema_model = self._clone_model(model)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _clone_model(self, model: nn.Module) -> nn.Module:
        from copy import deepcopy

        ema_model = deepcopy(model)
        for p in ema_model.parameters():
            p.detach_()

        return ema_model

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for ema_param, param in zip(
                self.ema_model.parameters(), model.parameters()
            ):
                ema_param.data.mul_(self.decay).add_(
                    param.data, alpha=1 - self.decay
                )

    def load(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.ema_model.load_state_dict(checkpoint["weights"])
        self.load_state_dict(checkpoint["ema"])

    def generate(
        self,
        label: Tensor,
        time_steps: int = 1_000,
        times: list[int] | None = [0, 15, 50, 100, 200, 300, 400, 550, 700, 999],
    ) -> list[Tensor]:

        images = []
        label = label.to(self.device)

        scheduler = DDPMScheduler(time_steps=time_steps)

        with torch.no_grad():
            model = self.ema_model.eval()
            z = torch.randn(1, 3, 32, 32)
            for t in reversed(range(1, time_steps)):
                t = torch.tensor([t]).long()
                temp = scheduler.beta[t] / (
                    (torch.sqrt(1 - scheduler.alpha[t]))
                    * (torch.sqrt(1 - scheduler.beta[t]))
                )
                z = (1 / (torch.sqrt(1 - scheduler.beta[t]))) * z - (
                    temp * model(z.to(self.device), t, label).cpu()
                )
                if times and t[0] in times:
                    images.append(z)
                e = torch.randn(1, 3, 32, 32)
                z = z + (e * torch.sqrt(scheduler.beta[t]))

            temp = self.scheduler.beta[0] / (
                (torch.sqrt(1 - scheduler.alpha[0]))
                * (torch.sqrt(1 - scheduler.beta[0]))
            )
            x = (
                1 / (torch.sqrt(1 - scheduler.beta[torch.tensor(0).to(self.device)]))
            ) * z - (
                temp
                * model(
                    z.to(self.device), torch.tensor([0]).to(self.device), label
                ).cpu()
            )

            images.append(x)

            return images


dataset = CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=v2.Compose(
        [
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    ),
)


class ModelTrainer:
    def __init__(
        self,
        batch_size: int = 64,
        time_steps: int = 1_000,
        ema_decay: float = 0.9999,
        lr: float = 2e-5,
    ) -> None:
        self.dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {self.device}")

        self.scheduler = DDPMScheduler(time_steps=time_steps)
        self.model = UNet().to(self.device)
        self.ema = EMA(self.model, decay=ema_decay)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.batch_size = batch_size
        self.time_steps = time_steps

    def train(
        self,
        n_epochs: int = 75,
        checkpoint_path: str | None = None,
        checkpoint_output_path: str = "checkpoint.pt",
    ) -> None:
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["weights"])
            self.ema.load_state_dict(checkpoint["ema"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        if torch.cuda.is_available():
            self.model = nn.DataParallel(self.model.to(self.device))
            self.ema.to(self.device)

        for i in range(n_epochs):
            total_loss = 0
            for X, y in tqdm(self.dataloader, desc=f"epoch {i + 1}/{n_epochs}"):
                # B, _, _, _ = X.shape
                X = X.to(self.device)
                t = torch.randint(0, self.time_steps, (self.batch_size,))
                epsilon = torch.randn_like(X, requires_grad=False)
                _, alpha = self.scheduler(X, t)
                alpha = alpha.view(self.batch_size, 1, 1, 1)
                X = (torch.sqrt(alpha) * X) + torch.sqrt(1 - alpha) * epsilon

                output = self.model(X, t, y.to(self.device))
                self.optimizer.zero_grad()
                loss = self.criterion(output, epsilon)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                self.ema.update(self.model.module if self.device == "cuda" else self.model)  # type: ignore

            print(
                f"epoch {i + 1} | loss {total_loss / (60000 / self.batch_size):.5f}"
            )

        checkpoint = {
            "weights": self.model.module.state_dict() if self.device == "cuda" else self.model.state_dict(),  # type: ignore
            "optimizer": self.optimizer.state_dict(),
            "ema": self.ema.state_dict(),
        }

        torch.save(checkpoint, checkpoint_output_path)


class LabelEncoder:
    def __init__(self, labels: Iterable[str]) -> None:
        self._labels = {
            label: torch.tensor(i).unsqueeze(-1) for i, label in enumerate(labels)
        }

    def __call__(self, label: str) -> Tensor:
        if not isinstance(label, str):
            raise TypeError(f"label must be a string, got {type(label)} instead")
        if label not in self._labels.keys():
            raise ValueError(
                f'Invalid label "{label}". Expected one of {self._labels.keys()}'
            )

        return self._labels[label]


def display_reverse(images: list[Tensor]) -> None:
    fig, axes = plt.subplots(1, 10, figsize=(20, 4))
    for i, ax in enumerate(axes.flat):
        x = images[i].squeeze(0)
        x = torch.clip(x, 0, 1)
        x = x.permute(1, 2, 0).detach()
        x = x.numpy()
        ax.imshow(x)
        ax.axis("off")
    plt.show()
