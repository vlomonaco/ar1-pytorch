import torch


__all__ = ["BatchRenorm1d", "BatchRenorm2d", "BatchRenorm3d"]

# torch.jit.ScriptModule
class BatchRenorm(torch.nn.Module):
    def __init__(self, num_features, eps=1e-3, momentum=0.01, affine=True,
                 weight=None, bias=None, running_mean=None, running_var=None,
                 num_batches_tracked=None
    ):
        super().__init__()
        if running_mean is not None:
            run_mean = running_mean
        else:
            run_mean = torch.zeros(num_features, dtype=torch.float)
        self.register_buffer(
            "running_mean", run_mean
        )
        if running_var is not None:
            run_std = torch.sqrt(running_var)
        else:
            run_std = torch.ones(num_features, dtype=torch.float)
        self.register_buffer(
            "running_std", run_std
        )
        if num_batches_tracked is not None:
            b_tracked = num_batches_tracked
        else:
            b_tracked = torch.tensor(0, dtype=torch.long)
        self.register_buffer(
            "num_batches_tracked", b_tracked
        )
        if weight is not None:
            weight = weight
        else:
            weight = torch.ones(num_features, dtype=torch.float)
        self.weight = torch.nn.Parameter(
            weight
        )
        if bias is not None:
            bias = bias
        else:
            bias = torch.zeros(num_features, dtype=torch.float)
        self.bias = torch.nn.Parameter(
            bias
        )
        self.affine = affine
        self.eps = eps
        self.step = 0
        self.momentum = momentum

    def _check_input_dim(self, x: torch.Tensor) -> None:
        raise NotImplementedError()  # pragma: no cover

    @property
    def rmax(self) -> torch.Tensor:
        return (2 / 35000 * self.num_batches_tracked + 25 / 35).clamp_(
            1.0, 3.0
        )

    @property
    def dmax(self) -> torch.Tensor:
        return (5 / 20000 * self.num_batches_tracked - 25 / 20).clamp_(
            0.0, 5.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)
        if x.dim() > 2:
            x = x.transpose(1, -1)
        if self.training:
            dims = [i for i in range(x.dim() - 1)]
            batch_mean = x.mean(dims)
            batch_std = x.std(dims, unbiased=False) + self.eps
            r = (
                batch_std.detach() / self.running_std.view_as(batch_std)
            ).clamp_(1 / self.rmax, self.rmax)
            d = (
                (batch_mean.detach() - self.running_mean.view_as(batch_mean))
                / self.running_std.view_as(batch_std)
            ).clamp_(-self.dmax, self.dmax)
            x = (x - batch_mean) / batch_std * r + d
            self.running_mean += self.momentum * (
                batch_mean.detach() - self.running_mean
            )
            self.running_std += self.momentum * (
                batch_std.detach() - self.running_std
            )
            self.num_batches_tracked += 1
        else:
            x = (x - self.running_mean) / self.running_std
        if self.affine:
            x = self.weight * x + self.bias
        if x.dim() > 2:
            x = x.transpose(1, -1)
        return x


class BatchRenorm1d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() not in [2, 3]:
            raise ValueError("expected 2D or 3D input (got {x.dim()}D input)")


class BatchRenorm2d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() != 4:
            raise ValueError("expected 4D input (got {x.dim()}D input)")


class BatchRenorm3d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() != 5:
            raise ValueError("expected 5D input (got {x.dim()}D input)")


if __name__ == "__main__":

    # Create batch renormalization layer
    br = BatchRenorm2d(3)

    # Create some example data with dimensions N x C x H x W
    x = torch.randn(1, 3, 10, 10)

    # Batch renormalize the data
    x = br(x)

    print("All good!")