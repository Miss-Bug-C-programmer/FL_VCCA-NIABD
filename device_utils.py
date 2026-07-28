import torch


class _NullGradScaler:
    def scale(self, loss):
        return loss

    def unscale_(self, optimizer) -> None:
        return None

    def step(self, optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        return None


def has_mps() -> bool:
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )


def default_main_device() -> str:
    """Prefer CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if has_mps():
        return "mps"
    return "cpu"

def resolve_ae_device(ae_device, main_device=None) -> str:
    raw = str(ae_device).strip().lower() if ae_device is not None else "auto"
    if raw in {"auto", "same", "main"}:
        return normalize_device(main_device or default_main_device()).type
    return normalize_device(raw).type


def use_amp_for_device(device) -> bool:
    dev = normalize_device(device)
    return dev.type == "cuda" and torch.cuda.is_available()


def supports_pin_memory(device) -> bool:
    dev = normalize_device(device)
    return dev.type == "cuda" and torch.cuda.is_available()


def make_grad_scaler(device, enabled: bool = True):
    enabled = bool(enabled) and use_amp_for_device(device)
    if not enabled:
        return _NullGradScaler()
    amp_mod = getattr(torch, "amp", None)
    if amp_mod is not None and hasattr(amp_mod, "GradScaler"):
        try:
            return amp_mod.GradScaler("cuda", enabled=True)
        except TypeError:
            return amp_mod.GradScaler(enabled=True)
    return torch.cuda.amp.GradScaler(enabled=True)


def normalize_device(device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    return torch.device(str(device))


def device_as_str(device) -> str:
    return normalize_device(device).type
