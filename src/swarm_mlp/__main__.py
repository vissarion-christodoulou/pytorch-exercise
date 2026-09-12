"""Entry point: report whether PyTorch can see a CUDA device."""

import torch


def main() -> None:
    print(f"torch version: {torch.__version__}")

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"CUDA is available ({torch.cuda.device_count()} device(s))")
        print(f"Current device: {torch.cuda.get_device_name(device)}")
    else:
        print("CUDA is not available; running on CPU")


if __name__ == "__main__":
    main()
