# pytorch-exercise

A small, installable Python project for practicing PyTorch.

## Setup

Create and activate a virtual environment, then install the project:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

## Usage

Run the entry point, which prints the installed torch version and whether CUDA is available:

```powershell
pytorch-exercise
```

Or run it as a module:

```powershell
python -m pytorch_exercise
```

## Layout

```
src/pytorch_exercise/
    __init__.py
    __main__.py    # main(): prints torch version and CUDA availability
```
