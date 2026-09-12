"""The model under training.

``SimpleMLP`` is reproduced verbatim from the assignment - the architecture is
mandated, not chosen, and the pipeline split is defined in terms of these exact
attribute names. It lives in its own module because both sides of the project
need it: the single-process reference trains the whole thing, and the workers
will each own a contiguous slice of it. One definition, imported twice, is what
keeps the two runs comparable.
"""

from __future__ import annotations

import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
