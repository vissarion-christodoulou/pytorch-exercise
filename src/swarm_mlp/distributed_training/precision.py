"""The wire precisions the ablation sweeps, for both channels that carry tensors.

The two channels take their precision differently. Activations cross the
trainer/worker RPCs and are typed by the worker's *schema*, which hivemind
publishes over the DHT so the trainer serialises to match - one declaration
governs both ends. Gradients cross the replica all-reduce and are typed by a
compression *instance* handed to the averager. Same three schemes either way, so
they are named once here.

There is no FP8 in this hivemind. The wire enum offers NONE, FLOAT16,
MEANSTD_16BIT and three 8-bit *quantisations* - no e4m3, no e5m2. The low tier is
therefore uniform integer quantisation, named ``int8`` rather than fp8 so the
write-up cannot claim a float format it never used.
"""

from __future__ import annotations

import numpy as np
import torch

from hivemind.compression import Float16Compression, NoCompression, Uniform8BitQuantization
from hivemind.compression.quantization import Quantization
from hivemind.proto import runtime_pb2
from hivemind.proto.runtime_pb2 import CompressionType

#: For ``BatchTensorDescriptor(compression=...)``, which wants the wire enum.
ACTIVATION_PRECISIONS = {
    "fp32": CompressionType.NONE,
    "fp16": CompressionType.FLOAT16,
    "int8": CompressionType.UNIFORM_8BIT,
}

#: For ``GradientAverager(compression=...)``, which wants a CompressionBase.
GRADIENT_PRECISIONS = {
    "fp32": NoCompression(),
    "fp16": Float16Compression(),
    "int8": Uniform8BitQuantization(),
}


def _extract(self: Quantization, serialized_tensor: runtime_pb2.Tensor) -> torch.Tensor:
    """``Quantization.extract`` with the one line NumPy 2 broke repaired.

    hivemind is pinned to a 2023 commit and reads its codebook size with
    ``int(np.frombuffer(buffer, count=1, dtype=np.int64))``. ``frombuffer``
    returns a one-element *array*, and NumPy 2.0 removed the implicit array ->
    scalar conversion that made this work (deprecated in 1.25). We are on 2.4, so
    this raises ``TypeError``. Everything below the first line is verbatim.
    """
    codebook_size = int(np.frombuffer(serialized_tensor.buffer, count=1, dtype=np.int64)[0])
    codebook = np.frombuffer(serialized_tensor.buffer, offset=8, count=codebook_size, dtype=self.codebook_dtype)
    quantized = np.frombuffer(serialized_tensor.buffer, offset=8 + codebook.nbytes, dtype=self.indices_dtype)
    quantized = torch.as_tensor(quantized, dtype=torch.int64).reshape(tuple(serialized_tensor.size))
    codebook = torch.as_tensor(codebook).to(dtype=getattr(torch, serialized_tensor.dtype))
    return codebook[quantized].requires_grad_(serialized_tensor.requires_grad)


def enable_int8() -> None:
    """Make 8-bit tensors decodable in this process. Call before serving.

    Only ``extract`` is affected, so without this an int8 tensor compresses fine
    and then raises on whoever receives it - the peer replica for gradients,
    both parties for activations. The whole int8 tier is unrunnable otherwise.

    Patching the base class covers every path at once: the instance handed to the
    averager and the registry instances ``deserialize_torch_tensor`` dispatches
    through are all ``Quantization`` subclasses. Patched here rather than in
    site-packages because the assignment pins hivemind to an exact hash, and an
    edited checkout would make these results irreproducible for anyone who
    follows the setup script. A no-op for fp32 and fp16.
    """
    Quantization.extract = _extract
