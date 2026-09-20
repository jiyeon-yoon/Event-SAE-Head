"""Fixed-prefix output-head sensitivity for a frozen OpenVLA SAE.

Numerical and simulator dependencies are deliberately imported by their callers,
not on package import. CPU planning does not initialize a model or CUDA.
"""

SCHEMA_VERSION = "output_head_sensitivity_v1"
