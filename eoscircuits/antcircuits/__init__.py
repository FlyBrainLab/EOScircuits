"""Antenna Circuit

This module supports:

1. Changing the affinity values of each of the odorant-receptor pairs characterizing the input of the Odorant Transduction Process.
2. Changing parameter values of the Biological Spike Generators (BSGs) associated with each OSN.
3. Changing the number of OSNs expressing the same Odorant Receptor (OR) type.
"""
from .circuit import ANTCircuit, ANTConfig