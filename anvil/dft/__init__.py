"""DFT labeling: pluggable engines, submission, collection, convergence checks.

`anvil.dft.engines` holds the backend abstraction (VASP, Quantum ESPRESSO via
QEpy or pw.x, or any ASE calculator); `anvil.dft.vasp` holds the VASP input
writers. See DESIGN.md §4.3.
"""
