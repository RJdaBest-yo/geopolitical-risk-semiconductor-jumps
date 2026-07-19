"""
Multi-factor Jump Diffusion analysis runner
"""
import sys, os

# Add paths
_main_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _main_root)

from mf_src.multifactor_jump import _run_self_tests as t
t()
