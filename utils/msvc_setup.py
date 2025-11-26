"""
MSVC Environment Setup for torch.compile on Windows.

This module configures the environment to use MSVC compiler for PyTorch's
torch.compile feature, which can provide 1.5-2x additional speedup.

Usage:
    from utils.msvc_setup import setup_msvc_for_torch_compile
    setup_msvc_for_torch_compile()
    import torch
    # torch.compile will now work
"""

import os
import sys


def setup_msvc_for_torch_compile():
    """
    Configure environment for MSVC compiler to work with torch.compile.

    This must be called BEFORE importing torch to ensure the compiler
    is properly configured.

    Returns:
        bool: True if setup successful, False otherwise
    """
    # MSVC paths
    msvc_root = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207"
    msvc_bin = os.path.join(msvc_root, r"bin\Hostx64\x64")
    msvc_lib = os.path.join(msvc_root, r"lib\x64")
    msvc_include = os.path.join(msvc_root, "include")

    # Windows SDK paths (using latest version)
    winsdk_root = r"C:\Program Files (x86)\Windows Kits\10"
    winsdk_version = "10.0.26100.0"
    winsdk_include = os.path.join(winsdk_root, "Include", winsdk_version)
    winsdk_lib = os.path.join(winsdk_root, "Lib", winsdk_version, "um", "x64")
    winsdk_ucrt_lib = os.path.join(winsdk_root, "Lib", winsdk_version, "ucrt", "x64")

    # Verify paths exist
    if not os.path.exists(msvc_bin):
        print(f"Warning: MSVC not found at {msvc_bin}")
        return False

    if not os.path.exists(winsdk_include):
        print(f"Warning: Windows SDK not found at {winsdk_include}")
        return False

    # Add to PATH
    path_entries = [
        msvc_bin,
        os.environ.get('PATH', '')
    ]
    os.environ['PATH'] = ';'.join(path_entries)

    # Setup INCLUDE paths
    include_entries = [
        msvc_include,
        os.path.join(winsdk_include, "ucrt"),
        os.path.join(winsdk_include, "um"),
        os.path.join(winsdk_include, "shared"),
    ]
    os.environ['INCLUDE'] = ';'.join(include_entries)

    # Setup LIB paths
    lib_entries = [
        msvc_lib,
        winsdk_lib,
        winsdk_ucrt_lib,
    ]
    os.environ['LIB'] = ';'.join(lib_entries)

    return True


def is_msvc_available():
    """
    Check if MSVC is available on the system.

    Returns:
        bool: True if MSVC is found, False otherwise
    """
    msvc_root = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207"
    msvc_bin = os.path.join(msvc_root, r"bin\Hostx64\x64")
    return os.path.exists(msvc_bin)


def print_msvc_status():
    """Print MSVC setup status."""
    if is_msvc_available():
        print("✓ MSVC compiler available")
        print("  Version: 14.44.35207")
        print("  torch.compile will be enabled")
    else:
        print("✗ MSVC compiler not found")
        print("  torch.compile will be disabled")
        print("  Install Visual Studio 2022 with C++ support to enable")
