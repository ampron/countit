# countIt
Python application to assist in state assignment of noisy data from a system displaying time-dependent switching between quantized states

---

**Author:** *Alex M. Pronschinske*

**Version:** *1.5.0*

Software Requirements
=====================

This package has the following dependencies:

 * python 2.7
 * numpy >= 1.5.1 (lower versions are untested)
 * scipy >= 0.9.0 (lower versions are untested)
 * matplotlib >= 1.1.0
 * scikit-learn >= 0.15.2
 * pyMTRX >= 1.6.0 (optional)


Installation Instructions (Windows 7)
=====================================

 1. Install python (32-bit, ignore OS type), and make sure to check the option for adding python.exe to system PATH
 2. Open command prompt
 3. Install nose by typing in the command prompt (starting after `$`): `$pip install nose`
 4. Install numpy (always the official 32-bit package) from .exe
 5. Install scipy (always the official 32-bit package) from .exe
 5. Install all matplotlib dependencies, using `$pip install [package_name_here]`:
  * six
  * python-dateutil
  * pytz
  * pyparsing
  * Pillow
 6. Install matplotlib (always the official 32-bit package) from .exe
 8. Install pyMTRX: `$pip install pyMTRX`
 8. Install scikit-learn: `$pip install scikit-learn`
 9. Install countIt from .exe

Usage Instructions
==================

**Graph Hotkeys**

 a: Set selection area to cover all data
 ctrl + z: undo last assignment operation
 u: unassign selected data
 0: assign selected data to next available automatic label
 1 - 9: assign selected data to the label of the number pressed
