"""
 __      __  _____    _   _____
 \ \    / / | ___ \  | | |   __|
  \ \  / /  | |  | | | | |  |_
   \ \/ /   | |  | | | | |   _]
    \  /    | |__| | | | |  |
     \/     |_____/  |_| |__|

VDIF VLBI format readers.

Example to copy a VDIF file.  The data should be identical, though frames
will be ordered by thread_id.

>>> from scintellometry.io import vdif
>>> with vdif.open('vlba.m5a', 'r') as fr, vdif.open(
...         'try.vdif', 'w', header=fr.header0, nthread=fr.nthread) as fw:
...     while(True):
...         try:
...             fw.write_frame(*fr.read_frame())
...         except:
...             break

For small files, one could just do:
>>> with vdif.open('vlba.m5a', 'r') as fr, vdif.open(
...         'try.vdif', 'w', header=fr.header0, nthread=fr.nthread) as fw:
...     fw.write(fr.read())

This copies everything to memory, though, and some header information is lost.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from .base import open
from .header import VDIFFrameHeader
from .data import VDIFData
