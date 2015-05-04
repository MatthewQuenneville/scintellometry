
# __      __  _____    _   _____
# \ \    / / | ___ \  | | |   __|
#  \ \  / /  | |  | | | | |  |_
#   \ \/ /   | |  | | | | |   _]
#    \  /    | |__| | | | |  |
#     \/     |_____/  |_| |__|

from __future__ import division
import os
import struct
import warnings

import numpy as np
from astropy.time import Time, TimeDelta
import astropy.units as u

from . import SequentialFile, header_defaults

# the high mag value for 2-bit reconstruction
OPTIMAL_2BIT_HIGH = 3.3359
FOUR_BIT_1_SIGMA = 2.95


class VDIFData(SequentialFile):

    telescope = 'vdif'

    def __init__(self, raw_files, comm=None):
        """Pulsar data stored in the VDIF format"""

        checkfile = open(raw_files[0], 'rb')
        header = VDIFFrameHeader(checkfile)
        self.header0 = header
        # for complex is bits/complex component
        bips = header.bits_per_sample
        if header['complex_data']:
            if bips == 8:
                dtype = 'ci1,ci1'
            elif bips == 2:
                dtype = 'ci2bit,ci2bit'
            else:
                raise ValueError("Can only do 8bit and 2bit complex")
        else:
            if bips == 2:
                dtype = '4bit'
            else:
                raise ValueError("Only 2bit real supported thus far.")

        self.header_size = header.size
        # Frame size includes header_size.
        framesize = header.framesize
        blocksize = framesize - self.header_size
        # could also do one second worth of data
        # blocksize = (framesize - self.header_size) * frames_per_sec
        nchan = header.nchan
        self.time0 = header.time
        self.npol = 2
        self.station = header.station
        self.thread_ids = get_thread_ids(checkfile, framesize)
        self.n_threads = len(self.thread_ids)
        self.samplerate = header.samplerate
        if not self.samplerate:  # not known
            frames_per_sec = count_frames_per_sec(checkfile, framesize,
                                                  header['seconds_from_ref'])
            self.samplerate = ((blocksize // nchan) *
                               (8 // bips) * frames_per_sec * u.Hz)
            # FIX ME!
            # if blocksize=data_per_second need to adjust samplerate!

        self.dtsample = (nchan * 2 / self.samplerate).to(u.s)
        # might have multiple threads with each one holding one channel
        if nchan > 1 and self.n_threads > 1:
            raise ValueError("Multi-channel, multi-threaded vdif "
                             "not supported.")
        if nchan == 1 and self.n_threads > 1:
            nchan = self.n_threads
        # in that case will need to get the freq of the different threads.
        # also for single thread multi-channel need to get freq from somewhere
        # probably via config file? check out dada.py for freq stuff
        if comm is None or comm.rank == 0:
            print("In VDIFData, calling super")
            print("Start time: ", self.time0.iso)
        self.files = raw_files
        checkfile.close()

        super(VDIFData, self).__init__(raw_files, blocksize, dtype, nchan,
                                       comm=comm)

    def open(self, number=0):
        """Open a raw file, and search for the start of the first frame."""
        if number == self.current_file_number:
            return self.fh_raw

        super(VDIFData, self).open(number)
        return self.fh_raw

    def close(self):
        """Close the whole file reader, unlinking links if needed."""
        if self.current_file_number is not None:
            self.fh_raw.close()

    def read(self, size):
        """Read size bytes, returning an ndarray with np.int8 dtype.

        Incorporate information from multiple underlying files if necessary.
        The current file pointer are assumed to be pointing at the right
        locations, i.e., just before the first bit of data that will be read.
        """
        if size % self.recordsize != 0:
            raise ValueError("Cannot read a non-integer number of records")

        # ensure we do not read beyond end
        size = min(size, len(self.files) * self.filesize - self.offset)
        if size <= 0:
            raise EOFError('At end of file in DADA.read')

        # allocate buffer for MPI read
        z = np.empty(size, dtype=np.int8)

        # read one or more pieces
        iz = 0
        while(iz < size):
            block, already_read = divmod(self.offset, self.filesize)
            fh_size = min(size - iz, self.filesize - already_read)
            z[iz:iz+fh_size] = np.fromstring(self.fh_raw.read(fh_size),
                                             dtype=z.dtype)
            self._seek(self.offset + fh_size)
            iz += fh_size

        return z

    def _seek(self, offset):
        assert offset % self.recordsize == 0
        file_number = offset // self.filesize
        file_offset = offset % self.filesize
        self.open(self.files, file_number)
        self.fh_raw.seek(file_offset + self.header_size)
        self.offset = offset

    def ntint(self, nchan):
        assert self.blocksize % (self.itemsize * nchan) == 0
        return self.blocksize // (self.itemsize * nchan)

    def __str__(self):
        return ('<DADAData nchan={0} dtype={1} blocksize={2}\n'
                'current_file_number={3}/{4} current_file={5}>'
                .format(self.nchan, self.dtype, self.blocksize,
                        self.current_file_number, len(self.files),
                        self.files[self.current_file_number]))

VDIF_header = {  # tuple has word-index, start-bit-index, bit-length
    'standard': {
        'invalid_data': (0, 31, 1),
        'legacy_mode': (0, 30, 1),
        'seconds_from_ref': (0, 0, 29),
        'ref_epoch': (1, 24, 6),
        'frame_nr': (1, 0, 24),
        'vdif_version': (2, 29, 3),
        'lg2_nchan': (2, 24, 5),
        'frame_length': (2, 0, 24),
        'complex_data': (3, 31, 1),
        'bits_per_sample': (3, 26, 5),
        'thread_id': (3, 16, 10),
        'station_id': (3, 0, 16),
        'edv': (4, 24, 8)},
    1: {'sampling_unit': (4, 23, 1),
        'sample_rate': (4, 0, 23),
        'sync_pattern': (5, 0, 32),
        'das_id': (6, 0, 32),
        'ua': (7, 0, 32)},
    3: {'sampling_unit': (4, 23, 1),
        'sample_rate': (4, 0, 23),
        'sync_pattern': (5, 0, 32),
        'loif_tuning': (6, 0, 32),
        'dbe_unit': (7, 24, 4),
        'if_nr': (7, 20, 4),
        'subband': (7, 17, 3),
        'sideband': (7, 16, 1),
        'major_rev': (7, 12, 4),
        'minor_rev': (7, 8, 4),
        'personality': (7, 0, 8)},
    4: {'sampling_unit': (4, 23, 1),
        'sample_rate': (4, 0, 23),
        'sync_pattern': (5, 0, 32)}}


def _make_parser(word_index, bit_index, bit_length):
    if bit_length == 1:
        return lambda x: bool((x[word_index] >> bit_index) & 1)
    else:
        mask = (1 << bit_length) - 1  # e.g., bit_length=8 -> 0xff
        if bit_index == 0:
            return lambda x: x[word_index] & mask
        else:
            return lambda x: (x[word_index] >> bit_index) & mask


VDIF_header_parsers = {vk: {k: _make_parser(*v) for k, v in vv.items()}
                       for vk, vv in VDIF_header.items()}


_eight_words = struct.Struct('<8I')

ref_epochs = Time(['{year:04d}-{month:02d}-{day:02d}'
                   .format(year=2000 + ref // 2,
                           month=1 if ref % 2 == 0 else 7,
                           day=1)
                   for ref in range(256)], format='isot', scale='utc')


class VDIFFrameHeader(object):
    def __init__(self, fh):
        """Read a VDIF Frame Header from a file, allowing parsing as needed."""
        # Get eight words and interpret them as unsigned 4-byte integer.
        self.data = _eight_words.unpack(fh.read(32))
        if self['legacy_mode']:
            # Legacy headers are only 4 words, so rewind.
            fh.seek(-16, 1)
            self.data = self.data[:4]
            self.edv = None
        else:
            self.edv = self['edv']
        self.size = len(self.data) * 4

    def __getitem__(self, item):
        try:
            return VDIF_header_parsers['standard'][item](self.data)
        except KeyError:
            if self.edv:
                try:
                    edv_parsers = VDIF_header_parsers[self.edv]
                except KeyError:
                    raise KeyError("VDIF Header of unsupported edv {0}"
                                   .format(self.edv))
                try:
                    return edv_parsers[item](self.data)
                except KeyError:
                    pass

        raise KeyError("VDIF Frame Header does not contain {0}".format(item))

    def keys(self):
        keys = VDIF_header_parsers['standard'].keys()
        if self.edv:
            keys += VDIF_header_parsers[self.edv].keys()
        return keys

    def __in__(self, key):
        return key in self.keys()

    def __repr__(self):
        return ("<VDIFFrameHeader {0}>"
                .format(",\n                 ".join(
                    ["{0}: {1}".format(k, self[k]) for k in self.keys()])))

    @property
    def bits_per_sample(self):
        return self['bits_per_sample'] + 1

    @property
    def framesize(self):
        return self['frame_length'] * 8

    @property
    def payloadsize(self):
        return self.framesize - self.size

    @property
    def nchan(self):
        return 2**self['lg2_nchan']

    @property
    def station(self):
        msb = self['station_id'] >> 8
        if 48 <= msb < 128:
            return chr(msb) + chr(self['station_id'] & 0xff)
        else:
            return self['station_id']

    @property
    def samplerate(self):
        if not self['legacy_mode'] and self.edv:
            return (self['sample_rate'] *
                    (u.MHz if self['sampling_unit'] else u.kHz))
        else:
            return None

    @property
    def time(self):
        """
        Convert ref_epoch and seconds_from_ref to Time object.

        Uses 'ref_epoch', which stores the number of half-years from 2000,
        and 'seconds_from_ref'.
        """
        return (ref_epochs[self['ref_epoch']] +
                TimeDelta(self['seconds_from_ref'], format='sec', scale='tai'))


def get_thread_ids(infile, framesize, searchsize=None):
    """
    Get the number of threads and their ID's in a vdif file.
    """
    # go to end of infile to get its size on disk
    if searchsize is None:
        searchsize = 1024 * framesize

    n_total = searchsize // framesize

    thread_ids = set()
    for n in range(n_total):
        infile.seek(n * framesize)
        try:
            thread_ids.add(VDIFFrameHeader(infile)['thread_id'])
        except:
            break

    return thread_ids


def count_frames_per_sec(fh, thread_id=None):
    """Returns the number of frames

    Can be for a specific thread_id (by default just the first thread in
    the first header).
    """
    fh.seek(0)
    header = VDIFFrameHeader(fh)
    assert header['frame_nr'] == 0
    sec0 = header['seconds_from_ref']
    thread_id0 = thread_id if thread_id is not None else header['thread_id']
    k = 0
    while(header['seconds_from_ref'] == sec0):
        fh.seek(header.payloadsize, 1)
        header = VDIFFrameHeader(fh)
        if header['thread_id'] == thread_id0:
            k += 1

    if header['seconds_from_ref'] != sec0 + 1:
        raise ValueError("Time in file has changed by more than 1 second.")

    return k


def init_luts():
    """Set up the look-up tables for levels as a function of input byte."""
    lut2level = np.array([-1.0, 1.0], dtype=np.float32)
    lut4level = np.array([-OPTIMAL_2BIT_HIGH, -1.0, 1.0, OPTIMAL_2BIT_HIGH],
                         dtype=np.float32)
    lut16level = (np.arange(16) - 8.)/FOUR_BIT_1_SIGMA

    b = np.arange(256)[:, np.newaxis]
    # 1-bit mode
    i = np.arange(8)
    lut1bit = lut2level[(b >> i) & 1]
    # 2-bit mode
    i = np.arange(0, 8, 2)
    lut2bit = lut4level[(b >> i) & 3]
    # 4-bit mode
    i = np.arange(0, 8, 4)
    lut4bit = lut16level[(b >> i) & 0xf]
    return lut1bit, lut2bit, lut4bit

lut1bit, lut2bit, lut4bit = init_luts()


# DADA defaults for psrfits HDUs
# Note: these are largely made-up at this point
header_defaults['vdif'] = {
    'PRIMARY': {'TELESCOP':'VDIF',
                'IBEAM':1, 'FD_POLN':'LIN',
                'OBS_MODE':'SEARCH',
                'ANT_X':0, 'ANT_Y':0, 'ANT_Z':0, 'NRCVR':1,
                'FD_HAND':1, 'FD_SANG':0, 'FD_XYPH':0,
                'BE_PHASE':0, 'BE_DCC':0, 'BE_DELAY':0,
                'TRK_MODE':'TRACK',
                'TCYCLE':0, 'OBSFREQ':300, 'OBSBW':100,
                'OBSNCHAN':0, 'CHAN_DM':0,
                'EQUINOX':2000.0, 'BMAJ':1, 'BMIN':1, 'BPA':0,
                'SCANLEN':1, 'FA_REQ':0,
                'CAL_FREQ':0, 'CAL_DCYC':0, 'CAL_PHS':0, 'CAL_NPHS':0,
                'STT_IMJD':54000, 'STT_SMJD':0, 'STT_OFFS':0},
    'SUBINT': {'INT_TYPE': 'TIME',
               'SCALE': 'FluxDen',
               'POL_TYPE': 'AABB',
               'NPOL':1,
               'NBIN':1, 'NBIN_PRD':1,
               'PHS_OFFS':0,
               'NBITS':1,
               'ZERO_OFF':0, 'SIGNINT':0,
               'NSUBOFFS':0,
               'NCHAN':1,
               'CHAN_BW':1,
               'DM':0, 'RM':0, 'NCHNOFFS':0,
               'NSBLK':1}}
