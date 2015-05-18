
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

with vdif.open_vdif('vlba.m5a', 'r') as fr, vdif.open_vdif(
        'try.vdif', 'w', header=fr.header0, nthread=fr.nthread) as fw:
    while(True):
        try:
            fw.write_frame(*fr.read_frame())
        except:
            break

For small files, one could just do:
fw.write(fr.read())
This copies everything to memory, though, and some header information is lost.
"""

from __future__ import division, unicode_literals
import warnings

import numpy as np
from astropy.utils import lazyproperty
from astropy.time import Time, TimeDelta
import astropy.units as u

from . import SequentialFile, header_defaults
from .vlbi_helpers import (get_frame_rate, make_parser, four_word_struct,
                           eight_word_struct)

# the high mag value for 2-bit reconstruction
OPTIMAL_2BIT_HIGH = 3.3359
FOUR_BIT_1_SIGMA = 2.95


# Check code on 2015-MAY-10
# 00000000  77 2c db 00 00 00 00 1c  75 02 00 20 fc ff 01 04  # header 0 - 3
# 00000010  10 00 80 03 ed fe ab ac  00 00 40 33 83 15 03 f2  # header 4 - 7
# 00000020  2a 0a 7c 43 8b 69 9d 59  cb 99 6d 9a 99 96 5d 67  # data 0 - 3
# NOTE: thread_id = 1
# 2a = 00 10 10 10 = (lsb first) 1,  1,  1, -3
# 0a = 00 00 10 10 =             1,  1, -3, -3
# 7c = 01 11 11 00 =            -3,  3,  3, -1
# m5d evn/Fd/GP052D_FD_No0006.m5a VDIF_5000-512-1-2 100
# Mark5 stream: 0x16cd140
#   stream = File-1/1=evn/Fd/GP052D_FD_No0006.m5a
#   format = VDIF_5000-512-1-2 = 3
#   start mjd/sec = 56824 21367.000000000
#   frame duration = 78125.00 ns
#   framenum = 0
#   sample rate = 256000000 Hz
#   offset = 0
#   framebytes = 5032 bytes
#   datasize = 5000 bytes
#   sample granularity = 4
#   frame granularity = 1
#   gframens = 78125
#   payload offset = 32
#   read position = 0
#   data window size = 1048576 bytes
#  1  1  1 -3  1  1 -3 -3 -3  3  3 -1  -> OK
# fh = vdif.VDIFData(['evn/Fd/GP052D_FD_No0006.m5a'], channels=range(8),
#                    fedge=0, fedge_at_top=False, blocksize=8*5000*32)
# d = fh.record_read(fh.blocksize)
# d.astype(int)[:, 1][:12]  # thread id = 1!!
# Out[8]: array([ 1,  1,  1, -3,  1,  1, -3, -3, -3,  3,  3, -1])  -> OK
# Also, next frame (thread #3)
# m5d evn/Fd/GP052D_FD_No0006.m5a VDIF_5000-512-1-2 12 5032
# -1  1 -1  1 -3 -1  3 -1  3 -3  1  3
# d.astype(int)[:, 3][:12]
# Out[14]: array([-1,  1, -1,  1, -3, -1,  3, -1,  3, -3,  1,  3])
# And first thread #0
# m5d evn/Fd/GP052D_FD_No0006.m5a VDIF_5000-512-1-2 12 20128
# -1 -1  3 -1  1 -1  3 -1  1  3 -1  1
# d.astype(int)[:, 0][:12]
# Out[14]: array([-1, -1,  3, -1,  1, -1,  3, -1,  1,  3, -1,  1])


class VDIFData(SequentialFile):

    telescope = 'vdif'

    def __init__(self, raw_files, channels, fedge, fedge_at_top,
                 blocksize=None, comm=None):
        """VDIF Data reader.

        Parameters
        ----------
        raw_files : list of string
            full file names of the VDIF data files.
        channels : list of int
            channel numbers to read; should be at the same frequency,
            i.e., 1 or 2 polarisations.
        fedge : Quantity
            Frequency at the edge of the requested VLBI channel
        fedge_at_top : bool
            Whether the frequency is at the top of the channel.
        blocksize : int or None
            Number of bytes typically read in one go
            (default: nthread*framesize, though for VDIF data this is rather
            low, so better to pass on a larger number).
        comm : MPI communicator
            For consistency with other readers.
        """
        if len(raw_files) > 1:
            raise ValueError("Can only handle single file for now.")
        self.fedge = fedge
        self.fedge_at_top = fedge_at_top
        with open(raw_files[0], 'rb') as checkfile:
            header = VDIFFrameHeader.fromfile(checkfile)
            self.header0 = header
            self.thread_ids = get_thread_ids(checkfile, header.framesize)
            self.nthread = len(self.thread_ids)

        if header.nchan > 1:
            # This needs more thought, though a single thread with multiple
            # channels should be easy, as it is similar to other formats
            # (just need to calculate frequencies).  But multiple channels
            # over multiple threads may not be so easy.
            raise ValueError("Multi-channel vdif not yet supported.")

        self.channels = channels
        # For normal folding, 1 or 2 channels should be given, but for other
        # reading, it may be useful to have all channels available.
        if channels is None:
            self.npol = self.nthread
            self.thread_indices = range(self.nthread)
        else:
            self.thread_indices = [None] * self.nthread
            try:
                self.npol = len(channels)
            except TypeError:
                self.npol = 1
                self.thread_indices[channels] = 0
            else:
                for i, channel in enumerate(channels):
                    self.thread_indices[channel] = i

        if not (1 <= self.npol <= 2):
            warnings.warn("Should use 1 or 2 channels for folding!")

        # Decoder for given bits per sample; see bottom of file.
        self._decode = DECODERS[header.bps, header['complex_data']]

        self.framesize = header.framesize
        self.payloadsize = header.payloadsize
        if blocksize is None:
            blocksize = header.payloadsize
        # Each "virtual record" is one sample for every thread.
        record_bps = header.bps * self.nthread
        if record_bps in (1, 2, 4):
            dtype = '{0}bit'.format(record_bps)
        elif record_bps % 8 == 0:
            dtype = '({0},)u1'.format(record_bps // 8)
        else:
            raise ValueError("VDIF with {0} bits per sample is not supported."
                             .format(header.bps))
        # SOMETHING LIKE THIS NEEDED FOR MULTIPLE FILES!
        # PROBABLY SHOULD MAKE SEQUENTIALFILE READER SEEK
        # OFFSET IN TOTAL BYTE SIZE, IGNORING HEADERS
        # self.totalfilesizes = np.array([os.path.getsize(fil)
        #                                 for fil in raw_files], dtype=np.int)
        # assert np.all(self.totalfilesizes // header.framesize == 0)
        # self.totalpayloads = (self.totalfilesizes // header.framesize *
        #                       header.payloadsize)
        # self.payloadranges = self.totalpayloads.cumsum()
        self.time0 = header.time()
        if header.bandwidth:
            self.samplerate = header.bandwidth * 2.
        else:  # bandwidth not known (e.g., legacy header)
            frame_rate = get_frame_rate(checkfile, VDIFFrameHeader) * u.Hz
            self.samplerate = ((header.payloadsize // 4) * (32 // header.bps) *
                               frame_rate).to(u.MHz)
        if header['complex_data']:
            self.samplerate /= 2.
        self.dtsample = (header.nchan / self.samplerate).to(u.ns)
        if comm is None or comm.rank == 0:
            print("In VDIFData, calling super")
            print("Start time: ", self.time0.iso)
        super(VDIFData, self).__init__(raw_files, blocksize, dtype,
                                       header.nchan, comm=comm)

    def _seek(self, offset):
        assert offset % self.recordsize == 0
        # Seek in the raw file using framesize, i.e., including headers.
        self.fh_raw.seek(offset // self.payloadsize * self.framesize)
        self.offset = offset

    def record_read(self, count):
        """Read and decode count bytes.

        The range retrieved can span multiple frames and files.

        Parameters
        ----------
        count : int
            Number of bytes to read.

        Returns
        -------
        data : array of float
            Dimensions are [sample-time, vlbi-channel].
        """
        # for now only allow integer number of frames
        assert count % (self.recordsize * self.nthread) == 0
        data = np.empty((count // self.recordsize, self.npol),
                        dtype=np.float32)
        sample = 0
        while count > 0:
            # Validate frame we're reading from.
            full_set, full_set_offset = divmod(
                self.offset, self.payloadsize * self.nthread)
            payload_offset = full_set_offset // self.nthread
            self.seek(self.payloadsize * self.nthread * full_set)
            frame_start = self.fh_raw.tell()
            to_read = min(count, self.payloadsize - payload_offset)
            nsample = to_read * self.nthread // self.recordsize
            for i in range(self.nthread):
                self.fh_raw.seek(frame_start + self.framesize * i)
                header = VDIFFrameHeader.fromfile(self.fh_raw,
                                                  self.header0.edv)
                index = self.thread_indices[header['thread_id']]
                if index is None:  # Do not need this thread
                    continue

                if header['invalid_data']:
                    data[sample:sample + nsample, index] = 0.

                if payload_offset:
                    # Reading the header left file pointer at start of payload.
                    self.fh_raw.seek(payload_offset, 1)

                raw = np.fromstring(self.fh_raw.read(to_read), np.uint8)
                data[sample:sample + nsample, index] = self._decode(raw)

            # ensure offset pointers from raw and virtual match again.
            self.offset += to_read * self.nthread
            sample += nsample
            count -= to_read * self.nthread

        if self.npol == 2:
            data = data.view('{0},{0}'.format(data.dtype.str))

        return data

    # def _seek(self, offset):
    #     assert offset % self.recordsize == 0
    #     # Find the correct file.
    #     file_number = np.searchsorted(self.payloadranges, offset)
    #     self.open(self.files, file_number)
    #     if file_number > 0:
    #         file_offset = offset
    #     else:
    #         file_offset = offset - self.payloadranges[file_number - 1]
    #     # Find the correct frame within the file.
    #     frame_nr, frame_offset = divmod(file_offset, self.payloadsize)
    #     self.fh_raw.seek(frame_nr * self.framesize + self.header_size)
    #     self.offset = offset

    def __str__(self):
        return ('<VDIFData nthread={0} dtype={1} blocksize={2}\n'
                'current_file_number={3}/{4} current_file={5}>'
                .format(self.nthread, self.dtype, self.blocksize,
                        self.current_file_number, len(self.files),
                        self.files[self.current_file_number]))

# VDIF defaults for psrfits HDUs
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


VDIF_header = {  # (key name, (word-index, bit-start, nbit [, default]))
    'standard': (('invalid_data', (0, 31, 1, False)),
                 ('legacy_mode', (0, 30, 1, False)),
                 ('seconds', (0, 0, 30)),
                 ('ref_epoch', (1, 24, 6)),
                 ('frame_nr', (1, 0, 24, 0x0)),
                 ('_2_30_2', (2, 30, 2, 0x0)),
                 ('vdif_version', (2, 29, 3, 0x1)),
                 ('lg2_nchan', (2, 24, 5)),
                 ('frame_length', (2, 0, 24)),
                 ('complex_data', (3, 31, 1)),
                 ('bits_per_sample', (3, 26, 5)),
                 ('thread_id', (3, 16, 10, 0x0)),
                 ('station_id', (3, 0, 16)),
                 ('edv', (4, 24, 8))),
    1: (('sampling_unit', (4, 23, 1)),
        ('sample_rate', (4, 0, 23)),
        ('sync_pattern', (5, 0, 32, 0xACABFEED)),
        ('das_id', (6, 0, 32, 0x0)),
        ('_7_0_32', (7, 0, 32, 0x0))),
    3: (('frame_length', (2, 0, 24, 629)),  # Repeat, to set default.
        ('sampling_unit', (4, 23, 1)),
        ('sample_rate', (4, 0, 23)),
        ('sync_pattern', (5, 0, 32, 0xACABFEED)),
        ('loif_tuning', (6, 0, 32, 0x0)),
        ('_7_28_4', (7, 28, 4, 0x0)),
        ('dbe_unit', (7, 24, 4, 0x0)),
        ('if_nr', (7, 20, 4, 0x0)),
        ('subband', (7, 17, 3, 0x0)),
        ('sideband', (7, 16, 1, False)),
        ('major_rev', (7, 12, 4, 0x0)),
        ('minor_rev', (7, 8, 4, 0x0)),
        ('personality', (7, 0, 8))),
    4: (('sampling_unit', (4, 23, 1)),
        ('sample_rate', (4, 0, 23)),
        ('sync_pattern', (5, 0, 32)))}

# Also have mark5b over vdif (edv = 0xab)
# http://www.vlbi.org/vdif/docs/vdif_extension_0xab.pdf

# These need to be very fast look-ups, so do not use OrderedDict here.
VDIF_header_parsers = {}
for vk, vv in VDIF_header.items():
    VDIF_header_parsers[vk] = {}
    for k, v in vv:
        VDIF_header_parsers[vk][k] = make_parser(*v[:3])


ref_max = int(2. * (Time.now().jyear - 2000.)) + 1
ref_epochs = Time(['{y:04d}-{m:02d}-01'.format(y=2000 + ref // 2,
                                               m=1 if ref % 2 == 0 else 7)
                   for ref in range(ref_max)], format='isot', scale='utc',
                  precision=9)


class VDIFFrameHeader(object):
    def __init__(self, data, edv=None, verify=True):
        """Interpret a tuple of words as a VDIF Frame Header."""
        self.data = data
        if edv is None:
            self.edv = False if self['legacy_mode'] else self['edv']
        else:
            self.edv = edv

        if verify:
            self.verify()

    def verify(self):
        """Basic checks of header integrity."""
        if self.edv is False:
            assert self['legacy_mode']
            assert len(self.data) == 4
        else:
            assert not self['legacy_mode']
            assert self.edv == self['edv']
            assert len(self.data) == 8
            if self.edv == 1:
                assert self['sync_pattern'] == 0xACABFEED
            elif self.edv == 3:
                assert self['frame_length'] == 629
                assert self['sync_pattern'] == 0xACABFEED

    def same_stream(self, other):
        """Whether header is consistent with being from the same stream."""
        # Words 2 and 3 should be invariant, and edv should be the same.
        if not (self.edv == other.edv and
                all(self[key] == other[key]
                    for key in ('ref_epoch', 'vdif_version', 'frame_length',
                                'complex_data', 'bits_per_sample',
                                'station_id'))):
            return False

        if self.edv:
            # For any edv, word 4 should be invariant.
            return self.data[4] == other.data[4]
        else:
            return True

    @classmethod
    def frombytes(cls, s, edv=None, verify=True):
        """Read VDIF Header from bytes."""
        try:
            return cls(eight_word_struct.unpack(s), edv, verify)
        except:
            if edv:
                raise
            else:
                return cls(four_word_struct.unpack(s), False, verify)

    def tobytes(self):
        if len(self.data) == 8:
            return eight_word_struct.pack(*self.data)
        else:
            return four_word_struct.pack(*self.data)

    @classmethod
    def fromfile(cls, fh, edv=None, verify=True):
        """Read VDIF Header from file."""
        # Assume non-legacy header to ensure those are done fastest.
        s = fh.read(32)
        if len(s) != 32:
            raise EOFError
        self = cls(eight_word_struct.unpack(s), edv, verify=False)
        if self.edv is False:
            # Legacy headers are 4 words, so rewind, and remove excess data.
            fh.seek(-16, 1)
            self.data = self.data[:4]
        if verify:
            self.verify()

        return self

    @classmethod
    def fromvalues(cls, **kwargs):
        """Initialise a header from parsed values.

        Here, the parsed values must be given as keyword arguments, i.e.,
        for any header = cls(<somedata), cls.fromvalues(**header) == header.

        However, unlike for the 'fromkeys' class method, defaults for some
        keywords can be inferred from arguments named after header methods
        such as 'bps' and 'time'.

        Given defaults for standard header keywords:

        invalid_data : `False`
        legacy_mode : `False`
        vdif_version : 1
        thread_id: 0
        frame_nr: 0

        Defaults inferred from other keyword arguments (if present):

        bits_per_sample : from 'bps'
        frame_length : from 'payloadsize' (and 'legacy_mode')
        lg2_nchan : from 'nchan'

        Given defaults for edv 1 and 3:

        sync_pattern: 0xACABFEED (for edv = 1 and 3)

        Defaults inferred from other keyword arguments for all edv:

        station_id : from 'station'
        sample_rate, sample_unit : from 'bandwidth' or 'framerate'
        ref_epoch, seconds : from 'time'
        frame_nr : from 'time' (or 'seconds'), 'bps', 'chan', and 'bandwidth'
        """
        bps = kwargs.pop('bps', None)
        if bps is not None:
            kwargs.setdefault('bits_per_sample',
                              bps // (2 if kwargs['complex_data'] else 1) - 1)

        payloadsize = kwargs.pop('payloadsize', None)
        if payloadsize is not None:
            headersize = 16 if kwargs['legacy_mode'] else 32
            kwargs.setdefault('frame_length', (payloadsize + headersize) // 8)

        nchan = kwargs.pop('nchan', None)
        if nchan is not None:
            assert np.log2(nchan) % 1 == 0
            kwargs.setdefault('lg2_nchan', int(np.log2(nchan)))

        # Now create a legacy header such that we can access properties like
        # bps and nchan without concern.
        legacy_kwargs = {k: kwargs[k] for k, v in VDIF_header['standard']
                         if v[0] < 4}
        legacy_kwargs['legacy_mode'] = True
        legacy = cls.fromkeys(**legacy_kwargs)

        station = kwargs.pop('station', None)
        if station is not None:
            try:
                station_id = ord(station[0]) << 8 + ord(station[1])
            except TypeError:
                station_id = station
            assert int(station_id) == station_id
            kwargs.setdefault('station_id', station_id)

        framerate = kwargs.pop('framerate', None)
        if framerate is not None:
            framerate = framerate.to(u.Hz)
            assert framerate.value % 1 == 0
            kwargs.setdefault('bandwitdth',
                              framerate * legacy.samples_per_frame /
                              (2 * legacy.nchan))

        bandwidth = kwargs.pop('bandwidth', None)
        if bandwidth is not None:
            if bandwidth.unit == u.kHz or bandwidth.to(u.MHz).value % 1 != 0:
                assert bandwidth.to(u.kHz).value % 1 == 0
                kwargs.setdefault('sample_rate', bandwidth.to(u.kHz).value)
                kwargs.setdefault('sampling_unit', False)
            else:
                kwargs.setdefault('sample_rate', bandwidth.to(u.MHz).value)
                kwargs.setdefault('sampling_unit', True)

        time = kwargs.pop('time', None)
        if time is not None:
            assert time > ref_epochs[0]
            if time > ref_epochs[-1]:
                ref_index = len(ref_epochs) - 1
            else:
                ref_index = np.searchsorted((ref_epochs - time < 0).sec)
            kwargs.setdefault('ref_epoch', ref_index)

            kwargs.setdefault('seconds',
                              (time - ref_epochs[kwargs['ref_epoch']]).sec)

        if bandwidth is not None:
            int_sec, frac_sec = divmod(kwargs['seconds'], 1)
            kwargs.setdefault('frame_nr', frac_sec / legacy.samples_per_frame *
                              legacy.bandwidth.to(u.Hz).value)
            kwargs['seconds'] = int(int_sec)

        # Now should have everything set up for constructing the binary data.
        return cls.fromkeys(**kwargs)

    @classmethod
    def fromkeys(cls, **kwargs):
        """Like fromvalues, but without any interpretation of keywords."""
        # Get all required values.
        headers = VDIF_header['standard']
        if kwargs['legacy_mode']:
            words = [0] * 4
            # effectively, remove 'edv' from headers, which is in word4.
            headers = tuple((k, v) for (k, v) in headers if v[0] < 4)
        else:
            words = [0] * 8
            headers += VDIF_header[kwargs['edv']]

        for k, v in headers:
            if len(v) > 3:  # Set any default.
                kwargs.setdefault(k, v[3])
            assert k in kwargs and kwargs[k] & ((1 << v[2]) - 1) == kwargs[k]
            words[v[0]] |= int(kwargs.pop(k)) << v[1]

        if kwargs:
            warnings.warn("Some keywords unused in header initialisation: {0}"
                          .format(kwargs))

        return cls(words)

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
        for item in VDIF_header['standard']:
            yield item[0]
        if self.edv in VDIF_header:
            for item in VDIF_header[self.edv]:
                yield item[0]

    def __eq__(self, other):
        return (type(self) is type(other) and
                list(self.data) == list(other.data))

    def __contains__(self, key):
        return key in self.keys()

    def __repr__(self):
        return ("<VDIFFrameHeader {0}>".format(",\n                 ".join(
            ["{0}: {1}".format(k, (hex(self[k]) if k == 'sync_pattern' else
                                   self[k])) for k in self.keys()])))

    @property
    def size(self):
        return len(self.data) * 4

    @property
    def framesize(self):
        return self['frame_length'] * 8

    @property
    def payloadsize(self):
        return self.framesize - self.size

    @property
    def bps(self):
        bps = self['bits_per_sample'] + 1
        if self['complex_data']:
            bps *= 2
        return bps

    @property
    def nchan(self):
        return 2**self['lg2_nchan']

    @property
    def samples_per_frame(self):
        # Values are not split over word boundaries.
        values_per_word = 32 // self.bps
        # samples are not split over payload boundaries.
        return self.payloadsize // 4 * values_per_word // self.nchan

    @property
    def station(self):
        msb = self['station_id'] >> 8
        if 48 <= msb < 128:
            return chr(msb) + chr(self['station_id'] & 0xff)
        else:
            return self['station_id']

    @property
    def bandwidth(self):
        if not self['legacy_mode'] and self.edv:
            return u.Quantity(self['sample_rate'],
                              u.MHz if self['sampling_unit'] else u.kHz)
        else:
            return NotImplemented

    @property
    def framerate(self):
        # Could use self.bandwidth here, but speed up the calculation by
        # changing to a Quantity only at the end.
        if not self['legacy_mode'] and self.edv:
            return u.Quantity(self['sample_rate'] *
                              (1000000 if self['sampling_unit'] else 1000) *
                              2 * self.nchan / self.samples_per_frame, u.Hz)
        else:
            return NotImplemented

    @property
    def seconds(self):
        return self['seconds']

    def time(self, frame_nr=None, framerate=None):
        """
        Convert ref_epoch, seconds, and possibly frame_nr to Time object.

        Uses 'ref_epoch', which stores the number of half-years from 2000,
        and 'seconds'.  By default, it also calculates the offset using
        the current frame number.  For non-zero frame_nr, this requires the
        framerate, which is calculated from the header.  It can be passed on
        if this is not available (e.g., for a legacy VDIF header).

        Set frame_nr=0 to just get the header time from ref_epoch and seconds.
        """
        if frame_nr is None:
            frame_nr = self['frame_nr']

        if frame_nr == 0:
            offset = 0.
        else:
            if framerate is None:
                framerate = self.framerate
            offset = (frame_nr / framerate).to(u.s).value
        return (ref_epochs[self['ref_epoch']] +
                TimeDelta(self.seconds, offset, format='sec', scale='tai'))


class VDIFFileBase(object):
    """VDIF file wrapper."""
    def __init__(self, fh_raw):
        self.fh_raw = fh_raw
        self.nchan = self.header0.nchan
        # Set up buffer to hold frame being read or written.
        self._frame_nr = None
        self._headers = [None] * len(self.thread_ids)
        self._frame = np.empty(
            (self.header0.samples_per_frame, len(self.thread_ids), self.nchan),
            dtype='c8' if self.header0['complex_data'] else 'f4')
        self.offset = 0

    # Providing normal File IO properties.
    def readable(self):
        return self.fh_raw.readable()

    def writable(self):
        return self.fh_raw.writable()

    def seekable(self):
        return self.fh_raw.readable()

    def tell(self):
        """Return offset (in samples)."""
        return self.offset

    def close(self):
        return self.fh_raw.close()

    @property
    def closed(self):
        return self.fh_raw.closed

    def __repr__(self):
        return '<{0} name={1}>'.format(type(self).__name__, self.fh_raw.name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class VDIFFileReader(VDIFFileBase):
    """VLBI VDIF format reader.

    This wrapper is allows one to access a VDIF file as a continues series of
    samples.  Invalid data are marked, but possible gaps in the data stream
    are not yet filled in.

    Parameters
    ----------
    name : str
        file name
    thread_ids: list of int
        Specific threads to read.  By default, all threads are read.
    nthread: int
        Number of threads in file (if known; by default, the first 1024
        frames in a file will be scanned to determine the number of threads).
    """
    def __init__(self, name, thread_ids=None, nthread=None):
        fh_raw = open(name, 'rb')
        self.header0 = VDIFFrameHeader.fromfile(fh_raw)
        if nthread is None:
            file_thread_ids = get_thread_ids(fh_raw, self.header0.framesize)
        else:
            file_thread_ids = range(nthread)
        self.nthread = len(file_thread_ids)
        if thread_ids is None:
            self.thread_ids = sorted(file_thread_ids)
        else:
            assert all(thread_id in file_thread_ids
                       for thread_id in thread_ids)
            self.thread_ids = thread_ids
        self._decode = DECODERS[self.header0.bps, self.header0['complex_data']]
        super(VDIFFileReader, self).__init__(fh_raw)
        self.framerate = self.header0.framerate
        if self.framerate is NotImplemented:  # Not known; e.g., legacy header.
            self.framerate = get_frame_rate(fh_raw, VDIFFrameHeader) * u.Hz
        fh_raw.seek(0)

    @lazyproperty
    def header1(self):
        raw_offset = self.fh_raw.tell()
        self.fh_raw.seek(-self.header0.framesize, 2)
        header1 = find_frame(self.fh_raw, template_header=self.header0,
                             maximum=10*self.header0.framesize, forward=False)
        self.fh_raw.seek(raw_offset)
        if header1 is None:
            raise TypeError("Corrupt VDIF? No frame in last {0} bytes."
                            .format(10*self.header0.framesize))
        return header1

    @property
    def size(self):
        n_frames = round(((self.header1.time() - self.header0.time()).to(u.s) *
                          self.framerate).to(1).value) + 1
        return n_frames * self.header0.samples_per_frame

    def seek(self, offset, from_what=0):
        """Like normal seek, but with the offset in samples."""
        if from_what == 0:
            self.offset = offset
        elif from_what == 1:
            self.offset += offset
        elif from_what == 2:
            self.offset = self.size + offset
        return self.offset

    def read_frame(self, frame_nr=None):
        """Read a single frame (for all threads).

        Parameters
        ----------
        frame_nr : int
            Frame number, counting from the start of the file.  By default,
            read from the current file position.

        Returns
        -------
        data : array of float or complex
            Dimensions are (sample-time, thread, channel).
        headers : list of VDIFFrameHeader
            Length is number of threads (matching data's second dimension).
        """
        if frame_nr is None:
            frame_nr, extra = divmod(self.offset,
                                     self.header0.samples_per_frame)
            if extra:
                raise ValueError("Can only start reading a frame if "
                                 "at a frame boundary.")

        if frame_nr != self._frame_nr:
            frame_start = self.header0.framesize * self.nthread * frame_nr
            # Read the frames in the threads wanted.
            for i in range(self.nthread):
                self.fh_raw.seek(frame_start + self.header0.framesize * i)
                header = VDIFFrameHeader.fromfile(self.fh_raw,
                                                  self.header0.edv)
                try:
                    index = self.thread_ids.index(header['thread_id'])
                except ValueError:  # not one of the ones we want.
                    continue

                self._headers[index] = header
                raw = self._decode(np.fromstring(
                    self.fh_raw.read(self.header0.payloadsize), np.uint8))
                self._frame[:, index] = raw.reshape(-1, self.nchan)
            self._frame_nr = frame_nr

        # Place offset in virtual data stream after the frame.
        self.offset = self.header0.samples_per_frame * (frame_nr + 1)
        return self._frame, self._headers

    def read(self, count=None, fill_value=0., squeeze=True):
        """Read count samples.

        The range retrieved can span multiple frames and files.

        Parameters
        ----------
        count : int
            Number of samples to read.  If omitted or negative, the whole
            file is read.
        fill_value : float or complex
            Value to use for invalid or missing data.
        squeeze : bool
            If `True` (default), remove channel and thread dimensions if unity.

        Returns
        -------
        data : array of float or complex
            Dimensions are (sample-time, vlbi-thread, channel).
        """
        if count is None or count < 0:
            count = self.size - self.offset

        data = np.empty((count, len(self.thread_ids), self.nchan),
                        dtype=self._frame.dtype)

        offset0 = self.offset
        sample = 0
        while count > 0:
            full_frame_nr, sample_offset = divmod(
                offset0 + sample, self.header0.samples_per_frame)
            # Read relevant frame (or return buffer if already read).
            frame, _ = self.read_frame(full_frame_nr)
            # Copy relevant data from it.
            nsample = min(count,
                          self.header0.samples_per_frame - sample_offset)
            data[sample:sample + nsample] = frame[sample_offset:
                                                  sample_offset + nsample]
            sample += nsample
            count -= nsample

        # Ensure pointer is at right place.
        self.seek(offset0 + sample)

        return data.squeeze() if squeeze else data


class VDIFFileWriter(VDIFFileBase):
    """VLBI VDIF format writer.

    Parameters
    ----------
    name : str
        file name
    nthread : int
        number of threads the VLBI data has (e.g., 2 for 2 polarisations)
    header : VDIFFrameHeader
        Header for the first frame, holding time information, etc.

    If no header is give, an attempt is made to construct the header from the
    remaining keyword arguments.  For a standard header, this would include:

    time : `~astropy.time.Time` instance
        (or 'ref_epoch' + 'seconds')
    nchan : number of FFT channels within stream (default 1).
            (note: different # of channels per thread is not supported).
    frame_length : number of long words for header plus payload
                   (default 629 for edv=3)
    complex_data : whether data is complex
    bps : bits per sample (or 'bits_per_sample', which is bps - 1)
    station_id : 2 characters or unsigned 2-byte integer
    edv : 1, 3, or 4

    For edv = 1, 3, or 4, in addition, a required keyword is

    bandwidth : Quantity in Hz (or 'sampling_unit' + 'sample_rate')

    For other edv, one requires

    framerate : number of frames per second.
    """
    def __init__(self, name, nthread=1, header=None, **kwargs):
        self.nthread = nthread
        self.thread_ids = range(nthread)
        if header is None:
            header = VDIFFrameHeader.fromvalues(**kwargs)
        self.header0 = header
        self._encode = ENCODERS[header.bps, header['complex_data']]
        fh_raw = open(name, 'wb')
        super(VDIFFileWriter, self).__init__(fh_raw)

    def write_frame(self, data=None, headers=None):
        """Write a single frame.

        Parameters
        ----------
        data : float or complex array
            Should have dimension (samples_per_frame, nthread, nchan).
            By default, the current buffer is written.
        header : list of VDIFFrameHeader instances
            Should have length nthread.  By default, the current header list
            is written.
        """
        if data is None:
            data = self._frame
        if headers is None:
            headers = self._headers

        assert data.shape == (self.header0.samples_per_frame,
                              self.nthread,
                              self.nchan)
        assert len(headers) == self.nthread

        for header, payload in zip(headers, data.transpose(1, 0, 2)):
            self.fh_raw.write(header.tobytes())
            self.fh_raw.write(self._encode(payload.ravel()).tostring())

        self._frame_nr = (header['frame_nr'] +
                          int(self.header0.framerate.value) *
                          (header['seconds'] - self.header0['seconds']))
        self.offset = self._frame_nr * self.header0.samples_per_frame

    def write(self, data, squeezed=True, invalid_data=False):
        """Write data, buffering by frames as needed."""
        if squeezed:
            if self.nthread == 1:
                data = np.expand_dims(data, axis=1)
            if self.nchan == 1:
                data = np.expand_dims(data, axis=-1)

        assert data.shape[1] == self.nthread
        assert data.shape[2] == self.nchan

        count = data.shape[0]
        sample = 0
        offset0 = self.offset
        while count > 0:
            full_frame_nr, sample_offset = divmod(
                self.offset, self.header0.samples_per_frame)
            if sample_offset == 0:
                # set up headers.
                header_kwargs = dict(self.header0)
                dt, frame_nr = divmod(full_frame_nr,
                                      int(self.header0.framerate.value))
                header_kwargs['seconds'] += dt
                header_kwargs['frame_nr'] = frame_nr
                header_kwargs['invalid_data'] = invalid_data
                for i, thread_id in enumerate(self.thread_ids):
                    header_kwargs['thread_id'] = thread_id
                    self._headers[i] = VDIFFrameHeader.fromkeys(
                        **header_kwargs)
            nsample = min(count,
                          self.header0.samples_per_frame - sample_offset)
            sample_end = sample_offset + nsample
            self._frame[sample_offset:sample_end] = data[sample:
                                                         sample + nsample]
            if sample_end == self.header0.samples_per_frame:
                self.write_frame()

            sample += nsample
            self.offset = offset0 + sample
            count -= nsample

    def close(self):
        frame_nr, extra = divmod(self.offset, self.header0.samples_per_frame)
        if extra != 0:
            assert frame_nr == self._frame_nr
            warnings.warn("Closing with partial buffer remaining."
                          "Writing padded frame, marked as invalid.")
            self.write(np.zeros((self.nthread, extra, self.nchan)),
                       invalid_data=True)
            assert self.offset % self.header0.samples_per_frame == 0
        return super(VDIFFileWriter, self).close()


def open_vdif(name, mode='r', **kwargs):
    """Open VLBI VDIF format file for reading or writing.

    This wrapper is allows one to access a VDIF file as a series of samples.

    Parameters
    ----------
    name : str
        File name
    mode : str ('r' or 'w')
        Whether to open for reading or writing.

    For reading
    -----------
    thread_ids: list of int
        Specific threads to read.  By default, all threads are read.

    For writing
    -----------
    nthread : int
        number of threads the VLBI data has (e.g., 2 for 2 polarisations)
    header : VDIFFrameHeader
        Header for the first frame, holding time information, etc.
        (or keywords that can be used to construct a header).

    Returns
    -------
    Filehandler : VDIFFileReader or VDIFFileWriter instance

    Raises
    ------
    ValueError if an unsupported mode is chosen.
    """
    if mode[0] == 'r':
        return VDIFFileReader(name, **kwargs)
    elif mode[0] == 'w':
        return VDIFFileWriter(name, **kwargs)
    else:
        raise ValueError("Only support opening VDIF file for reading "
                         "or writing (mode='r' or 'w').")


def find_frame(fh, template_header=None, framesize=None, maximum=None,
               forward=True):
    """Look for the first occurrence of a frame, from the current position.

    Search for a valid header at a given position which is consistent with
    `other_header` or with a header a framesize ahead.   Note that the latter
    turns out to be an unexpectedly weak check on real data!
    """
    if template_header:
        framesize = template_header.framesize

    if maximum is None:
        maximum = 2 * framesize
    # Loop over chunks to try to find the frame marker.
    file_pos = fh.tell()
    # First check whether we are right at a frame marker (usually true).
    if template_header:
        try:
            header = VDIFFrameHeader.fromfile(fh, verify=True)
            if template_header.same_stream(header):
                fh.seek(-header.size, 1)
                return header
        except:
            pass

    if forward:
        iterate = range(file_pos, file_pos + maximum)
    else:
        iterate = range(file_pos, file_pos - maximum, -1)
    for frame in iterate:
        fh.seek(frame)
        try:
            header1 = VDIFFrameHeader.fromfile(fh, verify=True)
        except(AssertionError, IOError, EOFError):
            continue

        if template_header:
            if template_header.same_stream(header1):
                fh.seek(frame)
                return header1
            continue

        # if no comparison header given, get header from a frame further up or
        # down and check those are consistent.
        fh.seek(frame + (framesize if forward else -framesize))
        try:
            header2 = VDIFFrameHeader.fromfile(fh, verify=True)
        except AssertionError:
            continue
        except:
            break

        if(header2.same_stream(header1) and
           abs(header2.seconds - header1.seconds) <= 1 and
           abs(header2['frame_nr'] - header1['frame_nr']) <= 1):
            fh.seek(frame)
            return header1

    # Didn't find any frame.
    fh.seek(file_pos)
    return None


def get_thread_ids(infile, framesize, searchsize=None):
    """
    Get the number of threads and their ID's in a vdif file.
    """
    if searchsize is None:
        searchsize = 1024 * framesize

    n_total = searchsize // framesize

    thread_ids = set()
    for n in range(n_total):
        infile.seek(n * framesize)
        try:
            thread_ids.add(VDIFFrameHeader.fromfile(infile)['thread_id'])
        except:
            break

    return thread_ids


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


# Decoders keyed by bits_per_sample, complex_data:
DECODERS = {
    (2, False): lambda x: lut2bit[x].ravel(),
    (4, True): lambda x: lut2bit[x].reshape(-1, 2).view(np.complex64).squeeze()
}


def encode_2bit(values):
    if values.dtype.kind == 'c':
        values = values.astype(np.complex64).view(np.float32)

    values = values.reshape(-1, 4)

    bitvalues = np.sign(np.trunc(values / 2.)).astype(np.int32)
    # value < -2:     -1
    # -2 < value < 2:  0
    # value > 2:       1
    bitvalues += np.where(values < 0, 1, 2)
    # value < -2:      0
    # -2 < value < 0:  1
    #  0 < value < 2:  2
    #  value > 2:      3
    return (bitvalues << np.arange(0, 8, 2)).sum(-1).astype(np.uint8)

ENCODERS = {
    (2, False): encode_2bit,
    (4, True): encode_2bit
}
