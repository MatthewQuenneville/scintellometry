from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import warnings

import numpy as np
import astropy.units as u

from astropy.time import Time, TimeDelta

from ..vlbi_helpers import make_parser, four_word_struct, eight_word_struct


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
