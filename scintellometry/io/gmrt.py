#    ____  ___ ___  ____  ______
#   /    ||   |   ||    \|      |
#  |   __|| _   _ ||  D  )      |
#  |  |  ||  \_/  ||    /|_|  |_|
#  |  |_ ||   |   ||    \  |  |
#  |     ||   |   ||  .  \ |  |
#  |___,_||___|___||__|\_| |__|
#
from __future__ import division

import numpy as np
from scipy.fftpack import fftfreq, fftshift
from astropy.time import Time, TimeDelta
import astropy.units as u
from astropy.coordinates import Angle

from . import MultiFile, header_defaults
from .fromfile import fromfile


class GMRTBase(MultiFile):
    """This is a base class for GMRTPhasedData and GMRTRawDumpData.
    It contains all common initialisation and methods.  Do not use directly.
    """
    def __init__(self, raw_files, blocksize, nchan,
                 samplerate, fedge, fedge_at_top, dtype, comm=None):
        self.samplerate = samplerate
        self.fedge = fedge
        self.fedge_at_top = fedge_at_top
        f = fftshift(fftfreq(nchan, (2./samplerate).to(u.s).value)) * u.Hz
        if fedge_at_top:
            self.frequencies = fedge - (f-f[0])
        else:
            self.frequencies = fedge + (f-f[0])

        super(GMRTBase, self).__init__(raw_files, blocksize, dtype, nchan,
                                       comm=comm)
        self.dtsample = (nchan * (2 if self.data_is_complex else 1) /
                         samplerate).to(u.s)

    def __repr__(self):
        return ("<open raw_voltage_file(s) {0} "
                "using timestamp file '{1}' at index {2} (time {3})>"
                .format(self.fh_raw, self.timestamp_file,
                        self.offset, self.time().iso))


class GMRTPhasedData(GMRTBase):

    telescope = 'gmrt'

    def __init__(self, timestamp_file, raw_files, blocksize, nchan,
                 samplerate, fedge, fedge_at_top, dtype='ci1',
                 utc_offset=5.5*u.hr, time_offset=0.0*u.s, comm=None):
        """GMRT phased data stored in blocks holding 0.25 s worth of data,
        separated over two streams (each with 0.125s).  For 16MHz BW, each
        block is 4 MiB with 2Mi complex samples split in 256 or 512 channels.
        Complex samples consist of two signed ints (custom 'ci1' dtype).
        """
        self.timestamp_file = timestamp_file
        (self.indices, self.timestamps,
         self.gsb_start) = read_timestamp_file_phased(timestamp_file,
                                                      utc_offset)
        self.time0 = self.timestamps[0] + time_offset
        # GMRT time is off by one 32MB record ---- remove for now
        # self.time0 -= (2.**25/samplerate).to(u.s)
        super(GMRTPhasedData, self).__init__(raw_files, blocksize, nchan,
                                             samplerate, fedge, fedge_at_top,
                                             dtype, comm)


class GMRTRawDumpData(GMRTBase):

    telescope = 'gmrt-raw'

    def __init__(self, timestamp_file, raw_files, blocksize, nchan=1,
                 samplerate=(200./6.)*u.MHz, fedge=None, fedge_at_top=None,
                 sample_offsets=None, dtype='4bit', utc_offset=5.5*u.hr,
                 comm=None, feeds=None):
        """GMRT raw dump data stored in blocks holding 0.251 s worth of data,
        in a single streams.  For 16MHz BW, each block is 4 MiB w/ 8Mi 4-bit
        baseband samples.
        """
        self.timestamp_file = timestamp_file
        self.timestamps = read_timestamp_file_rawdump(timestamp_file,
                                                      utc_offset)
        self.time0 = self.timestamps[0]
        self.nchan = 1
        self.npol = len(raw_files)
        self.feeds = feeds if isinstance(feeds,list) else [feeds]
        # GMRT time is off by one 32MB record ---- remove for now
        # self.time0 -= (2.**25/samplerate).to(u.s)
        super(GMRTRawDumpData, self).__init__(raw_files, blocksize, nchan,
                                              samplerate, fedge, fedge_at_top,
                                              dtype, comm)
        from scintellometry.phasing.fringeStopClass import FringeStopper
        self.fringe_stop = FringeStopper(Time('2015-04-27T15:01:45.360721848', format='isot', scale='utc'))
        self.update_delays(self.time0)
        """
        print(self.sample_offsets)
        if sample_offsets is None:
            self.sample_offsets = (0,) * len(raw_files)
        else:
            self.sample_offsets = []
            self.cable_delays = []
            print(self.sample_offsets)
            for sample_offset in sample_offsets:
                if not hasattr(sample_offset, 'unit'):
                    sample_offset = self.tell(sample_offset, u.s).value

                one_byte = self.tell(1, u.s).value
                offset, delay = divmod(sample_offset, one_byte)
                print offset, delay
                self.sample_offsets.append(int(offset))
                self.cable_delays.append(delay * u.s)
            self.cable_delays=[0.0*u.s,6e-8*u.s]
            self.sample_offsets=[0,77]
            print(self.sample_offsets)
            """
         
    def update_delays(self, time):
        delays_and_phases=self.fringe_stop.get_delay_and_phase(time, self.feeds)
        self.sample_offsets = [int(round(delays_and_phases[feed[:-1]][0].value))
                               for feed in self.feeds]
        self.phases = [delays_and_phases[feed[:-1]][1] for feed in self.feeds]

    def _seek(self, offset):
        if offset % self.recordsize != 0:
            raise ValueError("Cannot offset to non-integer number of records")

        # determine index in units of the blocksize
        block = offset // self.blocksize
        if block >= len(self.timestamps):
            raise EOFError('At end of file in MultiFile.read')

        # actual seeks in files
        for fh_raw, sample_offset in zip(self.fh_raw, self.sample_offsets):
            fh_raw.seek(offset + sample_offset)

        self.offset = offset

    def read(self, size):
        """Read size bytes, returning an ndarray with np.int8 dtype.

        Incorporate multiple files with different polarisation.

        The individual file pointers are assumed to be pointing at the right
        locations, i.e., just before data that will be read here.
        """
        if size % self.recordsize != 0:
            raise ValueError("Cannot read a non-integer number of records")

        # ensure we do not read beyond end
        size = min(size, len(self.timestamps) * self.blocksize - self.offset)
        if size <= 0:
            raise EOFError('At end of file in MultiFile.read')

        z = np.empty((self.npol, size), dtype=np.int8)

        # read one or more pieces.
        for zpol, fh_raw, sample_offset in zip(z, self.fh_raw,
                                               self.sample_offsets):
            iz = 0
            offset = self.offset
            while(iz < size):
                block, already_read = divmod(offset + sample_offset,
                                             self.blocksize)
                fh_size = min(size - iz, self.blocksize - already_read)
                zpol[iz:iz+fh_size] = np.fromstring(fh_raw.read(fh_size),
                                                    dtype=zpol.dtype)

                offset += fh_size
                iz += fh_size

        self.offset += size

        return z.T  # Ensure output has shape (size, npol)

    def record_read(self, count):
        out = fromfile(self, self.dtype, count)
        if self.npol == 1:
            return out
        else:
            # fromfile decodes each byte into 2 samples and thus the decoded
            # output has shape (size, npol, 2).  We need to get npol as the
            # last dimension.
            return out.reshape(-1, self.npol, 2).transpose(0, 2, 1).flatten().view(
                '{0},{0}'.format(out.dtype.str))


# GMRT defaults for psrfits HDUs
# Note: these are largely made-up at this point
header_defaults['gmrt'] = {
    'PRIMARY': {'TELESCOP':'GMRT',
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

header_defaults['gmrt-raw'] = header_defaults['gmrt']


def read_timestamp_file_phased(filename, utc_offset=5.5*u.hr):
    """Read timestamps from GMRT phased array timestamp file.

    Parameters
    ----------
    filename : str
        full path to the timestamp file
    utc_offset : Quantity or TimeDelta
        offset from UTC, subtracted from the times in the timestamp file.
        Default: 5.5*u.hr

    Returns
    -------
    indices : array of int
        list of indices (alternating 0 and 1) into the two raw data files
    timestamps : Time array
        UTC times associated with the data blocks
    gsb_start : Time
        UTC time at which the GMRT software correlator was started

    Notes
    -----

    A typical first line of a timestamp file is:

    2014 01 20 02 28 10 0.811174 2014 01 20 02 28 10 0.622453760 5049 1

    Here, the first set is the time as given by the PC that received the
    block, the second that from GPS.  This is followed by a sequence number
    and a sub-integration number.  These should increase monotonically.

    The code checks that the time difference PC-GPS is (roughly) constant,
    and that the first sequence was triggered on an integer GPS minute.

    The actual data are stored in two interleaved streams, and the routine
    returns a Time array that is twice the length of the time-stamp file,
    having interpolated the times for the second data stream.
    """

    utc_offset = TimeDelta(utc_offset)

    str2iso = lambda str: '{}-{}-{}T{}:{}:{}'.format(
        str[:4], str[5:7], str[8:10], str[11:13], str[14:16], str[17:19])
    dtype = np.dtype([('pc', 'S19'), ('pc_frac', np.float),
                      ('gps', 'S19'), ('gps_frac', np.float),
                      ('seq', np.int), ('sub', np.int)])
    timestamps = np.genfromtxt(filename, dtype=dtype,
                               delimiter=(19, 10, 20, 12, 5, 2),  # col lengths
                               converters={0: str2iso, 2: str2iso})

    # check if last line was corrupted
    if timestamps[-1]['sub'] < 0:
        timestamps = timestamps[:-1]

    # should have continuous series, of subintegrations at least
    assert np.all(np.diff(timestamps['sub']) % 8 == 1)  # either 1 or -7

    pc_times = (Time(timestamps['pc'], scale='utc', format='isot') +
                TimeDelta(timestamps['pc_frac'], format='sec') - utc_offset)
    gps_times = (Time(timestamps['gps'], scale='utc', format='isot') +
                 TimeDelta(timestamps['gps_frac'], format='sec') - utc_offset)

    gps_pc = gps_times - pc_times
    assert np.allclose(gps_pc.sec, gps_pc[0].sec, atol=5.e-3)

    # GSB should have started on whole minute
    gsb_start = gps_times[0] - timestamps[0]['seq'] * (gps_times[1] -
                                                       gps_times[0])
    assert '00.000' in gsb_start.isot

    # still, the sequence can have holes of 8, which need to be filled
    seq = timestamps['seq']
    dseq = np.diff(seq)
    holes = np.where(dseq > 1)
    # hole points to time just before hole
    for hole in holes[0][::-1]:  # reverse order since we are adding stuff
        hole_dt = gps_times[hole+1] - gps_times[hole]
        hole_frac = np.arange(1, dseq[hole], dtype=np.int) / float(dseq[hole])
        hole_times = gps_times[hole] + hole_frac * hole_dt
        gps_times = Time([gps_times[:hole+1], hole_times,
                          gps_times[hole+1:]])
        seq = np.hstack((seq[:hole+1], -np.ones(len(hole_frac)), seq[hole+1:]))

    # time differences between subsequent samples should now be (very) similar
    dt = gps_times[1:] - gps_times[:-1]
    assert np.allclose(dt.sec, dt[0].sec, atol=1.e-5)

    indices = np.repeat([[0,1]], len(gps_times), axis=0)
    # double the number of timestamps
    times = Time(np.repeat(gps_times.jd1, 2), np.repeat(gps_times.jd2, 2),
                 format='jd', scale='utc', precision=9)
    times = times + indices.flatten() * (dt[0] / 2.)
    # mark bad indices
    indices[seq < 0] = np.array([-1,-1])

    return indices.flatten(), times, gsb_start


def read_timestamp_file_rawdump(filename, utc_offset=5.5*u.hr):
    """Read timestamps from GMRT raw dump timestamp file.

    Parameters
    ----------
    filename : str
        full path to the timestamp file
    utc_offset : Quantity or TimeDelta
        offset from UTC, subtracted from the times in the timestamp file.
        Default: 5.5*u.hr

    Returns
    -------
    timestamps : Time array
        UTC times associated with the data blocks

    Notes
    -----

    A typical first line of a timestamp file for rawdump is:
    1234567899012345678990123456789
    2015 04 27 18 45 00 0.000000240

    This the time as given by the PC that received the block.
    """
    utc_offset = TimeDelta(utc_offset)

    str2iso = lambda str: '{}-{}-{}T{}:{}:{}'.format(
        str[:4], str[5:7], str[8:10], str[11:13], str[14:16], str[17:19])
    dtype = np.dtype([('pc', 'S19'), ('pc_frac', np.float)])
    timestamps = np.genfromtxt(filename, dtype=dtype,
                               delimiter=(19, 10),  # col lengths
                               converters={0: str2iso})

    pc_times = (Time(timestamps['pc'], scale='utc', format='isot',
                     precision=9) +
                TimeDelta(timestamps['pc_frac'], format='sec') - utc_offset)

    # time differences between subsequent samples should now be (very) similar
    dt = pc_times[1:] - pc_times[:-1]
    assert np.allclose(dt.sec, dt[0].sec, atol=1.e-5)

    return pc_times

nodes_and_voltages = {
    'C00R':(0,1), 'C00L':(8,1),
    'C01R':(0,2), 'C01L':(8,2),
    'C02R':(0,3), 'C02L':(8,3),
    'C03R':(0,4), 'C03L':(8,4),
    'C04R':(1,1), 'C04L':(9,1),
    'C05R':(1,2), 'C05L':(9,2),
    'C06R':(1,3), 'C06L':(9,3),
    'C08R':(1,4), 'C08L':(9,4),
    'C09R':(2,1), 'C09L':(10,1),
    'C10R':(2,2), 'C10L':(10,2),
    'C11R':(2,3), 'C11L':(10,3),
    'C12R':(2,4), 'C12L':(10,4),
    'C13R':(3,1), 'C13L':(11,1),
    'C14R':(3,2), 'C14L':(11,2),
    'W01R':(3,3), 'W01L':(11,3),
    'W02R':(3,4), 'W02L':(11,4),
    'W03R':(4,1), 'W03L':(12,1),
    'W04R':(4,2), 'W04L':(12,2),
    'W05R':(4,3), 'W05L':(12,3),
    'W06R':(4,4), 'W06L':(12,4),
    'E02R':(5,1), 'E02L':(13,1),
    'E03R':(5,2), 'E03L':(13,2),
    'E04R':(5,3), 'E04L':(13,3),
    'E05R':(5,4), 'E05L':(13,4),
    'E06R':(6,1), 'E06L':(14,1),
    'S01R':(6,2), 'S01L':(14,2),
    'S02R':(6,3), 'S02L':(14,3),
    'S03R':(6,4), 'S03L':(14,4),
    'S04R':(7,1), 'S04L':(15,1),
    'S06R':(7,2), 'S06L':(15,2)
}
