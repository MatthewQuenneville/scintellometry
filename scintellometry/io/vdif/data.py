from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import warnings

import astropy.units as u

from .. import SequentialFile, header_defaults
from .base import open as open_vdif


class VDIFData(SequentialFile):

    telescope = 'vdif'

    def __init__(self, raw_files, channels, fedge, fedge_at_top,
                 blocksize=None, open_raw=open_vdif, comm=None):
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
        comm : MPI communicator
            For consistency with other readers.
        """
        if len(raw_files) > 1:
            raise ValueError("Can only handle single file for now.")
        self.fedge = fedge
        self.fedge_at_top = fedge_at_top
        if channels is not None and not isinstance(channels, list):
            channels = list(channels)
        with open_raw(raw_files[0], 'r', thread_ids=channels) as checkfile:
            self.header0 = header = checkfile.header0
            self.npol = len(checkfile.thread_ids)
            dtype = checkfile._frame.dtype.str[1:]
            self.samplerate = (checkfile.framerate *
                               header.samples_per_frame).to(u.MHz)
            self.time0 = header.time()
            if blocksize is None:
                blocksize = header.payloadsize * checkfile.nthread
            # convert blocksize from raw vdif file bytes to
            # number of bytes of real/complex output.
            blocksize = (blocksize // 4 * (32 // header.bps) //
                         checkfile.nthread *
                         self.npol * checkfile._frame.dtype.itemsize)

        if header.nchan > 1:
            # This needs more thought, though a single thread with multiple
            # channels should be easy, as it is similar to other formats
            # (just need to calculate frequencies).  But multiple channels
            # over multiple threads may not be so easy.
            raise ValueError("Multi-channel vdif not yet supported.")

        # For normal folding, 1 or 2 channels should be given, but for other
        # reading, it may be useful to have all channels available.
        if not (1 <= self.npol <= 2):
            warnings.warn("Should use 1 or 2 channels for folding!")

        # SOMETHING LIKE THIS NEEDED FOR MULTIPLE FILES!
        # PROBABLY SHOULD MAKE SEQUENTIALFILE READER SEEK
        # OFFSET IN TOTAL BYTE SIZE, IGNORING HEADERS
        # self.totalfilesizes = np.array([os.path.getsize(fil)
        #                                 for fil in raw_files], dtype=np.int)
        # assert np.all(self.totalfilesizes // header.framesize == 0)
        # self.totalpayloads = (self.totalfilesizes // header.framesize *
        #                       header.payloadsize)
        # self.payloadranges = self.totalpayloads.cumsum()
        self.dtsample = (header.nchan / self.samplerate).to(u.ns)
        if comm is None or comm.rank == 0:
            print("In VDIFData, calling super")
            print("Start time: ", self.time0.iso)
        super(VDIFData, self).__init__(raw_files, blocksize, dtype,
                                       header.nchan, open_raw=open_raw,
                                       comm=comm)

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
        # for now ignore multiple possible files.
        assert count % self.recordsize == 0
        data = self.fh_raw.read(count // self.recordsize)
        assert data.shape[0] == count // self.recordsize
        if self.npol == 2:
            data = data.view('{0},{0}'.format(data.dtype.str))

        return data

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
