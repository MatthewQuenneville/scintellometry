import io
import warnings

import numpy as np
import astropy.units as u
from astropy.utils import lazyproperty

from ..vlbi_helpers import get_frame_rate
from .header import VDIFFrameHeader
from .coders import DECODERS, ENCODERS

# Check code on 2015-MAY-29
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
# fh = vdif.open('evn/Fd/GP052D_FD_No0006.m5a', 'r')
# d, h = fh.read_frame()
# d.astype(int)[:, 1, 0][:12]  # thread id = 1!!
# -> array([ 1,  1,  1, -3,  1,  1, -3, -3, -3,  3,  3, -1])  -> OK
# Also, next frame (thread #3)
# m5d evn/Fd/GP052D_FD_No0006.m5a VDIF_5000-512-1-2 12 5032
# -1  1 -1  1 -3 -1  3 -1  3 -3  1  3
# d.astype(int)[:, 3, 0][:12]
# -> array([-1,  1, -1,  1, -3, -1,  3, -1,  3, -3,  1,  3])
# And first thread #0
# m5d evn/Fd/GP052D_FD_No0006.m5a VDIF_5000-512-1-2 12 20128
# -1 -1  3 -1  1 -1  3 -1  1  3 -1  1
# d.astype(int)[:, 0, 0][:12]
# -> array([-1, -1,  3, -1,  1, -1,  3, -1,  1,  3, -1,  1])
# sanity check that we can also just read 12 samples
# fh.seek(0)
# fh.read(12).astype(int)[:, 0]
# -> array([-1, -1,  3, -1,  1, -1,  3, -1,  1,  3, -1,  1])


class VDIFFileBase(object):
    """VDIF file wrapper.  Base for VDIFFileReader and VDIFFileWriter."""
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
        fh_raw = io.open(name, 'rb')
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

        The range retrieved can span multiple frames.

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
        fh_raw = io.open(name, 'wb')
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


def open(name, mode='r', **kwargs):
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
