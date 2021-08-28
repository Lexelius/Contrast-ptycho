"""
Notes:
Subclasses of PtyScan can be made to override to tweak the methods of base class PtyScan.
Methods defined in PtyScan(object) are:
    ¨def __init__(self, pars=None, **kwargs):
    def initialize(self):
    ¨def _finalize(self):
    ^def load_weight(self):
    ^def load_positions(self):
    ^def load_common(self):
    def post_initialize(self):
    ¨def _mpi_check(self, chunksize, start=None):
    ¨def _mpi_indices(self, start, step):
    def get_data_chunk(self, chunksize, start=None):
    def auto(self, frames):
    ¨def _make_data_package(self, chunk):
    ¨def _mpi_pipeline_with_dictionaries(self, indices):
    ^def check(self, frames=None, start=None):
    ^def load(self, indices):
    ^def correct(self, raw, weights, common):
    ¨def _mpi_autocenter(self, data, weights):
    def report(self, what=None, shout=True):
    ¨def _mpi_save_chunk(self, kind='link', chunk=None):

¨: Method is protected (or private if prefix is __).
^: Description explicitly says **Override in subclass for custom implementation**.
"""

import numpy as np
import zmq
from zmq.utils import jsonapi as json
import time
import bitshuffle
import struct
from ptypy.core.data import PtyScan
from ptypy import utils as u
from ptypy.utils import parallel
from ptypy import defaults_tree
from ptypy.experiment import register

# save_path = '/tmp/ptypy/sim/'
#
# geofilepath = save_path + 'geometry.txt'
# print(geofilepath)
# print(''.join([line for line in open(geofilepath, 'r')]))
#
# positionpath = save_path + 'positions.txt'
# print(positionpath)
#
# print(''.join([line for line in open(positionpath, 'r')][:6])+'....')
logger = u.verbose.logger


##@defaults_tree.parse_doc('scandata.LiveScan')
@register()
class LiveScan(PtyScan):
    """
    A PtyScan subclass to extract data from a numpy array.

    Defaults:

    [name]
    type = str
    default = LiveScan
    help =

    [host]
    default = '127.0.0.1'
    type = str
    help = Name of the publishing host
    doc =

    [port]
    default = 5556
    type = int
    help = Port number on the publishing host
    doc =
    """

    def __init__(self, pars=None, **kwargs):
        p = self.DEFAULT.copy(depth=99)
        p.update(pars)
        p.update(kwargs)
        #
        # with open(p.base_path+'geometry.txt') as f:
        #     for line in f:
        #         key, value = line.strip().split()
        #         # we only replace Nones or missing keys
        #         if p.get(key) is None:
        #             p[key] = eval(value)
        # super(LiveScan, self).__init__(p, **kwargs)
        super(LiveScan, self).__init__(p, **kwargs)
        self.context = zmq.Context()

        # main socket
        socket = self.context.socket(zmq.SUB)
        socket.connect("tcp://%s:%u" % (self.info.host, self.info.port))
        socket.setsockopt(zmq.SUBSCRIBE, b"")  # subscribe to all topics
        self.socket = socket

        self.latest_pos_index_received = -1
        self.incoming = {}

    def check(self, frames=None, start=None):
        """
        Only called on the master node.
        """
        logger.info("Enters LiveScan.check(..)")  ##
        end_of_scan, end_of_det_stream = False, False

        # get all frames from the main socket
        ## This counts the "stutus: Heartbeat" message as a frame as well
        ## and seems to proccess this as actual data, thus causing error!!
        while True:
            try:
                msg = self.socket.recv_pyobj(flags=zmq.NOBLOCK) ##rde c§§1    NOBLOCK returns None if a message is not ready
                logger.info('inside "try:", msg is: %s' % msg)##
                ##headers = ('path' in msg.keys())
                ##emptymsg = ('heartbeat' in msg.values()) # just a message from contrast to keep connection alive
                if ('running' in msg.values()): # if zmq did not send a path: then save this message
                    self.latest_pos_index_received += 1
                    self.incoming[self.latest_pos_index_received] = msg
                    logger.info('inside "try: if ..:"')##
                elif msg['status'] == self.EOS:
                    end_of_scan = True
                    logger.info('inside "try: elif ..:"')##
                    break
            except zmq.ZMQError:
                logger.info('inside "except zmq.ZMQError:"')##
                # no more data available - working around bug in ptypy here
                if self.latest_pos_index_received < self.info.min_frames * parallel.size:
                    logger.info('self.latest_pos_index_received = %u , self.info.min_frames = %d , parallel.size = %d' % (self.latest_pos_index_received, self.info.min_frames, parallel.size))  ##
                    logger.info('have %u frames, waiting...' % (self.latest_pos_index_received + 1))
                    time.sleep(1)
                else:
                    logger.info('inside "except/else, will break')##
                    break

        logger.info('-------------------------------------')
        logger.info(self.incoming.keys())
        logger.info('-------------------------------------')
        ind = self.latest_pos_index_received
        return (ind - start + 1), (end_of_scan and end_of_det_stream)


    def load(self, indices):
        # indices are generated by PtyScan's _mpi_indices method.
        # It is a diffraction data index lists that determine
        # which node contains which data.
        raw, weight, pos = {}, {}, {}

        # communication
        if parallel.master:
            # send data to each node
            for node in range(1, parallel.size):
                node_inds = parallel.receive(source=node)
                dct = {i: self.incoming[i] for i in node_inds}
                parallel.send(dct, dest=node)
                for i in node_inds:
                    del self.incoming[i]
            # take data for this node
            dct = {i: self.incoming[i] for i in indices}
            for i in indices:
                del self.incoming[i]
        else:
            # receive data from the master node
            parallel.send(indices, dest=0)
            dct = parallel.receive(source=0)
        logger.info(dct)

        # repackage data and return
        for i in indices:
            try:
                raw[i] = dct[i]['diff']
                # raw[i] = dct[i][self.info.detector]
                # #            pos[i] = np.array([
                # #                        dct[i][self.info.xMotor],
                # #                        dct[i][self.info.yMotor],
                # #                        ]) * 1e-6
                # x = dct[i][self.info.xMotor]
                # y = dct[i][self.info.yMotor]
                x = dct[i]['x']
                y = dct[i]['y']
                pos[i] = -np.array((y, x)) * 1e-6
                weight[i] = np.ones_like(raw[i])
                weight[i][np.where(raw[i] == 2 ** 32 - 1)] = 0

                # d = msg['iterable'][0]
                # dct = {'x': d['position'][0],
                #        'y': d['position'][1],
                #        'diff': d['data']}
            except:
                break

        return raw, pos, weight


LS = LiveScan()
LS.initialize()
# # The decorator extracts information from the docstring of the subclass and
# # parent classes about the expected input parameters. Currently the docstring
# # of `LiveScan` does not contain anything special, thus the only parameters
# # registered are those of the parent class, `PtyScan`:
# print(defaults_tree['scandata.livescan'].to_string())
#
# # As you can see, there are already many parameters documented in `PtyScan`'s
# # class. For each parameter, most important are the *type*, *default* value and
# # *help* string. The decorator does more than collect this information: it also
# # generates from it a class variable called `DEFAULT`, which stores all defaults:
# print(u.verbose.report(LiveScan.DEFAULT, noheader=True))
#
# # We now need a new input parameter called `base_path`, so we documented it
# # in the docstring after the section header "Defaults:".
# print(defaults_tree['scandata.livescan.base_path'])
#
# # As you can see, the first step in `__init__` is to build a default
# # parameter structure to ensure that all input parameters are available.
# # The next line updates this structure to overwrite the entries specified by
# # the user.
#
# # Good! Next, we need to implement how the class finds out about
# # the positions in the scan. The method
# # :py:meth:`~ptypy.core.data.PtyScan.load_positions` can be used
# # for this purpose.
# print(PtyScan.load_positions.__doc__)
#
# # One nice thing about rewriting ``self.load_positions`` is that
# # the maximum number of frames will be set and we do not need to
# # manually adapt :py:meth:`~ptypy.core.data.PtyScan.check`
#
# # The last step is to overwrite the actual loading of data.
# # Loading happens (MPI-compatible) in
# # :py:meth:`~ptypy.core.data.PtyScan.load`
# print(PtyScan.load.__doc__)
#
# # Load seems a bit more complex than ``self.load_positions`` for its
# # return values. However, we can opt-out of providing weights (masks)
# # and positions, as we have already adapted ``self.load_positions``
# # and there were no bad pixels in the (linear) detector
#

# %%
from ptypy.core import Ptycho
from ptypy import utils as u

# PtyScan.__subclasses__()
# print('')
# import ptypy
# for name in (u.all_subclasses(ptypy.core.data.PtyScan, names=True)):
#     print(name)

p = u.Param()

# for verbose output
p.verbose_level = 3

# set home path
p.io = u.Param()
p.io.home = "/tmp/ptypy/" ## '/Users/lexelius/Documents/Contrast/temp/'
##p.io.autosave = None

# max 200 frames (128x128px) of diffraction data
p.scans = u.Param()
p.scans.contrast = u.Param()
# now you have to specify which ScanModel to use with scans.XX.name,
# just as you have to give 'name' for engines and PtyScan subclasses.
p.scans.contrast.name = 'Full'
p.scans.contrast.data = u.Param()
p.scans.contrast.data.name = 'LiveScan'  # 'LiveScan'
p.scans.contrast.data.min_frames = 5
## p.scans.contrast.data.xMotor = 'x'
## p.scans.contrast.data.yMotor = 'y'
p.scans.contrast.data.host = 'localhost'
p.scans.contrast.data.port = 5556

# attach a reconstrucion engine
p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM'
p.engines.engine00.numiter = 300
p.engines.engine00.numiter_contiguous = 2

# prepare and run
P = Ptycho(p, level=5)

# fig = u.plot_storage(list(P.diff.storages.values())[0], 0, slices=(slice(2), slice(None), slice(None)), modulus='log')
#
# rfile0 = '/tmp/ptypy/dumps/pydevconsole/pydevconsole_DM_0010.ptyr'
# rfile1 = '/tmp/ptypy/recons/pydevconsole/pydevconsole_DM_0300_first_completedlivescan_reconstruction.ptyr'
# load_ptyr0 = u.scripts.load_from_ptyr(rfile0)
# load_ptyr1 = u.scripts.load_from_ptyr(rfile1)
# from ptypy.utils import imsave
# from matplotlib import pyplot as plt
# pil0 = imsave(load_ptyr0[0,:,:], filename=rfile0.split('.ptyr')[0] + '.png', vmin=None, vmax=None, cmap=None)
# pil1 = imsave(load_ptyr1[0,:,:], filename=rfile1.split('.ptyr')[0] + '.png', vmin=None, vmax=None, cmap=None)
#
#
# # imsave doesn't want to accept data type of hsv for some reason
# hsv0 = u.plot_utils.complex2hsv(load_ptyr0[0,:,:], vmin=0., vmax=None)
# hsv1 = u.plot_utils.complex2hsv(load_ptyr1[0,:,:], vmin=0., vmax=None)
# pil0 = imsave(hsv0, filename=rfile0.split('.ptyr')[0] + '_hsv.png', vmin=None, vmax=None, cmap=None)
# pil1 = imsave(hsv1, filename=rfile0.split('.ptyr')[0] + '_hsv.png', vmin=None, vmax=None, cmap=None)
# plt.imshow(pil0)
# plt.show()
#
#



#%%
# # Loading the data
# # ----------------
#
# # With the subclass we create a scan only using defaults
# NPS = LiveScan()
# NPS.initialize()
#
# # In order to process the data. We need to call
# # :py:meth:`~ptypy.core.data.PtyScan.auto` with the chunk size
# # as arguments. It returns a data chunk that we can inspect
# # with :py:func:`ptypy.utils.verbose.report`. The information is
# # concatenated, but the length of iterables or dicts is always indicated
# # in parantheses.
# print(u.verbose.report(NPS.auto(80), noheader=True))
# print(u.verbose.report(NPS.auto(80), noheader=True))
#
# # We observe the second chunk was not 80 frames deep but 34
# # as we only had 114 frames of data.
#
# # So where is the *.ptyd* data-file? As default, PtyScan does not
# # actually save data. We have to manually activate it in in the
# # input paramaters.
# data = NPS.DEFAULT.copy(depth=2)
# data.save = 'append'
# NPS = LiveScan(pars=data)
# NPS.initialize()
# for i in range(50):
#     msg = NPS.auto(20)
#     if msg == NPS.EOS:
#         break
#
# # We can analyse the saved ``npy.ptyd`` with
# # :py:func:`~ptypy.io.h5IO.h5info`
# from ptypy.io import h5info
# print(h5info(NPS.info.dfile))
#
