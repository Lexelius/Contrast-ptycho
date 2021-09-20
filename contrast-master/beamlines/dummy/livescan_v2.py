"""
To be run before second cell- and after first cell of 'v2_dummy_beamline.py'.

----------------------------------------------------------------------------
Things to fix / troubleshooting notes:
----------------------------------------------------------------------------
* If num_iter have been reached before all patterns have been collected then
    Ptycho stops because it thinks it's finished!
    -> Define a number of iterations for recon that starts counting only after scan is over: yes!
* If 'p.scans.contrast.data.shape = 128' is included in the livescan.py script then
    ptycho-reconstruction becomes super fast and stops before scan is over as above.
* Check: how to specify " kind = 'full_flat' " as input for save_run(..) that is called by Ptycho.
    Or rather, is there a way to save the pods to the .ptyr files as well?
* Check if there is a way to see how many pods/frames that are included in each .ptyr file.
 """

"""
Notes:
-----------------------------------------------------------
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
from ptypy.utils.verbose import headerline

logger = u.verbose.logger
def logger_info(*arg):
    return

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

        logger.info(headerline('', 'c', '#'))
        logger.info(headerline('Entering LiveScan().init()', 'c', '#'))
        logger.info(headerline('', 'c', '#'))
        p = self.DEFAULT.copy(depth=99)
        p.update(pars)
        p.update(kwargs)

        super(LiveScan, self).__init__(p, **kwargs)
        self.context = zmq.Context()

        # main socket
        socket = self.context.socket(zmq.SUB)
        socket.connect("tcp://%s:%u" % (self.info.host, self.info.port))
        socket.setsockopt(zmq.SUBSCRIBE, b"")  # subscribe to all topics
        self.socket = socket

        self.latest_pos_index_received = -1
        self.incoming = {}
        logger.info(headerline('', 'c', '#'))
        logger.info(headerline('Leaving LiveScan().init()', 'c', '#'))
        logger.info(headerline('', 'c', '#') + '\n')

    def check(self, frames=None, start=None):
        """
        Only called on the master node.
        """
        logger.info(headerline('', 'c', '#'))
        logger.info(headerline('Entering LiveScan().check()', 'c', '#'))
        logger.info(headerline('', 'c', '#'))
        self.end_of_scan = False

        # get all frames from the main socket
        ## This counts the "stutus: Heartbeat" message as a frame as well
        while True:
            try:
                msg = self.socket.recv_pyobj(flags=zmq.NOBLOCK)  ## NOBLOCK returns None if a message is not ready
                logger.info('######## Received a message')  ##
                ##headers = ('path' in msg.keys())
                ##emptymsg = ('heartbeat' in msg.values()) # just a message from contrast to keep connection alive
                ######if 'running' in msg.values():  # if zmq did not send a path: then save this message
                if msg['status'] == 'running':
                    self.latest_pos_index_received += 1
                    self.incoming[self.latest_pos_index_received] = msg
                    logger.info('############ Frame nr. %d received' % self.latest_pos_index_received)  ##
                elif msg['status'] == 'msgEOS':  # self.EOS:
                    self.end_of_scan = True
                    logger.info('############ RecorderFooter received; END OF SCAN!')  ##
                    break
                else:
                    logger.info('############ Message was not important')  ##
            except zmq.ZMQError:
                logger.info('######## Waiting for messages')  ##
                # no more data available - working around bug in ptypy here
                if self.latest_pos_index_received < self.info.min_frames * parallel.size:
                    logger.info('############ self.latest_pos_index_received = %u , self.info.min_frames = %d , parallel.size = %d' % (self.latest_pos_index_received, self.info.min_frames, parallel.size))  ##
                    logger.info('############ Not enough frames received, have %u frames, waiting...' % (self.latest_pos_index_received + 1))
                    time.sleep(1)
                else:
                    logger.info('############ Will process gathered data')  ##
                    break

        ind = self.latest_pos_index_received
        logger.info('#### latest_pos_index_received = %d' % ind)
        logger.info(headerline('', 'c', '#'))
        logger.info(headerline('Leaving LiveScan().check()', 'c', '#'))
        logger.info(headerline('', 'c', '#') + '\n')
        return (ind - start + 1), self.end_of_scan

    def load(self, indices):
        # indices are generated by PtyScan's _mpi_indices method.
        # It is a diffraction data index lists that determine
        # which node contains which data.
        raw, weight, pos = {}, {}, {}
        logger.info(headerline('', 'c', '#'))
        logger.info(headerline('Entering LiveScan().load()', 'c', '#'))
        logger.info(headerline('', 'c', '#'))

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
        #logger_info(dct)

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
                pos[i] = np.array((x, y))
                # pos[i] = -np.array((y, x)) * 1e-6
                logger_info(pos[i])
                weight[i] = np.ones_like(raw[i])
                weight[i][np.where(raw[i] == 2 ** 32 - 1)] = 0

                # d = msg['iterable'][0]
                # dct = {'x': d['position'][0],
                #        'y': d['position'][1],
                #        'diff': d['data']}
                logger.info('######## Repackaged data from frame nr. : %d' % i)
            except:
                break
        logger.info(headerline('', 'c', '#'))
        logger.info(headerline('Leaving LiveScan().load()', 'c', '#'))
        logger.info(headerline('', 'c', '#') + '\n')

        return raw, pos, weight


LS = LiveScan()
LS.initialize()


from ptypy.core import Ptycho
from ptypy import utils as u

p = u.Param()

# for verbose output
p.verbose_level = 3

# set home path
p.io = u.Param()
p.io.home = '/Users/lexelius/Documents/Contrast/temp/' ## "/tmp/ptypy/"
p.io.autoplot = u.Param()
p.io.autoplot.active = False
# p.io.autoplot.dump = True
##p.io.autosave = None

# max 200 frames (128x128px) of diffraction data
p.scans = u.Param()
p.scans.contrast = u.Param()
# now you have to specify which ScanModel to use with scans.XX.name,
# just as you have to give 'name' for engines and PtyScan subclasses.
p.scans.contrast.name = 'Full'
p.scans.contrast.data = u.Param()
p.scans.contrast.data.name = 'LiveScan'  # 'LiveScan'
p.scans.contrast.data.min_frames = 1
#p.scans.contrast.data.num_frames = 150
p.scans.contrast.data.shape = 128
## p.scans.contrast.data.xMotor = 'x'
## p.scans.contrast.data.yMotor = 'y'
p.scans.contrast.data.host = 'localhost'
p.scans.contrast.data.port = 5556
p.scans.contrast.data.energy= 6.2
p.scans.contrast.data.distance = 7


# attach a reconstrucion engine
p.engines = u.Param()

p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM'
p.engines.engine00.numiter = 200## this is in total, including the iterations before all patterns have been collected
p.engines.engine00.numiter_contiguous = 1##2  # default = 1, nr of iterations without interruption

## including the below gives a bit different probe and sample, but still doesn't look good.
p.engines.engine00.probe_support = 0.7##None # default: 0.7
p.engines.engine00.probe_update_start = 2 # default: 2
p.engines.engine00.probe_inertia = 0.01 # default: 1e-09
p.engines.engine00.object_inertia = 0.0001##0.1 # default: 0.0001
p.engines.engine00.update_object_first = True # default: True
p.engines.engine00.overlap_converge_factor = 0.05##0.5 # default: 0.05
p.engines.engine00.overlap_max_iterations = 10##100 # default: 10
p.engines.engine00.fourier_relax_factor = 0.05 # default: 0.05

# prepare and run
# p.io.kind = 'full_flat'
P = Ptycho(p, level=5)

# P.plot_overview()

# u.plot_storage(list(P.obj.storages.values())[0], fignum=100, modulus='linear',
#                slices=(slice(1), slice(None), slice(None)), si_axes='x', mask=None)
# u.plot_client.MPLplotter.create_plot_from_tile_list(fignum=1, num_shape_list=[(4, (2, 2))], figsize=(8, 8))
# fig = u.plot_storage(list(P.diff.storages.values())[0], 0, slices=(slice(4), slice(None), slice(None)), modulus='log')
# fig = u.plot_client.MPLplotter.plot_storage(list(P.diff.storages.values())[0], 0,
#                                             slices=(slice(4), slice(None), slice(None)), modulus='log')

#%% plots from pods
from ptypy.utils import imsave
import matplotlib.pyplot as plt
# fig, axs = plt.subplots(nrows=2, ncols=3, sharex='col', sharey='row',figsize=(16,4))
fig, axs = plt.subplots(nrows=2, ncols=10, figsize=(16,4))

fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.9, wspace=0.05, hspace=0.2)
plt.setp(axs[0,0], title='Object')
plt.setp(axs[1,0], title='Probe')
k=0
axs[0,k].imshow(imsave(P.pods['P0000'].object))
axs[1,k].imshow(imsave(P.pods['P0000'].probe))
for i in range(10,90,10):
    k += 1
    axs[0,k].imshow(imsave(P.pods[f"P{i:04d}"].object))
    axs[1,k].imshow(imsave(P.pods[f"P{i:04d}"].probe))
k += 1
axs[0,k].imshow(imsave(P.pods['P0091'].object))
axs[1,k].imshow(imsave(P.pods['P0091'].probe))

#%%  Plot probe and object sample through out reconstruction. f"pydevconsole_DM_{i:04d}.ptyr"
from ptypy import io
from ptypy.utils import imsave
import matplotlib.pyplot as plt
import numpy as np
# fig, axs = plt.subplots(nrows=3, ncols=11, sharex='col', sharey='row',figsize=(16,4))
fig, axs = plt.subplots(nrows=3, ncols=12, figsize=(19,4.5))
# plt.rcParams['xtick.labelsize'] = 6
fig.subplots_adjust(left=0.04, bottom=0.05, right=0.999, top=0.9, wspace=0.36, hspace=0.2)
plt.setp(axs[0, 0], ylabel='Object')
plt.setp(axs[1, 0], ylabel='Phase')
plt.setp(axs[2, 0], ylabel='Probe')
plt.setp(axs[0, 0], title=f"DM_0000")
c_obj, c_probe = [], []
rfile0 = '/Users/lexelius/Documents/Contrast/temp/dumps/pydevconsole/pydevconsole_None_0000.ptyr'

content = io.h5read(rfile0, 'content')['content']
c_obj.append(content['obj']['ScontrastG00']['data'])
c_probe.append(content['probe']['ScontrastG00']['data'])
k=0
axs[0, k].imshow(imsave(c_obj[k][0], vmax=1.3))
axs[1, k].imshow(np.angle(c_obj[k][0]))
axs[2, k].imshow(imsave(c_probe[k][0]))
rfile10 = '/Users/lexelius/Documents/Contrast/temp/dumps/pydevconsole/pydevconsole_DM_0010.ptyr'
pref = rfile10.split('0')[0]
suff = '.ptyr'
for i in range(10,200,20):
    k += 1
    filename = pref + f"{i:04d}" + suff
    content = io.h5read(filename, 'content')['content']
    c_obj.append(content['obj']['ScontrastG00']['data'])
    c_probe.append(content['probe']['ScontrastG00']['data'])
    axs[0, k].imshow(imsave(c_obj[k][0], vmax=1.3))
    axs[1, k].imshow(np.angle(c_obj[k][0]))
    axs[2, k].imshow(imsave(c_probe[k][0]))
    plt.setp(axs[0, k], title=f"DM_{i:04d}")
k += 1
filename = '/Users/lexelius/Documents/Contrast/temp/recons/pydevconsole/pydevconsole_DM_0200.ptyr'
content = io.h5read(filename, 'content')['content']
c_obj.append(content['obj']['ScontrastG00']['data'])
c_probe.append(content['probe']['ScontrastG00']['data'])
last_obj = axs[0, k].imshow(imsave(c_obj[k][0], vmax=1.3))
last_phase = axs[1, k].imshow(np.angle(c_obj[k][0]))
axs[2, k].imshow(imsave(c_probe[k][0]))
plt.setp(axs[0, k], title=f"DM_0200")
plt.rcParams['xtick.labelsize'] = 8 ## default= 10
plt.rcParams['ytick.labelsize'] = 8 ## default= 10
plt.show()

c_pos = content['positions']['ScontrastG00']
figpos = plt.figure(num=2, figsize=(5, 5))
plt.plot(c_pos[:, 0], c_pos[:, 1], '-X')
plt.title('Scan trajectory, 92 positions')
plt.show()

## plt.figure(num=3)
## plt.imshow(abs(c_obj[k][0]), vmax=2)


#%%
#
# #########
# from ptypy import utils as u
#
# pars = u.Param()
# pars.interactive = True
# content1 = list(io.h5read(rfile0, 'content').values())[0]  # ['content']
# runtime = content['runtime']
# probes = u.Param()
# probes.update(content['probe'], Convert=True)
# objects = u.Param()
# objects.update(content['obj'], Convert=True)
#
# Plotter = u.MPLplotter(pars=pars, probes=probes, objects=objects, runtime=runtime)
# Plotter._set_autolayout('default')
# Plotter.update_plot_layout()
# Plotter.plot_all()
# Plotter.draw()
# Plotter.plot_all()
# Plotter.draw()
# from ptypy.utils import plot_client
#
# #########
rfile0 = '/Users/lexelius/Documents/Contrast/temp/dumps/pydevconsole/pydevconsole_DM_0010.ptyr' #'/tmp/ptypy/dumps/pydevconsole/pydevconsole_DM_0010.ptyr'
rfile1 = '/tmp/ptypy/recons/pydevconsole/pydevconsole_DM_0300_first_completedlivescan_reconstruction.ptyr'
rfile1 = '/tmp/ptypy/recons/pydevconsole/pydevconsole_DM_0100.ptyr'

from ptypy import io
ld0 = io.h5read('/Users/lexelius/Documents/Contrast/temp/linkdata1.ptyd.part000')
ld91 = io.h5read('/Users/lexelius/Documents/Contrast/temp/linkdata1.ptyd.part091')
ld = io.h5read('/Users/lexelius/Documents/Contrast/temp/linkdata1.ptyd')
h5_0 = io.h5read('/Users/lexelius/Documents/Contrast/temp/000000.h5')
pdc0 = io.h5read('/private/tmp/ptypy/dumps/pydevconsole/pydevconsole_None_0000.ptyr')
pdc150 = io.h5read('/private/tmp/ptypy/recons/pydevconsole/pydevconsole_DM_0150.ptyr')

# load_ptyr0 = u.scripts.load_from_ptyr(rfile0)
# #load_ptyr1 = u.scripts.load_from_ptyr(rfile1)
# from ptypy.utils import imsave
# from matplotlib import pyplot as plt
#
# pil0 = imsave(load_ptyr0[0, :, :], filename=None, vmin=None, vmax=None, cmap=None)
# plt.imshow(pil0)
# pil0 = imsave(load_ptyr0[0, :, :], filename=rfile0.split('.ptyr')[0] + '.png', vmin=None, vmax=None, cmap=None)
# pil1 = imsave(load_ptyr1[0,:,:], filename=rfile1.split('.ptyr')[0] + '.png', vmin=None, vmax=None, cmap=None)
#
