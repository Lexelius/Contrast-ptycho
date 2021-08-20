"""
To be run after v2_dummy_beamline.py
Combines "sim_ptycho_scan.py" and "/Users/lexelius/Documents/PtyPy/ptypy-master/tutorial/ownengine.py".
*In progress*

"I'm trying to make a reconstruction using data loaded
from a variable (and eventually directly from the zmq-recorder)."
"""

import sys
import time
import ptypy
from ptypy import utils as u

from contrast.environment import macro, env
from contrast.recorders import active_recorders, RecorderHeader, RecorderFooter

import os
def allfiles(*oldfiles):
    """
    :param oldfiles: list of files obtained by the function previously.
    :return: all files in the Contrast directory (locally).
    Does not check for replaced files with the same filename!
    """
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk('/Users/lexelius/Documents/Contrast'):
        for file in f:
            files.append(os.path.join(r, file))
#/Users/lexelius/Documents/PtyPy/ptypy-master
    if oldfiles:
        oldfiles = list(oldfiles)
        removed = list(set(oldfiles[0]) - set(files))
        added = list(set(files) - set(oldfiles[0]))
        print(f'{removed.__len__()} Removed files: {removed}')
        print(f'{added.__len__()} New files: {added}')
        return files, removed, added
    else:
        return files
allfiles0 = allfiles()



# @macro
# class Dummy_Ptycho(object):
#     """
#     Dummy macro which produces data from ptypy's MoonFlowerScan.
#
#     Does not use actual detectors or motors, but puts data in all active
#     recorders.
#     """
#
#     def __init__(self):
#         """
#         The constructor should parse parameters.
#         """

scannr = env.nextScanID
env.nextScanID += 1

# for verbose output
u.verbose.set_level(3) ##1

# create data parameter branch
data = u.Param()
data.shape = 128##256
data.num_frames = 200
data.density = .1
data.min_frames = 1
data.label=None
data.psize=172e-6
data.energy= 6.2
data.center='fftshift'
data.distance = 7
data.auto_center = True ##None
data.orientation = None
## data.model = 'raster'
data.save = 'link' ##
data.dfile = '/Users/lexelius/Documents/Contrast/temp/linkdata1.ptyd' ## with save='link', this creates a new .ptyd file.

# create PtyScan instance
MF = ptypy.core.data.MoonFlowerScan(data)
MF.initialize()

allfiles1, allremoved1, allnew1 = allfiles(allfiles0)
#%%
#%% Set up reconstruction
#ptypy.core.data.PtyScan(pars=None, **kwargs) ;;;; DEFAULT = {'add_poisson_noise': False, 'auto_center': None, 'center': 'fftshift', 'chunk_format': '.chunk%02d', 'dfile': None, 'distance': 7.19, 'energy': 7.2, 'experimentID': None, 'label': None, 'load_parallel': 'data', 'min_frames': 1, 'name': 'PtyScan', 'num_frames': None, 'orientation': None, 'positions_theory': None, 'psize': 0.000172, 'rebin': None, 'save': None, 'shape': 256, 'version': 0.1}

p = u.Param()
p.verbose_level = 5
p.io = u.Param()
p.io.home = "/Users/lexelius/Documents/PtyPy/ptypy-master/ptypy/"  # # "/tmp/ptypy/"
p.io.interaction = u.Param()
p.io.interaction.server = u.Param() ##
p.io.interaction.server.active = True ##
p.io.interaction.server.address = 'tcp://127.0.0.1' ##
p.io.interaction.server.port = 5556 ##5560

# data.density = .1
scandata = data
scandata.__delitem__('density') ## also deletes data.density!!
p.scans = u.Param()
p.scans.MF = u.Param()
p.scans.MF.data= scandata ##
# p.scans.MF.data= u.Param()
p.scans.MF.name = 'Full' # ScanModel, Vanilla, Full, Bragg3dModel
p.scans.MF.data.name = 'PtyScan'##'PtydScan'
# p.scans.MF.data.source = 'file'
# p.scans.MF.data.dfile = 'sample.ptyd'

p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM'
p.engines.engine00.numiter = 80

P1 = ptypy.core.Ptycho(p, level=1)
P1.diff.new_storage('S0000', shape=(MF.num_frames,data.shape,data.shape), psize=data.psize)
diff_storage = P1.diff.storages['S0000']

allfiles2, allremoved2, allnew2 = allfiles(allfiles1)
#%%

# P = ptypy.core.Ptycho(MF.geo.p, level=2)

    # def run(self):
    #     """
    #     This method does all the serious interaction with motors,
    #     detectors, and data recorders.
    #     """
print('\nScan #%d starting at %s\n' % (scannr, time.asctime()))
print('#     x          y          data')
print('-----------------------------------------------')

# send a header to the recorders
snap = env.snapshot.capture()
for r in active_recorders():
    r.queue.put(RecorderHeader(scannr=scannr, path=env.paths.directory,
                               snapshot=snap, description=_command))
msgs = []
sample = u.Param()
sample.info = MF.info
diff_patterns = []
import numpy
try:
    n = 0
    while True:
        # generate the next position
        msg = MF.auto(1)
        msgs.append(msg)
        if msg == MF.EOS:
            break
        d = msg['iterable'][0]
        dct = {'x': d['position'][0],
               'y': d['position'][1],
               'diff': d['data'],
               'mask': d['mask']}
        diff_patterns.append(d['data'])
        #diff_storage.fill(d['data'])  ## diff_storage.fill(flower_obj(storage.shape[-2:]))
        # diff_patterns.append(dct['diff'])
        sample.chunk = msg['chunk']
        sample.common = msg['common']['center']

        # pass data to recorders
        for r in active_recorders():
            r.queue.put(dct)

        # print spec-style info
        print('%-6u%-10.4f%-10.4f%10s' % (n, dct['x']*1e6, dct['y']*1e6, dct['diff'].shape))
        n += 1
        #time.sleep(.0001) ## .2

except KeyboardInterrupt:
    print('\nScan #%d cancelled at %s' % (scannr, time.asctime()))

# tell the recorders that the scan is over
for r in active_recorders():
    r.queue.put(RecorderFooter(scannr=scannr, path=env.paths.directory))

allfiles3, allremoved3, allnew3 = allfiles(allfiles2)
#%%
import numpy
diff_storage.fill(numpy.asarray(diff_patterns)) # insert diffraction patterns in P1.diff.storages['S0000']!

##P1 = ptypy.core.Ptycho(p, level=1)
###diff_storage = P1.diff.new_storage('S0000')
###diff_storage = P1.diff.storages['S0000']
###diff_storage.fill(diff_patterns) ## diff_storage.fill(flower_obj(storage.shape[-2:]))

P1.new_data = P1.model.new_data() # P1.model.ptycho.obj.reformat()
P1._redistribute_data(div = 'rect', obj_storage=None)
P1.new_data = P1.model.new_data() # P1.model.ptycho.obj.reformat()
P1.init_data() ## P2
## P2 gives error from P1.model.ptycho.obj.reformat()
P1.init_communication() ## P3
P1.init_engine() ## P4
P1.run() ## P5
P1.finalize() ## P5

P2 = ptypy.core.Ptycho(p, level=2)
#P3 = ptypy.core.Ptycho(p, level=3)

#fig = u.plot_storage(list(P2.diff.storages.values())[0], 0) ##
diff_storage1 = list(P1.diff.storages.values())[0]
fig = u.plot_storage(diff_storage1, 0, slices=(slice(2), slice(None), slice(None)), modulus='log')

