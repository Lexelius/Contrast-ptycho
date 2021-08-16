"""
To be run after v2_dummy_beamline.py
Combines "sim_ptycho_scan.py" and "/Users/lexelius/Documents/PtyPy/ptypy-master/tutorial/ownengine.py".
*In progress*
"""

import sys
import time
import ptypy
from ptypy import utils as u

from contrast.environment import macro, env
from contrast.recorders import active_recorders, RecorderHeader, RecorderFooter
#
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
u.verbose.set_level(1)

# create data parameter branch
data = u.Param()
data.shape = 256
data.num_frames = 400
data.density = .1
data.min_frames = 1
data.label=None
data.psize=172e-6
data.energy= 6.2
data.center='fftshift'
data.distance = 7
data.auto_center = None
data.orientation = None
## data.model = 'raster'

# create PtyScan instance
MF = ptypy.core.data.MoonFlowerScan(data)
MF.initialize()

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
               'diff': d['data']}
        sample.chunk = msg['chunk']
        sample.common = msg['common']['center']


        # pass data to recorders
        for r in active_recorders():
            r.queue.put(dct)

        # print spec-style info
        print('%-6u%-10.4f%-10.4f%10s' % (n, dct['x']*1e6, dct['y']*1e6, dct['diff'].shape))
        n += 1
        time.sleep(.05) ## .2

except KeyboardInterrupt:
    print('\nScan #%d cancelled at %s' % (scannr, time.asctime()))

# tell the recorders that the scan is over
for r in active_recorders():
    r.queue.put(RecorderFooter(scannr=scannr, path=env.paths.directory))


#%%

p = u.Param()
p.verbose_level = 3
p.io = u.Param()
p.io.home = "/Users/lexelius/Documents/PtyPy/ptypy-master/ptypy/"  # # "/tmp/ptypy/"

scandata = data
scandata.__delitem__('density') ## also deletes data.density!!
p.scans = u.Param()
p.scans.MF = u.Param()
p.scans.MF.data= scandata ##
# p.scans.MF.data= u.Param()
p.scans.MF.name = 'Vanilla'
p.scans.MF.data.name = 'PtydScan'
# p.scans.MF.data.source = 'file'
# p.scans.MF.data.dfile = 'sample.ptyd'

p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM'
p.engines.engine00.numiter = 80

# P = ptypy.core.Ptycho(p, level=2)