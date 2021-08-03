"""
To be run after v2_dummy_beamline.py
Combines "sim_ptycho_scan.py" and "/Users/lexelius/Documents/PtyPy/ptypy-master/tutorial/ownengine.py".
"""

import sys
import time
import ptypy
from ptypy import utils as u

from contrast.environment import macro, env
from contrast.recorders import active_recorders, RecorderHeader, RecorderFooter

@macro
class Dummy_Ptycho_v1(object):
    """
    Dummy macro which produces data from ptypy's MoonFlowerScan.

    Does not use actual detectors or motors, but puts data in all active
    recorders.dr
    """

    def __init__(self):
        """
        The constructor should parse parameters.
        """

        self.scannr = env.nextScanID
        env.nextScanID += 1

        # for verbose output
        u.verbose.set_level(1)

        # create data parameter branch
        p = u.Param()
        p.scans = u.Param()##
        p.scans.MF = u.Param()##
        p.scans.MF.shape = 256
        p.scans.MF.num_frames = 400
        p.scans.MF.density = .1
        p.scans.MF.min_frames = 1
        p.scans.MF.label=None
        p.scans.MF.psize=172e-6
        p.scans.MF.energy= 6.2
        p.scans.MF.center='fftshift'
        p.scans.MF.distance = 7
        p.scans.MF.auto_center = None
        p.scans.MF.orientation = None

        # create PtyScan instance
        self.MF = ptypy.core.data.MoonFlowerScan(p)
        self.MF.initialize()

        # P = ptypy.core.Ptycho(p, level=2)

    def run(self):
        """
        This method does all the serious interaction with motors,
        detectors, and data recorders.
        """
        print('\nScan #%d starting at %s\n' % (self.scannr, time.asctime()))
        print('#     x          y          data')
        print('-----------------------------------------------')

        # send a header to the recorders
        snap = env.snapshot.capture()
        for r in active_recorders():
            r.queue.put(RecorderHeader(scannr=self.scannr, path=env.paths.directory,
                                       snapshot=snap, description=self._command))

        try:
            n = 0
            while True:
                # generate the next position
                msg = self.MF.auto(1)
                if msg == self.MF.EOS:
                    break
                d = msg['iterable'][0]
                dct = {'x': d['position'][0],
                       'y': d['position'][1],
                       'diff': d['data']}

                # pass data to recorders
                for r in active_recorders():
                    r.queue.put(dct)

                # print spec-style info
                print('%-6u%-10.4f%-10.4f%10s' % (n, dct['x']*1e6, dct['y']*1e6, dct['diff'].shape))
                n += 1
                time.sleep(.2)

        except KeyboardInterrupt:
            print('\nScan #%d cancelled at %s' % (self.scannr, time.asctime()))

        # tell the recorders that the scan is over
        for r in active_recorders():
            r.queue.put(RecorderFooter())
