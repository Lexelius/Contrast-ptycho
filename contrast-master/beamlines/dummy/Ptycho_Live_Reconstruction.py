## Script gives error due to some misfit in Cobj, trying to figure out why this misfit occurs..
## TO DO: test if problems can be solved if writing a custum subclass ( @register class MyScan(PtyScan): )

import sys
import time
import ptypy
from ptypy import utils as u

from contrast.environment import macro, env
from contrast.recorders import active_recorders, RecorderHeader, RecorderFooter

data = u.Param()
data.shape = 128#256 ## 128 gives error "You requested 2424387039356.97M pixels.", 256 gives "You requested 9697548157427.89M pixels."
data.num_frames = 100#400
#data.density = .1
data.min_frames = 1
data.label=None
data.psize=172e-6
data.energy= 6.2
data.center='fftshift'
data.distance = 7
data.auto_center = False
data.orientation = None
data.save = 'link'
data.dfile = '/Users/lexelius/Documents/Contrast/temp/linkdata1.ptyd' ## with save='link', this creates a new .ptyd file.


p = u.Param()
p.verbose_level = 5
p.io = u.Param()
p.io.home = "/Users/lexelius/Documents/PtyPy/ptypy-master/ptypy/"
p.io.interaction = u.Param()
p.io.interaction.server = u.Param() ##
p.io.interaction.server.active = True ## Activates ZeroMQ interaction server
p.io.interaction.server.address = 'tcp://127.0.0.1' ##
p.io.interaction.server.port = 5556 ## 5560 is default for PtyPy, 5556 default for Contrast ZMQ.

# data.density = .1
#scandata = data
#scandata.__delitem__('density') ## also deletes data.density!!
p.scans = u.Param()
p.scans.MF = u.Param()
p.scans.MF.name = 'Full' # ScanModel, Vanilla, Full, Bragg3dModel,, 'BLOCKFULL'
p.scans.MF.data= data ##
#p.scans.MF.data = u.Param()
p.scans.MF.data.name = 'PtyScan' ## 'MoonFlowerScan', 'PtydScan', 'PtyScan', 'QuickScan', 'SimScan'
# p.scans.MF.data.source = 'file'
# p.scans.MF.data.dfile = 'sample.ptyd'

p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM'
p.engines.engine00.numiter = 80

P = ptypy.core.Ptycho(p, level=5)