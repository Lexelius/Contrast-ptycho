"""
Sets up a mock beamline with dummy motors and detectors.
"""

# need this main guard here because Process.start() (so our recorders)
# import __main__, and we don't want the subprocess to start new sub-
# processes etc.
if __name__=='__main__':

    import contrast
    from contrast.motors import DummyMotor, MotorMemorizer, ExamplePseudoMotor
    from contrast.scans import *
    from contrast.detectors import DummyDetector, Dummy1dDetector, DummyWritingDetector, DummyWritingDetector2
    from contrast.environment import env, register_shortcut
    from contrast.recorders import Hdf5Recorder, StreamRecorder
    import os

    # if you have ptypy installed, you can generate mock ptycho data
    #from sim_ptycho_scan import *

    env.userLevel = 1 # we're not experts!

    samx = DummyMotor(name='samx')
    samx.dial_limits = (0, 10)

    samy = DummyMotor(name='samy')
    samy.dial_limits = (-5, 5)

    samr = DummyMotor(name='samr')
    samr.dial_limits = (-180, 180)
    samr.velocity = 30

    basex = DummyMotor(name='basex')
    basex.dial_limits = (-8000, 8000)
    basex.velocity      = 10000
    basey = DummyMotor(name='basey')
    basey.dial_limits = (-8000, 8000)
    basey.velocity      = 10000
    basez = DummyMotor(name='basez')
    basez.dial_limits = (-8000, 8000)
    basez.velocity      = 10000

    sx = DummyMotor(name='sx')
    sx.dial_limits = (-50, 50)
    sy = DummyMotor(name='sy')
    sy.dial_limits = (-50, 50)
    sz = DummyMotor(name='sz')
    sz.dial_limits = (-50, 50)
    sr = DummyMotor(name='sr')
    sr.dial_limits = (-180, 180)
    sr.velocity = 30

    energy = DummyMotor(name='energy')
    energy.dial_limits   = (5000, 35000)
    energy.velocity      = 50000
    energy.dial_position = 10000

    attenuator1_x = DummyMotor(name='attenuator1_x')
    attenuator2_x = DummyMotor(name='attenuator2_x')
    attenuator3_x = DummyMotor(name='attenuator3_x')
    attenuator4_x = DummyMotor(name='attenuator4_x')
    attenuator1_x.dial_limits = (-42000, 42000)
    attenuator2_x.dial_limits = (-42000, 42000)
    attenuator3_x.dial_limits = (-42000, 42000)
    attenuator4_x.dial_limits = (-42000, 42000)
    attenuator1_x.velocity = 20000
    attenuator2_x.velocity = 20000
    attenuator3_x.velocity = 20000
    attenuator4_x.velocity = 20000

    gap = DummyMotor(name='gap', userlevel=5, user_format='%.5f')

    diff = ExamplePseudoMotor([samx, samy], name='diff')

    det1 = DummyDetector(name='det1')
    det2 = DummyWritingDetector(name='det2')
    det3 = Dummy1dDetector(name='det3')
    det4 = DummyWritingDetector2(name='det4')

##    env.paths.directory = "/tmp"
    env.paths.directory = "/Users/lexelius/Documents/Contrast/temp"

    # remove old files
    files = os.listdir(env.paths.directory)
    for file in files:
        if file.endswith(".h5"):
            os.remove(os.path.join(env.paths.directory, file))



    ##
    def zmq_fix1():
        """
        Check if zmqrec is already running and stop it if it is.
        """
        try:
            zmqrec.stop()
        except:
            pass
    ##

    # the Hdf5Recorder later gets its path from the env object
    h5rec = Hdf5Recorder(name='h5rec')
    h5rec.start()

    zmq_fix1()  ##
    zmqrec = StreamRecorder(name='zmqrec')
    print("Defined zmqrec!") ##
    zmqrec.start()


    # this MotorMemorizer keeps track of motor user positions and
    # limits, and dumps this to file when they are changed.
##    memorizer = MotorMemorizer(name='memorizer', filepath='/tmp/.dummy_beamline_motors')
    memorizer = MotorMemorizer(name='memorizer', filepath='/Users/lexelius/Documents/Contrast/temp/.dummy_beamline_motors')

    # handy shortcuts
    register_shortcut('wsample', 'wm samx samy')
    register_shortcut('waaa', 'wa')
    register_shortcut('zero_sample', 'umv samx 0 samy 0')

    # define pre- and post-scan actions, per base class
    def pre_scan_stuff(slf):
        print("Maybe open a shutter here?")
    def post_scan_stuff(slf):
        print("Maybe close that shutter again?")

    SoftwareScan._before_scan = pre_scan_stuff
    SoftwareScan._after_scan = post_scan_stuff
    Ct._before_ct = pre_scan_stuff
    Ct._after_ct = post_scan_stuff

    contrast.wisdom()

#%% # Run a cell with [option]+[Shift]+[Enter]

# Run the macro defined in sim_ptycho_scan.py
from contrast.environment import runCommand
from sim_ptycho_scan import *
runCommand('dummy_ptycho')
# from Ptycho_scan_v1 import *
# runCommand('dummy_ptycho_v1')
#%%
# Read content of output file

import h5py
outfile = env.paths.directory + "/000000.h5"
try:
    f = h5py.File(outfile,'r')
    filekeys = list(f.keys())
    dset = f[filekeys[0]]
    print(dset.name) ## = /entry


    def printname(name):
        print(name)

    f.visit(printname)

except:
    print("Error trying to open the output file, try running the cell again.")


# read_h5(fn) and h5py_dataset_iterator() is taken from: https://www.programcreek.com/python/example/28022/h5py.Dataset
def read_h5(fn):
    """Read h5 file into dict.
    Dict keys are the group + dataset names, e.g. '/a/b/c/dset'. All keys start
    with a leading slash even if written without (see :func:`write_h5`).
    Parameters
    ----------
    fn : str
        filename
    Examples
    --------
    >>> read_h5('foo.h5').keys()
    ['/a/b/d1', '/a/b/d2', '/a/c/d3', '/x/y/z']
    """
    fh = h5py.File(outfile, mode='r')
    dct = {}
    def get(name, obj, dct=dct):
        if isinstance(obj, h5py.Dataset):
            _name = name if name.startswith('/') else '/'+name
            dct[_name] = obj[()]
    fh.visititems(get)
    fh.close()
    return dct
##To do: organize the below to more structured/nested variables
h5datakeys=read_h5(outfile).keys()
h5data=read_h5(outfile)
h5data['/entry/measurement/diff']

# def h5py_dataset_iterator(self, g, prefix=''):
#     """Group recursive iterator
#
#     Iterate through all groups in all branches and return datasets in dicts)
#     """
#     for key in g.keys():
#         item = g[key]
#         path = '{}/{}'.format(prefix, key)
#         keys = [i for i in item.keys()]
#         if isinstance(item[keys[0]], h5py.Dataset):  # test for dataset
#             data = {'path': path}
#             for k in keys:
#                 if not isinstance(item[k], h5py.Group):
#                     dataset = np.array(item[k][()])
#
#                     if isinstance(dataset, np.ndarray):
#                         if dataset.size != 0:
#                             if isinstance(dataset[0], np.bytes_):
#                                 dataset = [a.decode('ascii')
#                                            for a in dataset]
#                     data.update({k: dataset})
#             yield data
#         else:  # test for group (go down)
#             yield from self.h5py_dataset_iterator(item, path)



# Separate diffraction array
diffraction_patterns = h5data['/entry/measurement/diff'].reshape((int(h5data['/entry/measurement/diff'].shape[0]/h5data['/entry/measurement/diff'].shape[1]),h5data['/entry/measurement/diff'].shape[1],h5data['/entry/measurement/diff'].shape[1]))

import numpy as np
import matplotlib.pyplot as plt
plt.matshow(np.log(diffraction_patterns[222]),0)

# ZMQ: figure out why zmq not active in lsrec, make a test zmq server & client
# Perform ptychography on diffraction patterns created by dummy_ptycho
# Divide ptypy data according to contrast motor position/scan
# Make plot of the position trajectory


