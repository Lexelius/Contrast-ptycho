"""
Sets up a mock beamline with dummy motors and detectors.
"""
"""
What to do after vacation:
* save output
* check individual diffraction patterns in ptypy
"""

"""
With PyCharm, run script within the "Contrast-project" through Python Console:
**open new console/click on 'rerun' button**
import os
os.getcwd()
os.chdir('/Users/lexelius/Documents/Contrast/contrast-master/beamlines/dummy')
os.getcwd()
run v1_dummy_beamline.py
run dummy_beamline.py
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
    from sim_ptycho_scan import *


##
    # import signal
    # def kill_streamrecorders():
    #     for r in StreamRecorder.getinstances():
    #         print("Killing %s" % r.name)
    #         os.kill(r.pid, signal.SIGTERM)
    #
    # kill_streamrecorders()
    ##

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

    env.paths.directory = '/Users/lexelius/Documents/Contrast/temp' ## "/tmp"

    # remove old files
    files = os.listdir(env.paths.directory)
    for file in files:
        if file.endswith(".h5"):
            os.remove(os.path.join(env.paths.directory, file))

    # the Hdf5Recorder later gets its path from the env object
    h5rec = Hdf5Recorder(name='h5rec')
    h5rec.start()

    def zmq_fix1():
        """
        Check if zmqrec is already running and stop it if it is.
        """
        try:
            zmqrec.stop()
        except:
            pass
    """
def zmq_fix1():
    try:
        zmqrec.stop()
    except:
        pass
    ##############################################
def zmq_fix2():
    if 'zmqrec' in locals():
        zmqrec.stop()
    ##############################################
    ## Results when zmqrec not yet defined:
    %timeit zmq_fix1()
    358 ns ± 3.26 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
    %timeit zmq_fix2()
    116 ns ± 0.236 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)
    ############################################## 
    ## Results when zmqrec is defined:

    """
    zmqrec = StreamRecorder(name='zmqrec')
    print("Defined zmqrec!") ##
    zmq_fix1()
    zmqrec.start()

    Dummy_Ptycho  ##

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

#%%
"""
run dummy_beamline.py
Backend MacOSX is interactive backend. Turning interactive mode on.
Welcome to contrast,
   "CONTinuous RASTer"
Process zmqrec:
Process h5rec:
Traceback (most recent call last):
Traceback (most recent call last):
  File "/opt/anaconda3/envs/ptypy-clone_contrast/lib/python3.7/multiprocessing/process.py", line 297, in _bootstrap
    self.run()
  File "/opt/anaconda3/envs/ptypy-clone_contrast/lib/python3.7/site-packages/contrast-0.0.1-py3.7.egg/contrast/recorders/StreamRecorder.py", line 57, in run
    super(StreamRecorder, self).run()
  File "/opt/anaconda3/envs/ptypy-clone_contrast/lib/python3.7/site-packages/contrast-0.0.1-py3.7.egg/contrast/recorders/Recorder.py", line 84, in run
    self._process_queue()
  File "/opt/anaconda3/envs/ptypy-clone_contrast/lib/python3.7/site-packages/contrast-0.0.1-py3.7.egg/contrast/recorders/Recorder.py", line 64, in _process_queue
    dcts = [self.queue.get() for i in range(self.queue.qsize())] # ok since only we are reading from self.queue
  File "/opt/anaconda3/envs/ptypy-clone_contrast/lib/python3.7/multiprocessing/queues.py", line 117, in qsize
    return self._maxsize - self._sem._semlock._get_value()
  File "/opt/anaconda3/envs/ptypy-clone_contrast/lib/python3.7/multiprocessing/process.py", line 297, in _bootstrap
    self.run()
  File "/opt/anaconda3/envs/ptypy-clone_contrast/lib/python3.7/site-packages/contrast-0.0.1-py3.7.egg/contrast/recorders/Recorder.py", line 84, in run
    self._process_queue()
NotImplementedError
  File "/opt/anaconda3/envs/ptypy-clone_contrast/lib/python3.7/site-packages/contrast-0.0.1-py3.7.egg/contrast/recorders/Recorder.py", line 64, in _process_queue
    dcts = [self.queue.get() for i in range(self.queue.qsize())] # ok since only we are reading from self.queue
  File "/opt/anaconda3/envs/ptypy-clone_contrast/lib/python3.7/multiprocessing/queues.py", line 117, in qsize
    return self._maxsize - self._sem._semlock._get_value()
NotImplementedError
"""