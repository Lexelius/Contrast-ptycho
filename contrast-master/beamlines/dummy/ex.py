print("This is a test script!")

if __name__=='__main__':

    import contrast
    from contrast.motors import DummyMotor, MotorMemorizer, ExamplePseudoMotor
    from contrast.scans import *
    from contrast.detectors import DummyDetector, Dummy1dDetector, DummyWritingDetector, DummyWritingDetector2
    from contrast.environment import env, register_shortcut
    from contrast.recorders import Hdf5Recorder, StreamRecorder, Recorder
    import os

    import os, signal
    def kill_all_recorders():
        for r in Recorder.getinstances():
            print("Killing %s" % r.name)
            os.kill(r.pid, signal.SIGTERM)

    import atexit
    atexit.register(kill_all_recorders)

print("End of script!")
