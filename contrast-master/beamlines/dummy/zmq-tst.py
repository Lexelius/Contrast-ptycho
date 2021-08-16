import zmq
import numpy as np
import matplotlib.pyplot as plt
import ptypy

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect('tcp://127.0.0.1:5556')
socket.setsockopt(zmq.SUBSCRIBE, b"")


x = []
y = []
diff = []
status = []
np.seterr(divide = 'ignore') # ignore error from taking the log of zero. Default is 'warn'
# alternatevly you could instead plot with:
# plt.matshow(np.log(np.where(diff[-1] > 1.0e-10, diff[-1], np.nan)), 0)
while True:
    message = socket.recv_pyobj()
    if len(list(message.values())) == 4:
        x.append(list(message.values())[0])
        y.append(list(message.values())[1])
        diff.append(list(message.values())[2])
        status.append(list(message.values())[3])
        P = ptypy.core.Ptycho(message, level=2)
        ##plt.matshow(np.log(diff[-1]), 0) ## to inefficient
        print(f"x = {x[-1]}, y = {y[-1]}")
    ##print(message)

#%% Figuring out how to send data to perform ptychography
# Stuff from ptypy-master/tutorial/simupod.py

import matplotlib as mpl
import numpy as np
import ptypy
from ptypy import utils as u
from ptypy.core import View, Container, Storage, Base, POD
plt = mpl.pyplot
import sys
scriptname = sys.argv[0].split('.')[0]

# As the ``Base`` class is slotted starting with version 0.3.0, we
# can't attach attributes after initialisation as we could with a normal
# python class. To reenable ths feature we have to subclasse once.
class Base2(Base):
    pass

P = Base2()

# Now we can start attaching attributes
P.CType = np.complex128
P.FType = np.float64

# Set experimental setup geometry and create propagator
# -----------------------------------------------------

# Here, we accept a little help from the :any:`Geo` class to provide
# a propagator and pixel sizes for sample and detector space.
from ptypy.core import geometry
g = u.Param()
g.energy = None  # u.keV2m(1.0)/6.32e-7
g.lam = u.keV2nm(6.2)##5.32e-7
g.distance = 7 ##15e-2
g.psize = 172e-6 ##24e-6
g.shape = 256
g.propagation = "farfield"
G = geometry.Geo(owner=P, pars=g)

# The Geo instance ``G`` has done a lot already at this moment. For
# example, we find forward and backward propagators at ``G.propagator.fw``
# and ``G.propagator.bw``. It has also calculated the appropriate
# pixel size in the sample plane (aka resolution),
print(G.resolution)

# which sets the shifting frame to be of the following size:
fsize = G.shape * G.resolution
print("%.2fx%.2fmm" % tuple(fsize*1e3))

# Create probing illumination
# ---------------------------

# Next, we need to create a probing illumination.
# We start with a suited container that we call *probe*
P.probe = Container(P, 'Cprobe', data_type='complex')
pr_shape = (1,)+ tuple(G.shape)
pr3 = P.probe.new_storage(shape=pr_shape, psize=G.resolution)
y, x = pr3.grids()
apert = u.smooth_step(fsize[0]/5-np.abs(x), 3e-5)*u.smooth_step(fsize[1]/5-np.abs(y), 3e-5)
pr3.fill(apert)

# In order to put some physics in the illumination we set the number of
# photons to 1 billion
pr3.data *= np.sqrt(1e9/np.sum(pr3.data*pr3.data.conj()))

# and we quickly check if the propagation works.
ill = pr3.data[0]
propagated_ill = G.propagator.fw(ill)
fig = plt.figure(3)
ax = fig.add_subplot(111)
im = ax.imshow(np.log10(np.abs(propagated_ill)+1))
plt.colorbar(im)

# Next, we need to model a sample through an object transmisson function.
# We start of with a suited container which we call *obj*.
P.obj = Container(P, 'Cobj', data_type='complex')

# As we have learned from the previous :ref:`ptypyclasses`\ ,
# we can use :any:`View`\ 's to create a Storage data buffer of the
# right size.
oar = View.DEFAULT_ACCESSRULE.copy()
oar.storageID = 'S00'
oar.psize = G.resolution
oar.layer = 0
oar.shape = G.shape
oar.active = True

## needs to be fixed for when x,y is read from zmq in realtime
arr=[x,y]
positions=np.transpose(np.asarray(arr))
##

for pos in positions:
    # the rule
    r = oar.copy()
    r.coord = pos
    V = View(P.obj, None, r)

P.obj.reformat()
print(P.obj.formatted_report())

storage = P.obj.storages['S00']
####storage.fill(flower_obj(storage.shape[-2:])) ## check if this needs to be filled andif so, with what.


# First we create the missing :py:class:`~ptypy.core.classes.Container`'s.
P.exit = Container(P, 'Cexit', data_type='complex')
P.diff = Container(P, 'Cdiff', data_type='real')
P.mask = Container(P, 'Cmask', data_type='real')

# We start with one POD and its views.
objviews = list(P.obj.views.values())
obview = objviews[0]


probe_ar = View.DEFAULT_ACCESSRULE.copy()
probe_ar.psize = G.resolution
probe_ar.shape = G.shape
probe_ar.active = True
probe_ar.storageID = pr3.ID
prview = View(P.probe, None, probe_ar)

# We construct the exit wave View. This construction is shorter as we only
# change a few bits in the access rule.
exit_ar = probe_ar.copy()
exit_ar.layer = 0
exit_ar.active = True
exview = View(P.exit, None, exit_ar)

# We construct diffraction and mask Views. Even shorter is the
# construction of the mask View as, for the mask, we are
# essentially using the same access as for the diffraction data.
diff_ar = probe_ar.copy()
diff_ar.layer = 0
diff_ar.active = True
diff_ar.psize = G.psize
mask_ar = diff_ar.copy()
maview = View(P.mask, None, mask_ar)
diview = View(P.diff, None, diff_ar)

# Now we can create the POD.
pods = []
views = {'probe': prview, 'obj': obview, 'exit': exview, 'diff': diview, 'mask': maview}
pod = POD(P, ID=None, views=views, geometry=G)
pods.append(pod)


# The :any:`POD` is the most important class in |ptypy|. Its instances
# are used to write the reconstruction algorithms using
# its attributes as local references. For example we can create and store an exit
# wave in the following convenient fashion:
pod.exit = pod.probe * pod.object

# The result of the calculation above is stored in the appropriate
# storage of ``P.exit``.
# Therefore we can use this command to plot the result.
exit_storage = list(P.exit.storages.values())[0]
fig = u.plot_storage(exit_storage, 6)

# The diffraction plane is also conveniently accessible with
pod.diff = np.abs(pod.fw(pod.exit))**2

# The result is stored in the diffraction container.
diff_storage = list(P.diff.storages.values())[0]
fig = u.plot_storage(diff_storage, 7, modulus='log')