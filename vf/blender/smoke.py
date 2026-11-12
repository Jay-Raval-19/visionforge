import blenderproc as bproc  # must be the first import inside `blenderproc run`
import sys

import numpy as np

out = sys.argv[-1]
bproc.init()
cube = bproc.object.create_primitive("CUBE", scale=[1, 1, 1], location=[0, 0, 1])
cube.set_cp("category_id", 1)
light = bproc.types.Light()
light.set_location([4, -4, 6])
light.set_energy(800)
bproc.camera.add_camera_pose(bproc.math.build_transformation_mat([6, -6, 4], [1.1, 0, 0.8]))
bproc.renderer.set_max_amount_of_samples(32)
bproc.renderer.enable_segmentation_output(map_by=["instance"])
data = bproc.renderer.render()
np.save(out + "_rgb.npy", np.asarray(data["colors"][0]))
np.save(out + "_seg.npy", np.asarray(data["instance_segmaps"][0]))
print("SMOKE_OK", np.asarray(data["colors"][0]).shape)
