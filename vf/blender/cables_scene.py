import blenderproc as bproc  # must be the first line (BlenderProc requirement)
# Blender (BlenderProc) renderer for simulated cable scenes.
#
#     blenderproc run vf/blender/cables_scene.py spec1.json [spec2.json ...]
#
# Each spec JSON describes the table, hubs with labelled ports, connectors, cable centrelines (from vf/ropesim.py),
# camera and lights. Outputs per spec: <out>_rgb.npy, <out>_seg.npy (pixel value = cable id, 0 = not a cable) and
# <out>_proj.json (pixel positions of requested 3D points, for the answer checker).
import json
import math
import sys

import bpy
import numpy as np


def material(name, color, rough=0.5, metal=0.0, noise=None):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    if noise:                                   # procedural texture (wood grain / desk mat)
        tex = nt.nodes.new("ShaderNodeTexNoise")
        tex.inputs["Scale"].default_value = noise["scale"]
        tex.inputs["Detail"].default_value = 8
        if "stretch" in noise:
            mapn = nt.nodes.new("ShaderNodeMapping")
            mapn.inputs["Scale"].default_value = noise["stretch"]
            coord = nt.nodes.new("ShaderNodeTexCoord")
            nt.links.new(coord.outputs["Object"], mapn.inputs["Vector"])
            nt.links.new(mapn.outputs["Vector"], tex.inputs["Vector"])
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].color = (*noise["c1"], 1)
        ramp.color_ramp.elements[1].color = (*noise["c2"], 1)
        nt.links.new(tex.outputs["Fac"], ramp.inputs["Fac"])
        nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def mesh_obj(obj, cat):
    mo = bproc.types.MeshObject(obj)
    mo.set_cp("category_id", cat)
    return mo


def box(name, center, size, mat, cat=0, rot_z=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center, rotation=(0, 0, rot_z))
    o = bpy.context.active_object
    o.name = name
    o.scale = size
    o.data.materials.append(mat)
    return mesh_obj(o, cat)


def cylinder(name, center, radius, depth, rot, mat, cat=0):
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=depth, location=center, rotation=rot, vertices=24)
    o = bpy.context.active_object
    o.name = name
    o.data.materials.append(mat)
    return mesh_obj(o, cat)


def cable(name, pts, radius, mat, cat):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius
    cu.bevel_resolution = 4
    cu.use_fill_caps = True
    sp = cu.splines.new("POLY")
    sp.points.add(len(pts) - 1)
    for p, q in zip(sp.points, pts):
        p.co = (q[0], q[1], q[2], 1)
    obj = bpy.data.objects.new(name, cu)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(mat)
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(obj.evaluated_get(dg))
    mobj = bpy.data.objects.new(name + "_mesh", me)
    bpy.context.scene.collection.objects.link(mobj)
    bpy.data.objects.remove(obj)
    for poly in me.polygons:
        poly.use_smooth = True
    return mesh_obj(mobj, cat)


def label(text, loc, size, mat, rot_z=0.0):
    cu = bpy.data.curves.new("lbl_" + text, "FONT")
    cu.body = text
    cu.size = size
    cu.align_x = "CENTER"
    cu.align_y = "CENTER"
    cu.extrude = 0.0004
    obj = bpy.data.objects.new("lbl_" + text, cu)
    obj.location = loc
    obj.rotation_euler = (0, 0, rot_z)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(mat)
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(obj.evaluated_get(dg))
    mobj = bpy.data.objects.new("lblm_" + text, me)
    mobj.location, mobj.rotation_euler = obj.location, obj.rotation_euler
    bpy.context.scene.collection.objects.link(mobj)
    bpy.data.objects.remove(obj)
    return mesh_obj(mobj, 0)


def build_and_render(spec):
    W, H = spec["res"]
    bproc.camera.set_resolution(W, H)
    t = spec["table"]
    tmat = material("table", t["c1"], rough=t.get("rough", .55),
                    noise=dict(scale=t["scale"], c1=t["c1"], c2=t["c2"], stretch=t.get("stretch", (1, 1, 1))))
    box("table", (0, 0, -0.01), (2.4, 2.0, 0.02), tmat)
    hub_mat = material("hub", (0.03, 0.03, 0.035), rough=.35)
    port_mat = material("port", (0.005, 0.005, 0.006), rough=.8)
    lbl_mat = material("label", (0.92, 0.92, 0.9), rough=.6)
    shell = material("shell", (0.75, 0.76, 0.78), rough=.25, metal=1.0)
    for hb in spec["hubs"]:
        box("hub", hb["center"], hb["size"], hub_mat)
        for p in hb["ports"]:
            box("port", p["pos"], (0.0012, 0.011, 0.0045), port_mat)   # flush slot: visible only on the side face
            label(p["label"], p["label_pos"], hb["label_size"], lbl_mat)
    for k, c in enumerate(spec["cables"]):
        cmat = material(f"cable{k}", c["color"], rough=c.get("rough", .42))
        cable(f"cable{k}", c["pts"], spec["radius"], cmat, c["id"])
        for end in c["connectors"]:             # plug body + strain relief, same mask id as its cable
            ang = math.atan2(end["dir"][1], end["dir"][0])
            box("plug", end["body"], (0.022, 0.011, 0.0065), cmat if c.get("plug_same", True) else hub_mat,
                c["id"], rot_z=ang)
            box("shell", end["shell"], (0.006, 0.0085, 0.0045), shell, c["id"], rot_z=ang)
    for o in spec.get("occluders", []):
        omat = material("occ", o["color"], rough=o.get("rough", .5))
        if o["kind"] == "box":
            box("occ", o["center"], o["size"], omat, rot_z=o.get("rot", 0))
        else:
            cylinder("occ", o["center"], o["radius"], o["height"], (0, 0, 0), omat)
    for li in spec["lights"]:
        L = bproc.types.Light(light_type=li["type"])
        L.set_location(li["loc"])
        L.set_energy(li["energy"])
        L.set_color(li.get("color", (1, 1, 1)))
        if li["type"] == "AREA":
            L.blender_obj.data.size = li.get("size", 0.8)
        if "rot" in li:
            L.set_rotation_euler(li["rot"])
    bpy.context.scene.world.use_nodes = True
    bg = bpy.context.scene.world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (*spec.get("world", (0.6, 0.62, 0.66)), 1)
    bg.inputs["Strength"].default_value = spec.get("world_strength", 0.6)
    cam = spec["camera"]
    bproc.camera.set_intrinsics_from_blender_params(lens=math.radians(cam["fov"]), lens_unit="FOV")
    fwd = np.array(cam["look"]) - np.array(cam["loc"])
    pose = bproc.math.build_transformation_mat(cam["loc"], bproc.camera.rotation_from_forward_vec(fwd))
    bproc.camera.add_camera_pose(pose)
    bproc.renderer.set_max_amount_of_samples(spec.get("samples", 64))
    # BlenderProc numbers objects for the mask when segmentation is enabled; our objects are created later,
    # so number them now (otherwise every object maps to the background)
    from blenderproc.python.utility.BlenderUtility import get_all_blender_mesh_objects
    for i, o in enumerate(get_all_blender_mesh_objects()):
        o.pass_index = i + 1
    data = bproc.renderer.render()
    out = spec["out"]
    np.save(out + "_rgb.npy", np.asarray(data["colors"][0]).astype(np.uint8))
    np.save(out + "_seg.npy", np.asarray(data["category_id_segmaps"][0]).astype(np.uint8))
    proj = bproc.camera.project_points(np.array(spec["project"], float), frame=0) if spec.get("project") else []
    with open(out + "_proj.json", "w") as f:
        json.dump(np.asarray(proj).tolist(), f)
    print("RENDERED", out)


bproc.init()
bproc.renderer.enable_segmentation_output(map_by=["category_id"], default_values={"category_id": 0})
for path in sys.argv[1:]:
    if not path.endswith(".json"):
        continue
    with open(path) as f:
        spec = json.load(f)
    build_and_render(spec)
    bproc.clean_up(clean_up_camera=True)
