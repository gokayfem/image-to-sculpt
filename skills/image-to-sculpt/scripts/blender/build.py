"""Replay a case's recorded build in Blender and render it (Blender 5.2+, headless).

  blender -b --factory-startup --python build.py -- <case>/blender/job.json

job.json (written by `i2s.py blender`): steps (npz paths + labels), final mesh, cameras (front, turntable, views:
matrix_world, lens, shift, width, height), reference image, centre, outputs. Needs only bpy and numpy.

Writes into <case>/blender/: sculpt.blend (the final sculpt, the reference as the camera background, a turntable
rig), frames/tl_NNNN.png (every recorded step from the photo's camera), frames/tt_NNNN.png (360 degrees),
frames/view_<name>.png (each registered view), and <case>/out/sculpt.glb/.obj/.stl (centred, base at z = 0).
"""
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix

JOB = json.load(open(sys.argv[sys.argv.index("--") + 1]))
FR = JOB["frames_dir"]
os.makedirs(FR, exist_ok=True)


def clear():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)


def setup():
    scn = bpy.context.scene
    scn.render.engine = "BLENDER_WORKBENCH"
    sh = scn.display.shading
    sh.light = "MATCAP"
    sh.studio_light = "clay_studio.exr"
    sh.color_type = "SINGLE"
    sh.single_color = tuple(JOB.get("clay", (0.8, 0.72, 0.62)))
    sh.show_cavity = True
    sh.cavity_type = "BOTH"
    scn.display.render_aa = "16"
    if scn.world is None:
        scn.world = bpy.data.worlds.new("World")
    scn.world.color = (0.035, 0.035, 0.04)
    scn.render.film_transparent = False
    scn.render.image_settings.file_format = "PNG"
    cd = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cd)
    scn.collection.objects.link(cam)
    scn.camera = cam
    if JOB.get("reference"):
        img = bpy.data.images.load(JOB["reference"])
        cd.show_background_images = True
        bg = cd.background_images.new()
        bg.image = img
        bg.alpha = 0.6
    return scn, cam


def set_cam(cam, c, scale=1.0):
    cam.matrix_world = Matrix(c["matrix"])
    cam.data.sensor_fit = "HORIZONTAL"
    cam.data.sensor_width = 36.0
    cam.data.lens = c["lens"]
    cam.data.shift_x, cam.data.shift_y = c["shift_x"], c["shift_y"]
    scn = bpy.context.scene
    scn.render.resolution_x = int(round(c["width"] * scale))
    scn.render.resolution_y = int(round(c["height"] * scale))
    scn.render.resolution_percentage = 100


def mesh_from(path, name="Sculpt"):
    z = np.load(path, allow_pickle=True)
    V, F = z["verts"].astype(np.float64), z["faces"].astype(np.int64)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(V))
    me.vertices.foreach_set("co", V.ravel())
    me.loops.add(F.size)
    me.loops.foreach_set("vertex_index", F.ravel())
    me.polygons.add(len(F))
    me.polygons.foreach_set("loop_start", np.arange(0, F.size, 3))
    me.polygons.foreach_set("use_smooth", np.ones(len(F), bool))
    me.update(calc_edges=True)
    me.validate()
    return me


def put(me, name="Sculpt"):
    if name in bpy.data.objects:
        ob = bpy.data.objects[name]
        old = ob.data
        ob.data = me
        bpy.data.meshes.remove(old)
    else:
        ob = bpy.data.objects.new(name, me)
        bpy.context.scene.collection.objects.link(ob)
    return ob


def render(path):
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def main():
    clear()
    scn, cam = setup()
    s = JOB.get("scale", 1.0)
    what = JOB.get("what", ["timelapse", "turntable", "views", "export"])
    if "timelapse" in what:
        set_cam(cam, JOB["cameras"]["front"], s)
        for k, st in enumerate(JOB["steps"]):
            put(mesh_from(st["path"]))
            render(os.path.join(FR, f"tl_{k:04d}.png"))
            print(f"timelapse {k + 1}/{len(JOB['steps'])} {st['label']}", flush=True)
    ob = put(mesh_from(JOB["final"]))
    if "views" in what:
        for name, c in JOB["cameras"]["views"].items():
            set_cam(cam, c, s)
            render(os.path.join(FR, f"view_{name}.png"))
    # turntable rig: an empty on the vertical axis through the centre, the sculpt parented to it
    piv = bpy.data.objects.new("TurntablePivot", None)
    scn.collection.objects.link(piv)
    piv.location = JOB["center"]
    bpy.context.view_layer.update()  # the pivot's matrix_world is stale until the view layer updates
    ob.parent = piv
    ob.matrix_parent_inverse = piv.matrix_world.inverted()
    if "turntable" in what:
        set_cam(cam, JOB["cameras"]["turntable"], s)
        n = JOB.get("turntable_frames", 180)
        for k in range(n):
            piv.rotation_euler = (0.0, 0.0, 2 * math.pi * k / n)
            render(os.path.join(FR, f"tt_{k:04d}.png"))
        piv.rotation_euler = (0.0, 0.0, 0.0)
        for k in range(n):  # keyframe it too, so the saved .blend plays the turntable
            piv.rotation_euler = (0.0, 0.0, 2 * math.pi * k / n)
            piv.keyframe_insert("rotation_euler", frame=k + 1)
        scn.frame_start, scn.frame_end = 1, n
        print("turntable done", flush=True)
    set_cam(cam, JOB["cameras"]["front"], 1.0)
    bpy.ops.wm.save_as_mainfile(filepath=JOB["blend"])
    if "export" in what:
        z = np.load(JOB["final"])
        V = z["verts"].astype(np.float64)
        o = np.array([JOB["center"][0], JOB["center"][1], V[:, 2].min()])
        me = mesh_from(JOB["final"], "SculptExport")
        me.vertices.foreach_set("co", (V - o).ravel())
        me.update()
        ex = bpy.data.objects.new("SculptExport", me)
        scn.collection.objects.link(ex)
        bpy.ops.object.select_all(action="DESELECT")
        ex.select_set(True)
        bpy.context.view_layer.objects.active = ex
        out = JOB["out_dir"]
        for fn, kw in (("export_scene.gltf", dict(filepath=os.path.join(out, "sculpt.glb"), export_format="GLB",
                                                  use_selection=True)),
                       ("wm.obj_export", dict(filepath=os.path.join(out, "sculpt.obj"), export_selected_objects=True,
                                              forward_axis="NEGATIVE_Y", up_axis="Z")),
                       ("wm.stl_export", dict(filepath=os.path.join(out, "sculpt.stl"), export_selected_objects=True))):
            try:
                mod, op = fn.split(".")
                getattr(getattr(bpy.ops, mod), op)(**kw)
                print("exported", kw["filepath"], flush=True)
            except Exception as e:  # an exporter missing from this Blender build is reported, not fatal
                print("export failed", fn, e, flush=True)
        bpy.data.objects.remove(ex, do_unlink=True)


main()
