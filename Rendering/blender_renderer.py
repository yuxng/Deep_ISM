#!/usr/bin/env python3

import os
import bpy
import bpy_extras
from mathutils import Matrix, Vector
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io

RENDERING_PATH = './'
MAX_CAMERA_DIST = 2
MAX_DEPTH = 1e8

def camPosToQuaternion(cx, cy, cz):
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = cx / camDist
    cy = cy / camDist
    cz = cz / camDist
    axis = (-cz, 0, cx)
    angle = math.acos(cy)
    a = math.sqrt(2) / 2
    b = math.sqrt(2) / 2
    w1 = axis[0]
    w2 = axis[1]
    w3 = axis[2]
    c = math.cos(angle / 2)
    d = math.sin(angle / 2)
    q1 = a * c - b * d * w1
    q2 = b * c + a * d * w1
    q3 = a * d * w2 + b * d * w3
    q4 = -b * d * w2 + a * d * w3
    return (q1, q2, q3, q4)

def quaternionFromYawPitchRoll(yaw, pitch, roll):
    c1 = math.cos(yaw / 2.0)
    c2 = math.cos(pitch / 2.0)
    c3 = math.cos(roll / 2.0)    
    s1 = math.sin(yaw / 2.0)
    s2 = math.sin(pitch / 2.0)
    s3 = math.sin(roll / 2.0)    
    q1 = c1 * c2 * c3 + s1 * s2 * s3
    q2 = c1 * c2 * s3 - s1 * s2 * c3
    q3 = c1 * s2 * c3 + s1 * c2 * s3
    q4 = s1 * c2 * c3 - c1 * s2 * s3
    return (q1, q2, q3, q4)


def camPosToQuaternion(cx, cy, cz):
    q1a = 0
    q1b = 0
    q1c = math.sqrt(2) / 2
    q1d = math.sqrt(2) / 2
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = cx / camDist
    cy = cy / camDist
    cz = cz / camDist    
    t = math.sqrt(cx * cx + cy * cy) 
    tx = cx / t
    ty = cy / t
    yaw = math.acos(ty)
    if tx > 0:
        yaw = 2 * math.pi - yaw
    pitch = 0
    tmp = min(max(tx*cx + ty*cy, -1),1)
    #roll = math.acos(tx * cx + ty * cy)
    roll = math.acos(tmp)
    if cz < 0:
        roll = -roll    
    print("%f %f %f" % (yaw, pitch, roll))
    q2a, q2b, q2c, q2d = quaternionFromYawPitchRoll(yaw, pitch, roll)    
    q1 = q1a * q2a - q1b * q2b - q1c * q2c - q1d * q2d
    q2 = q1b * q2a + q1a * q2b + q1d * q2c - q1c * q2d
    q3 = q1c * q2a - q1d * q2b + q1a * q2c + q1b * q2d
    q4 = q1d * q2a + q1c * q2b - q1b * q2c + q1a * q2d
    return (q1, q2, q3, q4)

def camRotQuaternion(cx, cy, cz, theta): 
    theta = theta / 180.0 * math.pi
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = -cx / camDist
    cy = -cy / camDist
    cz = -cz / camDist
    q1 = math.cos(theta * 0.5)
    q2 = -cx * math.sin(theta * 0.5)
    q3 = -cy * math.sin(theta * 0.5)
    q4 = -cz * math.sin(theta * 0.5)
    return (q1, q2, q3, q4)

def quaternionProduct(qx, qy): 
    a = qx[0]
    b = qx[1]
    c = qx[2]
    d = qx[3]
    e = qy[0]
    f = qy[1]
    g = qy[2]
    h = qy[3]
    q1 = a * e - b * f - c * g - d * h
    q2 = a * f + b * e + c * h - d * g
    q3 = a * g - b * h + c * e + d * f
    q4 = a * h + b * g - c * f + d * e    
    return (q1, q2, q3, q4)

def obj_centened_camera_pos(dist, azimuth_deg, elevation_deg):
    phi = float(elevation_deg) / 180 * math.pi
    theta = float(azimuth_deg) / 180 * math.pi
    x = (dist * math.cos(theta) * math.cos(phi))
    y = (dist * math.sin(theta) * math.cos(phi))
    z = (dist * math.sin(phi))
    return (x, y, z)

class BlenderRenderer(object):

    def __init__(self, viewport_size_x, viewport_size_y):
        '''
        viewport_size_x, viewport_size_y: rendering viewport resolution
        '''
        # remove the default cube
        bpy.ops.object.select_pattern(pattern="Cube")
        bpy.ops.object.delete()

        render_context = bpy.context.scene.render
        world  = bpy.context.scene.world
        camera = bpy.data.objects['Camera']
        light_1  = bpy.data.objects['Lamp']
        light_1.data.type = 'HEMI'

        # set the camera postion and orientation so that it is in
        # the front of the object
        camera.location = (1, 0, 0)

        # render setting
        render_context.resolution_percentage = 100
        world.horizon_color = (1, 1, 1)  # set background color to be white

        # set file name for storing temporary rendering result
        self.result_fn = '%s/render_result_%d.png' % (RENDERING_PATH, os.getpid())
        bpy.context.scene.render.filepath = self.result_fn

        self.render_context = render_context
        self.camera = camera
        self.light = light_1
        self.model_loaded = False
        self._set_lighting()
        self.render_context.resolution_x = viewport_size_x
        self.render_context.resolution_y = viewport_size_y

    def _set_lighting(self):
        # Create new lamp datablock
        light_data = bpy.data.lamps.new(name="New Lamp", type='HEMI')

        # Create new object with our lamp datablock
        light_2 = bpy.data.objects.new(name="New Lamp", object_data=light_data)
        bpy.context.scene.objects.link(light_2)

        # put the light behind the camera. Reduce specular lighting
        self.light.location       = (0, -2, 2)
        self.light.rotation_mode  = 'ZXY'
        self.light.rotation_euler = (math.radians(45), 0, math.radians(90))
        self.light.data.energy = 0.7

        light_2.location       = (0, 2, 2)
        light_2.rotation_mode  = 'ZXY'
        light_2.rotation_euler = (-math.radians(45), 0, math.radians(90))
        light_2.data.energy = 0.7

    def setViewpoint(self, azimuth, altitude, yaw, distance_ratio, fov):
        self.light.location = (distance_ratio * (MAX_CAMERA_DIST + 2), 0, 0)

        cx, cy, cz = obj_centened_camera_pos(distance_ratio * MAX_CAMERA_DIST, azimuth, altitude)
        q1 = camPosToQuaternion(cx, cy, cz)
        q2 = camRotQuaternion(cx, cy, cz, yaw)
        q = quaternionProduct(q2, q1)

        self.camera.location[0] = cx
        self.camera.location[1] = cy 
        self.camera.location[2] = cz

        self.camera.rotation_mode = 'QUATERNION'
        self.camera.rotation_quaternion[0] = q[0]
        self.camera.rotation_quaternion[1] = q[1]
        self.camera.rotation_quaternion[2] = q[2]
        self.camera.rotation_quaternion[3] = q[3]

    def setTransparency(self, transparency='SKY'):
        """ transparency is either 'SKY', 'TRANSPARENT'
        If set 'SKY', render background using sky color."""
        self.render_context.alpha_mode = transparency

    def selectModel(self):
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_pattern(pattern="RotCenter")
        bpy.ops.object.select_pattern(pattern="Lamp*")
        bpy.ops.object.select_pattern(pattern="Camera")
        bpy.ops.object.select_all(action='INVERT')

    def printSelection(self):
        print(bpy.context.selected_objects)

    def clearModel(self):
        self.selectModel()
        bpy.ops.object.delete()

        # The meshes still present after delete
        for item in bpy.data.meshes:
            bpy.data.meshes.remove(item)
        for item in bpy.data.materials:
            bpy.data.materials.remove(item)

    def loadModel(self, file_path):
        self.model_loaded = True
        try:
            if file_path.endswith('obj'):
                bpy.ops.import_scene.obj(filepath=file_path)
            elif file_path.endswith('3ds'):
                bpy.ops.import_scene.autodesk_3ds(filepath=file_path)
            elif file_path.endswith('dae'):
                # Must install OpenCollada. Please read README.md for installation
                bpy.ops.wm.collada_import(filepath=file_path)
            else:
                # TODO
                # Other formats not supported yet
                self.model_loaded = False
                raise Exception("Loading failed: %s" % (file_path))
        except Exception:
            self.model_loaded = False

    # Build intrinsic camera parameters from Blender camera data
    def compute_intrinsic(self):

        w = self.render_context.resolution_x * self.render_context.resolution_percentage / 100.
        h = self.render_context.resolution_y * self.render_context.resolution_percentage / 100.
        K = Matrix().to_3x3()
        K[0][0] = w/2. / math.tan(self.camera.data.angle/2)
        ratio = w/h
        K[1][1] = h/2. / math.tan(self.camera.data.angle/2) * ratio
        K[0][2] = w / 2.
        K[1][2] = h / 2.
        K[2][2] = 1.

        print('Focal length')
        print(self.camera.data.lens)

        return K

    # Returns camera rotation and translation matrices from Blender.
    # There are 3 coordinate systems involved:
    #    1. The World coordinates: "world"
    #       - right-handed
    #    2. The Blender camera coordinates: "bcam"
    #       - x is horizontal
    #       - y is up
    #       - right-handed: negative z look-at direction
    #    3. The desired computer vision camera coordinates: "cv"
    #       - x is horizontal
    #       - y is down (to align to the actual pixel coordinates 
    #         used in digital images)
    #       - right-handed: positive z look-at direction
    def compute_rotation_translation(self):
        # bcam stands for blender camera
        R_bcam2cv = Matrix(
            ((1, 0,  0),
             (0, -1, 0),
             (0, 0, -1)))

        # Transpose since the rotation is object rotation, 
        # and we want coordinate rotation
        # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
        # T_world2bcam = -1*R_world2bcam * location
        #
        # Use matrix_world instead to account for all constraints
        location, rotation = self.camera.matrix_world.decompose()[0:2]
        R_world2bcam = rotation.to_matrix().transposed()

        # Convert camera location to translation vector used in coordinate changes
        # T_world2bcam = -1*R_world2bcam*cam.location
        # Use location from matrix_world to account for constraints:     
        T_world2bcam = -1*R_world2bcam * location

        # Build the coordinate transform matrix from world to computer vision camera
        R_world2cv = R_bcam2cv*R_world2bcam
        T_world2cv = R_bcam2cv*T_world2bcam

        # put into 3x4 matrix
        RT = Matrix((
            R_world2cv[0][:] + (T_world2cv[0],),
            R_world2cv[1][:] + (T_world2cv[1],),
            R_world2cv[2][:] + (T_world2cv[2],)
             ))
        return RT

    def compute_projection_matrix(self):
        K = self.compute_intrinsic()
        RT = self.compute_rotation_translation()
        return K*RT

    # backproject pixels into 3D points
    def backproject(self, depth):
        # compute projection matrix
        P = self.compute_projection_matrix()
        P = np.matrix(P)
        Pinv = np.linalg.pinv(P)

        # compute the 3D points        
        width = depth.shape[1]
        height = depth.shape[0]
        points = np.zeros((height, width, 3), dtype=np.float32)

        # camera location
        C = self.camera.location
        C = np.matrix(C).transpose()
        Cmat = np.tile(C, (1, width*height))

        # construct the 2D points matrix
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        ones = np.ones((height, width), dtype=np.float32)
        x2d = np.stack((x, y, ones), axis=2).reshape(width*height, 3)

        # backprojection
        x3d = Pinv * x2d.transpose()
        x3d[0,:] = x3d[0,:] / x3d[3,:]
        x3d[1,:] = x3d[1,:] / x3d[3,:]
        x3d[2,:] = x3d[2,:] / x3d[3,:]
        x3d = x3d[:3,:]

        # compute the ray
        R = x3d - Cmat

        # compute the norm
        N = np.linalg.norm(R, axis=0)
        
        # normalization
        R = np.divide(R, np.tile(N, (3,1)))

        # compute the 3D points
        X = Cmat + np.multiply(np.tile(depth.reshape(1, width*height), (3, 1)), R)
        points[y, x, 0] = X[0,:].reshape(height, width)
        points[y, x, 1] = X[2,:].reshape(height, width)
        points[y, x, 2] = X[1,:].reshape(height, width)

        #for x in range(width):
        #    for y in range(height):
        #        if (depth[y, x] < MAX_DEPTH):
        #            x2d = np.matrix([x, y, 1]).transpose()
        #            x3d = Pinv * x2d
        #            x3d = x3d / x3d[3]
        #            x3d = x3d[:3]
        #            # compute the ray
        #            R = x3d - C
        #            # normalization
        #            R = R / np.linalg.norm(R)
        #            # point in 3D
        #            X = C + depth[y, x] * R
        #            # reverse y and z
        #            points[y, x, 0] = X[0]
        #            points[y, x, 1] = X[2]
        #            points[y, x, 2] = X[1]

        return points
            
    def render(self, return_image=True,
               image_path=os.path.join(RENDERING_PATH, 'tmp.png')):
        '''
        Render the object
        '''
        if not self.model_loaded:
            print('Model not loaded.')
            return

        # switch on nodes
        bpy.context.scene.use_nodes = True
        tree = bpy.context.scene.node_tree
        links = tree.links
  
        # clear default nodes
        for n in tree.nodes:
            tree.nodes.remove(n)
  
        # create input render layer node
        rl = tree.nodes.new('CompositorNodeRLayers')      
 
        # create output node
        v = tree.nodes.new('CompositorNodeViewer')
 
        # Links
        links.new(rl.outputs[2], v.inputs[0])  # link Image output to Viewer input

        self.result_fn = image_path
        bpy.context.scene.render.filepath = image_path
        bpy.ops.render.render(write_still=True)  # save straight to file

        # get viewer pixels
        pixels = bpy.data.images['Viewer Node'].pixels
 
        # compute depth map
        depth = np.array(pixels[:])
        width = bpy.data.images['Viewer Node'].size[0]
        height = bpy.data.images['Viewer Node'].size[1]
        depth = depth.reshape((height, width, 4))
        depth = depth[::-1,:,0]
        print(depth.shape, depth.max(), depth.min())
 
        # compute 3D points
        points = self.backproject(depth)

        data = {'depth': depth, 'points': points}
        scipy.io.savemat('data.mat', data)
        # plt.imshow(points)

        # project object
        # count = 0
        # for item in bpy.data.objects:
        #    print(item.name)
        #    if item.type == 'MESH':
        #        count = count + 1
        #        for vertex in item.data.vertices:
        #            x2d = P * Vector((vertex.co[0], vertex.co[2], vertex.co[1], 1))
        #            x2d = x2d / x2d[2]
        #            plt.plot(x2d[0], x2d[1], 'ro')
        #        if count > 1:
        #            break

        # plt.show()

        if return_image:
            im = np.array(Image.open(self.result_fn))  # read the image
            # Last channel is the alpha channel (transparency)
            return im[:, :, :3], im[:, :, 3]

def main():
    '''Test function'''
    dn = '/var/Projects/ShapeNetCore.v1/02958343/'
    model_id = [line.strip('\n') for line in open(dn + 'models.txt')]
    file_paths = [os.path.join(dn, line, 'model.obj') for line in model_id]
    renderer = BlenderRenderer(500, 500)
    for ind, curr_model_id in enumerate(model_id):
        print('Rendering model %s' % curr_model_id)
        az, el, depth_ratio = list(
            *([360, 5, 0.3] * np.random.rand(1, 3) + [0, 25, 0.65]))

        renderer.loadModel(file_paths[ind])
        renderer.setViewpoint(120, 30, 0, 0.7, 25)
        rendering, alpha = renderer.render()

        renderer.clearModel()
        break


if __name__ == "__main__":
    main()
