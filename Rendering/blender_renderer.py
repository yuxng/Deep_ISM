#!/usr/bin/env python3

import os
import random
import bpy
import bpy_extras
from mathutils import Matrix, Vector
import math
import numpy as np
import scipy.io
import pickle
import png

RENDERING_PATH = './'
MAX_CAMERA_DIST = 2
MAX_DEPTH = 1e8
FACTOR_DEPTH = 10000
g_shape_synset_name_pairs = [('02691156', 'aeroplane'),
                             ('02747177', 'ashtray'),
                             ('02773838', 'backpack'),
                             ('02801938', 'basket'),
                             ('02808440', 'tub'),  # bathtub
                             ('02818832', 'bed'),
                             ('02828884', 'bench'),
                             ('02834778', 'bicycle'),
                             ('02843684', 'mailbox'), # missing in objectnet3d, birdhouse, use view distribution of mailbox
                             ('02858304', 'boat'),
                             ('02871439', 'bookshelf'),
                             ('02876657', 'bottle'),
                             ('02880940', 'plate'), # missing in objectnet3d, bowl, use view distribution of plate
                             ('02924116', 'bus'),
                             ('02933112', 'cabinet'),
                             ('02942699', 'camera'),
                             ('02946921', 'can'),
                             ('02954340', 'cap'),
                             ('02958343', 'car'),
                             ('02992529', 'cellphone'),
                             ('03001627', 'chair'),
                             ('03046257', 'clock'),
                             ('03085013', 'keyboard'),
                             ('03207941', 'dishwasher'),
                             ('03211117', 'tvmonitor'),
                             ('03261776', 'headphone'),
                             ('03325088', 'faucet'),
                             ('03337140', 'filing_cabinet'),
                             ('03467517', 'guitar'),
                             ('03513137', 'helmet'),
                             ('03593526', 'jar'),
                             ('03624134', 'knife'),
                             ('03636649', 'lamp'),
                             ('03642806', 'laptop'),
                             ('03691459', 'speaker'),
                             ('03710193', 'mailbox'),
                             ('03759954', 'microphone'),
                             ('03761084', 'microwave'),
                             ('03790512', 'motorbike'),
                             ('03797390', 'cup'),  # missing in objectnet3d, mug, use view distribution of cup
                             ('03928116', 'piano'),
                             ('03938244', 'pillow'),
                             ('03948459', 'rifle'),  # missing in objectnet3d, pistol, use view distribution of rifle
                             ('03991062', 'pot'),
                             ('04004475', 'printer'),
                             ('04074963', 'remote_control'),
                             ('04090263', 'rifle'),
                             ('04099429', 'road_pole'),  # missing in objectnet3d, rocket, use view distribution of road_pole
                             ('04225987', 'skateboard'),
                             ('04256520', 'sofa'),
                             ('04330267', 'stove'),
                             ('04379243', 'diningtable'),  # use view distribution of dining_table
                             ('04401088', 'telephone'),
                             ('04460130', 'road_pole'),  # missing in objectnet3d, tower, use view distribution of road_pole
                             ('04468005', 'train'),
                             ('04530566', 'washing_machine'),
                             ('04554684', 'dishwasher')]  # washer, use view distribution of dishwasher

g_shape_synsets = [x[0] for x in g_shape_synset_name_pairs]
g_shape_names = [x[1] for x in g_shape_synset_name_pairs]
g_view_distribution_files = dict(zip(g_shape_synsets, [name+'.txt' for name in g_shape_names]))

g_syn_light_num_lowbound = 3
g_syn_light_num_highbound = 6
g_syn_light_dist_lowbound = 8
g_syn_light_dist_highbound = 12
g_syn_light_azimuth_degree_lowbound = 0
g_syn_light_azimuth_degree_highbound = 360
g_syn_light_elevation_degree_lowbound = -90
g_syn_light_elevation_degree_highbound = 90
g_syn_light_energy_mean = 3
g_syn_light_energy_std = 1
g_syn_light_environment_energy_lowbound = 0
g_syn_light_environment_energy_highbound = 1

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

        # set the camera postion and orientation so that it is in
        # the front of the object
        camera.location = (1, 0, 0)

        # render setting
        render_context.resolution_percentage = 100
        world.horizon_color = (1, 1, 1)  # set background color to be white

        # set file name for storing temporary rendering result
        self.result_fn = '%s/render_result_%d.png' % (RENDERING_PATH, os.getpid())
        bpy.context.scene.render.filepath = self.result_fn

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

        self.render_context = render_context
        self.camera = camera
        self.model_loaded = False
        self.render_context.resolution_x = viewport_size_x
        self.render_context.resolution_y = viewport_size_y
        self.pngWriter = png.Writer(viewport_size_x, viewport_size_y, greyscale=True, alpha=False, bitdepth=16)

    def _set_lighting(self, azimuth, elevation):
        # clear default lights
        bpy.ops.object.select_by_type(type='LAMP')
        bpy.ops.object.delete(use_global=False)

        # set environment lighting
        bpy.context.scene.world.light_settings.use_environment_light = True
        bpy.context.scene.world.light_settings.environment_energy = np.random.uniform(g_syn_light_environment_energy_lowbound, g_syn_light_environment_energy_highbound)
        bpy.context.scene.world.light_settings.environment_color = 'PLAIN'

        # set point lights
        num_light = random.randint(g_syn_light_num_lowbound,g_syn_light_num_highbound)
        print(num_light)
        light_info = np.zeros((num_light, 4), dtype=np.float32)
        for i in range(num_light):
            light_azimuth_deg = np.random.uniform(g_syn_light_azimuth_degree_lowbound, g_syn_light_azimuth_degree_highbound)
            light_elevation_deg  = np.random.uniform(g_syn_light_elevation_degree_lowbound, g_syn_light_elevation_degree_highbound)
            light_dist = np.random.uniform(g_syn_light_dist_lowbound, g_syn_light_dist_highbound)
            lx, ly, lz = obj_centened_camera_pos(light_dist, light_azimuth_deg, light_elevation_deg)
            bpy.ops.object.lamp_add(type='POINT', view_align = False, location=(lx, ly, lz))
            light_energy = np.random.normal(g_syn_light_energy_mean, g_syn_light_energy_std)
            bpy.data.objects['Point'].data.energy = light_energy

            light_info[i, 0] = light_azimuth_deg
            light_info[i, 1] = light_elevation_deg
            light_info[i, 2] = light_dist
            light_info[i, 3] = light_energy

        self.light_info = light_info

    def setViewpoint(self, azimuth, altitude, yaw, distance_ratio, fov):
        self._set_lighting(azimuth, altitude)

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

        self.azimuth = azimuth
        self.elevation = altitude
        self.tilt = yaw
        self.distance = distance_ratio * MAX_CAMERA_DIST

    def setTransparency(self, transparency='SKY'):
        """ transparency is either 'SKY', 'TRANSPARENT'
        If set 'SKY', render background using sky color."""
        self.render_context.alpha_mode = transparency

    def selectModel(self):
        bpy.ops.object.select_all(action='DESELECT')
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
        return K*RT, RT, K

    # backproject pixels into 3D points
    def backproject(self, depth):
        # compute projection matrix
        P, RT, K = self.compute_projection_matrix()
        P = np.matrix(P)
        Pinv = np.linalg.pinv(P)

        # compute the 3D points        
        width = depth.shape[1]
        height = depth.shape[0]
        points = np.zeros((height, width, 3), dtype=np.float64)

        # camera location
        C = self.camera.location
        C = np.matrix(C).transpose()
        Cmat = np.tile(C, (1, width*height))

        # construct the 2D points matrix
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        ones = np.ones((height, width), dtype=np.float64)
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

        # naive way of computing the 3D points
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
            
    def render(self, return_depth=True,
               image_path=os.path.join(RENDERING_PATH, 'tmp.png')):
        '''
        Render the object
        '''
        if not self.model_loaded:
            print('Model not loaded.')
            return

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
        ind = np.where(depth > MAX_DEPTH)
        depth[ind] = 0

        # convert depth map
        depth = depth * FACTOR_DEPTH
        depth = depth.astype(np.uint16)

        if return_depth:
            return depth

    def save_meta_data(self, filename):
        P, RT, K = self.compute_projection_matrix()

        meta_data = {'projection_matrix' : np.array(P),
                     'rotation_translation_matrix': np.array(RT),
                     'intrinsic_matrix': np.array(K),
                     'azimuth': self.azimuth,
                     'elevation': self.elevation,
                     'tilt': self.tilt,
                     'distance': self.distance,
                     'viewport_size_x': self.render_context.resolution_x,
                     'viewport_size_y': self.render_context.resolution_y,
                     'camera_location': np.array(self.camera.location),
                     'factor_depth': FACTOR_DEPTH,
                     'light_info': self.light_info}

        scipy.io.savemat(filename+'.mat', meta_data)


def main():
    '''Test function'''

    synset = '04379243'
    view_num = 10

    shapenet_root = '/var/Projects/ShapeNetCore.v1'
    view_dists_root = '/var/Projects/Deep_ISM/ObjectNet3D/view_distributions'
    results_root = '/var/Projects/Deep_ISM/Rendering/data/' + synset
    if not os.path.exists(results_root):
        os.makedirs(results_root)

    # load 3D shape paths
    dn = os.path.join(shapenet_root, synset)
    model_id = [line.strip('\n') for line in open(dn + '/models.txt')]
    file_paths = [os.path.join(dn, line, 'model.obj') for line in model_id]

    # load viewpoint distributions
    filename = os.path.join(view_dists_root, g_view_distribution_files[synset])
    if not os.path.exists(filename):
        print('Failed to read view distribution files from %s for synset %s' % 
              (filename, synset))
        exit()
    view_params = open(filename).readlines()
    view_params = [[float(x) for x in line.strip().split(' ')] for line in view_params]

    # initialize the blender render
    renderer = BlenderRenderer(256, 256)

    # for each 3D shape
    for ind, curr_model_id in enumerate(model_id):
        print('Rendering model %s' % curr_model_id)
        renderer.loadModel(file_paths[ind])

        # create output directory
        dirname = os.path.join(results_root, curr_model_id)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # sample viewpoints
        for i in range(view_num): 
            index = random.randint(0, len(view_params)-1)
            azimuth = view_params[index][0]
            elevation = view_params[index][1]
            tilt = view_params[index][2]

            # set viewpoint
            renderer.setViewpoint(azimuth, elevation, tilt, 0.7, 25)

            # set transparency
            renderer.setTransparency('TRANSPARENT')

            # rendering
            filename = dirname + '%02d_rgba.png' % i
            depth = renderer.render(True, filename)

            # save depth image
            filename = dirname + '%02d_depth.png' % i
            pngfile = open(filename, 'wb')
            renderer.pngWriter.write(pngfile, depth)

            # save meta data
            filename = dirname + '%02d_meta' % i
            renderer.save_meta_data(filename)

        renderer.clearModel()

        if ind >= 199:
            break


if __name__ == "__main__":
    main()
