__author__ = 'yuxiang'

import os
import datasets
import datasets.shapenet
import datasets.imdb
import numpy as np
import subprocess
import cPickle

g_shape_synset_name_pairs = [('02691156', 'aeroplane'),
                             ('02747177', 'ashtray'),
                             ('02773838', 'backpack'),
                             ('02801938', 'basket'),
                             ('02808440', 'bathtub'),  # bathtub
                             ('02818832', 'bed'),
                             ('02828884', 'bench'),
                             ('02834778', 'bicycle'),
                             ('02843684', 'birdhouse'), # missing in objectnet3d, birdhouse, use view distribution of mailbox
                             ('02858304', 'boat'),
                             ('02871439', 'bookshelf'),
                             ('02876657', 'bottle'),
                             ('02880940', 'bowl'), # missing in objectnet3d, bowl, use view distribution of plate
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
                             ('03797390', 'mug'),  # missing in objectnet3d, mug, use view distribution of cup
                             ('03928116', 'piano'),
                             ('03938244', 'pillow'),
                             ('03948459', 'pistol'),  # missing in objectnet3d, pistol, use view distribution of rifle
                             ('03991062', 'pot'),
                             ('04004475', 'printer'),
                             ('04074963', 'remote_control'),
                             ('04090263', 'rifle'),
                             ('04099429', 'rocket'),  # missing in objectnet3d, rocket, use view distribution of road_pole
                             ('04225987', 'skateboard'),
                             ('04256520', 'sofa'),
                             ('04330267', 'stove'),
                             ('04379243', 'table'),  # use view distribution of dining_table
                             ('04401088', 'telephone'),
                             ('04460130', 'tower'),  # missing in objectnet3d, tower, use view distribution of road_pole
                             ('04468005', 'train'),
                             ('04530566', 'washing_machine'),
                             ('04554684', 'dishwasher')]  # washer, use view distribution of dishwasher
g_shape_synsets = [x[0] for x in g_shape_synset_name_pairs]
g_shape_names = [x[1] for x in g_shape_synset_name_pairs]
g_shape_synset_name_mapping = dict(zip(g_shape_synsets, g_shape_names))

class shapenet(datasets.imdb):
    def __init__(self, image_set, shapenet_path=None):
        datasets.imdb.__init__(self, 'shapenet_' + image_set)
        self._image_set = image_set
        self._shapenet_path = self._get_default_path() if shapenet_path is None \
                            else shapenet_path
        self._data_path = os.path.join(self._shapenet_path, 'data')
        self._classes = ('__background__', 'mug')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.png'
        self._image_index = self._load_image_set_index()
        self._roidb_handler = self.gt_roidb

        assert os.path.exists(self._shapenet_path), \
                'shapenet path does not exist: {}'.format(self._shapenet_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self.image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """

        image_path = os.path.join(self._data_path, index + '_rgba' + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def label_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.label_path_from_index(self.image_index[i])

    def label_path_from_index(self, index):
        """
        Construct an label path from the image's "index" identifier.
        """

        label_path = os.path.join(self._data_path, index + '_depth' + self._image_ext)
        assert os.path.exists(label_path), \
                'Path does not exist: {}'.format(label_path)
        return label_path

    def metadata_path_at(self, i):
        """
        Return the absolute path to metadata i in the image sequence.
        """
        return self.metadata_path_from_index(self.image_index[i])

    def metadata_path_from_index(self, index):
        """
        Construct an metadata path from the image's "index" identifier.
        """

        metadata_path = os.path.join(self._data_path, index + '_meta.mat')
        assert os.path.exists(metadata_path), \
                'Path does not exist: {}'.format(metadata_path)
        return metadata_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_set_file = os.path.join(self._shapenet_path, self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)

        with open(image_set_file) as f:
            image_index = [x.rstrip('\n') for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where KITTI is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'ShapeNet')


    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """

        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_shapenet_annotation(index)
                    for index in self.image_index]

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb


    def _load_shapenet_annotation(self, index):
        """
        Load class name and meta data
        """
        # image path
        image_path = self.image_path_from_index(index)

        # label path
        label_path = self.label_path_from_index(index)

        # metadata path
        metadata_path = self.metadata_path_from_index(index)

        # the first 8 digits in index
        synset = index[:8]
        cls = g_shape_synset_name_mapping[synset]
        gt_class = self._class_to_ind[cls]
        
        return {'image': image_path,
                'label': label_path,
                'meta_data': metadata_path,
                'gt_class': gt_class,
                'flipped' : False}


if __name__ == '__main__':
    d = datasets.shapenet('train')
    res = d.roidb
    from IPython import embed; embed()
