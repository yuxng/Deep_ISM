__author__ = 'yuxiang'

import os
import datasets
import datasets.rgbd_scenes
import datasets.imdb
import numpy as np
import subprocess
import cPickle

class rgbd_scenes(datasets.imdb):
    def __init__(self, image_set, rgbd_scenes_path=None):
        datasets.imdb.__init__(self, 'rgbd_scenes_' + image_set)
        self._image_set = image_set
        self._rgbd_scenes_path = self._get_default_path() if rgbd_scenes_path is None \
                            else rgbd_scenes_path
        self._data_path = os.path.join(self._rgbd_scenes_path, 'imgs')
        self._classes = ('__background__', 'bowl', 'cap', 'cereal_box', 'coffee_mug', 'coffee_table', 'office_chair', 'soda_can', 'sofa', 'table')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.png'
        self._image_index = self._load_image_set_index()
        self._roidb_handler = self.gt_roidb

        assert os.path.exists(self._rgbd_scenes_path), \
                'rgbd_scenes path does not exist: {}'.format(self._rgbd_scenes_path)
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

        image_path = os.path.join(self._data_path, index + '-color' + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def depth_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.depth_path_from_index(self.image_index[i])

    def depth_path_from_index(self, index):
        """
        Construct an depth path from the image's "index" identifier.
        """

        depth_path = os.path.join(self._data_path, index + '-depth' + self._image_ext)
        assert os.path.exists(depth_path), \
                'Path does not exist: {}'.format(depth_path)
        return depth_path

    def metadata_path_at(self, i):
        """
        Return the absolute path to metadata i in the image sequence.
        """
        return self.metadata_path_from_index(self.image_index[i])

    def metadata_path_from_index(self, index):
        """
        Construct an metadata path from the image's "index" identifier.
        """

        metadata_path = os.path.join(self._data_path, index + '-meta.mat')
        return metadata_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_set_file = os.path.join(self._rgbd_scenes_path, self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)

        with open(image_set_file) as f:
            image_index = [x.rstrip('\n') for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where KITTI is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'RGBD_Scenes', 'rgbd-scenes-v2')


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

        gt_roidb = [self._load_rgbd_scenes_annotation(index)
                    for index in self.image_index]

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb


    def _load_rgbd_scenes_annotation(self, index):
        """
        Load class name and meta data
        """
        # image path
        image_path = self.image_path_from_index(index)

        # depth path
        depth_path = self.depth_path_from_index(index)

        # metadata path
        metadata_path = self.metadata_path_from_index(index)

        boxes = []
        gt_class = []
        
        return {'image': image_path,
                'depth': depth_path,
                'meta_data': metadata_path,
                'boxes': boxes,
                'gt_classes': gt_class,
                'flipped' : False}


if __name__ == '__main__':
    d = datasets.rgbd_scenes('val')
    res = d.roidb
    from IPython import embed; embed()
