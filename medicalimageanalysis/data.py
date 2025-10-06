
class Data(object):
    image = {}
    rigid = {}
    deformable = {}
    dose = {}

    image_list = []
    roi_list = []
    rigid_list = []
    dose_list = []

    @classmethod
    def clear(cls):
        cls.image = {}
        cls.rigid = {}
        cls.deformable = {}
        cls.dose = {}

        cls.image_list = []
        cls.roi_list = []
        cls.rigid_list = []

    @classmethod
    def delete_image(cls, image_name):
        del cls.image[image_name]
        del cls.image_list[image_name]

    @classmethod
    def match_rois(cls):
        image_rois = [list(cls.image[image_name].rois.keys()) for image_name in list(cls.image.keys())]
        roi_names = list({x for r in image_rois for x in r})
        Data.roi_list = roi_names

        color = [[128, 128, 128]] * len(roi_names)
        visible = [False] * len(roi_names)
        for ii, roi_name in enumerate(roi_names):
            for image_name in list(cls.image.keys()):
                rois_on_image = list(cls.image[image_name].rois.keys())
                if roi_name in rois_on_image:
                    if cls.image[image_name].rois[roi_name].color is not None:
                        color[ii] = cls.image[image_name].rois[roi_name].color
                        visible[ii] = cls.image[image_name].rois[roi_name].visible

        for ii, roi_name in enumerate(roi_names):
            for image_name in list(cls.image.keys()):
                rois_on_image = list(cls.image[image_name].rois.keys())
                if roi_name not in rois_on_image:
                    cls.image[image_name].add_roi(roi_name=roi_name, color=color[ii], visible=visible[ii])

    @classmethod
    def match_pois(cls):
        image_pois = [list(cls.image[image_name].pois.keys()) for image_name in list(cls.image.keys())]
        poi_names = list({x for r in image_pois for x in r})

        color = [[128, 128, 128]] * len(poi_names)
        visible = [False] * len(poi_names)
        for ii, poi_name in enumerate(poi_names):
            for image_name in list(cls.image.keys()):
                pois_on_image = list(cls.image[image_name].pois.keys())
                if poi_name in pois_on_image:
                    if cls.image[image_name].pois[poi_name].color is not None:
                        color[ii] = cls.image[image_name].pois[poi_name].color
                        visible[ii] = cls.image[image_name].pois[poi_name].visible

        for ii, poi_name in enumerate(poi_names):
            for image_name in list(cls.image.keys()):
                pois_on_image = list(cls.image[image_name].pois.keys())
                if poi_name not in pois_on_image:
                    cls.image[image_name].add_poi(poi_name=poi_name, color=color[ii], visible=visible[ii])
