
class Data(object):
    images = {}
    rigid = {}
    deformable = []
    dose = []
    meshes = []

    image_list = []
    rigid_list = []

    @classmethod
    def clear(cls):
        cls.images = {}
        cls.rigid = {}
        cls.deformable = []
        cls.dose = []
        cls.meshes = []

        cls.image_list = []
        cls.rigid_list = []
