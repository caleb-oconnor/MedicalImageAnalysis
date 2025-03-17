
class Data(object):
    images = []
    rigid = []
    deformable = []
    dose = []
    meshes = []

    @classmethod
    def clear(cls):
        cls.images = []
        cls.rigid = []
        cls.deformable = []
        cls.dose = []
        cls.meshes = []
