__author__ = 'Caleb OConnor'

from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='MedicalImageConverter',
    author='Caleb OConnor',
    author_email='csoconnor@mdanderson.org',
    version='2.0.6',
    description='Reads in medical images and structures them into 3D arrays with associated ROI/POIs if they exist.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['MedicalImageConverter', 'MedicalImageConverter/DataClasses', 'MedicalImageConverter/ReadClasses'],
    include_package_data=True,
    url='https://github.com/caleb-oconnor/MedicalImageConverter',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
    ],
    install_requires=['numpy', 'pandas', 'psutil', 'pydicom', 'pyvista', 'python-gdcm', 'opencv-python', 'SimpleITK'],
)
