from setuptools import find_packages, setup

install_requires = [
    "future",
    "numpy",
    "Pillow",
    "torch",
    "torch-dct",
    "torchvision",
]

tests_require = [
    "scipy",
]


setup(name='gandetect',
      version='0.0.1',
      license='MIT',
      packages=find_packages(),
      install_requires=install_requires,
      tests_require=tests_require,
      test_suite="tests")
