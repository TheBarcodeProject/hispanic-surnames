import setuptools

with open('requirements.txt', 'r') as f:
    install_requires = f.read().splitlines()

setuptools.setup(name='central_america_panam',
                 packages=['central_america_panam'],
                 install_requires=install_requires)