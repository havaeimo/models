# -*- coding: utf-8 -*-
from distutils.core import setup

scripts = ['projects/perceptron/train_perceptron.py',
           ]

setup(
    name='smartmodels',
    version='0.0.1',
    author='Marc-Alexandre Côté, Adam Salvail, Mathieu Germain',
    author_email='smart-udes-dev@googlegroups.com',
    packages=['smartmodels'],
    scripts=scripts,
    url='https://github.com/SMART-Lab/smartmodels',
    license='LICENSE.txt',
    description='Repository containing all different models we developed in the SMART lab.',
    long_description=open('README.md').read(),
    install_requires=['theano', 'smartlearner'],
    dependency_links=['https://github.com/SMART-Lab/smartlearner/archive/master.zip#egg=smartlearner-0.0.1']
)
