from setuptools import setup


# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf8') as f:
    long_description = f.read()


setup(
   name='Fancy_aggregations',
   version='1.4.1',
   description='A collection of aggregations, such as OWAs, Choquet and Sugeno integrals, etc.',
   author='Javier Fumanal Idocin',
   url='https://github.com/Fuminides/Fancy_aggregations',
   author_email='javier.fumanal@unavarra.es',
   packages=['Fancy_aggregations'],  #same as name
   install_requires=['numpy'], #external packages as dependencies
   long_description=long_description,
   long_description_content_type='text/markdown'
)