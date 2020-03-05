from setuptools import setup

setup(
   name='Fancy_aggregations',
   version='1.0',
   description='A collection of aggregations, such as OWAs, Choquet and Sugeno integrals, etc.',
   author='Javier Fumanal Idocin',
   url='https://github.com/Fuminides/Fancy_aggregations',
   author_email='javier.fumanal@unavarra.es',
   packages=['Fancy_aggregations'],  #same as name
   install_requires=['numpy'], #external packages as dependencies
)