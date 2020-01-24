from setuptools import setup

setup(name='netMetrics',
      version='0.2',
      description='Functions to automate network feature creation',
      url='http://github.com/dbeskow/netMetrics',
      author='David Beskow',
      author_email='dnbeskow@gmail.com',
      license='MIT',
      packages=['netMetrics'],
      install_requires=[
              'tweepy',
              'pandas',
              'progressbar2',
              'networkx',
              'numpy',
              'python-louvain'
              ],
      zip_safe=False)
