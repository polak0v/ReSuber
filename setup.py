
from setuptools import setup
import resuber.utils as utils

setup(name='resuber',
      version='{}'.format(utils.args.get_version()),
      description='Automatic data fetcher from a remote server.',
      url='https://github.com/polak0v/ReSuber',
      download_url='https://github.com/polak0v/ReSuber/archive/v{}.tar.gz'.format(utils.args.get_version()),
      author='polak0v',
      author_email='the_polakov@protonmail.com',
      license='MIT',
      packages=['resuber'],
      scripts=['bin/resuber', 'bin/spleeter2resuber', 'bin/resuber-translate', 'bin/resuber-move', 'bin/resuber-merge'],
      install_requires=['tensorflow-gpu==2.3.0', 'scipy==1.4.1', 'matplotlib==3.1.2'],
      include_package_data=True,
      zip_safe=False)