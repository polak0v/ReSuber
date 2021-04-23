
from setuptools import setup
import resuber.utils as utils

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='resuber',
      version='{}'.format(utils.args.get_version()),
      description='Software toolbox to re-synchronize and/or translate SRT subtitles from a movie.',
      url='https://github.com/polak0v/ReSuber',
      download_url='https://github.com/polak0v/ReSuber/archive/v{}.tar.gz'.format(utils.args.get_version()),
      author='polak0v',
      author_email='the_polakov@protonmail.com',
      license='MIT',
      packages=['resuber'],
      scripts=['bin/resuber', 'bin/spleeter2resuber', 'bin/resuber-translate', 'bin/resuber-move', 'bin/resuber-merge'],
      install_requires=requirements,
      include_package_data=True,
      zip_safe=False)