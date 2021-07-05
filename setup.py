
import os
import setuptools
import resuber.utils as utils

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = f.read().splitlines()
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(name='resuber',
      version='{}'.format(utils.args.get_version()),
      description='Software toolbox to re-synchronize and/or translate SRT subtitles from a movie.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/polak0v/ReSuber',
      download_url='https://github.com/polak0v/ReSuber/archive/v{}.tar.gz'.format(utils.args.get_version()),
      author='polak0v',
      author_email='the_polakov@protonmail.com',
      license='MIT',
      packages=setuptools.find_packages(),
      scripts=['bin/resuber', 'bin/spleeter2resuber', 'bin/resuber-translate', 'bin/resuber-move', 'bin/resuber-merge'],
      install_requires=requirements,
      include_package_data=True,
      zip_safe=False)