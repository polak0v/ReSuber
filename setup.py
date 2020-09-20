
from setuptools import setup
from resuber.utils.args import get_module_name, get_version

setup(name='resuber',
      version='{}'.format(get_version()),
      description='Automatic data fetcher from a remote server.',
      url='https://github.com/polak0v/ReSuber',
      download_url='https://github.com/polak0v/ReSuber/archive/v{}.tar.gz'.format(get_version()),
      author='polak0v',
      author_email='the_polakov@protonmail.com',
      license='MIT',
      packages=['resuber'],
      scripts=['bin/resuber', 'bin/spleeter2resuber'],
    #   install_requires=[''],
      include_package_data=True,
      zip_safe=False)