import codecs
import os.path
import re

def read(relative_path):
    """Read the curent file.

    Parameters
    ----------
        relative_path : string, required
            relative path to the file to be read, from the directory of this file
    
    Returns
    -------
        string : content of the file at relative path
    """
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, relative_path), 'r') as fp:
        return fp.read()

def get_version():
    """Get the version of this software, as describe in the __init__.py file from the top module.
    
    Returns
    -------
        string : version of this software
    """
    relative_path = os.path.join("..", "__init__.py")
    for line in read(relative_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

def get_module_name():
    """Get the module name of this software, as describe in the __init__.py file from the top module.
    
    Returns
    -------
        string : name of this software
    """
    relative_path = os.path.join("..", "__init__.py")
    for line in read(relative_path).splitlines():
        if line.startswith('__name__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find name string.")

def remove_none_args(args):
    """Remove the `None` from the `argparse` argument object

    Parameters
    ----------
        args : `argparse.Namespace`, required
            `argparse` argument object
    
    Returns
    -------
        `argparse.Namespace` : `argparse` argument object without `None`
    """
    attributes = []

    for k in args.__dict__:
        if args.__dict__[k] is None:
            attributes += [k]
    for attribute in attributes:
        args.__delattr__(attribute)
        
    return args

def get_docstring_option_help(obj, arg):
    """Return the docstring help from an argument of a given object

    Parameters
    ----------
        obj : `class`, required
            `class` of the object from which to read the docstring
        arg : string, required
            name of the argument from which to read the docstring help
    
    Returns
    -------
        string : the docstring help
    """
    doc_as_list = obj.__init__.__doc__.split("\n")
    for i in range(len(doc_as_list)):
        if re.search(".*?" + arg + " : .*?", doc_as_list[i]):
            return doc_as_list[i+1]
    else:
        raise RuntimeError("Unable to find arg {} from generated docstring of {}".format(arg, type(obj)))