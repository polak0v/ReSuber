import os
import re

def get_files_list(dir, recursive=True, ext=None, exclude="(?!x)x"):
    """Return the list of path to files from a dir.
    
    Parameters
    ----------
        dir : string
            the directory to scan
        recursive : bool
            recursively scan all the content of the directory (default: True)
        ext : string
            the file extension to match (default: None)

    Returns
    -------
        `list` [string] : all the filepaths that matched
    """
    filepaths = []
    if recursive:
        for root, _, files in os.walk(dir):
            for file in files:
                if re.search(exclude, file) is not None:
                    continue
                filepath = os.path.join(root, file)
                if ext is not None:
                    if filepath.split('.')[-1] == ext:
                        filepaths += [filepath]
                else:
                    filepaths += [filepath]
    else:
        for file in os.listdir(dir):
            if re.search(exclude, file) is not None:
                continue
            filepath = os.path.join(dir, file)
            if os.path.isfile(filepath):
                if ext is not None:
                    if filepath.split('.')[-1] == ext:
                        filepaths += [filepath]
                else:
                    filepaths += [os.path.join(dir, file)]
    
    return filepaths

def filepath_match(all_filepaths, filepath):
    """Return a list of path to files that matches the filepath name.
    
    Parameters
    ----------
        all_filepaths : `list` [string]
            all the filepaths to be checked for a match
        filepath : string
            the filepath to match

    Returns
    -------
        `list` [string] : all the filepaths that matched
    """
    filepaths = []
    filepath_name = '.'.join(os.path.basename(filepath).split('.')[:-1])
    # escape parenthesis
    filepath_name = filepath_name.replace("(", "\\(")
    filepath_name = filepath_name.replace(")", "\\)")
    filepath_name = filepath_name.replace("[", "\\[")
    filepath_name = filepath_name.replace("]", "\\]")
    filepath_name = filepath_name.replace(".", "\\.")
    for fpath in all_filepaths:
        match = re.search(filepath_name, os.path.basename(fpath))
        if match:
            if match[0]:
                filepaths += [fpath]
    
    return filepaths

