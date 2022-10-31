import hashlib
import os
import shutil
from typing import Dict
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

from candle.generic_utils import Progbar
from candle.modac_utils import get_file_from_modac


def get_file(
    fname: str,
    origin: str,
    unpack: bool = False,
    md5_hash: str = None,
    cache_subdir: str = "common",
    datadir: str = None,
) -> str:
    """
    Downloads a file from a URL if it not already in the cache. Passing the
    MD5 hash will verify the file after download as well as if it is already
    present in the cache.

    :param string fname: name of the file
    :param string origin: original URL of the file
    :param bool unpack: whether the file should be decompressed
    :param string md5_hash: MD5 hash of the file for verification
    :param string cache_subdir: directory being used as the cache
    :param string datadir: if set, datadir becomes its setting (which could be e.g. an absolute path) \
        and cache_subdir no longer matters

    :return: Path to the downloaded file
    :rtype: string
    """
    if datadir is None and "CANDLE_DATA_DIR" in os.environ:
        datadir = os.environ["CANDLE_DATA_DIR"]
    elif datadir is None and os.environ["CANDLE_DATA_DIR"] is None:
        raise ValueError(
            "Need data directory. Either pass datadir or set CANDLE_DATA_DIR environment variable"
        )

    if cache_subdir is not None:
        datadir = os.path.join(datadir, cache_subdir)

    if not os.path.exists(datadir):
        os.makedirs(datadir)

    if fname.endswith(".tar.gz"):
        fnamesplit = fname.split(".tar.gz")
        unpack_fpath = os.path.join(datadir, fnamesplit[0])
        unpack = True
    elif fname.endswith(".tgz"):
        fnamesplit = fname.split(".tgz")
        unpack_fpath = os.path.join(datadir, fnamesplit[0])
        unpack = True
    elif fname.endswith(".zip"):
        fnamesplit = fname.split(".zip")
        unpack_fpath = os.path.join(datadir, fnamesplit[0])
        unpack = True
    else:
        unpack_fpath = None

    fpath = os.path.join(datadir, fname)
    if not os.path.exists(os.path.dirname(fpath)):
        os.makedirs(os.path.dirname(fpath))

    download = False
    if os.path.exists(fpath) or (
        unpack_fpath is not None and os.path.exists(unpack_fpath)
    ):
        # file found; verify integrity if a hash was provided
        if md5_hash is not None:
            if not validate_file(fpath, md5_hash):
                print(
                    "A local file was found, but it seems to be "
                    "incomplete or outdated."
                )
                download = True
    else:
        download = True

    # fix ftp protocol if needed
    """
    if origin.startswith('ftp://'):
        new_url = origin.replace('ftp://','http://')
        origin = new_url
    print('Origin = ', origin)
    """

    if download:
        if "modac.cancer.gov" in origin:
            get_file_from_modac(fpath, origin)
        else:
            print("Downloading data from", origin)
            global progbar
            progbar = None

            def dl_progress(count, block_size, total_size):
                global progbar
                if progbar is None:
                    progbar = Progbar(total_size)
                else:
                    progbar.update(count * block_size)

            error_msg = "URL fetch failure on {}: {} -- {}"
            try:
                try:
                    urlretrieve(origin, fpath, dl_progress)
                    # fpath = wget.download(origin)
                except URLError as e:
                    raise Exception(error_msg.format(origin, e.errno, e.reason))
                except HTTPError as e:
                    raise Exception(error_msg.format(origin, e.code, e.msg))
            except (Exception, KeyboardInterrupt) as e:
                print(f"Error {e}")
                if os.path.exists(fpath):
                    os.remove(fpath)
                raise
            progbar = None
            print()

    if unpack:
        if not os.path.exists(unpack_fpath):
            print("Unpacking file...")
            try:
                shutil.unpack_archive(fpath, datadir)
            except (Exception, KeyboardInterrupt) as e:
                print(f"Error {e}")
                if os.path.exists(unpack_fpath):
                    if os.path.isfile(unpack_fpath):
                        os.remove(unpack_fpath)
                    else:
                        shutil.rmtree(unpack_fpath)
                raise
        return unpack_fpath

    return fpath


def validate_file(fpath: str, md5_hash: str) -> bool:
    """
    Validates a file against a MD5 hash.

    :param string fpath: path to the file being validated
    :param string md5_hash: the MD5 hash being validated against

    :return: Whether the file is valid
    :rtype: boolean
    """
    hasher = hashlib.md5()
    with open(fpath, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    if str(hasher.hexdigest()) == str(md5_hash):
        return True
    else:
        return False


def directory_from_parameters(params: Dict, commonroot: str = "Output") -> str:
    """
    Construct output directory path with unique IDs from parameters.

    :param Dict params: Dictionary of parameters read
    :param string commonroot: String to specify the common folder to store results.

    :return: Path to the output directory
    :rtype: string
    """

    if commonroot in set([".", "./"]):  # Same directory --> convert to absolute path
        outdir = os.path.abspath(".")
    else:  # Create path specified
        if os.getenv("CANDLE_DATA_DIR"):
            outdir = os.getenv("CANDLE_DATA_DIR")
        else:
            outdir = os.path.abspath(os.path.join(".", commonroot))

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        outdir = os.path.abspath(os.path.join(outdir, params["experiment_id"]))
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        outdir = os.path.abspath(os.path.join(outdir, params["run_id"]))
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    return outdir
