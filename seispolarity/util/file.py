import ftplib
import requests
from pathlib import Path
from tqdm import tqdm
import logging

logger = logging.getLogger("seispolarity")

def download_http(
    url, target, progress_bar=True, desc="Downloading", precheck_timeout=3
):
    """
    Downloads file from http/https source. Raises a ValueError for non-200 status codes.
    Download file from http/https source. Raise ValueError if status code is not 200.
    """
    logger.info(f"Downloading file from {url} to {target}")

    precheck_url(url, timeout=precheck_timeout)

    req = requests.get(url, stream=True, headers={"User-Agent": "SeisPolarity"})

    if req.status_code != 200:
        raise ValueError(
            f"Invalid URL. Request returned status code {req.status_code}."
        )

    content_length = req.headers.get("Content-Length")
    total = int(content_length) if content_length is not None else None
    if progress_bar:
        pbar = tqdm(
            unit="B", total=total, desc=desc, unit_scale=True, unit_divisor=1024
        )
    else:
        pbar = None

    target = Path(target)

    with open(target, "wb") as f_target:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                if pbar is not None:
                    pbar.update(len(chunk))
                f_target.write(chunk)

    if progress_bar:
        pbar.close()


def precheck_url(url, timeout):
    """
    Checks whether the url is reachable and give a 200 or 300 HTTP response code.
    Check if URL is reachable and returns 200 or 300 HTTP response code.
    """
    if timeout <= 0:
        return

    error_description = None

    try:
        req = requests.head(url, timeout=timeout, headers={"User-Agent": "SeisPolarity"})

        if req.status_code >= 400:
            error_description = f"status code {req.status_code}"

    except requests.Timeout:
        error_description = "a timeout"

    except requests.ConnectionError:
        error_description = "a connection error"

    if error_description is not None:
        logger.warning(
            f"The download precheck failed with {error_description}. "
            f"This is not an error itself, but might indicate a subsequent error."
        )

def download_ftp(
    host,
    file,
    target,
    user="anonymous",
    passwd="",
    blocksize=8192,
    progress_bar=True,
    desc="Downloading",
):
    """
    Downloads file from ftp source.
    Download file from ftp source.
    """
    with ftplib.FTP(host, user, passwd) as ftp:
        ftp.voidcmd("TYPE I")
        total = ftp.size(file)

        if progress_bar:
            pbar = tqdm(
                unit="B", total=total, desc=desc, unit_scale=True, unit_divisor=1024
            )

        def callback(chunk):
            if progress_bar:
                pbar.update(len(chunk))
            fout.write(chunk)

        with open(target, "wb") as fout:
            ftp.retrbinary(f"RETR {file}", callback, blocksize=blocksize)

        if progress_bar:
            pbar.close()

def callback_if_uncached(files, callback, force=False, wait_for_file=False):
    """
    Checks if files exist. If not, calls callback.
    Check if file exists. If not, call the callback function.
    """
    if not isinstance(files, list):
        files = [files]
    
    files = [Path(f) for f in files]
    
    if force or any(not f.is_file() for f in files):
        callback(files)
