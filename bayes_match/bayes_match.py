# @Author: Ilija Medan
# @Filename: bayes_match.py
# @License: BSD 3-Clause
# @Copyright: Ilija Medan

import time
import pandas as pd
import numpy as np
from astropy.time import Time
import operator
from tqdm import tqdm_notebook
from scipy.stats import rankdata
from scipy.optimize import curve_fit
from tqdm import tnrange
from lmfit import Parameters, minimize
from astropy import units as u
from astropy.coordinates import SkyCoord
import matplotlib.pylab as plt
from functools import reduce
import os


def mjd_to_yr(mjds):
    """
    Convert a mjd epoch to decimal year

    Parameters
    ----------

    mjds: array
        Mean Julian Dates
    """
    t = Time(mjds, format='mjd')
    return t.decimalyear


def ang_sep(ra, dec, epoch, rag, decg, pmra, pmdec):
    """
    Calculate the angular seperation between two objects
    taking into account the proper motion and epoch of the object.
    This assumes the epoch is given as a mjd.

    Parameters
    ----------

    ra: array
        Right Ascension of external objects

    dec: array
        Declination of external objects

    epoch: array
        Epoch in MJD of external objects

    rag: float
        Right Ascension of Gaia objects

    decg: float
        Declination of Gaia objects

    pmra: float
        Proper Motion in Right Ascension of Gaia objects

    pmdec: float
        Proper Motion in Declination of Gaia objects
    """
    year = mjd_to_yr(epoch)
    rax = ra + 2.7778e-7 * (2015.5 - year) * pmra / np.cos(dec / 57.296)
    decx = dec + 2.7778e-7 * (2015.5 - year) * pmdec
    cosx = np.cos(decx / 57.296)
    dra = 3600 * (rag - rax) * cosx
    dde = 3600 * (decg - decx)
    return dra, dde


def ang_sep2(ra, dec, epoch, rag, decg, pmra, pmdec):
    """
    Calculate the angular seperation between two objects
    taking into account the proper motion and epoch of the object.
    This assumes the epoch is given as a decimal year.

    Parameters
    ----------

    ra: array
        Right Ascension of external objects

    dec: array
        Declination of external objects

    epoch: array
        Epoch in decimal years of external objects

    rag: float
        Right Ascension of Gaia objects

    decg: float
        Declination of Gaia objects

    pmra: float
        Proper Motion in Right Ascension of Gaia objects

    pmdec: float
        Proper Motion in Declination of Gaia objects
    """
    year = epoch  # mjd_to_yr(epoch)
    rax = ra + 2.7778e-7 * (2015.5 - year) * pmra / np.cos(dec / 57.296)
    decx = dec + 2.7778e-7 * (2015.5 - year) * pmdec
    cosx = np.cos(decx / 57.296)
    dra = 3600 * (rag - rax) * cosx
    dde = 3600 * (decg - decx)
    return dra, dde


def xyz(sinra1, cosra1, sindec1, cosdec1, ra2, dec2):
    """
    Calculate the cartesian seperation between two objects
    from the ra and dec of the objects
    """
    x1 = cosra1 * cosdec1
    y1 = sinra1 * cosdec1
    z1 = sindec1
    x2 = np.cos(ra2 / 57.295) * np.cos(dec2 / 57.295)
    y2 = np.sin(ra2 / 57.295) * np.cos(dec2 / 57.295)
    z2 = np.sin(dec2 / 57.295)
    return abs(x1 - x2), abs(y1 - y2), abs(z1 - z2)


def lines_to_skip(skip, file, ra, linemax):
    """
    Find the approximate location in the file for
    a match between two catalogs. Assumes both catalogs
    are sorted by right ascension and returns the rows
    in the file near objects of similar right ascension

    Parameters
    ----------

    skip: int
        number of lines to skip before seatching

    file: str
        file searching through

    ra: float
        Right Ascension you are trying to be close to

    linemax: float
        Number of lines in file
    """
    with open(file, 'r') as f3:
        if skip == 0:
            pass
        else:
            for i in range(skip):
                next(f3)
        skn = 0 + skip
        for x in f3:
            skn += 1
            line = x.split()
            if float(line[1]) > ra:
                break
    sk_st = skn - 50000
    sk_end = skn + 50000
    if sk_st < 0:
        sk_st = 0
    if sk_end > linemax:
        sk_end = linemax
    return skn, sk_st, sk_end


def lines_to_skip2(skip, file, ra, linemax):
    """
    Find the approximate location in the file for
    a match between two catalogs. Assumes both catalogs
    are sorted by right ascension and returns the rows
    in the file near objects of similar right ascension

    This differs from above as it does not open a file
    and knows where to start looking through an array instead.
    This is to save time looping through a file.

    Parameters
    ----------

    skip: int
        number of lines to skip before seatching

    file: array
        file searching through

    ra: float
        Right Ascension you are trying to be close to

    linemax: float
        Number of lines in file
    """
    for j in range(skip, linemax):
        if file[j][1] > ra or j == linemax - 1:
            skn = j
            break
    sk_st = skn - 50000
    sk_end = skn + 50000
    if sk_st < 0:
        sk_st = 0
    if sk_end > linemax:
        sk_end = linemax
    return skn, sk_st, sk_end


def lines_to_skip3(skip, file, ra, linemax):
    """
    Find the approximate location in the file for
    a match between two catalogs. Assumes both catalogs
    are sorted by right ascension and returns the rows
    in the file near objects of similar right ascension

    This is similar to version 2, except this is for when
    you have only loaded a chunk of the file into memory.
    If no lines seem close in right ascension, it tells the
    program to then load the next chunk.

    Parameters
    ----------

    skip: int
        number of lines to skip before seatching

    file: array
        file searching through

    ra: float
        Right Ascension you are trying to be close to

    linemax: float
        Number of lines in file
    """
    for j in range(skip, linemax):
        if file[j][1] > ra or j == linemax - 1:
            skn = j
            break
    if skn == linemax - 1 or skn + 50000 > linemax:
        return skn, -1, -1
    else:
        sk_st = skn - 50000
        sk_end = skn + 50000
        if sk_st < 0:
            sk_st = 0
        if sk_end > linemax:
            sk_end = linemax
        return skn, sk_st, sk_end


def min_year(ra, dec, rag, decg, pmra, pmdec):
    """
    This finds the epoch at which the star in an external catalog
    will be at the impact parameter along a provided proper
    motion vector.

    NOTE: This currently looks for a large range of epochs.
    This is probably unnecessary and can be made smaller to save
    time.

    Parameters
    ----------

    ra: float
        Right Ascension of external objects

    dec: float
        Declination of external objects

    rag: float
        Right Ascension of Gaia object

    decg: float
        Declination of Gaia object

    pmra: float
        Proper Motion in Right Ascension of Gaia object

    pmdec: float
        Proper Motion in Declination of Gaia object
    """
    epochs = np.arange(1950, 2050 + 1 / 12, 1 / 12)
    dras = np.zeros(len(epochs))
    ddes = np.zeros(len(epochs))
    for i in range(len(epochs)):
        dras[i], ddes[i] = ang_sep2(ra, dec, epochs[i], rag, decg, pmra, pmdec)
    dra_min = dras[np.argmin(dras ** 2 + ddes ** 2)]
    dde_min = ddes[np.argmin(dras ** 2 + ddes ** 2)]
    epoch_min = epochs[np.argmin(dras ** 2 + ddes ** 2)]
    return dra_min, dde_min, epoch_min


# order for file to match should always be:
# objID, RA, DEC, epoch, sinRA, cosRA, sinDEC, cosDEC, Mags+MagErrs alternating, line #
def cross_match(match_file, chunksize, total,
                idcol, usecols, match_save_file,
                match_save_file_all, magcols, main_match_file):
    """
    This function finds the initial best matches in an external catalog to
    a catalog of Gaia sources. Initial best match is just the source with the
    smallest impact paramater to the Gaia source. This assumes the Gaia catalog
    has columns matching the GAIADR2_star.header file.

    Parameters
    ----------

    match_file: str
       external catalog that has already been pre-queried
       around all of the Gaia sources out to at least 15 as.
       This file MUST be sorted by RA.

    chunksize: int
        Size of chunks to load match_file

    total: int
        number of rows (not including header) in match_file

    idcol: list
        list that contains column number where the catalog id
        is in match_file

    usecols: list
        column numbers for [ra, dec, epoch] in match_file.
        It is assumed that the epoch is in mjd.

    match_save_file: str
        file name where to store initial best matches

    match_save_file_all: str
        file where to store all matches within 15"

    magcols: list
        column numbers of all magntiudes and magntidue errors
        in match_file (alternating)

    main_match_file: str
        file name where the Gaia catalog is stored. Gaia catalog
        must have columns matching GAIADR2_star.header file.
    """
    print('STARTING')
    start = time.time()
    path = ''
    path1 = ''

    # load the data in a way that can be divided into chunks
    match_ids_chunks = pd.read_table(path + match_file,
                                     usecols=idcol,
                                     dtype=int,
                                     chunksize=chunksize,
                                     delim_whitespace=True)
    match_ids = match_ids_chunks.get_chunk(chunksize).to_numpy(dtype=int)

    match_array_chunks = pd.read_table(path + match_file,
                                       usecols=usecols,
                                       chunksize=chunksize,
                                       delim_whitespace=True)
    match_array = match_array_chunks.get_chunk(chunksize).to_numpy(dtype=float)
    match_array = np.column_stack((np.array(range(len(match_array))),
                                   match_array))
    match_array = np.column_stack((match_array,
                                   np.sin(match_array[:, 1] / 57.295),
                                   np.cos(match_array[:, 1] / 57.295),
                                   np.sin(match_array[:, 2] / 57.295),
                                   np.cos(match_array[:, 2] / 57.295)))

    match_mags_chunks = pd.read_table(path + match_file,
                                      usecols=magcols,
                                      chunksize=chunksize,
                                      delim_whitespace=True)
    match_mags = match_mags_chunks.get_chunk(chunksize).to_numpy(dtype=float)
    match_array = np.column_stack((match_array,
                                   match_mags,
                                   np.arange(0, len(match_array), 1)))

    num_to_skip = 0
    best_epochs = []
    with open(path1 + main_match_file, 'r') as f, open(path + match_save_file_all, 'w') as fall, open(path + match_save_file,'w') as fbest:
        i = 0
        skip = 0
        matches = 0
        tot = len(match_array)
        if tot < total:
            do_3 = True
        else:
            do_3 = False
        # begin iterating though the Gaia file
        for x in f:
            i += 1
            # can change this line if want to start match
            # where you left off in another run
            if i > 0:
            	# pull out needed data from the Gaia file
                line = x.split()
                rag = float(line[0])
                decg = float(line[2])
                pmrag = float(line[7])
                pmdecg = float(line[9])
                idg = int(line[4])
                plxg = float(line[5])
                plxerrg = float(line[6])
                Gg = float(line[11])
                # find out which lines in the external catalog are near the Gaia source
                if do_3:
                    skip, sk_st, sk_end = lines_to_skip3(skip, match_array, rag,tot)
                else:
                    skip, sk_st, sk_end = lines_to_skip2(skip, match_array, rag,tot)
                while sk_st < 0:
                	# load new chunk if at end of current one
                    print('NEW CHUNK')
                    num_to_skip += chunksize
                    
                    match_ids = np.concatenate((match_ids[int(chunksize - 100000):],
                                                match_ids_chunks.get_chunk(chunksize).to_numpy(dtype=int)))

                    match_array_stack = match_array[int(chunksize - 100000):,:].copy()
                    match_array_stack[:, 0] -= match_array_stack[0][0]
                    match_array = match_array_chunks.get_chunk(chunksize).to_numpy(dtype=float)
                    match_array = np.column_stack((np.array(range(int(match_array_stack[-1][0] + 1),
                                                   len(match_array) + len(match_array_stack))),
                                                   match_array))
                    match_array = np.column_stack((match_array,
                                                   np.sin(match_array[:, 1] / 57.295),
                                                   np.cos(match_array[:, 1] / 57.295),
                                                   np.sin(match_array[:, 2] / 57.295),
                                                   np.cos(match_array[:, 2] / 57.295)))

                    match_mags = match_mags_chunks.get_chunk(chunksize).to_numpy(dtype=float)
                    match_array = np.column_stack((match_array,
                                                   match_mags,
                                                   np.arange(num_to_skip,num_to_skip + len(match_array), 1)))
                    match_array = np.concatenate((match_array_stack, match_array))                    
                    tot = len(match_array)
                    skip = 0
                    skip, sk_st, sk_end = lines_to_skip3(skip, match_array, rag, tot)
                    # if you reach the end of the file, know to change line skip
                    # function that you use
                    if total - num_to_skip <= chunksize:
                        do_3 = False
                # slice the file justa round the Gaia source
                match_array1 = match_array[sk_st:sk_end, :]
                xd, yd, zd = xyz(match_array1[:, 4],
                                 match_array1[:, 5],
                                 match_array1[:, 6],
                                 match_array1[:, 7],
                                 rag,
                                 decg)
                # furthur refine the search to minimize time finding epoch at
                # the impact paramater
                match_array2 = match_array1[(xd < 0.001) & (yd < 0.001) & (zd < 0.001)]

                if len(match_array2) > 0:
                    match_array2 = match_array2[match_array2[:, 3] < 1e19]
                    # now fully calculate angular seperations of possible matches
                    dra, dde = ang_sep(match_array2[:, 1],
                                       match_array2[:, 2],
                                       match_array2[:, 3],
                                       rag, decg, pmrag, pmdecg)
                    if len(dra) > 0:
                        best_line = []
                        for j in range(len(dra)):
                            # possible macthes are only those within 15 arcseconds
                            if dra[j] ** 2 + dde[j] ** 2 < 15 ** 2:
                                match_line = [idg, rag, decg, plxg, plxerrg,
                                              pmrag, pmdecg, Gg,
                                              match_ids[int(match_array2[j][0])][0]] + [match_array2[j][k] for k in range(1,4)]
                                match_line.append(dra[j])
                                match_line.append(dde[j])
                                # find the angular seperation and epoch at impact parameter
                                # along the proper motion vector
                                dra_n, dde_n, epoch_n = min_year(match_array2[j][1], match_array2[j][2],
                                                                 rag, decg, pmrag, pmdecg)
                                match_line[12] = dra_n
                                match_line[13] = dde_n
                                match_line.append(epoch_n)
                                for k in range(8, len(magcols) + 8):
                                    match_line.append(match_array2[j][k])
                                match_line.append(match_array2[j][-1])
                                # always save possible matches
                                fall.write(' '.join(str(m) for m in match_line) + '\n')
                                # if a possible match is within 5 arcseconds it is the
                                # current best match to the Gaia source
                                if dra_n ** 2 + dde_n ** 2 < 5 ** 2:
                                    if len(best_line)>0:
                                        if dra_n ** 2 + dde_n ** 2 < best_line[12] ** 2 + best_line[13] ** 2:
                                            best_line = match_line
                                    else:
                                        best_line = match_line
                        # save the data depending on if best match was found or not
                        if len(best_line) > 0:
                            matches += 1
                            best_epochs.append(best_line[14])
                            fbest.write(' '.join(str(m) for m in best_line) + '\n')
                        else:
                            best_line = [idg, rag, decg, plxg, plxerrg, pmrag, pmdecg, Gg] + [9999.] * (7 + len(magcols) + 1)
                            fbest.write(' '.join(str(m) for m in best_line) + '\n')
                    else:
                        best_line = [idg, rag, decg, plxg, plxerrg, pmrag, pmdecg, Gg] + [9999.] * (7 + len(magcols) + 1)
                        fbest.write(' '.join(str(m) for m in best_line) + '\n')
                else:
                    best_line = [idg, rag, decg, plxg, plxerrg, pmrag, pmdecg, Gg] + [9999.] * (7 + len(magcols) + 1)
                    fbest.write(' '.join(str(m) for m in best_line) + '\n')
                if i % 10000 == 0:
                    print(i, rag, matches, '-', "{0:.3f}".format((time.time() - start) / 60),'minutes')
        print('')
        print((time.time() - start) / 60, 'minutes')
        return np.array(best_epochs)


def cross_match_dis(match_file, chunksize, total, idcol,
                    usecols, match_save_file, match_save_file_all,
                    magcols, main_match_file):
    """
    This function finds the initial best matches in an external catalog to
    a catalog of Gaia sources, where the query was displaced by +/- 2 arcminutes.
    Initial best match is just the source with the
    smallest impact paramater to the Gaia source. This assumes the Gaia catalog
    has columns matching the GAIADR2_star.header file.

    Parameters
    ----------

    match_file: str
       external catalog that has already been pre-queried
       around all of the Gaia sources out to at least 15 as.
       This file MUST be sorted by RA.

    chunksize: int
        Size of chunks to load match_file

    total: int
        number of rows (not including header) in match_file

    idcol: list
        list that contains column number where the catalog id
        is in match_file

    usecols: list
        column numbers for [ra, dec, epoch] in match_file.
        It is assumed that the epoch is in mjd.

    match_save_file: str
        file name where to store initial best matches

    match_save_file_all: str
        file where to store all matches within 15"

    magcols: list
        column numbers of all magntiudes and magntidue errors
        in match_file (alternating)

    main_match_file: str
        file name where the Gaia catalog is stored. Gaia catalog
        must have columns matching GAIADR2_star.header file.
    """
    print('STARTING')
    start = time.time()
    path = ''
    path1 = ''
    match_ids_chunks = pd.read_table(path + match_file,
                                     usecols=idcol,
                                     dtype=int,
                                     chunksize=chunksize,
                                     delim_whitespace=True)
    match_ids = match_ids_chunks.get_chunk(chunksize).to_numpy(dtype=int)

    match_array_chunks = pd.read_table(path + match_file,
                                       usecols=usecols,
                                       chunksize=chunksize,
                                       delim_whitespace=True)
    match_array = match_array_chunks.get_chunk(chunksize).to_numpy(dtype=float)
    match_array = np.column_stack((np.array(range(len(match_array))), match_array))
    match_array = np.column_stack((match_array,
                                   np.sin(match_array[:, 1] / 57.295),
                                   np.cos(match_array[:, 1] / 57.295),
                                   np.sin(match_array[:, 2] / 57.295),
                                   np.cos(match_array[:, 2] / 57.295)))

    match_mags_chunks = pd.read_table(path + match_file,
                                      usecols=magcols,
                                      chunksize=chunksize,
                                      delim_whitespace=True)
    match_mags = match_mags_chunks.get_chunk(chunksize).to_numpy(dtype=float)
    match_array = np.column_stack((match_array, match_mags, np.arange(0, len(match_array), 1)))
    num_to_skip = 0
    best_epochs = []
    with open(path1 + main_match_file, 'r') as f, open(path + match_save_file_all, 'w') as fall, open(path + match_save_file,'w') as fbest:
        i = 0
        skip = 0
        matches = 0
        tot = len(match_array)
        if tot < total:
            do_3 = True
        else:
            do_3 = False
        for x in f:
            i += 1
            # can change this line if want to start match
            # where you left off in another run
            if i > 0:
                line = x.split()
                rag = float(line[0])
                decg = float(line[2])
                if decg >= 0:
                    decg -= (2. / 60.)
                else:
                    decg += (2. / 60.)
                pmrag = float(line[7])
                pmdecg = float(line[9])
                idg = int(line[4])
                plxg = float(line[5])
                plxerrg = float(line[6])
                Gg = float(line[11])
                if do_3:
                    skip, sk_st, sk_end = lines_to_skip3(skip, match_array, rag, tot)
                else:
                    skip, sk_st, sk_end = lines_to_skip2(skip, match_array, rag, tot)
                while sk_st < 0:
                    print('NEW CHUNK')
                    num_to_skip += chunksize

                    match_ids = np.concatenate((match_ids[int(chunksize - 100000):],
                                                match_ids_chunks.get_chunk(chunksize).to_numpy(dtype=int)))

                    match_array_stack = match_array[int(chunksize - 100000):,:].copy()
                    match_array_stack[:, 0] -= match_array_stack[0][0]
                    match_array = match_array_chunks.get_chunk(chunksize).to_numpy(dtype=float)
                    match_array = np.column_stack((np.array(range(int(match_array_stack[-1][0] + 1), len(match_array) + len(match_array_stack))), match_array))
                    match_array = np.column_stack((match_array,
                                                   np.sin(match_array[:, 1] / 57.295),
                                                   np.cos(match_array[:, 1] / 57.295),
                                                   np.sin(match_array[:, 2] / 57.295),
                                                   np.cos(match_array[:, 2] / 57.295)))

                    match_mags = match_mags_chunks.get_chunk(chunksize).to_numpy(dtype=float)
                    match_array = np.column_stack((match_array,
                                                   match_mags,
                                                   np.arange(num_to_skip, num_to_skip + len(match_array), 1)))
                    match_array = np.concatenate((match_array_stack, match_array))                    
                    tot = len(match_array)
                    skip = 0
                    skip, sk_st, sk_end = lines_to_skip3(skip, match_array, rag,tot)

                    if total - num_to_skip <= chunksize:
                        do_3 = False
                match_array1 = match_array[sk_st:sk_end, :]
                xd, yd, zd = xyz(match_array1[:, 4],
                                 match_array1[:, 5],
                                 match_array1[:, 6],
                                 match_array1[:, 7],
                                 rag,
                                 decg)
                match_array2 = match_array1[(xd < 0.001) & (yd < 0.001) & (zd < 0.001)]
                if len(match_array2) > 0:
                    match_array2 = match_array2[match_array2[:, 3] < 1e19]
                    dra, dde = ang_sep(match_array2[:, 1],
                                       match_array2[:, 2],
                                       match_array2[:, 3],
                                       rag,
                                       decg,
                                       pmrag,
                                       pmdecg)
                    if len(dra) > 0:
                        best_line = []
                        for j in range(len(dra)):
                            if dra[j] ** 2 + dde[j] ** 2 < 15 ** 2:
                                match_line = [idg, rag, decg, plxg, plxerrg,
                                              pmrag, pmdecg, Gg,
                                              match_ids[int(match_array2[j][0])][0]] + [match_array2[j][k] for k in range(1, 4)]
                                match_line.append(dra[j])
                                match_line.append(dde[j])
                                dra_n, dde_n, epoch_n = min_year(match_array2[j][1],
                                                                 match_array2[j][2],
                                                                 rag,
                                                                 decg,
                                                                 pmrag,
                                                                 pmdecg)
                                match_line[12] = dra_n
                                match_line[13] = dde_n

                                match_line.append(epoch_n)
                                for k in range(8, len(magcols) + 8):
                                    match_line.append(match_array2[j][k])
                                match_line.append(match_array2[j][-1])
                                fall.write(' '.join(str(m) for m in match_line) + '\n')
                                if dra_n ** 2 + dde_n ** 2 < 5 ** 2:
                                    if len(best_line) > 0:
                                        if dra_n ** 2 + dde_n ** 2 < best_line[12] ** 2 + best_line[13] ** 2:
                                            best_line = match_line
                                    else:
                                        best_line = match_line
                        if len(best_line) > 0:
                            matches += 1
                            best_epochs.append(best_line[14])
                            fbest.write(' '.join(str(m) for m in best_line) + '\n')
                        else:
                            best_line = [idg, rag, decg, plxg, plxerrg, pmrag, pmdecg, Gg] + [9999.] * (7 + len(magcols) + 1)
                            fbest.write(' '.join(str(m) for m in best_line) + '\n')
                    else:
                        best_line = [idg, rag, decg, plxg, plxerrg, pmrag, pmdecg, Gg] + [9999.] * (7 + len(magcols) + 1)
                        fbest.write(' '.join(str(m) for m in best_line) + '\n')
                else:
                    best_line = [idg, rag, decg, plxg, plxerrg, pmrag, pmdecg, Gg] + [9999.] * (7 + len(magcols) + 1)
                    fbest.write(' '.join(str(m) for m in best_line) + '\n')
                if i % 10000 == 0:
                    print(i, rag, matches, '-', "{0:.3f}".format((time.time() - start) / 60), 'minutes')
    print('')
    print((time.time() - start) / 60, 'minutes')


def rank_match_check_dups(path, file, file_save,
                          mean_epoch, flag=None, mag_cols=None):
    """
    This function calculates the angular seperations at some mean epoch
    between Gaia sources and all sources in field of an external catalog.
    It then ranks the sources in the field around a Gaia source in
    increasing order of angular seperation

    Parameters
    ----------

    path: str
        path to the working directory

    file: str
        file where all matches stored

    file_save: str
        file where ranks will be stored

    mean_epoch: float
        mean epoch to calculate angular seperations at.
        This is in units of decimal years.

    flag: ???
        dont use this I think

    mag_cols: ???
        dont use this I think
    """
    with open(path + file, 'r') as f, open(path + file_save, 'w') as frank:
        temp = []
        check_dups = []
        ranks = []
        adj_dra = []
        adj_dde = []
        i = 0
        skip = 0
        for x in tqdm_notebook(f):
            line = x.split()
            # always add field stars on first line
            if i == 0:
                temp.append(line)
                if flag is not None:
                    check_dups.append(int(line[flag[0]]))
                i += 1
            # add field star if Gaia ID matches previous line
            elif line[0] == temp[i-1][0]:
                temp.append(line)
                if flag is not None:
                    check_dups.append(int(line[flag[0]]))
                i += 1
            # if Gaia ID has changed, caluclated ang seps
            # and find the ranks
            else:
                ang_seps = []
                adj_dra = []
                adj_dde = []
                for j in range(len(temp)):
                    dra, dde = ang_sep2(float(temp[j][9]),
                                        float(temp[j][10]),
                                        mean_epoch,
                                        float(temp[j][1]),
                                        float(temp[j][2]),
                                        float(temp[j][5]),
                                        float(temp[j][6]))
                    adj_dra.append(dra)
                    adj_dde.append(dde)
                    ang_seps.append((dra ** 2 + dde ** 2) ** 0.5)
                rank0 = rankdata(ang_seps, method='ordinal')
                ndups = 0
                # does this need to be here?
                for j in range(len(rank0)):
                    if flag is not None:
                        if check_dups[j] == flag[1]:
                            rank0[j] = 0
                            ndups += 1
                if ndups > 0:
                    rank = rankdata(rank0, method='dense') - 1
                else:
                    rank = rank0
                for j in range(len(rank)):
                    new_line = [rank[j], adj_dra[j], adj_dde[j]]
                    frank.write(' '.join(str(m) for m in new_line) + '\n')
                i = 1
                temp = []
                temp.append(line)
                check_dups = []
                if flag is not None:
                    check_dups.append(int(line[flag[0]]))
        # once at the end of the file, do calculations
        # for the last Gaia source
        ang_seps = []
        adj_dra = []
        adj_dde = []
        for j in range(len(temp)):
            dra, dde = ang_sep2(float(temp[j][9]),
                                float(temp[j][10]),
                                mean_epoch,
                                float(temp[j][1]),
                                float(temp[j][2]),
                                float(temp[j][5]),
                                float(temp[j][6]))
            adj_dra.append(dra)
            adj_dde.append(dde)
            ang_seps.append((dra ** 2 + dde ** 2) ** 0.5)
        rank0 = rankdata(ang_seps, method='ordinal')
        ndups = 0
        for j in range(len(rank0)):
            if flag is not None:
                if check_dups[j] == flag[1]:
                    rank0[j] = 0
                    ndups += 1
        if ndups > 0:
            rank = rankdata(rank0, method='dense') - 1
        else:
            rank = rank0
        for j in range(len(rank)):
            new_line = [rank[j], adj_dra[j], adj_dde[j]]
            frank.write(' '.join(str(m) for m in new_line) + '\n')


def create_mag_ang_dists(path, file, file_rank, file_dis,
                         file_dis_rank, name, mag_cols, xbins, ybins,
                         gaia_cuts, b_cuts):
    """
    This creates all of the frequency dsitribtuions needed
    to calculate the Bayesian probabilities. This requires
    a directory called "Distribution_Files" to be in the working
    directory.

    Parameters
    ----------

    path: str
        path to working directory

    file: str
        file where all matches stored

    file_rank: str
        file where ranks for all matches stored

    file_dis: str
        file where all matches stored for displaced
        sample

    file_rank_dis: str
        file where ranks for all matches stored for
        displaced sample

    name: str
        name of external catalog that is prefixed
        to the all matches file

    mag_cols: list
        list of column numbers for magnitudes in file

    xbins: array
        bins for frequency distirbution for angular seperation axis

    ybins: array
        bins for frequency distirbution for mag difference axis

    gaia_cuts: np.array
        G cuts used to create distirbutions. It is assumed
        these are evenly distributed. Also need to add max
        boundary in last index

    b_cuts: np.array
        b cuts used to create distirbutions. It is assumed
        these are not evenly distributed. Also need to add max
        boundary in last index (should be b=90)
    """
    if not os.path.isdir(path + 'Distribution_Files/'):
        os.mkdir(path + 'Distribution_Files/')

    path_save = path + 'Distribution_Files/' + name

    skip = 0
    chunk = int(2e6)
    contin = True

    # load data in chunks
    data_true = np.genfromtxt(path + file,
                              skip_header=skip,
                              max_rows=chunk,
                              usecols=mag_cols)
    data = np.genfromtxt(path + file_rank,
                         skip_header=skip,
                         max_rows=chunk)
    data_gaia = np.genfromtxt(path + file,
                              skip_header=skip,
                              max_rows=chunk,
                              usecols=(7, 1, 2))

    c = SkyCoord(ra=data_gaia[:, 1] * u.degree,
                 dec=data_gaia[:,2] * u.degree,
                 frame='icrs')
    b_gaia = np.array(c.galactic.b.deg)

    # these are the bins in G and latitude used
    # to create the various frequence dsitributions
    evals = {}
    for i in range(len(gaia_cuts)):
        if i == 0:
            evals[str(i)] = eval('(data_gaia[:, 0] < %f)' % gaia_cuts[i])
        elif i == len(gaia_cuts) - 1:
            evals[str(i)] = eval('(data_gaia[:, 0] >= %f)' % gaia_cuts[i - 1])
        else:
            evals[str(i)] = eval('(data_gaia[:, 0] >= %f) & (data_gaia[:, 0] < %f)' % (gaia_cuts[i - 1], gaia_cuts[i]))
    evals_b = {}
    for i in range(len(b_cuts)):
        if i == 0:
            evals_b[str(i)] = eval('(abs(b_gaia) < %f)' % b_cuts[i])
        elif i == len(b_cuts) - 1:
            evals_b[str(i)] = eval('(abs(b_gaia) >= %f)' % b_cuts[i - 1])
        else:
            evals_b[str(i)] = eval('(abs(b_gaia) >= %f) & (abs(b_gaia) < %f)' % (b_cuts[i - 1], b_cuts[i]))

    Hs={}

    # for each combination of magntiude, G cut and b cut,
    # calculate the histogram
    for i in range(len(mag_cols)):
        for j in range(len(evals)):
            for k in range(len(evals_b)):
                x = (data[:,1][evals[str(j)] & evals_b[str(k)] & (data[:,0]>0) & (data_true[:,i]>-100) & (data_true[:,i]<100)]**2+data[:,2][evals[str(j)] & evals_b[str(k)] & (data[:,0]>0) & (data_true[:,i]>-100) & (data_true[:,i]<100)]**2)**0.5
                y = (data_true[:,i][evals[str(j)] & evals_b[str(k)] & (data[:,0]>0) & (data_true[:,i]>-100) & (data_true[:,i]<100)]-data_gaia[:,0][evals[str(j)] & evals_b[str(k)] & (data[:,0]>0) & (data_true[:,i]>-100) & (data_true[:,i]<100)])
                H, xedges, yedges = np.histogram2d(x, y, bins=[xbins, ybins])
                Hs['%d_%d_%d' % (i, j, k)] = H.T
    skip += chunk

    # repeat above for all chunks
    while contin:
        try:
            data_true = np.genfromtxt(path + file,
                                      skip_header=skip,
                                      max_rows=chunk,
                                      usecols=mag_cols)
            data = np.genfromtxt(path + file_rank,
                                 skip_header=skip,
                                 max_rows=chunk)
            data_gaia = np.genfromtxt(path + file,
                                      skip_header=skip,
                                      max_rows=chunk,
                                      usecols=(7, 1, 2))

            c = SkyCoord(ra=data_gaia[:, 1] * u.degree,
                         dec=data_gaia[:, 2] * u.degree,
                         frame='icrs')
            b_gaia = np.array(c.galactic.b.deg)

            evals = {}
            for i in range(len(gaia_cuts)):
                if i == 0:
                    evals[str(i)] = eval('(data_gaia[:, 0] < %f)' % gaia_cuts[i])
                elif i == len(gaia_cuts) - 1:
                    evals[str(i)] = eval('(data_gaia[:, 0] >= %f)' % gaia_cuts[i - 1])
                else:
                    evals[str(i)] = eval('(data_gaia[:, 0] >= %f) & (data_gaia[:, 0] < %f)' % (gaia_cuts[i - 1], gaia_cuts[i]))
            evals_b = {}
            for i in range(len(b_cuts)):
                if i == 0:
                    evals_b[str(i)] = eval('(abs(b_gaia) < %f)' % b_cuts[i])
                elif i == len(b_cuts) - 1:
                    evals_b[str(i)] = eval('(abs(b_gaia) >= %f)' % b_cuts[i - 1])
                else:
                    evals_b[str(i)] = eval('(abs(b_gaia) >= %f) & (abs(b_gaia) < %f)' % (b_cuts[i - 1], b_cuts[i]))

            for i in range(len(mag_cols)):
                for j in range(len(evals)):
                    for k in range(len(evals_b)):
                        x = (data[:,1][evals[str(j)] & evals_b[str(k)] & (data[:,0]>0) & (data_true[:,i]>-100) & (data_true[:,i]<100)]**2+data[:,2][evals[str(j)] & evals_b[str(k)] & (data[:,0]>0) & (data_true[:,i]>-100) & (data_true[:,i]<100)]**2)**0.5
                        y = (data_true[:,i][evals[str(j)] & evals_b[str(k)] & (data[:,0]>0) & (data_true[:,i]>-100) & (data_true[:,i]<100)]-data_gaia[:,0][evals[str(j)] & evals_b[str(k)] & (data[:,0]>0) & (data_true[:,i]>-100) & (data_true[:,i]<100)])
                        H, xedges, yedges = np.histogram2d(x, y, bins=[xbins, ybins])
                        Hs['%d_%d_%d' % (i, j, k)] += H.T

            skip += chunk
        except (StopIteration, IndexError) as e:
            contin = False

    skip = 0
    chunk = int(2e6)
    contin = True

    # do the same for the displaced sample
    data_true = np.genfromtxt(path + file_dis,
                              skip_header=skip,
                              max_rows=chunk,
                              usecols=mag_cols)
    data = np.genfromtxt(path + file_dis_rank,
                         skip_header=skip,
                         max_rows=chunk)
    data_gaia = np.genfromtxt(path + file_dis,
                              skip_header=skip,
                              max_rows=chunk,
                              usecols=(7, 1, 2))

    c = SkyCoord(ra=data_gaia[:, 1] * u.degree,
                 dec=data_gaia[:, 2] * u.degree,
                 frame='icrs')
    b_gaia = np.array(c.galactic.b.deg)

    evals = {}
    for i in range(len(gaia_cuts)):
        if i == 0:
            evals[str(i)] = eval('(data_gaia[:, 0] < %f)' % gaia_cuts[i])
        elif i == len(gaia_cuts) - 1:
            evals[str(i)] = eval('(data_gaia[:, 0] >= %f)' % gaia_cuts[i - 1])
        else:
            evals[str(i)] = eval('(data_gaia[:, 0] >= %f) & (data_gaia[:, 0] < %f)' % (gaia_cuts[i - 1], gaia_cuts[i]))
    evals_b = {}
    for i in range(len(b_cuts)):
        if i == 0:
            evals_b[str(i)] = eval('(abs(b_gaia) < %f)' % b_cuts[i])
        elif i == len(b_cuts) - 1:
            evals_b[str(i)] = eval('(abs(b_gaia) >= %f)' % b_cuts[i - 1])
        else:
            evals_b[str(i)] = eval('(abs(b_gaia) >= %f) & (abs(b_gaia) < %f)' % (b_cuts[i - 1], b_cuts[i]))

    Hds = {}

    for i in range(len(mag_cols)):
        for j in range(len(evals)):
            for k in range(len(evals_b)):
                x = (data[:,1][evals[str(j)] & evals_b[str(k)] & (data[:,0]>0) & (data_true[:,i]>-100) & (data_true[:,i]<100)]**2+data[:,2][evals[str(j)] & evals_b[str(k)] & (data[:,0]>0) & (data_true[:,i]>-100) & (data_true[:,i]<100)]**2)**0.5
                y = (data_true[:,i][evals[str(j)] & evals_b[str(k)] & (data[:,0]>0) & (data_true[:,i]>-100) & (data_true[:,i]<100)]-data_gaia[:,0][evals[str(j)] & evals_b[str(k)] & (data[:,0]>0) & (data_true[:,i]>-100) & (data_true[:,i]<100)])
                H, xedges, yedges = np.histogram2d(x, y, bins=[xbins, ybins])
                Hds['%d_%d_%d' % (i, j, k)] = H.T
    skip += chunk

    while contin:
        try:
            data_true = np.genfromtxt(path + file_dis,
                                      skip_header=skip,
                                      max_rows=chunk,
                                      usecols=mag_cols)
            data = np.genfromtxt(path + file_dis_rank,
                                 skip_header=skip,
                                 max_rows=chunk)
            data_gaia = np.genfromtxt(path + file_dis,
                                      skip_header=skip,
                                      max_rows=chunk,
                                      usecols=(7, 1, 2))

            c = SkyCoord(ra = data_gaia[:, 1] * u.degree,
                         dec=data_gaia[:, 2] * u.degree,
                         frame='icrs')
            b_gaia = np.array(c.galactic.b.deg)

            evals = {}
            for i in range(len(gaia_cuts)):
                if i == 0:
                    evals[str(i)] = eval('(data_gaia[:, 0] < %f)' % gaia_cuts[i])
                elif i == len(gaia_cuts) - 1:
                    evals[str(i)] = eval('(data_gaia[:, 0] >= %f)' % gaia_cuts[i - 1])
                else:
                    evals[str(i)] = eval('(data_gaia[:, 0] >= %f) & (data_gaia[:, 0] < %f)' % (gaia_cuts[i - 1], gaia_cuts[i]))
            evals_b = {}
            for i in range(len(b_cuts)):
                if i == 0:
                    evals_b[str(i)] = eval('(abs(b_gaia) < %f)' % b_cuts[i])
                elif i == len(b_cuts) - 1:
                    evals_b[str(i)] = eval('(abs(b_gaia) >= %f)' % b_cuts[i - 1])
                else:
                    evals_b[str(i)] = eval('(abs(b_gaia) >= %f) & (abs(b_gaia) < %f)' % (b_cuts[i - 1], b_cuts[i]))

            for i in range(len(mag_cols)):
                for j in range(len(evals)):
                    for k in range(len(evals_b)):
                        x = (data[:,1][evals[str(j)] & evals_b[str(k)] & (data[:,0]>0) & (data_true[:,i]>-100) & (data_true[:,i]<100)]**2+data[:,2][evals[str(j)] & evals_b[str(k)] & (data[:,0]>0) & (data_true[:,i]>-100) & (data_true[:,i]<100)]**2)**0.5
                        y = (data_true[:,i][evals[str(j)] & evals_b[str(k)] & (data[:,0]>0) & (data_true[:,i]>-100) & (data_true[:,i]<100)]-data_gaia[:,0][evals[str(j)] & evals_b[str(k)] & (data[:,0]>0) & (data_true[:,i]>-100) & (data_true[:,i]<100)])
                        H, xedges, yedges = np.histogram2d(x, y, bins=[xbins, ybins])
                        Hds['%d_%d_%d' % (i,j,k)]+=H.T

            skip += chunk
        except (StopIteration, IndexError) as e:
            contin = False

    def find_min_and_max(H, xedges, yedges):
        """
        This finds the physical bounds of the
        frequency distirubtion for plotting purposes
        """
        xsums = np.sum(H, axis=0)
        ysums = np.sum(H, axis=1)

        x_ind_max = int(len(xsums) - 1)
        for i in range(len(xsums)):
            if np.sum(xsums[i:]) == 0.:
                x_ind_max = i
                break

        y_ind_max = int(len(ysums) - 1)
        for i in range(1, len(ysums)):
            if np.sum(ysums[i:]) == 0.:
                y_ind_max = i
                break

        y_ind_min = 0
        for i in range(1, len(ysums) - 1):
            if ysums[i - 1] == 0. and ysums[i] == 0. and ysums[i + 1] > 0.:
                y_ind_min = i
                break

        if yedges[int(y_ind_min)] >= yedges[int(y_ind_max)]:
            return 0., 20, yedges[0], yedges[int(len(xsums) - 1)]
        else:
            return 0., 20, yedges[int(y_ind_min)], yedges[int(y_ind_max)]

    # save all of the frequency distributions
    X, Y = np.meshgrid(xbins, ybins)
    for i in range(len(mag_cols)):
        for j in range(len(evals)):
            for k in range(len(evals_b)):

                x_true_min, x_true_max, min_true, max_true = find_min_and_max(Hs['%d_%d_%d' % (i, j, k)], xedges, yedges)

                ntrue = np.sum(Hs['%d_%d_%d' % (i, j, k)])


                np.savez(path_save + '_extrap_match_color_dist_true_data_mag_' + str(i + 1) + '_gaia_cut_' + str(j + 1) + '_lat_cut_' + str(k + 1) + '.npz',
                         Htrue=Hs['%d_%d_%d' % (i, j, k)], xedges=xedges, yedges=yedges, ntrue=ntrue, min_true=min_true, max_true=max_true)

                x_dis_min, x_dis_max, min_dis, max_dis = find_min_and_max(Hds['%d_%d_%d' % (i, j, k)], xedges, yedges)

                ndis = np.sum(Hds['%d_%d_%d' % (i, j, k)])


                np.savez(path_save + '_extrap_match_color_dist_dis_data_mag_' + str(i + 1) + '_gaia_cut_' + str(j + 1) + '_lat_cut_' + str(k + 1) + '.npz',
                         Hdis=Hds['%d_%d_%d' % (i, j, k)], xedges=xedges, yedges=yedges, ndis=ndis, min_dis=min_dis, max_dis=max_dis)


def line(x, m, b):
    """
    Its a line
    """
    return m * x + b


def scale(x, A):
    """
    Its a constant scaling
    """
    return A * x


def gauss(x, A, mu, s):
    """
    its a normal distribution
    """
    return A * np.exp(-(x - mu) ** 2 / (2 * s ** 2))


def n_gauss(pars, inps, data, ncomps):
    """
    This creates a model of a the displaced sample
    distribution, which is a linear distribution multiplied
    by a sum of normal distributions of ncomps.
    Specifically, this model is meant to
    be fit by the package lmfit.

    Parameters
    ----------

    pars: dict
        dictonary of paramaters that will be fit by lmfit

    inps: 2D array
        2D array with the bins in the x and y directions

    data: 2D array
        the observed distribution

    ncomps: int
        the number of gaussian components
    """
    x = inps[:, 0]
    y = inps[:, 1]
    Z = np.zeros(len(x))
    for i in range(ncomps):
        A = pars['A' + str(i)]
        mu = pars['mu' + str(i)]
        s = pars['s' + str(i)]
        Z += gauss(y, A, mu, s)
    Z *= line(x, pars['m'], pars['b'])
    Z *= pars['C']
    return (data - Z) / data ** 0.5


def n_gauss1d(pars, x, data, ncomps):
    """
    This creates a model of the displaced sample
    distribution in the y direction, which is a sum
    of normal distributions of ncomps.
    Specifically, this model is meant to
    be fit by the package lmfit.

    Parameters
    ----------

    pars: dict
        dictonary of paramaters that will be fit by lmfit

    inps: array
        array with the bins in the y direction

    data: 2D array
        the observed distribution

    ncomps: int
        the number of gaussian components
    """
    Z = np.zeros(len(x))
    for i in range(ncomps):
        A = pars['A' + str(i)]
        mu = pars['mu' + str(i)]
        s = pars['s' + str(i)]
        Z += gauss(x, A, mu, s)
    return (data - Z) / data ** 0.5


def n_gauss1d_BIC(pars, x, data, ncomps):
    """
    This calculates the Bayesian information criteria for
    a model of a the displaced sample
    distributionin the y direction, which is a sum
    of normal distributions of ncomps.
    Specifically, this model is meant to
    be fit by the package lmfit.

    Parameters
    ----------

    pars: dict
        dictonary of paramaters that will be fit by lmfit

    inps: array
        array with the bins in the y direction

    data: 2D array
        the observed distribution

    ncomps: int
        the number of gaussian components
    """
    Z = np.zeros(len(x))
    for i in range(ncomps):
        A = pars['A' + str(i)]
        mu = pars['mu' + str(i)]
        s = pars['s' + str(i)]
        Z += gauss(x, A, mu, s)
    return len(x) * np.log(np.sum((data - Z) **2 ) / len(x)) + ncomps * 3 * np.log(len(x))


def n_gauss1d_eval(x, pars, ncomps):
    """
    This provides the model of a the displaced sample
    distributionin the y direction, which is a sum
    of normal distributions of ncomps, for some params
    found by lmfit.

    Parameters
    ----------

    pars: dict
        dictonary of paramaters that will be fit by lmfit

    inps: array
        array with the bins in the y direction

    ncomps: int
        the number of gaussian components
    """
    Z = np.zeros(x.shape)
    for i in range(ncomps):
        A = pars['A' + str(i)]
        mu = pars['mu' + str(i)]
        s = pars['s' + str(i)]
        Z += gauss(x, A, mu, s)
    return Z


def n_gauss_eval(inps, pars, ncomps):
    """
    This provides the model of a the displaced sample
    distributionin the 2D, which is a linear distribution multiplied
    by a sume of normal distributions of ncomps, for some params
    found by lmfit.

    Parameters
    ----------

    pars: dict
        dictonary of paramaters that will be fit by lmfit

    inps: 2D array
        2D array with the bins in the x and y directions

    ncomps: int
        the number of gaussian components
    """
    x = inps[:, 0]
    y = inps[:, 1]
    Z = np.zeros(len(x))
    for i in range(ncomps):
        A = pars['A' + str(i)]
        mu = pars['mu' + str(i)]
        s = pars['s' + str(i)]
        Z += gauss(y, A, mu, s)
    Z *= line(x, pars['m'], pars['b'])
    Z *= pars['C']
    return Z


def rebin(a, shape):
    sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
    return a.reshape(sh).sum(-1).sum(1)


def back_mod_2d_gauss(name, mag_strs, g_strs, b_strs, plot_model=True):
    """
    This function models the frequency distributions for the displaced
    sample.

    Parameters
    ----------

    name: str
        name of the external catalog that is prefixed
        to the all matches file

    mag_strs: list
        list strings for the names of the magntiudes
        in the external catalog (for plotting purposes)

    g_strs: list
        list of strings for names of cuts in G
        (needed for plotting and knowing number of cuts)

    b_strs: list
        list of strings for names of cuts in b latitude
        (needed for plotting and knowing number of cuts)

    plot_model: booleen
        option to plot the resulting models and plots
        that evaluate the staus of the models
    """
    path = 'Distribution_Files/' + name

    # iterate through all of the distributions
    for i in range(len(mag_strs)):
        for j in range(len(g_strs)):
            for k in range(len(b_strs)):
                print(name, mag_strs[i], g_strs[j], b_strs[k])
                data_dis = np.load(path + '_extrap_match_color_dist_dis_data_mag_' + str(i + 1) + '_gaia_cut_' + str(j + 1) + '_lat_cut_' + str(k + 1) + '.npz')
                Hdis = data_dis['Hdis']
                xedges = data_dis['xedges']
                yedges = data_dis['yedges']
                ndis = data_dis['ndis']
                min_dis = data_dis['min_dis']
                max_dis = data_dis['max_dis']

                edges_ang = xedges
                edges_mag = yedges

                X, Y = np.meshgrid(xedges, yedges)
                xdiff = xedges[1] - xedges[0]
                ydiff = yedges[1] - yedges[0]
                Xm, Ym = np.meshgrid(np.arange(xedges[0] + xdiff / 2, xedges[-1] + xdiff / 2, xdiff), np.arange(yedges[0] + ydiff / 2, yedges[-1] + ydiff / 2, ydiff))

                Hang = np.sum(Hdis, axis=0)
                Hmag = np.sum(Hdis, axis=1)

                # get the bin midpoints in x and y direction
                ang_mid = np.array([(xedges[j] + xedges[j + 1]) / 2 for j in range(len(xedges) - 1)])
                mag_mid = np.array([(yedges[j] + yedges[j + 1]) / 2 for j in range(len(yedges) - 1)])

                rebin_nx = (xedges[1] - xedges[0])

                # fit a linear distribution in the x direction
                # take of some of the edge of the distribtuion
                # as sometimes there is an edge effect that skews
                # the fit
                popt_ang, pcov_ang = curve_fit(line,
                                               ang_mid[:len(Hang) - int(12 / rebin_nx)],
                                               Hang[:len(Hang) - int(12 / rebin_nx)],
                                               p0=(100000,0.01))

                # in the y direction, fit a sum of gaussians of varrying weights
                # the optimal number of components found by BIC
                results = {}
                chis = []
                max_ind = np.unravel_index(Hdis.argmax(), Hdis.shape)
                for ncomps in tnrange(2, 11):
                    try:
                        pars = Parameters()
                        for l in range(ncomps):
                            pars.add('A' + str(l), value=1000, min=0, vary=True)
                            pars.add('mu' + str(l), value=mag_mid[np.argmax(Hmag)], vary=True)
                            pars.add('s' + str(l), value=1, vary=True, min=0)

                        result = minimize(n_gauss1d, pars, args=(mag_mid[Hmag > 0], Hmag[Hmag > 0], ncomps))
                        results[str(ncomps)] = result
                        best_params_dict = result.params.valuesdict()
                        chis.append(n_gauss1d_BIC(best_params_dict, mag_mid[Hmag>0], Hmag[Hmag>0], ncomps))
                    except TypeError:
                        results[str(ncomps)] = np.nan
                        chis.append(500000)

                ncomps = np.argmin(chis) + 2
                result = results[str(ncomps)]

                best_params_dict = result.params.valuesdict()

                if plot_model:
                    # plot the results
                    plt.figure(figsize=(7, 7))
                    plt.scatter(range(2, 11), chis)
                    plt.grid()
                    plt.axvline(ncomps, linestyle='--', c='r')
                    plt.ylim((np.min(chis) - 20, np.max(np.array(chis)[np.array(chis) < 500000]) + 20))
                    plt.show()

                    result.params.pretty_print()

                    plt.figure(figsize=(7, 7))
                    plt.hist(edges_ang[:len(Hang)], bins=edges_ang, weights=Hang)
                    plt.plot(np.arange(0, 20.2, .2),
                             line(np.arange(0, 20.2, .2), *popt_ang),
                             '--',
                             c='r')
                    plt.grid()
                    plt.xlim((0, 20))
                    plt.show()

                    plt.figure(figsize=(7, 7))
                    plt.hist(edges_mag[:len(Hmag)], bins=edges_mag, weights=Hmag)
                    plt.plot(np.arange(-50, 50.2, .2),
                             n_gauss1d_eval(np.arange(-50, 50.2, .2), best_params_dict, ncomps),
                             '--',
                             c='r')
                    plt.xlim((min_dis, max_dis))
                    plt.grid()
                    plt.show()

                Hmod = line(Xm, *popt_ang) * n_gauss1d_eval(Ym, best_params_dict, ncomps)

                popt, pcov = curve_fit(scale,
                                       Hmod.ravel()[Hdis.ravel() > 0],
                                       Hdis.ravel()[Hdis.ravel() > 0])
                if plot_model:
                    print(popt)

                    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

                    ax1.pcolormesh(X, Y, np.log10(Hmod * popt[0]),
                                   cmap='viridis', shading='flat',
                                   vmin=0, vmax=np.log10(np.nanmax(Hdis)))
                    ax1.set_xlim((0, 15))
                    ax1.set_ylim((min_dis, max_dis))

                    ax2.pcolormesh(X, Y, np.log10(Hdis),
                                   cmap='viridis', shading='flat',
                                   vmin=0, vmax=np.log10(np.nanmax(Hdis)))
                    ax2.set_xlim((0, 15))
                    ax2.set_ylim((min_dis, max_dis))
                    plt.show()

                    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

                    ax1.pcolormesh(X, Y, (Hmod * popt[0]),
                                   cmap='viridis', shading='flat',
                                   vmin=0, vmax=(np.nanmax(Hdis)))
                    ax1.set_xlim((0, 20))
                    ax1.set_ylim((min_dis, max_dis))

                    ax2.pcolormesh(X, Y, (Hdis),
                                   cmap='viridis', shading='flat',
                                   vmin=0, vmax=(np.nanmax(Hdis)))
                    ax2.set_xlim((0, 20))
                    ax2.set_ylim((min_dis, max_dis))
                    plt.show()

                Hmod *= popt[0]

                # save the modeled distribution
                np.savez(path + '_extrap_match_color_dist_dis_model_data_mag_' + str(i + 1) + '_gaia_cut_' + str(j + 1) + '_lat_cut_' + str(k + 1) + '.npz',
                         Hmod=Hmod, xedges=xedges, yedges=yedges, nmod=np.sum(Hmod), min_mod=0, max_mod=0,
                         mod_params=best_params_dict, ncomps=ncomps)


def calc_ind(x, x1, x2, dx):
    """
    Find the index in the 2D histogram that corresponds to some
    x value

    Parameters
    ----------

    x: array
        x values of indexes you are trying to find

    x1: float
        the value of first index

    x2: float
        the value of last index

    dx: float
        width of the bins
    """
    ind1 = (x - x1) / (x2 - x1)
    ind1[ind1 < 0] = 0
    ind1[ind1 >= 1] = int(1 / dx - 1)
    ind1[(ind1 >= 0) & (ind1 < 1)] = np.floor(ind1[(ind1 >= 0) & (ind1 < 1)] * (x2 - x1) / dx)
    return ind1


def calc_bayes_prob(name, smooth_func, args, mag_cols, all_file, rank_file,
                    gaia_cuts, b_cuts, g_strs, b_strs):
    """
    Calculate the Bayesian probability of a star in an external
    catalog being a match to a Gaia source

    Parameters
    ----------

    name: str
        name of the external catalog that is prefixed
        to the all matches file

    smooth_func: scipy function
        function used to smooth prue frequency dsitributions

    args: dict
        arguments for smooth_func

    mag_cols: list
        column numbers for magntiudes in all_file

    all_file: str
        file name where all matches are stored without name

    rank_file: str
        file name for ranks of all matches without name

    gaia_cuts: np.array
        G cuts used to create distirbutions. It is assumed
        these are evenly distributed. Also need to add max
        boundary in last index

    b_cuts: np.array
        b cuts used to create distirbutions. It is assumed
        these are not evenly distributed. Also need to add max
        boundary in last index (should be b=90)

    g_strs: list
        list of strings for names of cuts in G
        (needed for plotting)

    b_strs: list
        list of strings for names of cuts in b latitude
        (needed for plotting)
    """
    print(name)
    path = name
    path_dist = 'Distribution_Files/' + name
    if not os.path.isdir('Bayes_Probs/'):
        os.mkdir('Bayes_Probs/')
    path_fig = 'Bayes_Probs/' + name

    bayes_dists = {}

    start = time.time()

    match_save_file = '_bayes_probs_per_mag_gaia_cut_b_cut.txt'

    # loop through all distribution to calculate the bayesian probability distributions
    for i in tqdm_notebook(range(len(mag_cols))):
        for j in range(len(gaia_cuts)):
            for k in range(len(b_cuts)):
                data_dis = np.load(path_dist + '_extrap_match_color_dist_dis_data_mag_' + str(i + 1) + '_gaia_cut_' + str(j + 1) + '_lat_cut_' + str(k + 1) + '.npz')
                Hdis = data_dis['Hdis']
                xedges_dis = data_dis['xedges']
                yedges_dis = data_dis['yedges']
                ndis = data_dis['ndis']
                min_dis = data_dis['min_dis']
                max_dis = data_dis['max_dis']

                data_mod = np.load(path_dist + '_extrap_match_color_dist_dis_model_data_mag_' + str(i + 1) + '_gaia_cut_' + str(j + 1) + '_lat_cut_' + str(k + 1) + '.npz')
                Hmod = data_mod['Hmod']
                xedges_mod = data_mod['xedges']
                yedges_mod = data_mod['yedges']
                nmod = data_mod['nmod']
                min_mod = data_mod['min_mod']
                max_mod = data_mod['max_mod']

                X, Y = np.meshgrid(xedges_dis, yedges_dis)

                # plot the modeled background comapred to the true background
                plt.figure(figsize=(7, 7))
                plt.pcolormesh(X, Y, Hmod/Hdis,
                               shading='flat',
                               cmap='viridis',
                               vmin=0, vmax=5)
                plt.colorbar()
                plt.xlim((0, 20))
                plt.ylim((min_dis, max_dis))
                plt.title('%s - Mag %d - %s - %s' % (name, i, g_strs[j], b_strs[k]))
                plt.savefig(path_fig + '_compare_dis_model_%d_%d_%d' % (i, j, k), dpi=100)
                plt.show()
                
                data_true = np.load(path_dist + '_extrap_match_color_dist_true_data_mag_' + str(i + 1) + '_gaia_cut_' + str(j + 1) + '_lat_cut_' + str(k + 1) + '.npz')
                Htrue = data_true['Htrue']
                xedges_true = data_true['xedges']
                yedges_true = data_true['yedges']
                ntrue = data_true['ntrue']
                min_true = data_true['min_true']
                max_true = data_true['max_true']

                def line_fit(x, a, b, c, d):
                    one = a * x + b
                    two = c * x + d
                    return np.minimum(one, two)

                scales = np.arange(0, 5, 0.05)
                sums_neg = np.zeros(len(scales))
                for l in range(len(scales)):
                    Hsub_smooth = smooth_func(Htrue, **args) - smooth_func(Hmod * scales[l], **args)
                    sums_neg[l] = np.sum(Hsub_smooth[Hsub_smooth < 0])

                # find the scaling factor between true and displaced sample
                popt, pcov = curve_fit(line_fit, scales, sums_neg, p0=(-1, 1, -10000, 1))

                scale = (popt[3] - popt[1]) / (popt[0] - popt[2])

                # calculate bayes distribution
                Htrue_smooth = smooth_func(Htrue, **args)
                Hdis_smooth = smooth_func(Hmod * scale, **args)
                Hsub_smooth = Htrue_smooth - Hdis_smooth
                Hsub_smooth[Hsub_smooth < 0.] = 0.

                P1 = np.sum(Hsub_smooth) / np.sum(Htrue_smooth)
                P0 = 1. - P1

                bayes_dists['%d_%d_%d' % (i, j, k)] = (Hsub_smooth / np.sum(Hsub_smooth)) * P1 / ((Hsub_smooth / np.sum(Hsub_smooth)) * P1 + (Hdis_smooth / np.sum(Hdis_smooth)) * P0)

                np.savez(path_dist + '_bayes_prob_dist_mag_' + str(i + 1) + '_gaia_cut_' + str(j + 1) + '_lat_cut_' + str(k + 1) + '.npz',
                         Hbayes=bayes_dists['%d_%d_%d' % (i, j, k)], scale=scale, P1=P1, P0=P0)

                Htrue_smooth = smooth_func(Htrue, **args)
                Hdis_smooth = Hmod * scale
                Hsub_smooth = Htrue_smooth - Hdis_smooth

                # plot the resulting distributions
                f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(30, 20))

                ax1.pcolormesh(X, Y, np.log10(Htrue_smooth),
                               shading='flat', cmap='viridis',
                               vmin=0, vmax=np.nanmax(np.log10(Htrue_smooth)))
                ax1.set_xlim((0, 20))
                ax1.set_ylim((min_true, max_true))
                ax1.set_title('All - Smoothed')

                ax2.pcolormesh(X, Y, np.log10(Hsub_smooth),
                               shading='flat', cmap='viridis',
                               vmin=0, vmax=np.nanmax(np.log10(Htrue_smooth)))
                ax2.set_xlim((0, 20))
                ax2.set_ylim((min_true, max_true))
                ax2.set_title('Inferred True - Smoothed')

                ax3.pcolormesh(X, Y, np.log10(Hmod),
                               shading='flat', cmap='viridis',
                               vmin=0, vmax=np.nanmax(np.log10(Htrue_smooth)))
                ax3.set_xlim((0, 20))
                ax3.set_ylim((min_true, max_true))
                ax3.set_title('Displaced - Modeled')

                ax4.pcolormesh(X, Y, np.log10(-1 * Hsub_smooth),
                               shading='flat', cmap='viridis',
                               vmin=0, vmax=np.nanmax(np.log10(Htrue_smooth)))
                ax4.set_xlim((0, 20))
                ax4.set_ylim((min_true, max_true))
                ax4.set_title('Inferred True Negative Portion - Smoothed')

                ax5.scatter(scales, sums_neg, marker='.', c='k', label='_nolegend_')
                ax5.plot(scales, line_fit(scales, *popt), c='r', label='_nolegend_')
                ax5.grid()
                ax5.axvline(ntrue / ndis, linestyle='--', c='r', label='Scale - All')
                ax5.axvline((popt[3] - popt[1]) / (popt[0] - popt[2]),
                            linestyle='--',c='k', label='Scale - Elbow')
                ax5.axvline(nmod / ndis, linestyle='--', c='g', label=r'N$_{mod}$/N$_{dis}$')
                ax5.legend()
                ax5.set_title('Negative Portion Fit')

                Xm, Ym = np.meshgrid(np.arange(xedges_true[0] + (xedges_true[1] - xedges_true[0]) / 2, xedges_true[-1] + (xedges_true[1] - xedges_true[0]) / 2, xedges_true[1] - xedges_true[0]), np.arange(yedges_true[0] + (yedges_true[1] - yedges_true[0]) / 2, yedges_true[-1] + (yedges_true[1] - yedges_true[0]) / 2, yedges_true[1] - yedges_true[0]))

                ax6.pcolormesh(X, Y, bayes_dists['%d_%d_%d' % (i, j, k)],
                               shading='flat', cmap='viridis', vmin=0, vmax=1)
                ax6.contour(Xm, Ym, bayes_dists['%d_%d_%d' % (i, j, k)],
                            levels=[0.95], c='r', linewidth=3)
                ax6.set_xlim((0, 20))
                ax6.set_ylim((min_true, max_true))
                ax6.set_title('Bayes Prob W/ 95% Line')

                plt.suptitle('%s - Mag %d - %s - %s' % (name, i, g_strs[j], b_strs[k]))
                plt.savefig(path_fig + '_smooth_dists_%d_%d_%d' % (i, j, k), dpi=100)
                plt.show()

    skip = 0
    chunk = int(1e6)
    contin = True

    # grab chunks of data need to find which index in probability distribution
    # to use
    data_ang = np.genfromtxt(path + rank_file, skip_header=skip, max_rows=chunk)
    data_gaia = np.genfromtxt(path + all_file, skip_header=skip, max_rows=chunk, usecols=(7, 1, 2))
    data_mag = np.genfromtxt(path + all_file, skip_header=skip, max_rows=chunk, usecols=mag_cols)

    ang_sep = (data_ang[:, 1] ** 2 + data_ang[:, 2] ** 2) ** 0.5
    G = data_gaia[:, 0]
    c = SkyCoord(ra=data_gaia[:, 1] * u.degree,
                 dec=data_gaia[:, 2] * u.degree,
                 frame='icrs')
    # get the indexes in all dimensions of probability distribution
    # i.e. which mag, G cut, b cut, and where on 2D dist
    b = np.array(abs(c.galactic.b.deg))
    js = calc_ind(G, gaia_cuts[0], gaia_cuts[-1], gaia_cuts[1] - gaia_cuts[0])
    ks = np.zeros(len(b))
    for i in range(len(b_cuts)):
        if i == 0:
            ks[b <= b_cuts[i]] = i
        elif i == len(b_cuts) - 1:
            ks[b >= b_cuts[i - 1]] = i
        else:
            ks[(b > b_cuts[i - 1]) & (b <= b_cuts[i])] = i
    x = calc_ind(ang_sep, xedges_true[0], xedges_true[-1], xedges_true[1] - xedges_true[0])
    x = x.astype(int)

    # now calculate the baye probs for each mag
    bayes_probs = np.zeros(data_mag.shape)

    for i in range(len(mag_cols)):
        m = data_mag[:, i] - G
        y = calc_ind(m, yedges_true[0], yedges_true[-1], yedges_true[1] - yedges_true[0])
        y = y.astype(int)
        for j in range(len(gaia_cuts)):
            for k in range(len(b_cuts)):
                bayes_probs[:, i][(js == j) & (ks == k)] = bayes_dists['%d_%d_%d' % (i, j, k)][y[(js == j) & (ks == k)],x[(js == j) & (ks == k)]]
        bayes_probs[:, i][(abs(m - G) > 30) | (data_ang[:, 0] == 0)] = -9999
        if name == 'SDSS_DR12':
            bad_mags = [24.635, 25.114, 24.802, 24.362, 22.827]
            bayes_probs[:, i][np.around(bayes_probs[:, i], 3) == bad_mags[i]] = -9999

    with open(path + match_save_file, 'ab') as f:
        np.savetxt(f, bayes_probs)

    skip += chunk

    chunks_done = 1

    print('%d Chunks Done after %.3f minutes!' % (chunks_done, (time.time() - start) / 60))

    while contin:
        try:
            data_ang = np.genfromtxt(path + rank_file, skip_header=skip, max_rows=chunk)
            data_gaia = np.genfromtxt(path + all_file, skip_header=skip, max_rows=chunk, usecols=(7, 1, 2))
            data_mag = np.genfromtxt(path + all_file, skip_header=skip, max_rows=chunk, usecols=mag_cols)

            ang_sep = (data_ang[:,1] ** 2 + data_ang[:,2] ** 2) ** 0.5
            G = data_gaia[:,0]
            c = SkyCoord(ra=data_gaia[:, 1] * u.degree, dec=data_gaia[:, 2] * u.degree, frame='icrs')
            b = np.array(abs(c.galactic.b.deg))
            js = calc_ind(G, gaia_cuts[0], gaia_cuts[-1], gaia_cuts[1] - gaia_cuts[0])
            ks = np.zeros(len(b))
            for i in range(len(b_cuts)):
                if i == 0:
                    ks[b <= b_cuts[i]] = i
                elif i == len(b_cuts) - 1:
                    ks[b >= b_cuts[i - 1]] = i
                else:
                    ks[(b > b_cuts[i - 1]) & (b <= b_cuts[i])] = i
            x = calc_ind(ang_sep, xedges_true[0], xedges_true[-1], xedges_true[1] - xedges_true[0])
            x = x.astype(int)

            bayes_probs = np.zeros(data_mag.shape)

            for i in range(len(mag_cols)):
                m = data_mag[:,i] - G
                y = calc_ind(m, yedges_true[0], yedges_true[-1], yedges_true[1] - yedges_true[0])
                y = y.astype(int)
                for j in range(len(gaia_cuts)):
                    for k in range(len(b_cuts)):
                        bayes_probs[:, i][(js == j) & (ks == k)] = bayes_dists['%d_%d_%d' % (i, j, k)][y[(js == j) & (ks == k)], x[(js == j) & (ks == k)]]
                bayes_probs[:, i][(abs(m) > 50) | (data_ang[:, 0] == 0)] = -9999
                if name == 'SDSS_DR12':
                    bad_mags = [24.635, 25.114, 24.802, 24.362, 22.827]
                    bayes_probs[:, i][np.around(bayes_probs[:, i], 3) == bad_mags[i]] = -9999

            with open(path + match_save_file, 'ab') as f:
                np.savetxt(f, bayes_probs)

            skip += chunk

            chunks_done += 1

            print('%d Chunks Done after %.3f minutes!' % (chunks_done, (time.time() - start) / 60))

        except (StopIteration, IndexError) as e:
            contin = False


def prod(factors):
    return reduce(operator.mul, factors, 1)


def make_best_and_rank_2_sample(name, all_file, rank_file,
                                bayes_file, make_rank_2=True):
    """
    Create files with the best matches and optionally lower
    probability matches in the field.

    Parameters
    ----------

    name: str
        name of the external catalog that is prefixed
        to the all matches file

    all_file: str
        file name for all matches without name

    rank_file: str
        file name for all matches ranks without name

    bayes_file: str
        file name for all matches bayes probs without name

    make_rank_2: booleen
        whether or not you want to make file for lower probability
        matches in the field
    """
    print(name)
    path = name

    save_all = path + all_file[:-4]
    save_rank = path + rank_file[:-4]
    save_bayes = path + bayes_file[:-4]

    start = time.time()
    
    with open(path + all_file, 'r') as f_all, open(path + rank_file, 'r') as f_rank, open(path + bayes_file, 'r') as f_bayes, open(save_all + '_bayes_matches_best_match.txt', 'w') as f_save_all_best, open(save_rank + '_bayes_matches_best_match.txt', 'w') as f_save_rank_best, open(save_bayes + '_bayes_matches_best_match.txt', 'w') as f_save_bayes_best, open(save_all + '_bayes_matches_gt_eq_rank_2.txt', 'w') as f_save_all_2nd, open(save_rank + '_bayes_matches_gt_eq_rank_2.txt', 'w') as f_save_rank_2nd, open(save_bayes + '_bayes_matches_gt_eq_rank_2.txt', 'w') as f_save_bayes_2nd:
        all_temp = []
        rank_temp = []
        bayes_temp = []
        best_matches = 0
        num = 0
        for x_all, x_rank, x_bayes in tqdm_notebook(zip(f_all, f_rank, f_bayes)):
            num += 1
            l_all = x_all.split()
            l_rank = x_rank.split()
            l_bayes = x_bayes.split()
            bayes = np.array(l_bayes)
            bayes = bayes.astype(float)

            if (bayes == -9999).all():
                pass
            # check if at new Gaia ID
            elif len(all_temp) == 0:
                all_temp.append(l_all)
                rank_temp.append(l_rank)
                bayes_temp.append(bayes)
            # if at same Gaia ID, keep appending
            elif l_all[0] == all_temp[-1][0]:
                all_temp.append(l_all)
                rank_temp.append(l_rank)
                bayes_temp.append(bayes)
            # new gaia ID, check which is best match
            else:
                match_prob = np.zeros(len(bayes_temp))
                for i in range(len(bayes_temp)):
                    match_prob[i] = prod(bayes_temp[i][bayes_temp[i] >= 0.])
                if np.nanmax(match_prob) > 0.95:
                    best_matches += 1
                    ind_max = np.argmax(match_prob)
                    for i in range(len(match_prob)):
                        if i != ind_max:
                            if make_rank_2:
                                f_save_all_2nd.write(' '.join(str(m) for m in all_temp[i]) + '\n')
                                f_save_rank_2nd.write(' '.join(str(m) for m in rank_temp[i]) + '\n')
                                f_save_bayes_2nd.write(' '.join(str(m) for m in list(bayes_temp[i])) + '\n')
                        else:
                            f_save_all_best.write(' '.join(str(m) for m in all_temp[i]) + '\n')
                            f_save_rank_best.write(' '.join(str(m) for m in rank_temp[i]) + '\n')
                            f_save_bayes_best.write(' '.join(str(m) for m in list(bayes_temp[i])) + '\n')

                all_temp = []
                rank_temp = []
                bayes_temp = []

                all_temp.append(l_all)
                rank_temp.append(l_rank)
                bayes_temp.append(bayes)

            if num % 100000 == 0:
                print(num, best_matches, '-', "{0:.3f}".format((time.time() - start) / 60), 'minutes')

        # once at end of file, check final Gaia ID for
        # the best match
        match_prob = np.zeros(len(bayes_temp))
        for i in range(len(bayes_temp)):
            match_prob[i] = prod(bayes_temp[i][bayes_temp[i] >= 0.])
        if np.nanmax(match_prob) > 0.95:
            best_matches += 1
            ind_max = np.argmax(match_prob)
            for i in range(len(match_prob)):
                if i != ind_max:
                    if make_rank_2:
                        f_save_all_2nd.write(' '.join(str(m) for m in all_temp[i]) + '\n')
                        f_save_rank_2nd.write(' '.join(str(m) for m in rank_temp[i]) + '\n')
                        f_save_bayes_2nd.write(' '.join(str(m) for m in list(bayes_temp[i])) + '\n')
                else:
                    f_save_all_best.write(' '.join(str(m) for m in all_temp[i]) + '\n')
                    f_save_rank_best.write(' '.join(str(m) for m in rank_temp[i]) + '\n')
                    f_save_bayes_best.write(' '.join(str(m) for m in list(bayes_temp[i])) + '\n')
