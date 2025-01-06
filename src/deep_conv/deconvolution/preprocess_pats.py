import glob
import os 
import time
import hashlib
import subprocess
import gzip
import tempfile
import logging 
import argparse
import pandas as pd 
import numpy as np
import os.path as op
from multiprocessing import Pool

coord_cols = ['chr', 'start', 'end', 'startCpG', 'endCpG']


def pat2name(pat):
    return op.basename(op.splitext(op.splitext(pat)[0])[0])


def check_executable(cmd):
    for p in os.environ['PATH'].split(":"):
        if os.access(op.join(p, cmd), os.X_OK):
            return
    print(f'executable {cmd} not found in PATH')
    exit(1)


def validate_file(fpath):
    if not op.isfile(fpath):
        print('Invalid file', fpath)
        exit()
    return fpath


def mkdir_p(dirpath):
    if not op.isdir(dirpath):
        os.mkdir(dirpath)
    return dirpath


def clear_mem_file(tmp_dir_l):
    try:
        for d in os.listdir(tmp_dir_l):
            if not d.endswith('.bed'):
                continue
            dd = op.join(tmp_dir_l, d)
            if not op.isfile(dd):
                continue
            if (time.time() - op.getmtime(dd)) > 5 * 60 * 60:
                os.remove(dd)
    except Exception as e:
        # no reason to crash over this cleanup process
        return


def pat2homog_mp_wrap(markers, pats, tmp_dir_l, rlen, verb, force, nodump, debug, threads, wgbs_tools_exec):
    # multiprocess wrapper for the pat2homog method
    # return a dict {pat: homog table}
    # homog table is markers x [U, X, M]
    params = [(markers, p, tmp_dir_l, rlen, verb, force, nodump, debug, wgbs_tools_exec)
               for p in pats]
    p = Pool(threads)
    arr = p.starmap(pat2homog, params)
    p.close()
    p.join()
    res = {k: v for d in arr for k, v in d.items()}
    # validate results:
    for k in res.keys():
        if res[k].empty:
            print('Failed in homog_mem: ', k)
            exit(1)
    return res


def pat2memfile(pat, tmp_dir_l):
    # get memoization path for a given pat path
    dir_hash = hashlib.md5(op.dirname(op.realpath(pat))\
            .encode()).hexdigest()[:6]
    return op.join(tmp_dir_l, f'{pat2name(pat)}.mem.{dir_hash}.homog.gz')


def remove_files(files):
    for f in files:
        if op.isfile(f):
            os.remove(f)


def load_homog(homog_path):
    validate_file(homog_path)
    names = coord_cols + list('UXM')
    ddtype = {x: int for x in names}
    ddtype['chr'] = str
    try:
        df = pd.read_csv(homog_path, sep='\t', dtype=ddtype, names=names, comment='#')
    except:
        print(f'Failed loading memoization file {homog_path} .')
        # os.remove(homog_path)
        return pd.DataFrame()
    return df


def wrap_cpp_tool(pat, markers, tmp_dir_l, rlen, verb, debug, wgbs_tools_exec):
    # dump extended markers file (for tabix -R)
    validate_file(pat)
    # dump reducerd markers file
    uniq_name = pat2name(pat) + '.' + next(tempfile._get_candidate_names())
    uniq_name = op.join(tmp_dir_l, uniq_name)
    tmp_mpath = uniq_name + '.bed'
    markers[coord_cols].to_csv(tmp_mpath, sep='\t', header=None, index=None)
    # pat to homog.gz:
    cmd = f'{wgbs_tools_exec} homog -f --rlen {rlen} -b {tmp_mpath} {pat} --prefix {uniq_name}'
    if debug:
        cmd += ' -v '
        print(cmd)
    so = None if verb else subprocess.PIPE
    subprocess.check_call(cmd, shell=True, stderr=so, stdout=so)
    if not debug:
        remove_files([tmp_mpath])
    return uniq_name + '.uxm.bed.gz'


def pat2homog(markers, pat, tmp_dir_l, rlen, verb, force, nodump, debug, wgbs_tools_exec):
    mempat = pat2memfile(pat, tmp_dir_l)
    name = pat2name(pat)
    # load current markers if exist:
    omark = markers.copy()
    markers = markers.drop_duplicates(subset=coord_cols).reset_index(drop=True)
    # in case this pat file is unseen before, 
    # or it's mempat is older than the pat, or --force is specified:
    ignore_mem = False
    if not op.isfile(mempat):
        msg = f'[ {name} ] no memoization found'
        ignore_mem = True
    elif op.getctime(mempat) < op.getctime(pat):
        msg = f'[ {name} ] memoization is older than pat! deleting it'
        ignore_mem = True
    if force:
        msg = f'[ {name} ] overwriting existing memoization file (--force)'
        ignore_mem = True
    if ignore_mem:
        remove_files([mempat])
        if verb:
            print(msg)
        remain_mrk = markers
        homog = pd.DataFrame()
    # else, find markers not already present in mempat
    else:
        homog = load_homog(mempat).reset_index(drop=True)
        if homog.empty:
            if verb:
                print(f'loaded empty homog for {pat}: {mempat}')
            os.remove(mempat)
            return {pat: homog}
        remain_mrk = markers.merge(homog, how='left')
        remain_mrk = remain_mrk[remain_mrk['U'].isna()]
        if verb and not remain_mrk.empty:
            print(f'[ {name} ] found memoization, missing {remain_mrk.shape[0]} markers' )
    # if all markers are present, return them
    if remain_mrk.empty:
        if verb:
            print(f'[ {name} ] all markers found in memory')
        res = omark.merge(homog, how='left')
        # debug \ validation
        if res.shape[0] != omark.shape[0]:
            print(f'[ {name} ] Error:', res.shape, omark.shape)
            return {pat: pd.DataFrame()}
        return {pat: res[['U', 'X', 'M']]}
    # otherwise, run compute remaining homog values 
    tmp_homog_path = wrap_cpp_tool(pat, remain_mrk, tmp_dir_l, rlen, verb, debug, wgbs_tools_exec)
    # homog.gz to numpy array uxm:
    uxm = load_homog(tmp_homog_path)
    # cleanup 
    remove_files([tmp_homog_path])
    nodump = bool(nodump)
    if uxm[['U', 'X', 'M']].values.sum() == 0:
        if verb:
            print('\033[91m' + f'WARNING:' + '\033[0m' +
                    f' possibly failed in {pat} - all {uxm.shape[0]} ' \
                    'values are zero. memoization is not updated, to be safe')
        nodump = True
    all_markers = pd.concat([homog, uxm])
    all_markers.sort_values(by=['startCpG', 'endCpG'], inplace=True)
    res = omark.merge(all_markers, how='left')
    if not nodump:
        # dump
        with gzip.open(mempat, 'wt') as mp:
            mp.write(f'# {pat}\n')
        all_markers.to_csv(mempat, sep='\t', index=None, na_rep='NA', header=None, mode='a')
        # chmod to -rw-rw-r-- so all group users can share the same temp_dir
        os.chmod(mempat, 0o664)
    return {pat: res[['U', 'X', 'M']]}


def gen_homogs(markers, pats, tmp_dir, verb, rlen, force, nodump, debug, threads, wgbs_tools_exec):
    check_executable(wgbs_tools_exec)
    # arguments cleanup
    markers = markers[coord_cols]          # keep only coords columns
    # change to abs path (for mem files), validate existance, drop dups
    pats = sorted(set(map(validate_file, map(op.abspath, pats))))
    # create tmp_dir if needed
    tmp_dir_l = mkdir_p(op.join(mkdir_p(tmp_dir), f'l{rlen}'))
    # run compute homog values on missing markers and pats
    uxm_dict = pat2homog_mp_wrap(markers, pats, tmp_dir_l, rlen, verb, force, nodump, debug, threads, wgbs_tools_exec)
    # clean tmp_dir_l from old temp files:
    clear_mem_file(tmp_dir_l)
    return uxm_dict


def load_pats_homog(atlas, pats, tmp_dir, verb, rlen, force, nodump, debug, threads, wgbs_tools_exec):
    for pat in pats:
        if not pat.endswith('.pat.gz'):
            print(f'Invalid input file: {pat}. must end with .pat.gz')
            exit(1)
    pats = [op.abspath(p) for p in pats]
    uxm_dict = gen_homogs(atlas, pats, tmp_dir, verb, rlen, force, nodump, debug, threads, wgbs_tools_exec)
    samples_df = atlas[['name', 'direction']].copy()
    counts = atlas[['name', 'direction']].copy()
    for pat in pats:
        name = pat2name(pat)
        uxm = uxm_dict[pat].values
        counts[name] = uxm.sum(axis = 1)
        with np.errstate(divide='ignore', invalid='ignore'):
            uxm = uxm / uxm.sum(axis = 1)[:, np.newaxis]
        samples_df[name] = np.where(samples_df['direction'] == 'U', uxm[:, 0], uxm[:, 2])
    return samples_df, counts


def pats_to_homog(atlas_path, pats_path, wgbs_tools_exec_path, r_len = 4, output_path=None, threads=4):
    atlas = pd.read_csv(atlas_path, sep='\t')    
    pats = glob.glob(pats_path+"/*.pat.gz")
    marker_read_proportions, marker_read_coverage  = load_pats_homog(
        atlas=atlas,
        pats=pats,
        tmp_dir="/users/zetzioni/sharedscratch/temp",
        verb=False,
        rlen=r_len,
        force=False,
        nodump=False,
        debug=False,
        threads=threads,
        wgbs_tools_exec=wgbs_tools_exec_path,
    )
    if output_path is not None:
        marker_read_proportions.to_csv(output_path+"sf.csv",index=False)
        marker_read_coverage.to_csv(output_path+"counts.csv",index=False)
    return marker_read_proportions, marker_read_coverage

