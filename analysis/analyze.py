"""
SPDX-FileCopyrightText: © 2024 Trufo™ <tech@trufo.ai>
SPDX-License-Identifier: MIT

Summary statistics for a watermark.
"""

import numpy as np
import pandas as pd

from benchmark import robustness


def summarize_image(df):
    df = df.loc[df['content_format'].apply(lambda x: x.split('/')[0]) == 'image']
    df = df[~df['error']]

    # basic encoding + decoding
    edf = df[df['operation'] == "encode"][[
        'watermark', 'dataset', 'evaluation', 'content_id',
        'content_dimensions', 'time_taken_ms', 'error',
        'psnr', 'ssim', 'tpcp',
    ]]
    edf['content_size'] = edf['content_dimensions'].apply(lambda x: np.round(np.sqrt(x[0] * x[1])).astype(int))

    basic_edit_types = [type(edit).__name__ for edit in robustness.image_edits(robustness.ImageEvaluation.V1_BASIC)]
    ddf = df[df['edit_type'].isin(basic_edit_types)]
    ddf_png = ddf[ddf['edit_parameters'].apply(lambda x: len(x) == 0)][[
        'watermark', 'dataset', 'evaluation', 'content_id',
        'detected', 'decoded',
    ]]
    ddf_jpg = ddf[ddf['edit_parameters'].apply(lambda x: len(x) == 1)][[
        'watermark', 'dataset', 'evaluation', 'content_id',
        'detected', 'decoded',
    ]]
    ddf = pd.merge(ddf_png, ddf_jpg, how='outer', on=['watermark', 'dataset', 'evaluation', 'content_id'], suffixes=['_png', '_jpg'])

    xdf = pd.merge(edf, ddf, how='left', on=['watermark', 'dataset', 'evaluation', 'content_id'])

    ddf_all = df[df['operation'] == "decode"].copy()
    ddf_all['count'] = 1
    ddf_all = ddf_all[[
        'watermark', 'dataset', 'evaluation', 'content_id',
        'detected', 'decoded', 'count',
    ]].groupby(['watermark', 'dataset', 'evaluation', 'content_id']).sum()

    xdf = pd.merge(xdf, ddf_all, how='left', on=['watermark', 'dataset', 'evaluation', 'content_id'])

    # basic by edit type
    ydf = df.copy()
    ydf = pd.merge(ydf, xdf[[
        'watermark', 'dataset', 'evaluation', 'content_id', 'content_size',
    ]], how='left', on=['watermark', 'dataset', 'evaluation', 'content_id',])
    ydf['count'] = 1
    ydf['ntime'] = ydf['time_taken_ms'] / (256 + ydf['content_size'])

    ydf = ydf[[
        'watermark', 'dataset', 'evaluation', 'edit_type',
        'error', 'detected', 'decoded', 'count', 'ntime',
    ]].groupby(['watermark', 'dataset', 'evaluation', 'edit_type']).sum()
    ydf['ntime'] /= ydf['count']

    ### composite score

    # encoding
    enc_zdf = xdf.copy()
    enc_zdf['enc_count'] = 1
    enc_zdf['enc_ntime'] = enc_zdf['time_taken_ms'] / (256 + enc_zdf['content_size'])
    enc_zdf['enc_psnr'] = enc_zdf['psnr']

    enc_zdf['enc_tpcp'] = 10. / (10. + enc_zdf['tpcp'])

    enc_zdf = enc_zdf[[
        'watermark', 'dataset', 'evaluation', 'enc_count', 'enc_ntime', 'enc_psnr', 'enc_tpcp',
    ]].groupby(['watermark', 'dataset', 'evaluation']).sum()
    enc_zdf['enc_psnr'] = enc_zdf['enc_psnr'] / enc_zdf['enc_count']
    enc_zdf['enc_tpcp'] = 10. / (enc_zdf['enc_tpcp'] / enc_zdf['enc_count']) - 10.

    zdf = enc_zdf[['enc_count', 'enc_ntime', 'enc_psnr', 'enc_tpcp']]

    # decoding
    dec_zdf = ydf.reset_index()
    dec_zdf = dec_zdf[dec_zdf['edit_type'].isin(["BASE"])]
    dec_zdf['dec_ntime'] = dec_zdf['ntime'] * dec_zdf['count']
    dec_zdf = dec_zdf.groupby(['watermark', 'dataset', 'evaluation']).sum()
    dec_zdf['dec_base_count'] = dec_zdf['count']
    dec_zdf['dec_base_ntime'] = dec_zdf['dec_ntime'] / dec_zdf['count']
    dec_zdf['dec_base_score'] = dec_zdf['decoded'] / dec_zdf['count'] * 100

    zdf = pd.merge(zdf, dec_zdf[[
        'dec_base_count', 'dec_base_ntime', 'dec_base_score',
    ]], how='outer', on=['watermark', 'dataset', 'evaluation'])

    dec_zdf = ydf.reset_index()
    dec_zdf = dec_zdf[~dec_zdf['edit_type'].isin(["", "BASE"])].reset_index()
    dec_zdf['dec_ntime'] = dec_zdf['ntime'] * dec_zdf['count']
    dec_zdf = dec_zdf.groupby(['watermark', 'dataset', 'evaluation']).sum()
    dec_zdf['dec_edit_count'] = dec_zdf['count']
    dec_zdf['dec_edit_ntime'] = dec_zdf['dec_ntime'] / dec_zdf['count']
    dec_zdf['dec_edit_score'] = dec_zdf['decoded'] / dec_zdf['count'] * 100

    zdf = pd.merge(zdf, dec_zdf[[
        'dec_edit_count', 'dec_edit_ntime', 'dec_edit_score',
    ]], how='outer', on=['watermark', 'dataset', 'evaluation'])

    return (xdf, ydf, zdf)
