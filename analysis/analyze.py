"""
SPDX-FileCopyrightText: © 2024 Trufo™ <engineering@trufo.ai>
SPDX-License-Identifier: MIT

Summary statistics for a watermark.
"""

import os
import glob
from typing import Union

import numpy as np
import pandas as pd

from benchmark import durability
from benchmark.image.edit import ImageEditParams
from bench import BenchmarkEvaluation


class ImageAnalysis():
    """
    Analyze raw benchmark data.
    """
    def __init__(self, input: Union[str, pd.DataFrame]):
        # parsing input
        if isinstance(input, str):
            df_files = glob.glob(f"{os.getcwd()}/results/{input}.json")

            dfs = []
            for df_file in df_files:
                dfs.append(pd.read_json(df_file))
            df = pd.concat(dfs)
        else:
            df = input
        
        # pruning dataframe
        df = df.loc[df['dataset'].apply(lambda x: 'IMG' in x)]
        df = df[~df['error']]
        self.df = df

        self.summarize_per()
        self.summarize_edit()
        self.summarize_score()
    
    @staticmethod
    def get_evaluation_subset(df: pd.DataFrame, evaluation: BenchmarkEvaluation):
        return df[(df['edit_type'] != '') & (df['evaluation'] == evaluation.value)]

    def summarize_per(self):
        """
        Get a per-(watermark, dataset, evaluation, content_id) summary, with both encoding and decoding metrics.
        """
        # encoding metrics
        xdf = self.df[self.df['operation'] == "encode"][[
            'watermark', 'dataset', 'evaluation', 'content_id',
            'content_dimensions', 'time_taken_ms', 'error',
            'psnr', 'ssim', 'pcpa',
        ]]
        def get_size(shape):
            if len(shape) >= 2:
                return np.round(np.sqrt(np.prod(shape[:2]))).astype(int)
            return 0
        xdf['content_size'] = xdf['content_dimensions'].apply(get_size)

        # decoding metrics
        ddf_all = self.df[self.df['operation'] == "decode"].copy()
        ddf_all['total'] = 1
        ddf_all = ddf_all[[
            'watermark', 'dataset', 'evaluation', 'content_id',
            'detected', 'decoded', 'total',
        ]].groupby(['watermark', 'dataset', 'evaluation', 'content_id']).sum()
        xdf = pd.merge(xdf, ddf_all, how='left', on=['watermark', 'dataset', 'evaluation', 'content_id'])

        self.df_per = xdf

    def summarize_edit(self):
        """
        Get a per-edit breakdown, per-(watermark, dataset, evaluation).
        """
        # preprocessing
        ydf = self.df.copy()
        ydf = pd.merge(ydf, self.df_per[[
            'watermark', 'dataset', 'evaluation', 'content_id', 'content_size',
        ]], how='left', on=['watermark', 'dataset', 'evaluation', 'content_id',])
        ydf['count'] = 1
        ydf['ntime'] = ydf['time_taken_ms'] / (256 + ydf['content_size'])

        # aggregating
        ydf = ydf[[
            'watermark', 'dataset', 'evaluation', 'edit_type',
            'error', 'detected', 'decoded', 'count', 'ntime',
        ]].groupby(['watermark', 'dataset', 'evaluation', 'edit_type']).sum()
        ydf['ntime'] /= ydf['count']

        self.df_edit = ydf
    
    def summarize_score(self):
        """
        Get top-line scores, per-(watermark, dataset, evaluation).
        """
        # encoding composite
        enc_zdf = self.df_per.copy()
        enc_zdf['enc_count'] = 1
        enc_zdf['enc_ntime'] = enc_zdf['time_taken_ms'] / (256 + enc_zdf['content_size'])
        enc_zdf['enc_psnr'] = enc_zdf['psnr']
        enc_zdf['enc_ssim'] = enc_zdf['ssim']
        enc_zdf['enc_pcpa'] = 10. / (10. + enc_zdf['pcpa'])

        enc_zdf = enc_zdf[[
            'watermark', 'dataset', 'evaluation', 'enc_count', 'enc_ntime', 'enc_psnr', 'enc_ssim', 'enc_pcpa',
        ]].groupby(['watermark', 'dataset', 'evaluation']).sum()
        enc_zdf['enc_psnr'] = enc_zdf['enc_psnr'] / enc_zdf['enc_count']
        enc_zdf['enc_sdsm'] = -np.log2(1. - enc_zdf['enc_ssim'] / enc_zdf['enc_count'])
        enc_zdf['enc_pcpa'] = 10. / (enc_zdf['enc_pcpa'] / enc_zdf['enc_count']) - 10.

        zdf = enc_zdf[['enc_count', 'enc_ntime', 'enc_psnr', 'enc_sdsm', 'enc_pcpa']]

        # decoding composite
        dec_zdf = self.df_edit.reset_index()
        dec_zdf = dec_zdf[dec_zdf['edit_type'] != '']
        dec_zdf['dec_ntime'] = dec_zdf['ntime'] * dec_zdf['count']
        dec_zdf = dec_zdf.groupby(['watermark', 'dataset', 'evaluation']).sum()
        dec_zdf['dec_count'] = dec_zdf['count']
        dec_zdf['dec_ntime'] = dec_zdf['dec_ntime'] / dec_zdf['count']
        dec_zdf['dec_score'] = dec_zdf['decoded'] / dec_zdf['count'] * 100

        zdf = pd.merge(zdf, dec_zdf[[
            'dec_count', 'dec_ntime', 'dec_score',
        ]], how='outer', on=['watermark', 'dataset', 'evaluation'])

        self.df_score = zdf
