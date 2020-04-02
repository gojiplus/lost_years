#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
import pandas as pd

from pkg_resources import resource_filename

from .utils import column_exists, fixup_columns, closest

SSA_DATA = resource_filename(__name__, "data/ssa.csv")
SSA_COLS = ['age', 'male_life_expectancy', 'female_life_expectancy', 'year']


class LostYearsSSAData():
    __df = None

    @classmethod
    def lost_years_ssa(cls, df, cols=None):
        """Appends Life expectancycolumn from SSA data to the input DataFrame
        based on age, sex and year in the specific cols mapping

        Args:
            df (:obj:`DataFrame`): Pandas DataFrame containing the last name
                column.
            cols (dict or None): Column mapping for age, sex, and year
                in DataFrame
                (None for default mapping: {'age': 'age', 'sex': 'sex',
                                            'year': 'year'})
        Returns:
            DataFrame: Pandas DataFrame with life expectency column(s):-
                'ssa_age', 'ssa_year', 'ssa_life_expectancy'
        """
        df_cols = {}
        for col in ['age', 'sex', 'year']:
            tcol = col if cols is None else cols[col]
            if tcol not in df.columns:
                print("No column `{0!s}` in the DataFrame".format(tcol))
                return df
            df_cols[col] = tcol

        if cls.__df is None:
            cls.__df = pd.read_csv(SSA_DATA, usecols=SSA_COLS)

        out_df = pd.DataFrame()
        for i, r in df.iterrows():
            if r[df_cols['sex']].lower() in ['m', 'male']:
                ecol = 'male_life_expectancy'
            else:
                ecol = 'female_life_expectancy'
            sdf = cls.__df[['age', 'year', ecol]]
            for c in ['age', 'year']:
                sdf = sdf[sdf[c] == closest(sdf[c].unique(), r[df_cols[c]])]
            odf = sdf[['age', 'year', ecol]].copy()
            odf.columns = ['ssa_age', 'ssa_year', 'ssa_life_expectancy']
            odf['index'] = i
            out_df = pd.concat([out_df, odf])
        out_df.set_index('index', drop=True, inplace=True)
        rdf = df.join(out_df)
        return rdf


lost_years_ssa = LostYearsSSAData.lost_years_ssa


def main(argv=sys.argv[1:]):
    title = ('Appends Lost Years data column(s) by age, sex and year')
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('input', default=None,
                        help='Input file')
    parser.add_argument('-a', '--age', default='age',
                        help='Columns name of age in the input file'
                             '(default=`age`)')
    parser.add_argument('-s', '--sex', default='sex',
                        help='Columns name of sex in the input file'
                             '(default=`sex`)')
    parser.add_argument('-y', '--year', default='year',
                        help='Columns name of year in the input file'
                             '(default=`year`)')
    parser.add_argument('-o', '--output', default='lost-years-output.csv',
                        help='Output file with Lost Years data column(s)')

    args = parser.parse_args(argv)

    print(args)

    df = pd.read_csv(args.input)

    if not column_exists(df, args.age):
        print("Column: `{0!s}` not found in the input file".format(args.age))
        return -1

    if not column_exists(df, args.sex):
        print("Column: `{0!s}` not found in the input file".format(args.sex))
        return -1

    if not column_exists(df, args.year):
        print("Column: `{0!s}` not found in the input file".format(args.year))
        return -1

    rdf = lost_years_ssa(df, cols={'age': args.age, 'sex': args.sex,
                                   'year': args.year})

    print("Saving output to file: `{0:s}`".format(args.output))
    rdf.columns = fixup_columns(rdf.columns)
    rdf.to_csv(args.output, index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
