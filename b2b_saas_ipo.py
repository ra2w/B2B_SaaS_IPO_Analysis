'''
SaaS B2B IPO Analysis
Date created: 04/01/20
Author: Ramu Arunachalam
Email: ramu@acapital.com
'''

import matplotlib as mp
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
from scipy.stats import norm
import seaborn as sns
import re
from scipy import stats
import statsmodels.api as sm
import statsmodels.api as sm

from IPython.display import display, HTML

plt.style.use('fivethirtyeight')

class ChainedAssignent:
    def __init__(self, chained=None):
        acceptable = [None, 'warn', 'raise']
        assert chained in acceptable, "chained must be in " + str(acceptable)
        self.swcw = chained

    def __enter__(self):
        self.saved_swcw = pd.options.mode.chained_assignment
        pd.options.mode.chained_assignment = self.swcw
        return self

    def __exit__(self, *args):
        pd.options.mode.chained_assignment = self.saved_swcw


# column names to include from main csv
col_names = ['name',
    'founding_year',
    'ipo_year',
    'arr__m',
    'arr_growth',
    'net_cash'] #'net_dollar_retention','ltm_magic_number']


def normalize_column_names(df):
    # remove $ and whitespace from column names; add _ between words, add __m for $m
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('$','_').str.replace('/','_').str.replace('%','pct')
    if ('revenue_growth' in df.columns):
        df['rev_growth'] = df.revenue_growth


def merge_data(df,df_k,df_a,df_mkt):
    df_m = pd.merge(df,df_k,on=['ticker'],how='left')
    df_m = pd.merge(df_m,df_mkt,on=['ticker'],how='left')
    df_m.update(df_a)
    return df_m

def dump_table(df, filename):
    df.to_csv(filename)

def load_table(filename):
    df = pd.read_csv(filename)
    df = df.set_index('ticker')
    df = df.replace(to_replace ='\$', value = '', regex = True)
    df = df.replace(to_replace ='\%', value = '', regex = True)
    df = df.replace(to_replace ='\,', value = '', regex = True)
    df = df.replace(to_replace = '[]')
    normalize_column_names(df)

    for c in df.columns:
        if (c not in ['name','ticker','last_filing','earnings_link','type']):
            #print('c={}'.format(c))
            df[c] = pd.to_numeric(df[c],downcast='float')
    #if ('rrr_m' in df.columns):
    #    df.rrr__m = pd.to_numeric(df.rrr__m,downcast = 'float')
    #if ('rev_growth' in df.columns):
    #    df.rev_growth = pd.to_numeric(df.rev_growth,downcast = 'float')
    df['name']=df.index
# Remove companies that didn't have a SaaS business model @ IPO

    df = df[df.name != 'PANW']
    df = df[df.name != 'SPLK']
    df = df[df.name != 'FEYE']
    return df

large_file_name = '/Users/ramu/IPO/large_set_comp_04_01_20'
small_file_name = '/Users/ramu/IPO/small_set_comp_03_31_20'
augment_file_name = '/Users/ramu/IPO/augment_set_comp_04_01_20.csv'
USE_SMALL_DATA_SET = False
VERBOSE = False


# Revenue analysis
rev_req_cols =['rrr__m','rev_growth','net_cash','capital_raised'],
rev_filter_cols = ['rrr__m','rev_growth','net_cash','founding_year','ipo_year','capital_raised']

def remove_na(df,col_names):
    for c in col_names:
        print("{} companies missing: {}".format(df[df[c].isna()].shape[0],c))
        #df = df[df[c].notna()]
    return df

def add_burn_efficiency(df):
    def burn_calc(f,ef):
        # $ burned to get to $100M
        df[ef] = df.net_raised / df[f]*100
        #df[ef][df[ef].lt(0)] = max(df[ef])
        #df[ef][df[ef].gt(4.75)] = 4.75
    burn_calc('arr__m','arr_ef')
    burn_calc('rrr__m','rev_ef')

def add_net_raised(df):
    df['net_raised'] = (df.capital_raised-df.net_cash)


def saas_plot(df,x_col,y_col,p50,p75,xscale='linear',yscale='linear',w=8,h=6,reg=False,plot_percentile=1):
    df_pl=df[(df[x_col].notna()) & (df[y_col].notna())]

    df_l = df_pl[df_pl[x_col]<=df_pl[x_col].quantile(plot_percentile)]
    df_notl = df_pl[(df_pl[x_col]>df_pl[x_col].quantile(plot_percentile)) & (df_pl[x_col]<=df_pl[x_col].quantile(1))]

    with ChainedAssignent():
        df_l['label']=df_l.name
        df_notl['label']=''

    df_pl = pd.concat([df_l,df_notl])

    #print("Correlation ({},{}) = {}".format(y_col,x_col,stats.pearsonr(df_pl[y_col], df_pl[x_col])))
    g = sns.lmplot(data=df_l, x=x_col, y=y_col,fit_reg=reg,aspect =1)
    g.fig.set_figwidth(w)
    g.fig.set_figheight(h)
    plt.xscale(xscale)
    plt.yscale(yscale)

    def label_point(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            ax.text(point['x']+.02, point['y'], str(point['val']))


    label_point(df_l[x_col], df_l[y_col], df_pl.label, plt.gca())

    x = plt.gca().axes.get_xlim()
    xmed = p50
    x75 = p75


    # how to plot median line?
    #ax.plot(bot_sort,ymax, label='Max', linestyle='--',color = '#001c58')
    plt.plot(x, len(x) * [xmed], sns.xkcd_rgb["pale red"], label='Median',linestyle='--',)
    plt.plot(x, len(x) * [x75], sns.xkcd_rgb["denim blue"],label = '75^{th} percentile',linestyle='--',)
    plt.legend()
    plt.show()

#sns.jointplot(x='ltm_median_payback_period', y='arr_per_customer', kind="reg", stat_func=r2,data=df)
#ax.set_yscale('log')
#sns.regplot(x="ltm_median_payback_period", y="rev_growth", data=df);


def saas_reg(df,x_col,y_col,log_x=[False],log_y=False,use_b=False):

    df_reg = df[df[y_col].notna()]
    i = 0
    for c in x_col:
        df_reg = df_reg[df_reg[c].notna()]
        if (log_x[i]):
            df_reg[c]=np.log10(df_reg[c])
        i = i + 1

    if log_y:
        df_reg[y_col]=np.log10(df_reg[y_col])
    #df_reg=df[(df[x_col].notna()) & (df[y_col].notna())]
    X = df_reg[x_col]
    if use_b:
        X = sm.add_constant(X)
    y = df_reg[y_col]

    # Note the difference in argument order
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X) # make the predictions by the model
    # Print out the statistics
    print(model.summary())
    return model



def saas_plot_hist(title,df,x_min=2,x_lim=5,bins=8,param='rev_ef',market_cap_min=2000):
    plt.figure()
    if (market_cap_min):
        df_t = df[df.market_cap__m>=market_cap_min]
    df_t = df[(df.market_cap__m>=market_cap_min) & (df[param] <= x_lim) & (df[param].notna())]
    x,r = pd.qcut(df_t[param],bins,retbins=True)

    df_1 = df_t[df_t.type == 'bot_up_ent']

    fig1, f1_axes = plt.subplots(ncols=3, constrained_layout=True,figsize = (16, 4))
    sns.distplot(df_1[param],hist=True,kde_kws={'clip': (x_min, x_lim)},bins=r,ax=f1_axes[0])
    sns.catplot(x="all",y=param, hue="type", kind="box", data=df_t,ax=f1_axes[1]);

    #,hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "g"})

    df_2 = df_t[df_t.type == 'top_down_ent']
    sns.distplot(df_2[param],hist=True,kde_kws={'clip': (x_min, x_lim)},bins=r,ax=f1_axes[2])
    #hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "b"})

    plt.xlim(x_min,x_lim, None)
    plt.title(title)

    plt.tight_layout()
    plt.show()
    print('N = {}'.format(df_t.shape[0]))


def print_stats(str,df,col):
    print('[{}] {}'.format(str,col))
    count,mean,std,low,p_25,p_50,p_75,high = df[col].describe()
    print('\tN = {:.0f} [min,max] = [{:.1f},{:.1f}] mean = {:.1f} std = {:.1f}'.format(count,low,high,mean,std))
    print('\t[25%,50%,75%] = [{:.1f},{:.1f},{:1f}]'.format(p_25,p_50,p_75))

# print box plots for two variables side by side
# even though the name of the function in saas_box_plot it can also do violin plots!

def saas_box_plot(df,var1,x_label1,y_label1,axes,plot_type='box',title='',axis='left',showfliers=False):

    df_v1 = df[df[var1].notna()]
    if plot_type == 'box':
        ax1 = sns.boxplot(x='type', y=var1,data=df_v1, ax=axes,showfliers = False)
    else:
        ax1 = sns.violinplot(x=var1,y="all",hue="type",data=df,split=True,ax=axes,palette='deep',inner='quartile')

    if (axis == 'right'):
        ax1.yaxis.set_label_position("right")
        ax1.yaxis.tick_right()
    ax1.set_ylabel(y_label1)
    ax1.set_xlabel(x_label1)
    ax1.set_title(title)



def main(filename='data/ipo_db.csv',
         req_cols=['rev_growth','arr__m','arr_growth','net_cash','capital_raised'],
         filter_cols = ['rrr__m','rev_growth',
                        'arr__m','arr_growth',
                        'arr_ef','rev_ef',
                        'ltm_median_payback_period',
                        'net_cash',
                        'founding_year','ipo_year',
                        'net_dollar_retention','ltm_magic_number',
                        'arr_per_customer',
                        'capital_raised',
                        'net_raised',
                        's&m_pct',
                        'gross_margin',
                        'type',
                        'market_cap__m','name']):
    df = load_table(filename)
    df = remove_na(df,req_cols)
    add_net_raised(df)
    add_burn_efficiency(df)
    if (filter_cols):
        df = df[filter_cols]
    print("\nTotal companies (N) = {}".format(df.shape[0]))
    pd.options.display.float_format = '{:,.2f}'.format
    return df
