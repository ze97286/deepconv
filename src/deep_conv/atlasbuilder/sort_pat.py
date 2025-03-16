import pandas as pd

chrs = [f'chr{i}' for i in range(1,23)]

def sort_pat(pat_file):
    df = pd.read_csv(pat_file, sep="\t", names=['chr', 'start', 'pattern', 'count'])
    df = df[df.chr.isin(chrs)]
    df['chr_num'] = df['chr'].str.extract('(\d+)', expand=False).astype(int)
    df_sorted = df.sort_values(by=['chr_num', 'start']).drop(columns='chr_num')
    print(df_sorted)
    df_sorted.to_csv(
        pat_file,
        sep="\t",           
        header=False,       
        index=False,        
        compression="gzip"  
    )

