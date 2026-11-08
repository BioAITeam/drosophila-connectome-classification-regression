#!/usr/bin/env python3
"""Recompute Table 16 from selected predictions and published comparison values."""
from pathlib import Path
import argparse,csv,json,statistics
import numpy as np
from ki67.evaluation import classification_metrics

REPO=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--finetune-summary',type=Path,required=True)
    p.add_argument('--literature',type=Path,default=Path('6 Ki67/5 Outputs/paper-literature-comparison/literature.csv'))
    p.add_argument('--joint',type=Path,default=Path('6 Ki67/5 Outputs/joint-bcdata-shid-v1'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Choose a new output directory')
    args.output.mkdir(parents=True)
    keys=['accuracy','macro_precision','macro_recall','macro_f1']
    runs=[]
    for seed in [1729,2718,3141]:
        folder=args.joint/'runs/joint/phikonv2/multiclass/anatomical'/f'seed-{seed}'
        best=json.loads((folder/'best.json').read_text())
        with np.load(folder/best['prediction_path'],allow_pickle=False) as saved:
            mask=saved['source_dataset']=='shid'
            if mask.sum()!=700:raise ValueError('Expected 700 SHIDC-B-Ki-67 test images')
            runs.append(classification_metrics(saved['y_true'][mask],saved['probabilities'][mask]))
    fine=json.loads(args.finetune_summary.read_text())['summary']
    rows=[{'source':'This work','model':'BRAP-Net, anatomical, frozen Phikon-v2',
           **{k:100*statistics.mean(v[k] for v in runs) for k in keys},
           **{k+'_sd':100*statistics.stdev(v[k] for v in runs) for k in keys}},
          {'source':'This work','model':'BRAP-Net, anatomical, fully fine-tuned Phikon-v2',
           **{k:fine[k]['mean_percent'] for k in keys},
           **{k+'_sd':fine[k]['sample_sd_percent'] for k in keys}}]
    with args.literature.open() as stream:
        for v in csv.DictReader(stream):rows.append({'source':v['citation_key'],'model':v['model'],**{k:float(v[k]) for k in keys}})
    rows.sort(key=lambda v:-v['macro_recall'])
    lines=[r'\begin{table*}[!tbp]',r'\centering',
      r'\caption{Three-class Ki-67 classification on the SHIDC-B-Ki-67 test set (700 images). Rows are ordered by decreasing macro recall (BA). Results from this study are mean $\pm$ standard deviation across three training seeds. All metrics are expressed as percentages.}',
      r'\label{tab:ki67-vlm-comparison}',r'\small',r'\setlength{\tabcolsep}{3pt}',r'\renewcommand{\arraystretch}{1.2}',
      r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrr@{}}',r'\toprule',
      r'Source & Method & Accuracy (\%) & \makecell{Macro\\precision (\%)} & \makecell{Macro\\recall (\%)} & Macro-F1 (\%) \\',r'\midrule']
    for v in rows:
        source=v['source'] if v['source']=='This work' else r'\citep{'+v['source']+'}'
        values=[f"${v[k]:.2f}\\pm{v[k+'_sd']:.2f}$" if k+'_sd' in v else f'{v[k]:.2f}' for k in keys]
        lines.append(' & '.join([source,v['model'],*values])+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular*}',r'\end{table*}'])
    (args.output/'Table-16.tex').write_text('\n'.join(lines)+'\n')
    (args.output/'Table-16.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('Reconstructed Table 16 from three seeds per anatomical model.')

if __name__=='__main__':main()
