"""Render Ki-67 tables from reconstructed numerical summaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from .common import portable_path
from .paper_io import prepare_destination, sha256

HEADERS = {1: '\\begin{table*}[!tbp]\n\\centering\n\\caption{Proposed SHIDC-B-Ki-67-V2 partition with mutually exclusive patient groups. Percentages refer to the 2,356 images. Cell counts correspond to the existing positive and negative tumor-cell annotations.}\n\\label{tab:shid-v2-partition}\n\\small\n\\renewcommand{\\arraystretch}{1.15}\n\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrrrrrr@{}}\n\\toprule\nPartition & Patients & Images & Images (\\%) & Positive cells & Negative cells & Total cells \\\\\n\\midrule', 4: '\\begin{table*}[!tbp]\n\\centering\n\\caption{Joint three-class Ki-67 test performance. Models were trained jointly using BCData with either SHIDC-B-Ki-67 or SHIDC-B-Ki-67-V2. Selected gains are evaluated unchanged on the corresponding combined test set and dataset subsets. BCData-only rows refer to models trained jointly using BCData and SHIDC-B-Ki-67. The Phikon-v2 rows are the anatomical and rewired BRAP-Net variants. Initial denotes epoch zero. Selected checkpoints maximize combined validation BA. Selected BA is mean $\\pm$ sample standard deviation across three seeds. Other entries are means. Accuracy, macro-F1, and ROC-AUC also describe selected checkpoints. BA, accuracy, and macro-F1 are percentages. $\\Delta$ is the paired BA change in percentage points (pp) before rounding. ROC-AUC is macro one-versus-rest.}\n\\label{tab:ki67-multiclass}\n\\small\n\\setlength{\\tabcolsep}{3pt}\n\\renewcommand{\\arraystretch}{1.2}\n\\begin{tabular}{@{}llrrrrrr@{}}\n\\toprule\nTest set & Model & \\makecell{Initial\\\\BA} & \\makecell{Selected\\\\BA} & $\\Delta$ (pp) & Accuracy & Macro-F1 & ROC-AUC \\\\\n\\midrule', 7: '\\begin{table*}[!tb]\n\\centering\n\\caption{Test BA for separately trained binary Ki-67 models. Initial BA is the mean at epoch zero. Selected BA is mean $\\pm$ sample standard deviation across three seeds at checkpoints selected by validation BA. BA is expressed as a percentage. $\\Delta$ is the paired change in percentage points, computed before rounding. Each dataset and threshold has separate fitted gains. Seeds reuse the same test images within each dataset. Model names identify the frozen extractor and the anatomical or rewired graph.}\n\\label{tab:ki67-binary}\n\\small\n\\setlength{\\tabcolsep}{2.2pt}\n\\renewcommand{\\arraystretch}{1.15}\n\\begin{tabular}{@{}llrrrrrrrrr@{}}\n\\toprule\n& & \\multicolumn{3}{c}{PI $\\leq 5$ / $>5$} & \\multicolumn{3}{c}{PI $\\leq 10$ / $>10$} & \\multicolumn{3}{c}{PI $<30$ / $\\geq 30$} \\\\\n\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(l){9-11}\nDataset & Model & Initial & Selected & $\\Delta$ (pp) & Initial & Selected & $\\Delta$ (pp) & Initial & Selected & $\\Delta$ (pp) \\\\\n\\midrule', 8: '\\begin{table}[!tbp]\n\\centering\n\\caption{Cross-dataset test BA (\\%) for anatomical BRAP-Net with frozen Phikon-v2. Values are mean $\\pm$ sample standard deviation across three source-selected checkpoints per task. Target tests contain 700 images for each SHIDC-B-Ki-67 partition and 402 for BCData. Training and validation use the source dataset exclusively.}\n\\label{tab:ki67-cross-dataset}\n\\footnotesize\n\\setlength{\\tabcolsep}{1pt}\n\\renewcommand{\\arraystretch}{1.2}\n\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}lrrrr@{}}\n\\toprule\n\\makecell[l]{Training\\\\to test} & \\makecell{Three-\\\\class} & PI 5\\% & PI 10\\% & PI 30\\% \\\\\n\\midrule', 13: '\\begin{table*}[!tbp]\n\\centering\n\\caption{Exploratory binary Ki-67 test performance for PI $<30$ versus $\\geq30$, using fine-tuned Phikon-v2 with an anatomical connectome or a linear output layer. Values are percentages, reported as mean $\\pm$ sample standard deviation across three training runs per architecture.}\n\\label{tab:ki67-exploratory-finetuning}\n\\small\n\\renewcommand{\\arraystretch}{1.15}\n\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}llrrr@{}}\n\\toprule\nTest set & Model & BA (\\%) & Accuracy (\\%) & Macro-F1 (\\%) \\\\\n\\midrule'}

FOOTERS = {
    1: '\\bottomrule\n\\end{tabular*}\n\\end{table*}',
    4: '\\bottomrule\n\\end{tabular}\n\\end{table*}',
    7: '\\bottomrule\n\\end{tabular}\n\\end{table*}',
    8: '\\bottomrule\n\\end{tabular*}\n\\end{table}',
    13: '\\bottomrule\n\\end{tabular*}\n\\end{table*}',
}



def read(path):
    return json.loads(Path(path).read_text())


def pm(mean, sd, factor=100):
    return f"${factor * mean:.2f}\\pm{factor * sd:.2f}$"


def row(values):
    return " & ".join(map(str, values)) + r" \\"


def split_rows(data):
    rows = []
    ids = [set(v['patient_ids']) for v in data['splits'].values()]
    if any(a & b for i, a in enumerate(ids) for b in ids[i+1:]):
        raise ValueError('Patient sets overlap')
    total = data['total']
    for name in ['Train', 'Validation', 'Test', 'Total']:
        d = total if name == 'Total' else data['splits'][name]
        if name == 'Total':
            rows.append(r'\midrule')
        rows.append(row([name, d['patients'], f"{d['images']:,}", f"{100*d['images']/total['images']:.2f}",
                         *[f"{d[k]:,}" for k in ['positive_cells', 'negative_cells', 'total_cells']]]))
    return rows


def joint_rows(original, additional):
    blocks = [(original, 'pooled', r'Combined\\BCData +\\SHIDC-B-Ki-67', 1102),
              (original, 'bcdata', 'BCData', 402), (original, 'shid', 'SHIDC-B-Ki-67', 700),
              (additional, 'combined', r'Combined\\BCData +\\SHIDC-B-Ki-67-V2', 1102),
              (additional, 'shid', 'SHIDC-B-Ki-67-V2', 700)]
    result = []
    for index, (data, scope, name, count) in enumerate(blocks):
        group = [r for r in data if r['scope'] == scope]
        if len(group) != 4:
            raise ValueError(f'Expected four models in {name}')
        if index:
            result.extend([r'\cmidrule(l){2-8}', r'\addlinespace'])
        for i, d in enumerate(group):
            if d['n_seeds'] != 3:
                raise ValueError('Three seeds required')
            model = {'dinov2':'DINOv2', 'phikonv2':'Phikon-v2'}[d['representation']]
            model += ' + ' + ('anatomical' if d['graph'] == 'anatomical' else 'rewired')
            if d['representation'] == 'phikonv2':
                model += ' (BRAP-Net)'
            result.append(row([r'\multirow{4}{*}{\makecell[l]{'+name+rf'\\(n = {count:,})'+'}}' if i == 0 else '', model,
                               f"{100*d['initial_balanced_accuracy_mean']:.2f}",
                               pm(d['selected_balanced_accuracy_mean'], d['selected_balanced_accuracy_sample_sd']),
                               f"{d['balanced_accuracy_change_pp_mean']:+.2f}",
                               f"{100*d['selected_accuracy_mean']:.2f}", f"{100*d['selected_macro_f1_mean']:.2f}",
                               f"{d['selected_roc_auc_mean']:.4f}"]))
    return result


def binary_rows(original, additional):
    lookup = {(d['dataset'], d['representation'], d['graph'], d['task']): d for d in original}
    for d in additional:
        lookup[('shid_v2',d['representation'],d['graph'],d['task'])] = dict(d, selected_balanced_accuracy_sd=d['selected_balanced_accuracy_sample_sd'])
    result = []
    for n, (dataset, name) in enumerate([('bcdata','BCData'),('shid','SHIDC-B-Ki-67'),('shid_v2','SHIDC-B-Ki-67-V2')]):
        if n:
            result.append(r'\addlinespace')
        for i,(rep,graph) in enumerate([(r,g) for r in ['dinov2','phikonv2'] for g in ['anatomical','degree241']]):
            model={'dinov2':'DINOv2','phikonv2':'Phikon-v2'}[rep]+' + '+('anatomical' if graph == 'anatomical' else 'rewired')
            values=[r'\multirow{4}{*}{'+name+'}' if i == 0 else '',model]
            for task in ['binary5','binary10','binary30']:
                d=lookup[(dataset,rep,graph,task)]
                values.extend([f"{100*d['initial_balanced_accuracy_mean']:.2f}",
                               pm(d['selected_balanced_accuracy_mean'],d['selected_balanced_accuracy_sd']),
                               f"{d['balanced_accuracy_change_pp_mean']:+.2f}"])
            result.append(row(values))
    return result


def cross_rows(data):
    lookup={(d['source'],d['target'],d['task']):d for d in data}
    result=[]
    for source,target in [('BCData','SHIDC-B-Ki-67'),('BCData','SHIDC-B-Ki-67-V2'),
                          ('SHIDC-B-Ki-67','BCData'),('SHIDC-B-Ki-67-V2','BCData')]:
        first=lookup[(source,target,'multiclass')]
        values=[r'\makecell[l]{'+first['source']+r'\\to '+first['target']+'}']
        for task in ['multiclass','binary5','binary10','binary30']:
            d=lookup[(source,target,task)]
            values.append(pm(d['balanced_accuracy_mean'],d['balanced_accuracy_sd']))
        if result:
            result.append(r'\addlinespace')
        result.append(row(values))
    return result


def fine_rows(data):
    lookup={(d['test_set'],d['architecture']):d for d in data}
    result=[]
    for i,(subset,name,count) in enumerate([('combined','Combined',1102),('bcdata','BCData',402),('shid_v2','SHIDC-B-Ki-67-V2',700)]):
        if i:
            result.append(r'\addlinespace')
        for j,architecture in enumerate(['anatomical','linear']):
            d=lookup[(subset,architecture)]
            result.append(row([r'\multirow{2}{*}{'+name+rf' ($n={count:,}$)'+'}' if j == 0 else '',
                               'Anatomical connectome' if architecture == 'anatomical' else 'Linear output layer',
                               *[pm(d[k+'_mean_percent'],d[k+'_sd_percent'],1) for k in ['balanced_accuracy','accuracy','macro_f1']]]))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partition-summary',type=Path,default=Path('6 Ki67/5 Outputs/shid-v2-partition/split_summary.json'))
    parser.add_argument('--binary-summary',type=Path,default=Path('6 Ki67/5 Outputs/paper/summary.json'))
    for name in ['joint-summary','v2-summary','cross-summary','finetune-summary','destination']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    destination=prepare_destination(args.destination)
    v2=read(args.v2_summary)
    tables={1:split_rows(read(args.partition_summary)),4:joint_rows(read(args.joint_summary)['rows'],v2['joint_groups']),
            7:binary_rows(read(args.binary_summary)['groups'],v2['binary_groups']),
            8:cross_rows(read(args.cross_summary)['summary']),13:fine_rows(read(args.finetune_summary)['table'])}
    for number,rows in tables.items():
        (destination/f'Table-{number}.tex').write_text('\n'.join([HEADERS[number],*rows,FOOTERS[number]])+'\n')
    sources={key:{'path':portable_path(value),'sha256':sha256(value)} for key,value in vars(args).items() if key != 'destination'}
    (destination/'provenance.json').write_text(json.dumps({'tables':list(tables),'sources':sources},indent=2)+'\n')


if __name__ == '__main__':
    main()
