from pathlib import Path
import csv,json,hashlib,statistics,math
r=Path(__file__).resolve().parent;o=r/'results'
b=(r/'data/train.csv').read_bytes()
assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()=='a4d3682aa8ebde2035f3b9a8219ab337572eb0d6'
rows=list(csv.DictReader((o/'test_predictions.csv').open()))
sumr={x['model']:x for x in csv.DictReader((o/'summary.csv').open())}
folds=list(csv.DictReader((o/'fold_scores.csv').open()))
for name in sumr:
 tp=sum(int(x['y_true'])==1 and int(x[name])==1 for x in rows)
 fp=sum(int(x['y_true'])==0 and int(x[name])==1 for x in rows)
 fn=sum(int(x['y_true'])==1 and int(x[name])==0 for x in rows)
 f=2*tp/(2*tp+fp+fn)
 assert math.isclose(f,float(sumr[name]['Test_F1']),abs_tol=1e-12)
 values=[float(x[name]) for x in folds]
 assert math.isclose(statistics.mean(values),float(sumr[name]['CV_F1']),abs_tol=1e-12)
 assert math.isclose(statistics.stdev(values),float(sumr[name]['CV_std']),abs_tol=1e-12)
for name in ['grid','random']:
 cv=list(csv.DictReader((o/f'{name}_cv_results.csv').open()));assert len(cv)==48
 assert math.isclose(max(float(x['mean_test_score']) for x in cv),float(sumr[name.title()]['CV_F1']),abs_tol=1e-12)
manifest=list(csv.DictReader((o/'split_manifest.csv').open()))
assert len(manifest)==891 and len({x['Id'] for x in manifest})==891
tr=[x for x in manifest if x['split']=='train'];te=[x for x in manifest if x['split']=='test']
assert {x['group'] for x in tr}.isdisjoint({x['group'] for x in te})
for k in range(1,6):
 assert {x['group'] for x in tr if x['cv_fold']==str(k)}.isdisjoint({x['group'] for x in tr if x['cv_fold']!=str(k)})
sel=json.loads((o/'selection_before_test.json').read_text());assert sel['selected']==max(sumr,key=lambda n:float(sumr[n]['CV_F1']))
assert {x['Id'] for x in te}=={x['Id'] for x in rows}
nb=json.loads((r/'lab03.ipynb').read_text());assert all(c['execution_count'] for c in nb['cells'] if c['cell_type']=='code')
assert not any(out['output_type']=='error' for c in nb['cells'] for out in c.get('outputs',[]))
import nbformat
nbformat.validate(nb)
print('PASS: source Git blob, split/group isolation, search counts and best scores, CV mean/std, independently recomputed test F1, chosen winner, executed notebook schema.')
