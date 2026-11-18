"""
incertidumbre_bootstrap_mcnemar.py
Reproduce, con las bases MDT anonimizadas, las predicciones por estudiante del modelo elegido
(logística con el promedio de S1 y S2) y calcula:
  - intervalos de Clopper-Pearson para sensibilidad y especificidad,
  - bootstrap percentil (2000 remuestreos de Test, semilla 2025) para MAE y AUC,
  - prueba exacta de McNemar frente a la regla de la empresa (nivel I/1 en S2).
Uso:  python incertidumbre_bootstrap_mcnemar.py .     (desde la carpeta 03_Analisis)
Requiere: numpy, pandas, scipy, openpyxl.
La logística replica la de scikit-learn (penalización L2, C = 1, variable estandarizada).
"""
import numpy as np, pandas as pd, sys
from scipy import stats, optimize
RUTA=sys.argv[1]
fr=[]
for y in (2025,2026):
    d=pd.read_excel(f'{RUTA}/MDT_Simulacros_Saber11_{y}_anonimizada.xlsx',sheet_name='MDT'); d['COHORTE']=y
    a=pd.read_excel(f'{RUTA}/MDT_Simulacros_Saber11_{y}_anonimizada.xlsx',sheet_name='ALERTAS')
    d['IMP_S1']=d.ID_ESTUDIANTE.isin(a[a.Alerta=='Simulacro 1 imputado'].ID_ESTUDIANTE)
    d['IMP_S2']=d.ID_ESTUDIANTE.isin(a[a.Alerta=='Simulacro 2 imputado'].ID_ESTUDIANTE)
    fr.append(d)
D=pd.concat(fr,ignore_index=True); C=D[~(D.IMP_S1|D.IMP_S2)].copy()
TR=C[C.COHORTE==2025].reset_index(drop=True); TE=C[C.COHORTE==2026].reset_index(drop=True)
T='SB11_GLOBAL'
TR['R']=(TR[T]<=TR[T].quantile(.25)).astype(int); TE['R']=(TE[T]<=TE[T].quantile(.25)).astype(int)
print('n',len(TR),len(TE),'pos',TR.R.sum(),TE.R.sum())
pr=lambda d:(d.S1_GLOBAL+d.S2_GLOBAL)/2
print('r S1S2 pooled',round(C.S1_GLOBAL.corr(C.S2_GLOBAL),3),'train',round(TR.S1_GLOBAL.corr(TR.S2_GLOBAL),3),'test',round(TE.S1_GLOBAL.corr(TE.S2_GLOBAL),3))
# linear
x=pr(TR); b1,b0=np.polyfit(x,TR[T],1); pte=b0+b1*pr(TE); print('lin',round(b0,3),round(b1,4),'MAE test',round(np.mean(abs(TE[T]-pte)),3))
# logistic sklearn-like: standardized x, L2 C=1 on weight, intercept free
def fitlog(x,y):
    m,s=x.mean(),x.std(); z=(x-m)/s
    def f(p):
        w,b=p; t=w*z+b; return np.sum(np.logaddexp(0,t)-y*t)+0.5*w*w
    p=optimize.minimize(f,[0,0],method='BFGS',options={'gtol':1e-10}).x
    return lambda xx: 1/(1+np.exp(-(p[0]*(xx-m)/s+p[1])))
f=fitlog(pr(TR).values,TR.R.values); p=f(pr(TE).values)
from scipy.stats import mannwhitneyu
def auc(y,s): y=np.asarray(y); s=np.asarray(s); return mannwhitneyu(s[y==1],s[y==0]).statistic/((y==1).sum()*(y==0).sum())
print('AUC test',round(auc(TE.R,p),3),'train',round(auc(TR.R,f(pr(TR).values)),3))
fi=fitlog(pr(TE).values,TE.R.values); print('AUC inverso',round(auc(TR.R,fi(pr(TR).values)),3))
yh=(p>=0.232).astype(int); ex=(TE.S2_DESEMPEÑO=='I/1').astype(int)
y=TE.R.values
def cm(y,h): return dict(TP=int(((h==1)&(y==1)).sum()),FP=int(((h==1)&(y==0)).sum()),FN=int(((h==0)&(y==1)).sum()),TN=int(((h==0)&(y==0)).sum()))
print('modelo',cm(y,yh),'empresa',cm(y,ex.values))
# McNemar on correctness
cm_=(yh==y); ce=(ex.values==y)
b=int((cm_&~ce).sum()); c=int((~cm_&ce).sum()); print('McNemar discordantes: modelo acierta/empresa falla',b,'; al revés',c)
pm=stats.binomtest(min(b,c),b+c,0.5).pvalue; print('McNemar exacto p',round(pm,4))
# also only among no-riesgo (FP) and riesgo
for g in (0,1):
    k=y==g; bb=int((cm_&~ce&k).sum()); cc=int((~cm_&ce&k).sum()); print(' grupo',g,'b',bb,'c',cc, 'p',round(stats.binomtest(min(bb,cc),bb+cc,.5).pvalue,4) if bb+cc else None)
# bootstrap
rng=np.random.default_rng(2025); B=2000; maes=[];aucs=[];daucs=[];dmae=[]
e_lin=abs(TE[T].values-pte.values); e_exp=abs(TE[T].values-TE.S2_GLOBAL.values)
n=len(TE)
for _ in range(B):
    i=rng.integers(0,n,n); yy=y[i]
    maes.append(e_lin[i].mean()); dmae.append(e_exp[i].mean()-e_lin[i].mean())
    if 0<yy.sum()<n:
        a1=auc(yy,p[i]); a2=auc(yy,ex.values[i].astype(float)); aucs.append(a1); daucs.append(a1-a2)
q=lambda v:[round(x,3) for x in np.percentile(v,[2.5,97.5])]
print('MAE lin',round(e_lin.mean(),2),'IC',q(maes)); print('dif MAE empresa-lin IC',q(dmae))
print('AUC IC',q(aucs),'dif AUC modelo-empresa IC',q(daucs), 'n boot auc',len(aucs))
# Clopper-Pearson
def cp(k,nn,a=.05):
    lo=0.0 if k==0 else stats.beta.ppf(a/2,k,nn-k+1); hi=1.0 if k==nn else stats.beta.ppf(1-a/2,k+1,nn-k); return round(lo,2),round(hi,2)
for nm,(tp,fn,tn,fp) in {'base':(0,13,39,0),'empresa':(11,2,19,20),'modelo':(13,0,26,13),'L2':(12,1,27,12),'RF':(10,3,28,11)}.items():
    print(nm,'sens',round(tp/(tp+fn),2),cp(tp,tp+fn),'espec',round(tn/(tn+fp),2),cp(tn,tn+fp))
# priority groups test
pa=pr(TE); print('alta',(pa<=182.459).sum(),'media',((pa>182.459)&(pa<=202.434)).sum(),'baja',(pa>202.434).sum())
