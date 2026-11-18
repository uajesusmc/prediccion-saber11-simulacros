# interpretabilidad_coeficientes.py: coeficientes, IC y valores p (OLS y logística sin penalización) y coeficientes de Ridge por área, calculados en Train.
# Uso: python interpretabilidad_coeficientes.py .   (desde 03_Analisis). Requiere numpy, pandas, scipy, openpyxl.
import numpy as np, pandas as pd, sys
from scipy import stats, optimize
RUTA=sys.argv[1]; fr=[]
for y in (2025,2026):
    d=pd.read_excel(f'{RUTA}/MDT_Simulacros_Saber11_{y}_anonimizada.xlsx',sheet_name='MDT'); d['COHORTE']=y
    a=pd.read_excel(f'{RUTA}/MDT_Simulacros_Saber11_{y}_anonimizada.xlsx',sheet_name='ALERTAS')
    d['IMP']=d.ID_ESTUDIANTE.isin(a[a.Alerta.isin(['Simulacro 1 imputado','Simulacro 2 imputado'])].ID_ESTUDIANTE); fr.append(d)
D=pd.concat(fr,ignore_index=True); C=D[~D.IMP]
TR=C[C.COHORTE==2025].copy(); T='SB11_GLOBAL'
TR['S']=(TR.S1_GLOBAL+TR.S2_GLOBAL)/2; TR['R']=(TR[T]<=TR[T].quantile(.25)).astype(int)

def ols(X,y):
    X=np.column_stack([np.ones(len(y)),X]); n,k=X.shape
    b=np.linalg.lstsq(X,y,rcond=None)[0]; r=y-X@b; s2=r@r/(n-k); V=s2*np.linalg.inv(X.T@X); se=np.sqrt(np.diag(V))
    t=b/se; p=2*stats.t.sf(abs(t),n-k); q=stats.t.ppf(.975,n-k); R2=1-r@r/((y-y.mean())@(y-y.mean()))
    return b,se,p,np.column_stack([b-q*se,b+q*se]),R2
b,se,p,ci,R2=ols(TR[['S']].values,TR[T].values); print('OLS S̄: b',b.round(3),'se',se.round(3),'p',p,'CI',ci.round(3),'R2',round(R2,3))
b,se,p,ci,R2=ols(TR[['S1_GLOBAL','S2_GLOBAL']].values,TR[T].values); print('OLS S1,S2: b',b.round(3),'p',p.round(4),'CI',ci.round(3),'R2',round(R2,3))
def logit(x,y):
    X=np.column_stack([np.ones(len(y)),x])
    f=lambda w: np.sum(np.logaddexp(0,X@w)-y*(X@w))
    w=optimize.minimize(f,np.zeros(2),method='BFGS',options={'gtol':1e-10}).x
    pp=1/(1+np.exp(-X@w)); H=X.T@(X*(pp*(1-pp))[:,None]); se=np.sqrt(np.diag(np.linalg.inv(H))); z=w/se
    return w,se,2*stats.norm.sf(abs(z))
w,se,p=logit(TR['S'].values,TR.R.values); print('logit b',w.round(4),'se',se.round(4),'p',p,'OR10',round(np.exp(10*w[1]),3),'CI',np.exp(10*(w[1]+np.array([-1.96,1.96])*se[1])).round(3))
areas=[f'S{k}_{a}' for k in (1,2) for a in ['LECTURA','MATEMATICAS','SOCIALES','NATURALES','INGLES']]
Xs=(TR[areas]-TR[areas].mean())/TR[areas].std(ddof=0); yc=TR[T]-TR[T].mean()
best=None
for al in np.logspace(-2,3,30):
    Xm=Xs.values; H=Xm@np.linalg.solve(Xm.T@Xm+al*np.eye(10),Xm.T); r=yc.values-H@yc.values; loo=np.mean((r/(1-np.diag(H)))**2)
    if best is None or loo<best[0]: best=(loo,al)
al=best[1]; w=np.linalg.solve(Xs.values.T@Xs.values+al*np.eye(10),Xs.values.T@yc.values)
print('ridge alpha',round(al,2)); print(sorted(zip(areas,np.round(w,2)),key=lambda t:-abs(t[1])))
# check ridge test MAE replicates 16.39
TE=C[C.COHORTE==2026]; Xt=(TE[areas]-TR[areas].mean())/TR[areas].std(ddof=0); pr=Xt.values@w+TR[T].mean(); print('ridge MAE test',round(np.mean(abs(TE[T].values-pr)),2))
