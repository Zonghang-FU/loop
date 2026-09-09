# -*- coding: utf-8 -*-
"""
=========================================================================
 GM  -  RECIPE 2/3 : APPLICATION   (independante de la recipe 3)
=========================================================================
 INPUTS : BASE_ARTICLE  (dataset)  +  GM_MODELS  (managed folder)
 OUTPUT : GM_PREDICTIONS           (dataset)

 Le SCHEMA DE SORTIE est identique quel que soit DECODAGE : n'importe
 quelle version de la recipe 3 sait lire ce dataset.

 DECODAGE = 'independant'   chaque modele prend son argmax dans son coin
 DECODAGE = 'joint'         decodage hierarchique conjoint (voir plus bas)

 --- Decodage joint ---------------------------------------------------
 Les 3 modeles restent inchanges (aucun re-entrainement). On remplace les
 3 argmax independants par un choix de CHEMIN coherent niv1>niv2>niv3.
 Chaque classe fine implique ses parents par troncature, donc on peut
 noter un chemin complet :

   P3(c3) = w3.Z3[c3] + w2.Z2[c3[:3]] + w1.Z1[c3[:1]]
   P2(c2) = w2.Z2[c2] + w1.Z1[c2[:1]] + w3.max{Z3[c3] : c3[:3]==c2}
   P1(c1) = w1.Z1[c1] + w2.max{Z2[c2] : c2[:1]==c1}
                      + w3.max{Z3[c3] : c3[:1]==c1}

 Z = scores centres-reduits par ligne (les 3 modeles n'ont pas la meme
 echelle de decision_function : LinearSVC vs SGD).
 On prend n1 = argmax P1, puis les top-3 de niv2 et niv3 RESTREINTS a la
 descendance de n1 (repli sans contrainte si la descendance est vide).

 Motivation chiffree (backtest precedent, niveau 3) :
   lignes coherentes   85.2%  ->  top1 0.8424 / top3 0.9385
   lignes incoherentes 14.8%  ->  top1 0.2773 / top3 0.6104
 La perte est concentree sur les lignes ou les 3 modeles se contredisent.
=========================================================================
"""
import io, os, re, sys, time, unicodedata
import numpy as np
import pandas as pd
import joblib
import dataiku
from scipy.sparse import hstack

# ============================== CONFIG ==============================
IN_DATASET  = "BASE_ARTICLE"
IN_FOLDER   = "GM_MODELS"
OUT_DATASET = "GM_PREDICTIONS"

DECODAGE    = 'joint'          # 'joint' ou 'independant'
POIDS       = {'niv1': 1.0, 'niv2': 1.0, 'niv3': 1.0}   # w1, w2, w3 (a ajuster)
TEMPERATURE = 1.0              # >1 = scores adoucis, <1 = plus tranches
CHUNK       = 25000            # 50000 si DECODAGE='independant'

# On score TOUTES les lignes de BASE_ARTICLE, sans aucun filtre :
#   - quel que soit le contenu de MAKTX (meme vide ou aberrant)
#   - que MATKL soit deja renseigne ou non
# Le tri se fait en aval : la recipe 3 exclut les lignes non evaluables en
# s'appuyant sur les drapeaux maktx_ok / mtart_ok produits ici.
# MTART a exclure du calcul des metrics (aucune valeur en dur par defaut).
MTART_EXCLUS = set()           # ex. {'PROD'} si ce type sort du perimetre

LEVELS = ['niv1', 'niv2', 'niv3']
TOPK   = {'niv1': 1, 'niv2': 3, 'niv3': 3}

# --- Codes MATKL invalides / placeholders -----------------------------------
CODES_INVALIDES = {
    '', '0', '000', '00000',
    'XXX', 'YYY', 'ZZZ',
    'NA', 'N/A', 'NAN', 'NONE', 'NULL',
    '0M5',
    'Z00000',
}

GARBAGE = {'','test','testspn','test spn','na','n a','xxx','xxxxx','yyy','zzz',
           'sans','divers','autre','autres','neant','reserve','a definir',
           'article non defini','sans designation'}
RE_FILL  = re.compile(r'[.\-_*=~/#+]{2,}')
RE_PUNCT = re.compile(r'[^a-z0-9 ]')
RE_SP    = re.compile(r'\s+')

def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m)); sys.stdout.flush()

# ===================== HELPERS (identiques a la recipe 1) =============
def norm_text(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ''
    s = unicodedata.normalize('NFKD', str(s).lower())
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return RE_SP.sub(' ', RE_PUNCT.sub(' ', RE_FILL.sub(' ', s))).strip()

def matkl_valide(m):
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return None
    m = str(m).strip().upper()
    if m in CODES_INVALIDES:
        return None
    return m

def split_levels(m):
    m = matkl_valide(m)
    if m is None:
        return (None, None, None)
    return (m[:1], m[:3] if len(m) >= 3 else None, m if len(m) == 5 else None)

def prepare(df):
    df = df.copy()
    df['lib']   = df['MAKTX'].map(norm_text)
    df['MTART'] = df['MTART'].fillna('NA').astype(str).str.strip()
    lv = df['MATKL'].map(split_levels)
    df['niv1'] = [x[0] for x in lv]
    df['niv2'] = [x[1] for x in lv]
    df['niv3'] = [x[2] for x in lv]
    df['desc_ok'] = ((~df['lib'].isin(GARBAGE)) & (df['lib'].str.len() > 3) &
                     (~df['lib'].str.replace(' ', '', regex=False).str.isdigit()))
    return df

def folder_read(folder, name):
    try:
        return joblib.load(os.path.join(folder.get_path(), name))
    except Exception:
        with folder.get_download_stream(name) as f:
            return joblib.load(io.BytesIO(f.read()))

def topk(M, classes, k):
    """k colonnes ordonnees par score DECROISSANT.
    Les positions sans candidat valide valent None : soit parce que le modele
    a moins de k classes, soit -- cas du decodage contraint -- parce que la
    descendance retenue en compte moins de k. Ne JAMAIS remonter une classe de
    score -inf : elle est hors contrainte."""
    n, c = M.shape
    kk = min(k, c)
    idx = np.argpartition(-M, kth=kk - 1, axis=1)[:, :kk] if c > kk else \
          np.tile(np.arange(c), (n, 1))
    rows = np.arange(n)[:, None]
    idx = idx[rows, np.argsort(-M[rows, idx], axis=1)]
    out = np.asarray(classes)[idx].astype(object)
    out[~np.isfinite(M[rows, idx])] = None          # candidats hors contrainte
    if kk < k:
        out = np.concatenate([out, np.full((n, k - kk), None, dtype=object)], axis=1)
    return [out[:, i] for i in range(k)]

def marge(M):
    """Ecart top1 - top2, en ignorant les candidats hors contrainte (-inf).
    NaN quand il n'existe qu'un seul candidat valide."""
    if M.shape[1] < 2:
        return np.full(M.shape[0], np.nan)
    p = np.sort(M, axis=1)
    t1, t2 = p[:, -1], p[:, -2]
    out = t1 - t2
    out[~np.isfinite(t2)] = np.nan
    return out

def log_softmax(M, T=1.0):
    """decision_function -> log-probabilite. INDISPENSABLE : c'est la seule
    normalisation qui rende les 3 niveaux comparables.
    Un z-score ecraserait l'amplitude (un niv1 tres sur et un niv1 hesitant
    donnent le meme +-1) et avantagerait mecaniquement le niveau ayant le plus
    de classes -- donc niv3, le moins fiable des trois."""
    M = np.asarray(M, dtype=float) / T
    m = M.max(axis=1, keepdims=True)
    e = np.exp(M - m)
    return (M - m) - np.log(e.sum(axis=1, keepdims=True))

# ============================ CHARGEMENT =============================
log("chargement du modele depuis %s" % IN_FOLDER)
folder     = dataiku.Folder(IN_FOLDER)
vecs       = folder_read(folder, "vectorizers.joblib")
dico_train = folder_read(folder, "dico_train.joblib")
dico_full  = folder_read(folder, "dico_full.joblib")
matnr_test = folder_read(folder, "matnr_test.joblib")
meta       = folder_read(folder, "meta.joblib")
models     = {L: folder_read(folder, "model_%s.joblib" % L) for L in LEVELS}
PUR_L0     = meta['pur_l0']
log("modele du %s | decodage = %s" % (meta['date_entrainement'], DECODAGE))

CLS = {L: np.asarray(models[L].classes_) for L in LEVELS}
log("classes : niv1=%d niv2=%d niv3=%d" % tuple(len(CLS[L]) for L in LEVELS))

# --- table de parente (une fois pour toutes) --------------------------
idx1 = {c: i for i, c in enumerate(CLS['niv1'])}
idx2 = {c: i for i, c in enumerate(CLS['niv2'])}
par2_1 = np.array([idx1.get(c[:1], -1) for c in CLS['niv2']])
par3_2 = np.array([idx2.get(c[:3], -1) for c in CLS['niv3']])
par3_1 = np.array([idx1.get(c[:1], -1) for c in CLS['niv3']])
enfants2_de_1 = [np.where(par2_1 == j)[0] for j in range(len(CLS['niv1']))]
enfants3_de_1 = [np.where(par3_1 == j)[0] for j in range(len(CLS['niv1']))]
enfants3_de_2 = [np.where(par3_2 == j)[0] for j in range(len(CLS['niv2']))]
orphelins = int((par3_2 < 0).sum() + (par2_1 < 0).sum() + (par3_1 < 0).sum())
if orphelins:
    log("  %d classes sans parent dans le modele superieur (contribution nulle)"
        % orphelins)

log("chargement de %s" % IN_DATASET)
df = dataiku.Dataset(IN_DATASET).get_dataframe(
        columns=['MATNR', 'MTART', 'MATKL', 'MAKTX'])
df = prepare(df)
df['MATNR_s'] = df['MATNR'].astype(str)

# drapeaux de qualite : calcules ici, appliques (ou non) en aval
df['maktx_ok'] = df['desc_ok'].values
df['mtart_ok'] = ((df['MTART'].str.strip() != '') & (df['MTART'] != 'NA')
                  & (~df['MTART'].isin(MTART_EXCLUS))).values

# population : purement informatif, ne filtre rien
est_holdout = df['MATNR_s'].isin(matnr_test) if len(matnr_test) else False
df['split'] = np.where(est_holdout, 'HOLDOUT',
               np.where(df.niv1.notna(), 'ETIQUETE', 'A_COMPLETER'))

S = df.reset_index(drop=True)          # <-- AUCUN filtre : on score tout
log("a scorer : %d lignes (totalite de %s)" % (len(S), IN_DATASET))
log("  population : %s" % S['split'].value_counts().to_dict())
log("  maktx_ok %.1f%% | mtart_ok %.1f%%"
    % (100 * S['maktx_ok'].mean(), 100 * S['mtart_ok'].mean()))

# ====================== SCORING PAR PAQUETS ==========================
def gather(Z, parents):
    """Z[:, parents]. Les classes orphelines (parent absent du modele
    superieur) recoivent la log-proba uniforme -log(n) : neutre, ni bonus
    ni penalite."""
    neutre = -np.log(Z.shape[1])
    out = np.full((Z.shape[0], len(parents)), neutre)
    ok = parents >= 0
    out[:, ok] = Z[:, parents[ok]]
    return out

def maxima_enfants(Z, groupes, n_parents):
    """max-marginal (style Viterbi) : meilleur descendant de chaque parent.
    Parent sans descendance -> log-proba uniforme."""
    out = np.full((Z.shape[0], n_parents), -np.log(Z.shape[1]))
    for j, k in enumerate(groupes):
        if len(k):
            out[:, j] = Z[:, k].max(axis=1)
    return out

res = {L: {'top': [[] for _ in range(TOPK[L])], 'marge': []} for L in LEVELS}
cmp_indep = {L: [] for L in LEVELS}          # pour le controle joint vs independant

w1, w2, w3 = POIDS['niv1'], POIDS['niv2'], POIDS['niv3']
t0 = time.time()
for start in range(0, len(S), CHUNK):
    part = S.iloc[start:start + CHUNK]
    Xp = hstack([vecs['word'].transform(part['lib']),
                 vecs['char'].transform(part['lib']),
                 vecs['mtart'].transform(part[['MTART']]) * 0.5]).tocsr()

    brut = {}
    for L in LEVELS:
        sc = models[L].decision_function(Xp)
        if sc.ndim == 1:
            sc = np.c_[-sc, sc]
        brut[L] = sc

    # reference independante (toujours calculee, pour le controle)
    for L in LEVELS:
        cmp_indep[L].append(topk(brut[L], CLS[L], 1)[0])

    if DECODAGE == 'independant':
        final = {L: brut[L] for L in LEVELS}
        contrainte = None
    else:
        Z1 = log_softmax(brut['niv1'], TEMPERATURE)
        Z2 = log_softmax(brut['niv2'], TEMPERATURE)
        Z3 = log_softmax(brut['niv3'], TEMPERATURE)
        P3 = w3 * Z3 + w2 * gather(Z2, par3_2) + w1 * gather(Z1, par3_1)
        P2 = (w2 * Z2 + w1 * gather(Z1, par2_1)
              + w3 * maxima_enfants(Z3, enfants3_de_2, len(CLS['niv2'])))
        P1 = (w1 * Z1
              + w2 * maxima_enfants(Z2, enfants2_de_1, len(CLS['niv1']))
              + w3 * maxima_enfants(Z3, enfants3_de_1, len(CLS['niv1'])))
        final = {'niv1': P1, 'niv2': P2, 'niv3': P3}
        contrainte = P1.argmax(axis=1)          # racine imposee aux niveaux fins

    for L in LEVELS:
        M = final[L]
        if contrainte is not None and L != 'niv1':
            par = par2_1 if L == 'niv2' else par3_1
            ok = (par[None, :] == contrainte[:, None])
            Mc = np.where(ok, M, -np.inf)
            vide = ~np.isfinite(Mc).any(axis=1)     # racine sans descendance
            if vide.any():
                Mc[vide] = M[vide]
            M = Mc
        cols = topk(M, CLS[L], TOPK[L])
        for i, c in enumerate(cols):
            res[L]['top'][i].append(c)
        res[L]['marge'].append(marge(M))

    if (start // CHUNK) % 5 == 0:
        log("  %d / %d (%ds)" % (start, len(S), time.time() - t0))

for L in LEVELS:
    res[L]['top']   = [np.concatenate(x) for x in res[L]['top']]
    res[L]['marge'] = np.concatenate(res[L]['marge'])
    cmp_indep[L]    = np.concatenate(cmp_indep[L])

# ====================== SORTIE + COUCHE L0 ===========================
OUT = pd.DataFrame({
    'MATNR': S['MATNR'].values, 'MTART': S['MTART'].values,
    'MAKTX': S['MAKTX'].values, 'libelle_norm': S['lib'].values,
    'split': S['split'].values, 'MATKL_reel': S['MATKL'].values,
    'maktx_ok': S['maktx_ok'].values, 'mtart_ok': S['mtart_ok'].values,
    'niv1_reel': S['niv1'].values, 'niv2_reel': S['niv2'].values,
    'niv3_reel': S['niv3'].values,
})

# dico_train pour les lignes du holdout (evite la fuite), dico_full sinon.
# Si HOLDOUT=0 a l'entrainement, les deux dictionnaires sont identiques.
log("couche L0 (match exact du libelle)")
est_test_v = (S['split'] == 'HOLDOUT').values
for L in LEVELS:
    k = TOPK[L]
    dt  = dico_train[L].reindex(S['lib'].values)
    dfu = dico_full[L].reindex(S['lib'].values)
    d = pd.DataFrame({c: np.where(est_test_v, dt[c].values, dfu[c].values)
                      for c in ['top1', 'top2', 'top3', 'n_tot', 'purete']})
    pur = pd.to_numeric(d['purete'], errors='coerce').fillna(0).values
    hit = pd.notna(d['top1']).values & (pur >= PUR_L0)
    for i in range(k):
        col = ('%s_pred' % L) if k == 1 else ('%s_top%d' % (L, i + 1))
        OUT[col] = np.where(hit, d['top%d' % (i + 1)].values, res[L]['top'][i])
    OUT['%s_source' % L]     = np.where(hit, 'L0', 'ML')
    OUT['%s_confiance' % L]  = np.where(hit, pur, np.nan)
    OUT['%s_marge_ml' % L]   = res[L]['marge']
    log("  %s : L0 %.1f%% / ML %.1f%%" % (L, 100 * hit.mean(), 100 * (1 - hit.mean())))

OUT['decodage'] = DECODAGE

p1 = OUT['niv1_pred'].astype(str)
p2 = OUT['niv2_top1'].astype(str)
p3 = OUT['niv3_top1'].astype(str)
OUT['coherent'] = (p3.str[:1] == p1) & (p3.str[:3] == p2) & (p2.str[:1] == p1)
OUT['action'] = np.where(OUT['coherent'] & (OUT['niv1_source'] == 'L0'), 'AUTO_NIV3',
                 np.where(OUT['coherent'], 'AUTO_NIV1_PROPOSE_NIV3', 'A_VALIDER'))
log("coherence : %.1f%%" % (100 * OUT['coherent'].mean()))

# ---- controle immediat : joint vs independant -----------------------
# Sur le holdout s'il existe, sinon sur toutes les lignes etiquetees (dans ce
# cas le chiffre mesure la reproduction du referentiel, pas la generalisation).
m = est_test_v if est_test_v.sum() > 0 else (S['split'] == 'ETIQUETE').values
m = m & S['maktx_ok'].values
if m.sum() > 0:
    quoi = "holdout" if est_test_v.sum() > 0 else "lignes etiquetees (VUES a l'entrainement)"
    log("--- controle sur %d %s : couche ML seule, hors L0 ---" % (m.sum(), quoi))
    for L in LEVELS:
        vrai = S[L].values[m]
        dispo = pd.notna(vrai)
        if dispo.sum() < 100:
            continue
        a_ind = float((cmp_indep[L][m][dispo] == vrai[dispo]).mean())
        a_fin = float((res[L]['top'][0][m][dispo] == vrai[dispo]).mean())
        log("  %s : independant %.4f | %-11s %.4f | ecart %+.4f  (n=%d)"
            % (L, a_ind, DECODAGE, a_fin, a_fin - a_ind, dispo.sum()))

dataiku.Dataset(OUT_DATASET).write_with_schema(OUT)
log("APPLICATION TERMINEE : %d lignes -> %s" % (len(OUT), OUT_DATASET))





# -*- coding: utf-8 -*-
"""
=========================================================================
 GM  -  RECIPE 3/3 : METRICS      (independante de la recipe 2)
=========================================================================
 INPUT   : GM_PREDICTIONS
 OUTPUTS : GM_METRICS_SYNTHESE / GM_METRICS_PAR_RANG / GM_METRICS_PAR_SOURCE

 Ne lit que des colonnes produites par TOUTE version de la recipe 2
 (decodage independant OU decodage joint) -> les deux recipes restent
 interchangeables. Si les drapeaux maktx_ok / mtart_ok sont absents du
 dataset, ils sont recalcules ici.

 PERIMETRE EVALUE
 ----------------
 Toutes les lignes ayant :
   - un MAKTX exploitable   (hors liste noire, non vide, > 3 caracteres,
                             pas uniquement numerique)
   - un MTART exploitable   (non vide, hors MTART_EXCLUS)
   - un MATKL reel VALIDE   (sinon pas de verite terrain a comparer)
 Aucun filtre sur la population : les lignes deja etiquetees sont evaluees
 comme les autres.

 ATTENTION A LA LECTURE
 ----------------------
 Si l'entrainement a tourne avec HOLDOUT=0, le modele ET le dictionnaire L0
 ont vu chacune de ces lignes. Le chiffre obtenu mesure alors la CAPACITE A
 REPRODUIRE le referentiel existant (utile comme audit de coherence des
 saisies), et NON la performance attendue sur des articles nouveaux.
 La colonne `population` distingue HOLDOUT (jamais vu) de ETIQUETE (vu).

 DEUX MODES DE COMPTAGE, calcules cote a cote
 --------------------------------------------
 [hierarchique]  <- le mode demande, denominateur COMMUN aux 3 niveaux
   On compare a la profondeur  min(profondeur du niveau, profondeur de la
   verite).  Un article saisi seulement en 'C' est donc evaluable AUSSI au
   niveau 2 et au niveau 3 : il suffit que le niveau 1 contenu dans la
   prediction soit bon.
       verite 'C0101' -> niv3 juge sur 5 car., niv2 sur 3 car., niv1 sur 1
       verite 'C01'   -> niv3 juge sur 3 car., niv2 sur 3 car., niv1 sur 1
       verite 'C'     -> niv3 juge sur 1 car., niv2 sur 1 car., niv1 sur 1
   => n_evalues identique pour niv1, niv2 et niv3.

 [strict]  <- l'ancien mode, conserve pour comparaison
   Chaque niveau n'est evalue que sur les articles ayant une verite A CE
   niveau, et l'egalite doit etre complete. Denominateurs differents.

 Dans les deux modes : top2 / top3 sont CUMULATIFS.
 Ne sont evaluees que les lignes split == 'TEST' avec un MATKL reel valide.
 TAUX GLOBAL = bonne prediction du niveau 1 (regle metier retenue).
=========================================================================
"""
import sys, time
import numpy as np
import pandas as pd
import dataiku

IN_DATASET   = "GM_PREDICTIONS"
OUT_SYNTHESE = "GM_METRICS_SYNTHESE"
OUT_RANG     = "GM_METRICS_PAR_RANG"
OUT_SOURCE   = "GM_METRICS_PAR_SOURCE"
MIN_N        = 200

NIVEAUX          = ['niv1', 'niv2', 'niv3']
NIVEAU_LONGUEUR  = {'niv1': 1, 'niv2': 3, 'niv3': 5}
TOPCOLS          = {'niv1': ['niv1_pred'],
                    'niv2': ['niv2_top1', 'niv2_top2', 'niv2_top3'],
                    'niv3': ['niv3_top1', 'niv3_top2', 'niv3_top3']}

def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m)); sys.stdout.flush()

# ============================ CHARGEMENT =============================
log("chargement de %s" % IN_DATASET)
P = dataiku.Dataset(IN_DATASET).get_dataframe()

# --- drapeaux de qualite : lus si presents, recalcules sinon -----------
GARBAGE = {'','test','testspn','test spn','na','n a','xxx','xxxxx','yyy','zzz',
           'sans','divers','autre','autres','neant','reserve','a definir',
           'article non defini','sans designation'}
MTART_EXCLUS = set()          # ex. {'PROD'} ; aucune valeur en dur par defaut

if 'maktx_ok' not in P.columns:
    import re, unicodedata
    RE_FILL, RE_PUNCT, RE_SP = (re.compile(r'[.\-_*=~/#+]{2,}'),
                                re.compile(r'[^a-z0-9 ]'), re.compile(r'\s+'))
    def _norm(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ''
        x = unicodedata.normalize('NFKD', str(x).lower())
        x = ''.join(c for c in x if not unicodedata.combining(c))
        return RE_SP.sub(' ', RE_PUNCT.sub(' ', RE_FILL.sub(' ', x))).strip()
    _l = P['MAKTX'].map(_norm)
    P['maktx_ok'] = ((~_l.isin(GARBAGE)) & (_l.str.len() > 3) &
                     (~_l.str.replace(' ', '', regex=False).str.isdigit()))
    log("colonne maktx_ok absente -> recalculee")
if 'mtart_ok' not in P.columns:
    _m = P['MTART'].fillna('').astype(str).str.strip()
    P['mtart_ok'] = (_m != '') & (_m != 'NA') & (~_m.isin(MTART_EXCLUS))
    log("colonne mtart_ok absente -> recalculee")
if 'split' not in P.columns:
    P['split'] = 'ETIQUETE'

garde = (P['maktx_ok'].fillna(False).astype(bool)
         & P['mtart_ok'].fillna(False).astype(bool)
         & P['niv1_reel'].notna()
         & (P['niv1_reel'].astype(str).str.strip() != ''))
E = P[garde].copy().reset_index(drop=True)
log("%d lignes -> %d evaluables" % (len(P), len(E)))
log("  exclues : maktx %d | mtart %d | MATKL absent ou invalide %d"
    % ((~P['maktx_ok'].fillna(False).astype(bool)).sum(),
       (~P['mtart_ok'].fillna(False).astype(bool)).sum(),
       P['niv1_reel'].isna().sum()))
POPULATIONS = sorted(E['split'].dropna().unique().tolist())
log("  populations : %s" % E['split'].value_counts().to_dict())
if 'HOLDOUT' not in POPULATIONS:
    log("  !! aucune ligne HOLDOUT : toutes les lignes evaluees ont ete VUES a")
    log("     l'entrainement -> ce chiffre mesure la reproduction du referentiel,")
    log("     pas la generalisation. Mettre HOLDOUT=0.20 dans la recipe 1 pour")
    log("     obtenir la mesure de generalisation.")

TOUTES = [c for v in TOPCOLS.values() for c in v] + \
         ['niv1_reel', 'niv2_reel', 'niv3_reel']
for c in TOUTES:
    E[c] = E[c].astype('string')

# verite la plus fine disponible + sa profondeur (1, 3 ou 5 caracteres)
E['verite'] = E['niv3_reel'].fillna(E['niv2_reel']).fillna(E['niv1_reel'])
E['profondeur'] = E['verite'].str.len().fillna(0).astype(int)
E['granularite'] = np.where(E['niv3_reel'].notna(), 'niv3',
                     np.where(E['niv2_reel'].notna(), 'niv2', 'niv1'))
log("profondeurs de la verite : %s" % E['profondeur'].value_counts().to_dict())

# ====================== COMPTAGE : LES DEUX MODES =====================
def hits_hierarchiques(sub, niveau):
    """Cumules [top1, top<=2, top<=3]. Comparaison a la profondeur
    min(profondeur du niveau, profondeur de la verite).
    Denominateur = TOUTES les lignes de `sub`."""
    prof_niv = NIVEAU_LONGUEUR[niveau]
    d = np.minimum(prof_niv, sub['profondeur'].values)
    ver = sub['verite']
    cum, out = np.zeros(len(sub), bool), []
    for col in TOPCOLS[niveau]:
        pred = sub[col]
        hit = np.zeros(len(sub), bool)
        for dd in np.unique(d):
            if dd <= 0:
                continue
            m = (d == dd)
            hit[m] = (pred[m].str[:int(dd)] == ver[m].str[:int(dd)]).fillna(False).values
        cum = cum | hit
        out.append(cum.copy())
    return out

def hits_stricts(sub, niveau):
    """Cumules [top1, top<=2, top<=3] sur les seules lignes ayant une verite
    A CE niveau, egalite complete. Retourne aussi le masque de disponibilite."""
    dispo = sub['%s_reel' % niveau].notna().values
    ver = sub['%s_reel' % niveau]
    cum, out = np.zeros(len(sub), bool), []
    for col in TOPCOLS[niveau]:
        cum = cum | ((sub[col] == ver).fillna(False).values & dispo)
        out.append(cum.copy())
    return out, dispo

# ============================== BOUCLE ================================
# perimetre = *TOUS* / chaque MTART ; puis *TOUS* decline par population
DECOUPES = ([('*TOUS*', '*TOUTES*')]
            + [(m, '*TOUTES*') for m in sorted(E['MTART'].dropna().unique().tolist())]
            + [('*TOUS*', p) for p in POPULATIONS])
rows_syn, rows_rang, rows_src = [], [], []

for per, pop in DECOUPES:
    selP = np.ones(len(E), bool) if per == '*TOUS*' else (E['MTART'] == per).values
    if pop != '*TOUTES*':
        selP = selP & (E['split'] == pop).values
    if selP.sum() < MIN_N:
        continue
    sub = E[selP]
    n = len(sub)
    syn = dict(perimetre=per, population=pop, n_evalues=int(n))

    for niveau in NIVEAUX:
        # ---------- mode hierarchique : denominateur commun -----------
        accH = hits_hierarchiques(sub, niveau)
        for r, h in enumerate(accH, 1):
            syn['h_%s_top%d' % (niveau, r)] = round(float(h.mean()), 4)
            rows_rang.append(dict(mode='hierarchique', niveau=niveau, perimetre=per,
                                  population=pop,
                                  rang='top%d' % r, n_evalues=int(n),
                                  accuracy=round(float(h.mean()), 4),
                                  gain_vs_rang_precedent=round(
                                      float(h.mean() - accH[r-2].mean()), 4) if r > 1 else None))
        syn['h_%s_n' % niveau] = int(n)          # identique pour les 3 niveaux

        # ---------- mode strict : ancien comptage ---------------------
        accS, dispo = hits_stricts(sub, niveau)
        nd = int(dispo.sum())
        syn['s_%s_n' % niveau] = nd
        if nd >= MIN_N:
            for r, h in enumerate(accS, 1):
                a = float(h[dispo].mean())
                syn['s_%s_top%d' % (niveau, r)] = round(a, 4)
                rows_rang.append(dict(mode='strict', niveau=niveau, perimetre=per,
                                      population=pop,
                                      rang='top%d' % r, n_evalues=nd,
                                      accuracy=round(a, 4),
                                      gain_vs_rang_precedent=round(
                                          float(a - accS[r-2][dispo].mean()), 4) if r > 1 else None))

        # ---------- d'ou vient la performance -------------------------
        col_src = '%s_source' % niveau
        if col_src in sub.columns:
            for src in sorted(sub[col_src].dropna().unique()):
                m = (sub[col_src].values == src)
                if m.sum() < MIN_N:
                    continue
                rows_src.append(dict(niveau=niveau, perimetre=per, population=pop,
                                     dimension='source',
                                     modalite=str(src),
                                     couverture=round(float(m.mean()), 4),
                                     accuracy_top1=round(float(accH[0][m].mean()), 4),
                                     accuracy_topk=round(float(accH[-1][m].mean()), 4),
                                     n=int(m.sum())))
        if 'coherent' in sub.columns:
            for coh in [True, False]:
                m = (sub['coherent'].values == coh)
                if m.sum() < MIN_N:
                    continue
                rows_src.append(dict(niveau=niveau, perimetre=per, population=pop,
                                     dimension='coherence',
                                     modalite='coherent' if coh else 'incoherent',
                                     couverture=round(float(m.mean()), 4),
                                     accuracy_top1=round(float(accH[0][m].mean()), 4),
                                     accuracy_topk=round(float(accH[-1][m].mean()), 4),
                                     n=int(m.sum())))

    # ---------- taux global : regle metier = niveau 1 bon -------------
    syn['TAUX_GLOBAL_niv1_ok'] = round(
        float((sub['niv1_pred'] == sub['niv1_reel']).fillna(False).mean()), 4)

    # ---------- accuracy a la granularite native ----------------------
    ok = np.zeros(n, bool)
    for g, col in [('niv1', 'niv1_pred'), ('niv2', 'niv2_top1'), ('niv3', 'niv3_top1')]:
        m = (sub['granularite'].values == g)
        if m.any():
            ok[m] = (sub[col][m] == sub['%s_reel' % g][m]).fillna(False).values
    syn['acc_granularite_native'] = round(float(ok.mean()), 4)
    for g in NIVEAUX:
        syn['part_saisie_%s' % g] = round(float((sub['granularite'].values == g).mean()), 4)
    rows_syn.append(syn)

SYN  = pd.DataFrame(rows_syn)
RANG = pd.DataFrame(rows_rang)
SRC  = pd.DataFrame(rows_src)

ordre = (['perimetre', 'population', 'n_evalues',
          'TAUX_GLOBAL_niv1_ok', 'acc_granularite_native']
         # h_*_n : denominateur du mode hierarchique. Les trois DOIVENT etre
         # egaux entre eux et egaux a n_evalues -- c'est la propriete visee.
         + ['h_%s_n' % L for L in NIVEAUX]
         + ['h_%s_top%d' % (L, r) for L in NIVEAUX for r in (1, 2, 3)
            if not (L == 'niv1' and r > 1)]
         + ['s_%s_n' % L for L in NIVEAUX]
         + ['s_%s_top%d' % (L, r) for L in NIVEAUX for r in (1, 2, 3)
            if not (L == 'niv1' and r > 1)]
         + ['part_saisie_%s' % L for L in NIVEAUX])
SYN = SYN[[c for c in ordre if c in SYN.columns]]
SYN = pd.concat([SYN[(SYN.perimetre == '*TOUS*') & (SYN.population == '*TOUTES*')],
                 SYN[SYN.population != '*TOUTES*'],
                 SYN[(SYN.perimetre != '*TOUS*')].sort_values('n_evalues', ascending=False)])

print("\n=== SYNTHESE (h_ = hierarchique, denominateur commun ; s_ = strict) ===")
print(SYN.to_string(index=False))
print("\n=== PAR RANG (*TOUS*) ===")
print(RANG[(RANG.perimetre == '*TOUS*') & (RANG.population == '*TOUTES*')].to_string(index=False))
print("\n=== PAR SOURCE / COHERENCE (*TOUS*) ===")
print(SRC[(SRC.perimetre == '*TOUS*') & (SRC.population == '*TOUTES*')].to_string(index=False))

dataiku.Dataset(OUT_SYNTHESE).write_with_schema(SYN)
dataiku.Dataset(OUT_RANG).write_with_schema(RANG)
dataiku.Dataset(OUT_SOURCE).write_with_schema(SRC)
log("METRICS TERMINEES")



































# -*- coding: utf-8 -*-
"""
=========================================================================
 GM  -  RECIPE 1/3 : ENTRAINEMENT
=========================================================================
 INPUT  : BASE_ARTICLE                (dataset)
 OUTPUT : GM_MODELS                   (managed folder)

 Le dossier est VIDE puis reecrit -> il ne contient jamais qu'un seul
 modele, le dernier entraine.

 Architecture retenue (validee par backtests v1..v4) :
   * 3 modeles INDEPENDANTS (niv1 / niv2 / niv3), pas de cascade reelle
   * chaine : L0 match exact du libelle  ->  ML
     -> L1b (MTART+mot1) SUPPRIME : contribution nette mesuree -0.58pt
        au niv1, +0.58pt au niv2, -0.06pt au niv3 = nulle.
   * pas de vote inter-niveaux : gain mesure -0.40pt au niv1.

 Deux dictionnaires L0 sont sauvegardes :
   dico_train : construit sur le TRAIN seul  -> sert a scorer le TEST
                (sinon fuite : le libelle de test serait dans le dico)
   dico_full  : construit sur TOUT l'etiquete -> sert a scorer la CIBLE
=========================================================================
"""
import io, os, re, sys, time, unicodedata, threading
import numpy as np
import pandas as pd
import joblib

import dataiku
from joblib import Parallel, delayed
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from scipy.sparse import hstack

# ============================== CONFIG ==============================
IN_DATASET     = "BASE_ARTICLE"
OUT_FOLDER     = "GM_MODELS"
RANDOM_STATE   = 42
TEST_SIZE      = 0.20
MAX_CHAR_FEAT  = 200000        # baisser a 100000 si MemoryError
PARALLEL       = True
LEVELS         = ['niv1', 'niv2', 'niv3']
MODEL_BY_LEVEL = {'niv1': 'linsvc', 'niv2': 'sgd', 'niv3': 'sgd'}
PUR_L0         = 0.90          # 0.99 n'apporte rien (+0.07pt pour -1.6pt de couv.)

# --- Codes MATKL invalides / placeholders -----------------------------------
# Liste EXPLICITE des valeurs rencontrees dans les donnees qui ne sont pas de
# vrais groupes de marchandise. A completer a la main si de nouveaux codes
# apparaissent : la recipe 1 journalise les candidats suspects non listes.
CODES_INVALIDES = {
    '', '0', '000', '00000',          # zeros de remplissage
    'XXX', 'YYY', 'ZZZ',              # placeholders de saisie
    'NA', 'N/A', 'NAN', 'NONE', 'NULL',
    '0M5',                            # erreur de saisie (59 articles)
    'Z00000',                         # erreur de saisie (1 article)
}

GARBAGE = {'','test','testspn','test spn','na','n a','xxx','xxxxx','yyy','zzz',
           'sans','divers','autre','autres','neant','reserve','a definir',
           'article non defini','sans designation'}
RE_FILL  = re.compile(r'[.\-_*=~/#+]{2,}')
RE_PUNCT = re.compile(r'[^a-z0-9 ]')
RE_SP    = re.compile(r'\s+')

_lock = threading.Lock()
def log(m):
    with _lock:
        print("[%s] %s" % (time.strftime("%H:%M:%S"), m)); sys.stdout.flush()

# ===================== HELPERS (dupliques dans les 3 recipes) =========
def norm_text(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ''
    s = unicodedata.normalize('NFKD', str(s).lower())
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return RE_SP.sub(' ', RE_PUNCT.sub(' ', RE_FILL.sub(' ', s))).strip()

def matkl_valide(m):
    """Normalise un MATKL. Retourne None si la valeur est un placeholder."""
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return None
    m = str(m).strip().upper()
    if m in CODES_INVALIDES:
        return None
    return m

def split_levels(m):
    """MATKL -> (niv1, niv2, niv3). Un code court ne renseigne que les
    niveaux grossiers ; les niveaux absents valent None."""
    m = matkl_valide(m)
    if m is None:
        return (None, None, None)
    return (m[:1],
            m[:3] if len(m) >= 3 else None,
            m      if len(m) == 5 else None)

def prepare(df):
    df = df.copy()
    df['lib']   = df['MAKTX'].map(norm_text)
    df['MTART'] = df['MTART'].fillna('NA').astype(str).str.strip()
    lv = df['MATKL'].map(split_levels)
    df['niv1'] = [x[0] for x in lv]
    df['niv2'] = [x[1] for x in lv]
    df['niv3'] = [x[2] for x in lv]
    df['desc_ok'] = ((~df['lib'].isin(GARBAGE)) & (df['lib'].str.len() > 3) &
                     (~df['lib'].str.replace(' ', '', regex=False).str.isdigit()))
    return df

def build_dico_top3(d, level, key='lib'):
    """key -> (top1, top2, top3) par frequence decroissante + purete du top1."""
    d = d[d[level].notna()]
    g = d.groupby([key, level]).size().rename('n').reset_index()
    g = g.sort_values([key, 'n'], ascending=[True, False])
    g['rk'] = g.groupby(key).cumcount() + 1
    tot = g.groupby(key)['n'].sum().rename('n_tot')
    out = g[g.rk == 1].set_index(key)[[level, 'n']].rename(
              columns={level: 'top1', 'n': 'n1'})
    for r in (2, 3):
        out = out.join(g[g.rk == r].set_index(key)[level].rename('top%d' % r))
    out = out.join(tot)
    out['purete'] = out['n1'] / out['n_tot']
    return out[['top1', 'top2', 'top3', 'n_tot', 'purete']]

def folder_write(folder, name, obj):
    try:                                     # dossier local -> ecriture directe
        joblib.dump(obj, os.path.join(folder.get_path(), name), compress=3)
    except Exception:                        # dossier distant -> flux
        buf = io.BytesIO(); joblib.dump(obj, buf, compress=3); buf.seek(0)
        folder.upload_stream(name, buf)

# ============================ CHARGEMENT =============================
log("chargement de %s" % IN_DATASET)
df = dataiku.Dataset(IN_DATASET).get_dataframe(
        columns=['MATNR', 'MTART', 'MATKL', 'MAKTX'])
df = prepare(df)
assert 'N' not in set(df['niv1'].dropna()), "classe fantome 'N' : MATKL NaN mal gere"

# --- auto-controle : de nouveaux placeholders sont-ils apparus ? -------------
# CODES_INVALIDES fait autorite. Ce controle ne filtre RIEN : il signale les
# codes qui ressemblent a des placeholders mais ne sont pas encore listes,
# pour qu'on les ajoute a la main a CODES_INVALIDES si c'en est.
_brut = df['MATKL'].dropna().astype(str).str.strip().str.upper()
_brut = _brut[_brut != '']
_longueurs = _brut.str.len().value_counts().sort_index().to_dict()
log("longueurs de MATKL observees : %s" % {int(k): int(v) for k, v in _longueurs.items()})

_reste = _brut[~_brut.isin(CODES_INVALIDES)]
_suspect = ((~_reste.str.len().isin([1, 3, 5]))                     # longueur inattendue
            | (~_reste.str[0].str.isalpha())                        # ne commence pas par une lettre
            | ((_reste.str.len() > 1)                               # un seul caractere repete
               & _reste.map(lambda x: len(set(x)) == 1)))
_candidats = _reste[_suspect].value_counts()
if len(_candidats):
    log("ATTENTION - codes suspects NON listes dans CODES_INVALIDES : %s"
        % {k: int(v) for k, v in _candidats.head(20).items()})
    log("  -> les ajouter a CODES_INVALIDES s'il s'agit bien de placeholders")
else:
    log("aucun nouveau code suspect : CODES_INVALIDES est a jour")

_mtart_vus = sorted(df['MTART'].dropna().unique().tolist())
log("types d'article (MTART) presents : %d -> %s" % (len(_mtart_vus), _mtart_vus))

lab = df[df.desc_ok & df.niv1.notna()].copy()
log("pool etiquete : %d lignes" % len(lab))
for L in LEVELS:
    log("  %s : %d lignes / %d classes" % (L, lab[L].notna().sum(), lab[L].nunique()))

# ====== SPLIT GROUPE PAR LIBELLE ======================================
# Obligatoire : 62% des lignes partagent leur libelle avec une autre ligne.
# Un split aleatoire mettrait le meme texte des deux cotes et gonflerait
# les scores de 10 a 20 points.
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
itr, ite = next(gss.split(lab, groups=lab['lib']))
trG, teG = lab.iloc[itr].copy(), lab.iloc[ite].copy()
log("split groupe : train %d / test %d (libelles disjoints)" % (len(trG), len(teG)))

# ====== DICTIONNAIRES L0 ==============================================
log("construction des dictionnaires L0")
dico_train, dico_full = {}, {}
for L in LEVELS:
    dico_train[L] = build_dico_top3(trG, L)
    dico_full[L]  = build_dico_top3(lab, L)
    log("  %s : dico_train %d cles | dico_full %d cles"
        % (L, len(dico_train[L]), len(dico_full[L])))

# ====== VECTORISATION (une seule fois, partagee par les 3 niveaux) =====
log("vectorisation")
t0 = time.time()
vec_word = TfidfVectorizer(analyzer='word', ngram_range=(1, 2),
                           min_df=2, sublinear_tf=True)
vec_char = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=3,
                           sublinear_tf=True, max_features=MAX_CHAR_FEAT)
enc_mtart = OneHotEncoder(handle_unknown='ignore')
Xtr = hstack([vec_word.fit_transform(trG['lib']),
              vec_char.fit_transform(trG['lib']),
              enc_mtart.fit_transform(trG[['MTART']]) * 0.5]).tocsr()
log("  X_train %s (%ds)" % (Xtr.shape, time.time() - t0))

# ====== ENTRAINEMENT DES 3 NIVEAUX ====================================
def train_level(L, n_jobs):
    m = trG[L].notna().values
    clf = (LinearSVC(C=0.5, class_weight='balanced', max_iter=3000)
           if MODEL_BY_LEVEL[L] == 'linsvc' else
           SGDClassifier(loss='modified_huber', alpha=1e-6, max_iter=25, tol=1e-4,
                         class_weight='balanced', random_state=RANDOM_STATE,
                         n_jobs=n_jobs))
    t = time.time()
    clf.fit(Xtr[m], trG.loc[m, L].values)
    log("  %s entraine : %d lignes, %d classes, %ds"
        % (L, m.sum(), len(clf.classes_), time.time() - t))
    return L, clf

log("entrainement des 3 niveaux (%s)" % ("parallele" if PARALLEL else "sequentiel"))
pairs = (Parallel(n_jobs=3, prefer="threads")(delayed(train_level)(L, 1) for L in LEVELS)
         if PARALLEL else [train_level(L, -1) for L in LEVELS])
models = dict(pairs)

# ====== SAUVEGARDE : on VIDE puis on ecrit ============================
folder = dataiku.Folder(OUT_FOLDER)
existants = folder.list_paths_in_partition()
log("nettoyage du dossier %s (%d fichiers)" % (OUT_FOLDER, len(existants)))
for p in existants:
    folder.delete_path(p)

meta = dict(date_entrainement=time.strftime("%Y-%m-%d %H:%M:%S"),
            n_train=len(trG), n_test=len(teG), levels=LEVELS,
            model_by_level=MODEL_BY_LEVEL, pur_l0=PUR_L0,
            max_char_feat=MAX_CHAR_FEAT, random_state=RANDOM_STATE,
            test_size=TEST_SIZE,
            codes_invalides=sorted(CODES_INVALIDES),
            longueurs_observees={int(k): int(v) for k, v in _longueurs.items()},
            mtart_vus=_mtart_vus,          # decouverts, jamais imposes
            classes={L: list(models[L].classes_) for L in LEVELS})

log("sauvegarde")
folder_write(folder, "vectorizers.joblib",
             dict(word=vec_word, char=vec_char, mtart=enc_mtart))
for L in LEVELS:
    folder_write(folder, "model_%s.joblib" % L, models[L])
folder_write(folder, "dico_train.joblib", dico_train)
folder_write(folder, "dico_full.joblib",  dico_full)
folder_write(folder, "matnr_test.joblib", set(teG['MATNR'].astype(str)))
folder_write(folder, "meta.joblib", meta)

log("fichiers ecrits : %s" % folder.list_paths_in_partition())
log("ENTRAINEMENT TERMINE")







# -*- coding: utf-8 -*-
"""
=========================================================================
 GM  -  RECIPE 2/3 : APPLICATION
=========================================================================
 INPUTS : BASE_ARTICLE  (dataset)  +  GM_MODELS  (managed folder)
 OUTPUT : GM_PREDICTIONS           (dataset)

 Scorer deux populations, marquees par la colonne `split` :
   TEST  : articles etiquetes mis de cote a l'entrainement
           -> sert a la recipe 3 (metrics).  Dictionnaire = dico_train,
              sinon le libelle de test serait dans son propre dico (fuite).
   CIBLE : articles a completer (MATKL vide ou invalide)
           -> la livraison metier.  Dictionnaire = dico_full.

 Sorties par niveau :
   niv1 -> UNE seule prediction
   niv2 -> top1 / top2 / top3, ordonnes par score decroissant
   niv3 -> top1 / top2 / top3, ordonnes par score decroissant
=========================================================================
"""
import io, os, re, sys, time, unicodedata
import numpy as np
import pandas as pd
import joblib
import dataiku
from scipy.sparse import hstack

# ============================== CONFIG ==============================
IN_DATASET  = "BASE_ARTICLE"
IN_FOLDER   = "GM_MODELS"
OUT_DATASET = "GM_PREDICTIONS"
CHUNK       = 50000        # decision_function par paquets (niv3 = 276 classes)
LEVELS      = ['niv1', 'niv2', 'niv3']
TOPK        = {'niv1': 1, 'niv2': 3, 'niv3': 3}
SCORE_TRAIN = False        # True -> scorer aussi les lignes d'entrainement

# --- Codes MATKL invalides / placeholders -----------------------------------
# Liste EXPLICITE des valeurs rencontrees dans les donnees qui ne sont pas de
# vrais groupes de marchandise. A completer a la main si de nouveaux codes
# apparaissent : la recipe 1 journalise les candidats suspects non listes.
CODES_INVALIDES = {
    '', '0', '000', '00000',          # zeros de remplissage
    'XXX', 'YYY', 'ZZZ',              # placeholders de saisie
    'NA', 'N/A', 'NAN', 'NONE', 'NULL',
    '0M5',                            # erreur de saisie (59 articles)
    'Z00000',                         # erreur de saisie (1 article)
}

GARBAGE = {'','test','testspn','test spn','na','n a','xxx','xxxxx','yyy','zzz',
           'sans','divers','autre','autres','neant','reserve','a definir',
           'article non defini','sans designation'}
RE_FILL  = re.compile(r'[.\-_*=~/#+]{2,}')
RE_PUNCT = re.compile(r'[^a-z0-9 ]')
RE_SP    = re.compile(r'\s+')

def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m)); sys.stdout.flush()

# ===================== HELPERS (identiques a la recipe 1) =============
def norm_text(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ''
    s = unicodedata.normalize('NFKD', str(s).lower())
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return RE_SP.sub(' ', RE_PUNCT.sub(' ', RE_FILL.sub(' ', s))).strip()

def matkl_valide(m):
    """Normalise un MATKL. Retourne None si la valeur est un placeholder."""
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return None
    m = str(m).strip().upper()
    if m in CODES_INVALIDES:
        return None
    return m

def split_levels(m):
    """MATKL -> (niv1, niv2, niv3). Un code court ne renseigne que les
    niveaux grossiers ; les niveaux absents valent None."""
    m = matkl_valide(m)
    if m is None:
        return (None, None, None)
    return (m[:1],
            m[:3] if len(m) >= 3 else None,
            m      if len(m) == 5 else None)

def prepare(df):
    df = df.copy()
    df['lib']   = df['MAKTX'].map(norm_text)
    df['MTART'] = df['MTART'].fillna('NA').astype(str).str.strip()
    lv = df['MATKL'].map(split_levels)
    df['niv1'] = [x[0] for x in lv]
    df['niv2'] = [x[1] for x in lv]
    df['niv3'] = [x[2] for x in lv]
    df['desc_ok'] = ((~df['lib'].isin(GARBAGE)) & (df['lib'].str.len() > 3) &
                     (~df['lib'].str.replace(' ', '', regex=False).str.isdigit()))
    return df

def folder_read(folder, name):
    try:
        return joblib.load(os.path.join(folder.get_path(), name))
    except Exception:
        with folder.get_download_stream(name) as f:
            return joblib.load(io.BytesIO(f.read()))

def top_k_from_scores(S, classes, k):
    """Retourne k colonnes ordonnees par score DECROISSANT + la marge top1-top2."""
    n, c = S.shape
    kk = min(k, c)
    idx = np.argpartition(-S, kth=kk - 1, axis=1)[:, :kk] if c > kk else \
          np.tile(np.arange(c), (n, 1))
    rows = np.arange(n)[:, None]
    idx = idx[rows, np.argsort(-S[rows, idx], axis=1)]        # tri strict
    out = classes[idx].astype(object)
    if kk < k:                                                # padding
        out = np.concatenate([out, np.full((n, k - kk), None, dtype=object)], axis=1)
    part = np.partition(S, -2, axis=1) if c >= 2 else None
    marge = (part[:, -1] - part[:, -2]) if c >= 2 else np.zeros(n)
    return [out[:, i] for i in range(k)], marge

# ============================ CHARGEMENT =============================
log("chargement du modele depuis %s" % IN_FOLDER)
folder     = dataiku.Folder(IN_FOLDER)
vecs       = folder_read(folder, "vectorizers.joblib")
dico_train = folder_read(folder, "dico_train.joblib")
dico_full  = folder_read(folder, "dico_full.joblib")
matnr_test = folder_read(folder, "matnr_test.joblib")
meta       = folder_read(folder, "meta.joblib")
models     = {L: folder_read(folder, "model_%s.joblib" % L) for L in LEVELS}
PUR_L0     = meta['pur_l0']
log("modele du %s | train=%d test=%d" % (meta['date_entrainement'],
                                         meta['n_train'], meta['n_test']))

log("chargement de %s" % IN_DATASET)
df = dataiku.Dataset(IN_DATASET).get_dataframe(
        columns=['MATNR', 'MTART', 'MATKL', 'MAKTX'])
df = prepare(df)
df['MATNR_s'] = df['MATNR'].astype(str)

est_test  = df['MATNR_s'].isin(matnr_test) & df.desc_ok & df.niv1.notna()
est_cible = df.desc_ok & df.niv1.isna()
df['split'] = np.where(est_test, 'TEST', np.where(est_cible, 'CIBLE', 'TRAIN'))

keep = df['split'].isin(['TEST', 'CIBLE'] + (['TRAIN'] if SCORE_TRAIN else []))
S = df[keep].reset_index(drop=True)
log("a scorer : %s" % S['split'].value_counts().to_dict())

# ====== SORTIE : squelette ===========================================
OUT = pd.DataFrame({
    'MATNR':      S['MATNR'].values,
    'MTART':      S['MTART'].values,
    'MAKTX':      S['MAKTX'].values,
    'libelle_norm': S['lib'].values,
    'split':      S['split'].values,
    'MATKL_reel': S['MATKL'].values,
    'niv1_reel':  S['niv1'].values,
    'niv2_reel':  S['niv2'].values,
    'niv3_reel':  S['niv3'].values,
})

# ====== 1) COUCHE ML (par paquets pour la memoire) ====================
log("vectorisation + scoring ML")
ml_pred = {L: [[] for _ in range(TOPK[L])] for L in LEVELS}
ml_marge = {L: [] for L in LEVELS}
for start in range(0, len(S), CHUNK):
    part = S.iloc[start:start + CHUNK]
    Xp = hstack([vecs['word'].transform(part['lib']),
                 vecs['char'].transform(part['lib']),
                 vecs['mtart'].transform(part[['MTART']]) * 0.5]).tocsr()
    for L in LEVELS:
        clf = models[L]
        sc = clf.decision_function(Xp)
        if sc.ndim == 1:
            sc = np.c_[-sc, sc]
        cols, marge = top_k_from_scores(sc, clf.classes_, TOPK[L])
        for i, c in enumerate(cols):
            ml_pred[L][i].append(c)
        ml_marge[L].append(marge)
    if (start // CHUNK) % 5 == 0:
        log("  %d / %d" % (start, len(S)))

for L in LEVELS:
    ml_pred[L]  = [np.concatenate(x) for x in ml_pred[L]]
    ml_marge[L] = np.concatenate(ml_marge[L])

# ====== 2) COUCHE L0 (match exact) puis fusion ========================
# dico_train pour le TEST (pas de fuite), dico_full pour la CIBLE.
log("couche L0 (match exact du libelle)")
est_test_v = (S['split'] == 'TEST').values
for L in LEVELS:
    k = TOPK[L]
    dt = dico_train[L].reindex(S['lib'].values)
    dfu = dico_full[L].reindex(S['lib'].values)
    # choix du dictionnaire ligne par ligne
    d = pd.DataFrame({c: np.where(est_test_v, dt[c].values, dfu[c].values)
                      for c in ['top1', 'top2', 'top3', 'n_tot', 'purete']})
    hit = (pd.notna(d['top1']).values &
           (pd.to_numeric(d['purete'], errors='coerce').fillna(0).values >= PUR_L0))

    for i in range(k):
        col = 'niv%d_top%d' % (int(L[-1]), i + 1) if k > 1 else '%s_pred' % L
        OUT[col] = np.where(hit, d['top%d' % (i + 1)].values, ml_pred[L][i])
    OUT['%s_source' % L]   = np.where(hit, 'L0', 'ML')
    OUT['%s_confiance' % L] = np.where(hit,
            pd.to_numeric(d['purete'], errors='coerce').fillna(0).values,
            np.nan)
    OUT['%s_marge_ml' % L] = ml_marge[L]
    log("  %s : L0 %.1f%% / ML %.1f%%" % (L, 100*hit.mean(), 100*(1-hit.mean())))

# ====== 3) COHERENCE HIERARCHIQUE (signal de confiance gratuit) =======
# Mesure sur le backtest : coherent -> niv3 a 84.4% ; incoherent -> 31.0%.
p1 = OUT['niv1_pred'].astype(str)
p2 = OUT['niv2_top1'].astype(str)
p3 = OUT['niv3_top1'].astype(str)
OUT['coherent'] = (p3.str[:1] == p1) & (p3.str[:3] == p2) & (p2.str[:1] == p1)

OUT['action'] = np.where(OUT['coherent'] & (OUT['niv1_source'] == 'L0'), 'AUTO_NIV3',
                 np.where(OUT['coherent'], 'AUTO_NIV1_PROPOSE_NIV3',
                          'A_VALIDER'))

log("coherence : %.1f%%" % (100 * OUT['coherent'].mean()))
log("actions : %s" % OUT['action'].value_counts().to_dict())

dataiku.Dataset(OUT_DATASET).write_with_schema(OUT)
log("APPLICATION TERMINEE : %d lignes ecrites dans %s" % (len(OUT), OUT_DATASET))








# -*- coding: utf-8 -*-
"""
=========================================================================
 GM  -  RECIPE 3/3 : METRICS
=========================================================================
 INPUT   : GM_PREDICTIONS
 OUTPUTS : GM_METRICS_SYNTHESE     (1 ligne par perimetre : tout cote a cote)
           GM_METRICS_PAR_RANG     (format long : niveau x perimetre x rang)
           GM_METRICS_PAR_SOURCE   (d'ou vient la performance : L0 / ML / coherence)

 REGLES D'EVALUATION
 -------------------
 * On evalue UNIQUEMENT les lignes split == 'TEST' ayant un MATKL reel VALIDE.
   Les MATKL vides ou invalides (000 / XXX / YYY / 0M5 ...) sont exclus :
   il n'y a pas de verite terrain, on ne peut rien mesurer dessus.
 * Denominateur PROPRE A CHAQUE NIVEAU :
     - niv1 : tous les articles valides (tous ont au moins un niveau 1)
     - niv2 : uniquement ceux dont le MATKL reel fait >= 3 caracteres
     - niv3 : uniquement ceux dont le MATKL reel fait 5 caracteres
   Un article saisi seulement au niveau 1 n'est donc PAS penalise sur les
   niveaux 2 et 3 : il n'y entre pas.
 * top2 / top3 sont CUMULATIFS : top2 = "la verite est dans {top1, top2}".
 * TAUX GLOBAL = taux de bonne prediction du niveau 1 (regle metier retenue).
 * `acc_granularite_native` : chaque article juge a la finesse a laquelle il
   a reellement ete saisi (niv1 -> on juge niv1, niv3 -> on juge niv3 top1).
=========================================================================
"""
import sys, time
import numpy as np
import pandas as pd
import dataiku

IN_DATASET   = "GM_PREDICTIONS"
OUT_SYNTHESE = "GM_METRICS_SYNTHESE"
OUT_RANG     = "GM_METRICS_PAR_RANG"
OUT_SOURCE   = "GM_METRICS_PAR_SOURCE"
MIN_N        = 200          # perimetre ignore en dessous

def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m)); sys.stdout.flush()

log("chargement de %s" % IN_DATASET)
P = dataiku.Dataset(IN_DATASET).get_dataframe()

# ---------------- filtre : TEST + verite terrain valide ----------------
avant = len(P)
E = P[(P['split'] == 'TEST') & P['niv1_reel'].notna() &
      (P['niv1_reel'].astype(str).str.strip() != '')].copy()
log("%d lignes -> %d evaluables (TEST avec MATKL valide)" % (avant, len(E)))

for c in ['niv1_pred', 'niv2_top1', 'niv2_top2', 'niv2_top3',
          'niv3_top1', 'niv3_top2', 'niv3_top3',
          'niv1_reel', 'niv2_reel', 'niv3_reel']:
    E[c] = E[c].astype(object).where(E[c].notna(), None)

# masques de disponibilite de la verite, niveau par niveau
DISPO = {'niv1': E['niv1_reel'].notna().values,
         'niv2': E['niv2_reel'].notna().values,
         'niv3': E['niv3_reel'].notna().values}
TOPCOLS = {'niv1': ['niv1_pred'],
           'niv2': ['niv2_top1', 'niv2_top2', 'niv2_top3'],
           'niv3': ['niv3_top1', 'niv3_top2', 'niv3_top3']}

def hits_cumules(sub, level):
    """liste de masques booleens : [top1_ok, top<=2_ok, top<=3_ok]"""
    vrai = sub['%s_reel' % level].values
    acc, cum = [], np.zeros(len(sub), bool)
    for col in TOPCOLS[level]:
        cum = cum | (sub[col].values == vrai)
        acc.append(cum.copy())
    return acc

# granularite native de chaque article
E['granularite'] = np.where(E['niv3_reel'].notna(), 'niv3',
                     np.where(E['niv2_reel'].notna(), 'niv2', 'niv1'))

PERIMETRES = ['*TOUS*'] + sorted(E['MTART'].dropna().unique().tolist())

rows_syn, rows_rang, rows_src = [], [], []
for per in PERIMETRES:
    selP = np.ones(len(E), bool) if per == '*TOUS*' else (E['MTART'] == per).values
    if selP.sum() < MIN_N:
        continue
    syn = dict(perimetre=per, n_evalues=int(selP.sum()))

    # ---- par niveau et par rang -------------------------------------
    for L in ['niv1', 'niv2', 'niv3']:
        sel = selP & DISPO[L]
        if sel.sum() < MIN_N:
            syn['%s_n' % L] = int(sel.sum())
            continue
        sub = E[sel]
        acc = hits_cumules(sub, L)
        syn['%s_n' % L] = int(sel.sum())
        for r, h in enumerate(acc, 1):
            syn['%s_top%d' % (L, r)] = round(float(h.mean()), 4)
            rows_rang.append(dict(niveau=L, perimetre=per, rang='top%d' % r,
                                  n_evalues=int(sel.sum()),
                                  accuracy=round(float(h.mean()), 4),
                                  gain_vs_rang_precedent=round(
                                      float(h.mean() - acc[r-2].mean()), 4) if r > 1 else None))
        # ---- d'ou vient la performance : L0 vs ML -------------------
        for src in ['L0', 'ML']:
            m = (sub['%s_source' % L].values == src)
            if m.sum() < MIN_N:
                continue
            rows_src.append(dict(niveau=L, perimetre=per, dimension='source',
                                 modalite=src,
                                 couverture=round(float(m.mean()), 4),
                                 accuracy_top1=round(float(acc[0][m].mean()), 4),
                                 accuracy_topk=round(float(acc[-1][m].mean()), 4),
                                 n=int(m.sum())))
        # ---- stratification par coherence hierarchique ---------------
        for coh in [True, False]:
            m = (sub['coherent'].values == coh)
            if m.sum() < MIN_N:
                continue
            rows_src.append(dict(niveau=L, perimetre=per, dimension='coherence',
                                 modalite='coherent' if coh else 'incoherent',
                                 couverture=round(float(m.mean()), 4),
                                 accuracy_top1=round(float(acc[0][m].mean()), 4),
                                 accuracy_topk=round(float(acc[-1][m].mean()), 4),
                                 n=int(m.sum())))

    # ---- taux global : regle metier = le niveau 1 doit etre bon ------
    sub1 = E[selP]
    syn['TAUX_GLOBAL_niv1_ok'] = round(
        float((sub1['niv1_pred'].values == sub1['niv1_reel'].values).mean()), 4)

    # ---- accuracy a la granularite native ----------------------------
    ok = np.zeros(selP.sum(), bool)
    for g, col in [('niv1', 'niv1_pred'), ('niv2', 'niv2_top1'), ('niv3', 'niv3_top1')]:
        m = (sub1['granularite'].values == g)
        if m.any():
            ok[m] = (sub1[col].values[m] == sub1['%s_reel' % g].values[m])
    syn['acc_granularite_native'] = round(float(ok.mean()), 4)
    for g in ['niv1', 'niv2', 'niv3']:
        syn['part_saisie_%s' % g] = round(
            float((sub1['granularite'].values == g).mean()), 4)
    rows_syn.append(syn)

SYN  = pd.DataFrame(rows_syn)
RANG = pd.DataFrame(rows_rang)
SRC  = pd.DataFrame(rows_src)

# ordonner les colonnes de la synthese de facon lisible
ordre = (['perimetre', 'n_evalues', 'TAUX_GLOBAL_niv1_ok', 'acc_granularite_native',
          'niv1_n', 'niv1_top1',
          'niv2_n', 'niv2_top1', 'niv2_top2', 'niv2_top3',
          'niv3_n', 'niv3_top1', 'niv3_top2', 'niv3_top3',
          'part_saisie_niv1', 'part_saisie_niv2', 'part_saisie_niv3'])
SYN = SYN[[c for c in ordre if c in SYN.columns]]
SYN = pd.concat([SYN[SYN.perimetre == '*TOUS*'],
                 SYN[SYN.perimetre != '*TOUS*'].sort_values('n_evalues', ascending=False)])

print("\n=== SYNTHESE ===");                  print(SYN.to_string(index=False))
print("\n=== PAR RANG (extrait *TOUS*) ===")
print(RANG[RANG.perimetre == '*TOUS*'].to_string(index=False))
print("\n=== PAR SOURCE / COHERENCE (extrait *TOUS*) ===")
print(SRC[SRC.perimetre == '*TOUS*'].to_string(index=False))

dataiku.Dataset(OUT_SYNTHESE).write_with_schema(SYN)
dataiku.Dataset(OUT_RANG).write_with_schema(RANG)
dataiku.Dataset(OUT_SOURCE).write_with_schema(SRC)
log("METRICS TERMINEES")




