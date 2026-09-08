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




