# -*- coding: utf-8 -*-
"""
=========================================================================
 GM_COMMON  --  bibliotheque partagee
=========================================================================
 A deposer dans Dataiku :  Code  >  Libraries  >  python/gm_common.py
 Chaque recipe fait ensuite :  from gm_common import *

 Contient tout ce qui est commun aux filieres PROD et DEV :
   - nettoyage des libelles et des codes MATKL
   - chargement du referentiel Mapping (MATKL -> LIBEL)
   - recuperation des libelles courts mais reproductibles
   - dictionnaires L0 (match exact, top-3)
   - vectorisation, entrainement, decodage (independant ou joint)
   - comptage des metrics (modes hierarchique et strict)
=========================================================================
"""
import io, os, re, sys, time, unicodedata, threading
import numpy as np
import pandas as pd
import joblib

# ============================ PARAMETRES =============================
RANDOM_STATE  = 42
LEVELS        = ['niv1', 'niv2', 'niv3']
LONGUEURS     = {'niv1': 1, 'niv2': 3, 'niv3': 5}   # longueur du code par niveau
TOPK          = {'niv1': 1, 'niv2': 3, 'niv3': 3}   # niv1 -> 1 seule reponse
MODEL_BY_LEVEL = {'niv1': 'linsvc', 'niv2': 'sgd', 'niv3': 'sgd'}
MAX_CHAR_FEAT = 200000
PUR_L0        = 0.90        # purete minimale pour qu'une entree L0 soit utilisee

# --- Codes MATKL invalides / placeholders deja rencontres -------------------
# Liste explicite. Le referentiel Mapping sert de second filtre : un code
# absent du Mapping est aussi considere comme inutilisable (voir EXIGER_MAPPING).
CODES_INVALIDES = {
    '', '0', '000', '00000',
    'XXX', 'YYY', 'ZZZ',
    'NA', 'N/A', 'NAN', 'NONE', 'NULL',
    '0M5',
    'Z00000',
}
EXIGER_MAPPING = True       # True -> un MATKL absent du Mapping est rejete

# --- Libelles inexploitables ------------------------------------------------
GARBAGE = {'','test','testspn','test spn','na','n a','xxx','xxxxx','yyy','zzz',
           'sans','divers','autre','autres','neant','reserve','a definir',
           'article non defini','sans designation'}

# --- Recuperation des libelles courts --------------------------------------
# Un libelle de moins de LONGUEUR_MIN_LIBELLE caracteres est normalement
# rejete. Il est RECUPERE si le couple (MTART, libelle) revient au moins
# N_MIN_RECUP fois et porte TOUJOURS le meme MATKL : la saisie est alors
# reproductible, donc exploitable en entrainement comme en match exact L0.
LONGUEUR_MIN_LIBELLE = 4
LONGUEUR_MIN_RECUP   = 3    # plancher absolu : jamais en dessous
N_MIN_RECUP          = 2    # occurrences minimales du couple (MTART, libelle)

MTART_EXCLUS = set()        # aucun type d'article exclu par defaut

RE_FILL  = re.compile(r'[.\-_*=~/#+]{2,}')
RE_PUNCT = re.compile(r'[^a-z0-9 ]')
RE_SP    = re.compile(r'\s+')

_lock = threading.Lock()
def log(m):
    with _lock:
        print("[%s] %s" % (time.strftime("%H:%M:%S"), m)); sys.stdout.flush()

# ===================== NOMS DES OBJETS PAR FILIERE ===================
def noms(mode):
    """Tous les noms de datasets / folders d'une filiere, en un seul endroit."""
    assert mode in ('prod', 'dev'), "mode doit valoir 'prod' ou 'dev'"
    if mode == 'prod':
        return dict(
            mode='prod',
            entree_train='BASE_ARTICLE',        # entrainement sur tout l'etiquete
            entree_apply='BASE_ARTICLE',        # application sur la totalite
            folder='GM_MODELS_PROD',
            predictions='GM_PREDICTIONS_PROD',
            syn='GM_METRICS_PROD_SYNTHESE',
            rang='GM_METRICS_PROD_PAR_RANG',
            src='GM_METRICS_PROD_PAR_SOURCE')
    return dict(
        mode='dev',
        entree_train='BASE_ARTICLE_DEV_TRAIN',
        entree_apply='BASE_ARTICLE_DEV_TEST',  # jamais vu a l'entrainement
        folder='GM_MODELS_DEV',
        predictions='GM_PREDICTIONS_DEV',
        syn='GM_METRICS_DEV_SYNTHESE',
        rang='GM_METRICS_DEV_PAR_RANG',
        src='GM_METRICS_DEV_PAR_SOURCE')

# ========================= REFERENTIEL MAPPING =======================
def charger_mapping(nom='Mapping'):
    """Mapping(MATKL, LIBEL) -> dict de travail.
    Le referentiel fait autorite sur l'ensemble des codes possibles, y compris
    ceux qui n'apparaissent dans aucun article."""
    import dataiku
    m = dataiku.Dataset(nom).get_dataframe()
    corresp = {c.strip().upper(): c for c in m.columns}
    c_code = corresp.get('MATKL')
    c_lib = corresp.get('LIBEL') or corresp.get('LIBELLE') or corresp.get('LABEL')
    if c_code is None or c_lib is None:
        raise ValueError("Mapping doit contenir les colonnes MATKL et LIBEL "
                         "(trouve : %s)" % list(m.columns))
    m = m[[c_code, c_lib]].rename(columns={c_code: 'MATKL', c_lib: 'LIBEL'})
    m['MATKL'] = m['MATKL'].astype(str).str.strip().str.upper()
    m['LIBEL'] = m['LIBEL'].astype(str).str.strip()
    m = m[(m['MATKL'] != '') & (m['MATKL'].str.upper() != 'NAN')]
    doublons = int(m['MATKL'].duplicated().sum())
    m = m.drop_duplicates('MATKL', keep='first').reset_index(drop=True)

    codes = set(m['MATKL'])
    par_niveau = {L: {c for c in codes if len(c) == LONGUEURS[L]} for L in LEVELS}
    hors = sorted({c for c in codes if len(c) not in LONGUEURS.values()})

    log("Mapping : %d codes retenus%s" %
        (len(codes), (" (%d doublons ignores)" % doublons) if doublons else ""))
    log("  par niveau : niv1=%d  niv2=%d  niv3=%d"
        % tuple(len(par_niveau[L]) for L in LEVELS))
    if hors:
        log("  ATTENTION %d codes de longueur inattendue : %s"
            % (len(hors), hors[:10]))

    # Les codes niv2 se subdivisent-ils ? (explique pourquoi un modele niv3 ne
    # peut structurellement pas repondre pour un article saisi en 3 caracteres)
    prefixes3 = {c[:3] for c in par_niveau['niv3']}
    avec_enfants = par_niveau['niv2'] & prefixes3
    log("  codes niv2 ayant au moins un enfant niv3 : %d / %d"
        % (len(avec_enfants), len(par_niveau['niv2'])))
    if par_niveau['niv2'] and len(avec_enfants) < 0.5 * len(par_niveau['niv2']):
        log("  -> un groupe se subdivise OU ne se subdivise pas : les deux")
        log("     familles sont quasi disjointes. Le niveau 3 n'est donc pas")
        log("     applicable a tous les articles.")
    return dict(libel=dict(zip(m['MATKL'], m['LIBEL'])), codes=codes,
                par_niveau=par_niveau, avec_enfants=avec_enfants, df=m)

def ajouter_libelles(OUT, colonnes, libel):
    """Ajoute <colonne>_libel pour chaque colonne de codes demandee."""
    for c in colonnes:
        if c in OUT.columns:
            OUT[c + '_libel'] = pd.Series(OUT[c], index=OUT.index).map(
                lambda x: libel.get(x) if isinstance(x, str) else None)

# ======================== NETTOYAGE / NIVEAUX ========================
def norm_text(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ''
    s = unicodedata.normalize('NFKD', str(s).lower())
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return RE_SP.sub(' ', RE_PUNCT.sub(' ', RE_FILL.sub(' ', s))).strip()

def matkl_valide(m, codes_ok=None):
    """Normalise un MATKL ; None si placeholder ou absent du referentiel."""
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return None
    m = str(m).strip().upper()
    if m in CODES_INVALIDES:
        return None
    if EXIGER_MAPPING and codes_ok is not None and m not in codes_ok:
        return None
    return m

def niveaux(m):
    """Code propre -> (niv1, niv2, niv3). Un code court ne renseigne que les
    niveaux grossiers ; les niveaux absents valent None."""
    if m is None:
        return (None, None, None)
    n = len(m)
    return (m[:1] if n >= 1 else None,
            m[:3] if n >= 3 else None,
            m      if n == 5 else None)

def prepare(df, codes_ok=None, avec_recuperation=True, tracer=True):
    """Ajoute lib, matkl_clean, niv1..3, maktx_ok, mtart_ok, recupere."""
    df = df.copy()
    df['lib']   = df['MAKTX'].map(norm_text)
    df['MTART'] = df['MTART'].fillna('').astype(str).str.strip()
    df['matkl_clean'] = df['MATKL'].map(lambda m: matkl_valide(m, codes_ok))
    lv = [niveaux(m) for m in df['matkl_clean']]
    df['niv1'] = [x[0] for x in lv]
    df['niv2'] = [x[1] for x in lv]
    df['niv3'] = [x[2] for x in lv]

    lon          = df['lib'].str.len()
    pas_garbage  = ~df['lib'].isin(GARBAGE)
    pas_num      = ~df['lib'].str.replace(' ', '', regex=False).str.isdigit()
    exploitable  = pas_garbage & pas_num
    base_ok      = exploitable & (lon >= LONGUEUR_MIN_LIBELLE)

    # --- recuperation des libelles courts mais reproductibles --------------
    df['recupere'] = False
    if avec_recuperation:
        court = exploitable & (lon >= LONGUEUR_MIN_RECUP) & (lon < LONGUEUR_MIN_LIBELLE)
        cand = df.loc[court & df['matkl_clean'].notna(), ['MTART', 'lib', 'matkl_clean']]
        if len(cand):
            g = (cand.groupby(['MTART', 'lib'])['matkl_clean']
                     .agg(n='count', distincts='nunique'))
            retenus = g.index[(g['n'] >= N_MIN_RECUP) & (g['distincts'] == 1)]
            if len(retenus):
                cle = pd.MultiIndex.from_arrays(
                    [df['MTART'].values, df['lib'].values])
                df['recupere'] = (court.values & cle.isin(retenus))
        if tracer:
            log("libelles courts (%d-%d car.) recuperes : %d lignes / %d candidates"
                % (LONGUEUR_MIN_RECUP, LONGUEUR_MIN_LIBELLE - 1,
                   int(df['recupere'].sum()), int(court.sum())))

    df['maktx_ok'] = (base_ok | df['recupere']).values
    df['mtart_ok'] = ((df['MTART'] != '') & (df['MTART'] != 'NA')
                      & (~df['MTART'].isin(MTART_EXCLUS))).values
    if tracer:
        log("qualite : maktx_ok %.2f%% | mtart_ok %.2f%% | MATKL exploitable %.2f%%"
            % (100 * df['maktx_ok'].mean(), 100 * df['mtart_ok'].mean(),
               100 * df['matkl_clean'].notna().mean()))
    return df

def lignes_utilisables(df):
    """Lignes servant a l'entrainement et au calcul des metrics."""
    return (df['maktx_ok'].astype(bool) & df['mtart_ok'].astype(bool)
            & df['matkl_clean'].notna()).values

# ========================= DICTIONNAIRES L0 ==========================
def build_dico_top3(d, level, key='lib'):
    """key -> (top1, top2, top3) par frequence decroissante + purete du top1."""
    d = d[d[level].notna()]
    if not len(d):
        return pd.DataFrame(columns=['top1','top2','top3','n_tot','purete']) \
                 .rename_axis(key)
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

# ============================== FOLDER ===============================
def folder_write(folder, name, obj):
    try:
        joblib.dump(obj, os.path.join(folder.get_path(), name), compress=3)
    except Exception:
        buf = io.BytesIO(); joblib.dump(obj, buf, compress=3); buf.seek(0)
        folder.upload_stream(name, buf)

def folder_read(folder, name):
    try:
        return joblib.load(os.path.join(folder.get_path(), name))
    except Exception:
        with folder.get_download_stream(name) as f:
            return joblib.load(io.BytesIO(f.read()))

def folder_vider(folder):
    existants = folder.list_paths_in_partition()
    for p in existants:
        folder.delete_path(p)
    return len(existants)

# ======================= VECTORISATION / MODELES =====================
def construire_vectoriseurs():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import OneHotEncoder
    return dict(
        word=TfidfVectorizer(analyzer='word', ngram_range=(1, 2),
                             min_df=2, sublinear_tf=True),
        char=TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=3,
                             sublinear_tf=True, max_features=MAX_CHAR_FEAT),
        mtart=OneHotEncoder(handle_unknown='ignore'))

def vectoriser(vecs, d, ajuster=False):
    from scipy.sparse import hstack
    f = (lambda v, x: v.fit_transform(x)) if ajuster else (lambda v, x: v.transform(x))
    return hstack([f(vecs['word'], d['lib']),
                   f(vecs['char'], d['lib']),
                   f(vecs['mtart'], d[['MTART']]) * 0.5]).tocsr()

def construire_modele(niveau, n_jobs=1):
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import SGDClassifier
    if MODEL_BY_LEVEL[niveau] == 'linsvc':
        return LinearSVC(C=0.5, class_weight='balanced', max_iter=3000)
    return SGDClassifier(loss='modified_huber', alpha=1e-6, max_iter=25, tol=1e-4,
                         class_weight='balanced', random_state=RANDOM_STATE,
                         n_jobs=n_jobs)

# ============================= DECODAGE ==============================
def topk(M, classes, k):
    """k colonnes triees par score DECROISSANT. None quand il n'y a pas de
    candidat valide (moins de k classes, ou candidat hors contrainte a -inf)."""
    n, c = M.shape
    kk = min(k, c)
    idx = (np.argpartition(-M, kth=kk - 1, axis=1)[:, :kk] if c > kk
           else np.tile(np.arange(c), (n, 1)))
    rows = np.arange(n)[:, None]
    idx = idx[rows, np.argsort(-M[rows, idx], axis=1)]
    out = np.asarray(classes)[idx].astype(object)
    out[~np.isfinite(M[rows, idx])] = None
    if kk < k:
        out = np.concatenate([out, np.full((n, k - kk), None, dtype=object)], axis=1)
    return [out[:, i] for i in range(k)]

def marge(M):
    """Ecart top1-top2 en ignorant les -inf. NaN si un seul candidat valide."""
    if M.shape[1] < 2:
        return np.full(M.shape[0], np.nan)
    p = np.sort(M, axis=1)
    out = p[:, -1] - p[:, -2]
    out[~np.isfinite(p[:, -2])] = np.nan
    return out

def log_softmax(M, T=1.0):
    """decision_function -> log-probabilite. Seule normalisation qui rende les
    3 niveaux comparables : un z-score ecraserait l'amplitude et avantagerait
    mecaniquement le niveau ayant le plus de classes (donc niv3, le moins sur)."""
    M = np.asarray(M, dtype=float) / T
    m = M.max(axis=1, keepdims=True)
    e = np.exp(M - m)
    return (M - m) - np.log(e.sum(axis=1, keepdims=True))

def en_objet(a):
    a = np.asarray(a, dtype=object)
    return np.where(pd.isna(a), None, a)

class Decodeur:
    """Transforme les decision_function des 3 modeles en top-k par niveau.

    mode='independant' : chaque niveau prend son argmax, sans contrainte.
    mode='joint'       : score de CHEMIN, chaque classe fine impliquant ses
        parents par troncature :
          P3(c3) = w3.Z3[c3] + w2.Z2[c3[:3]] + w1.Z1[c3[:1]]
          P2(c2) = w2.Z2[c2] + w1.Z1[c2[:1]] + w3.max{Z3 enfants de c2}
          P1(c1) = w1.Z1[c1] + w2.max{Z2 enfants} + w3.max{Z3 descendants}
        La racine n1 = argmax P1 est ensuite imposee aux niveaux fins
        (repli sans contrainte si la descendance retenue est vide).
    """
    def __init__(self, classes, mode='joint', poids=None, temperature=1.0):
        self.cls = {L: np.asarray(classes[L]) for L in LEVELS}
        self.mode = mode
        self.w = poids or {L: 1.0 for L in LEVELS}
        self.T = temperature
        i1 = {c: i for i, c in enumerate(self.cls['niv1'])}
        i2 = {c: i for i, c in enumerate(self.cls['niv2'])}
        self.par2_1 = np.array([i1.get(c[:1], -1) for c in self.cls['niv2']])
        self.par3_2 = np.array([i2.get(c[:3], -1) for c in self.cls['niv3']])
        self.par3_1 = np.array([i1.get(c[:1], -1) for c in self.cls['niv3']])
        n1 = len(self.cls['niv1']); n2 = len(self.cls['niv2'])
        self.e2d1 = [np.where(self.par2_1 == j)[0] for j in range(n1)]
        self.e3d1 = [np.where(self.par3_1 == j)[0] for j in range(n1)]
        self.e3d2 = [np.where(self.par3_2 == j)[0] for j in range(n2)]

    @staticmethod
    def _gather(Z, parents):
        """Z[:, parents] ; log-proba uniforme pour les classes orphelines."""
        neutre = -np.log(max(Z.shape[1], 1))
        out = np.full((Z.shape[0], len(parents)), neutre)
        ok = parents >= 0
        out[:, ok] = Z[:, parents[ok]]
        return out

    @staticmethod
    def _max_enfants(Z, groupes, n_parents):
        out = np.full((Z.shape[0], n_parents), -np.log(max(Z.shape[1], 1)))
        for j, k in enumerate(groupes):
            if len(k):
                out[:, j] = Z[:, k].max(axis=1)
        return out

    def predire(self, brut):
        """brut : dict niveau -> matrice de scores. Retourne
        (dict niveau -> liste de top-k, dict niveau -> marge,
         dict niveau -> top1 du decodage independant pour comparaison)."""
        indep = {L: topk(brut[L], self.cls[L], 1)[0] for L in LEVELS}
        if self.mode == 'independant':
            final, racine = {L: brut[L] for L in LEVELS}, None
        else:
            Z = {L: log_softmax(brut[L], self.T) for L in LEVELS}
            w1, w2, w3 = (self.w['niv1'], self.w['niv2'], self.w['niv3'])
            P3 = (w3 * Z['niv3'] + w2 * self._gather(Z['niv2'], self.par3_2)
                  + w1 * self._gather(Z['niv1'], self.par3_1))
            P2 = (w2 * Z['niv2'] + w1 * self._gather(Z['niv1'], self.par2_1)
                  + w3 * self._max_enfants(Z['niv3'], self.e3d2, len(self.cls['niv2'])))
            P1 = (w1 * Z['niv1']
                  + w2 * self._max_enfants(Z['niv2'], self.e2d1, len(self.cls['niv1']))
                  + w3 * self._max_enfants(Z['niv3'], self.e3d1, len(self.cls['niv1'])))
            final = {'niv1': P1, 'niv2': P2, 'niv3': P3}
            racine = P1.argmax(axis=1)
        tops, marges = {}, {}
        for L in LEVELS:
            M = final[L]
            if racine is not None and L != 'niv1':
                par = self.par2_1 if L == 'niv2' else self.par3_1
                Mc = np.where(par[None, :] == racine[:, None], M, -np.inf)
                vide = ~np.isfinite(Mc).any(axis=1)
                if vide.any():
                    Mc[vide] = M[vide]
                M = Mc
            tops[L] = topk(M, self.cls[L], TOPK[L])
            marges[L] = marge(M)
        return tops, marges, indep

def suffixe(niveau, i):
    """Nom de colonne : niv1 -> 'pred' ; niv2/niv3 -> 'top1'..'top3'."""
    return 'pred' if TOPK[niveau] == 1 else ('top%d' % (i + 1))

# ============================== METRICS ==============================
def colonnes_pred(prefixe=''):
    """prefixe : '' (resultat livre), 'ml_' (modele seul), 'l0_' (dico seul)."""
    return {L: ['%s_%s%s' % (L, prefixe, suffixe(L, i)) for i in range(TOPK[L])]
            for L in LEVELS}

def hits_hierarchiques(sub, niveau, topcols):
    """Cumules [top1, top<=2, top<=3]. Comparaison a la profondeur
    min(profondeur du niveau, profondeur de la verite) : un article saisi
    seulement en 'C' est donc evaluable a tous les niveaux.
    Denominateur = TOUTES les lignes de sub -> identique pour les 3 niveaux."""
    d = np.minimum(LONGUEURS[niveau], sub['profondeur'].values)
    ver = sub['verite']
    cum, out = np.zeros(len(sub), bool), []
    for col in topcols[niveau]:
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

def hits_stricts(sub, niveau, topcols):
    """Cumules sur les seules lignes ayant une verite A CE niveau, egalite
    complete. Retourne aussi le masque de disponibilite."""
    dispo = sub['%s_reel' % niveau].notna().values
    ver = sub['%s_reel' % niveau]
    cum, out = np.zeros(len(sub), bool), []
    for col in topcols[niveau]:
        cum = cum | ((sub[col] == ver).fillna(False).values & dispo)
        out.append(cum.copy())
    return out, dispo







# -*- coding: utf-8 -*-
"""
=========================================================================
 DEV - RECIPE 0/3 : CONSTITUTION DU JEU DE DEV
=========================================================================
 INPUTS  : BASE_ARTICLE  +  Mapping
 OUTPUTS : BASE_ARTICLE_DEV_TRAIN  +  BASE_ARTICLE_DEV_TEST

 On ne garde que les lignes exploitables (MATKL present dans Mapping, libelle
 et MTART utilisables), puis on coupe en deux.

 METHODE_SPLIT
 -------------
 'realiste'  (defaut, recommande) : le jeu de test imite la population
     reellement a completer. Mesure : parmi les articles sans MATKL, environ
     PART_INEDIT_CIBLE ont un libelle qui n'apparait nulle part ailleurs. On
     reproduit ce taux -> le dictionnaire L0 fonctionne sur la bonne
     proportion de lignes, et le chiffre obtenu est directement transposable.
 'libelle'   : coupe par libelle, train et test totalement disjoints. Mesure
     la generalisation pure, mais le dictionnaire L0 ne peut JAMAIS matcher
     (son resultat sera 0% de couverture) -> ne mesure que le modele.
 'aleatoire' : coupe par ligne. Le meme libelle se retrouve des deux cotes ;
     scores optimistes, utile seulement comme borne haute.
=========================================================================
"""
import numpy as np
import pandas as pd
import dataiku
from gm_common import *

# ============================== CONFIG ==============================
IN_DATASET        = "BASE_ARTICLE"
IN_MAPPING        = "Mapping"
OUT_TRAIN         = "BASE_ARTICLE_DEV_TRAIN"
OUT_TEST          = "BASE_ARTICLE_DEV_TEST"

METHODE_SPLIT     = 'realiste'      # 'realiste' | 'libelle' | 'aleatoire'
PART_TEST         = 0.20
PART_INEDIT_CIBLE = 0.63            # part du test dont le libelle est absent du train
COLONNES          = ['MATNR', 'MTART', 'MATKL', 'MAKTX']

# ============================ CHARGEMENT =============================
mp = charger_mapping(IN_MAPPING)
log("chargement de %s" % IN_DATASET)
df = dataiku.Dataset(IN_DATASET).get_dataframe(columns=COLONNES)
df = prepare(df, codes_ok=mp['codes'])

ok = lignes_utilisables(df)
lab = df[ok].reset_index(drop=True)
log("%d lignes -> %d exploitables pour le dev" % (len(df), len(lab)))
log("  granularite de saisie : %s"
    % np.where(lab.niv3.notna(), 'niv3',
        np.where(lab.niv2.notna(), 'niv2', 'niv1')).tolist().count('niv3'))

rng = np.random.RandomState(RANDOM_STATE)
n = len(lab)
est_test = np.zeros(n, bool)

if METHODE_SPLIT == 'aleatoire':
    est_test = rng.rand(n) < PART_TEST

elif METHODE_SPLIT == 'libelle':
    libs = lab['lib'].unique()
    rng.shuffle(libs)
    tailles = lab.groupby('lib').size()
    cible, cumul, retenus = PART_TEST * n, 0, set()
    for L in libs:
        if cumul >= cible:
            break
        retenus.add(L); cumul += int(tailles[L])
    est_test = lab['lib'].isin(retenus).values

else:   # 'realiste'
    cible_test   = PART_TEST * n
    cible_inedit = cible_test * PART_INEDIT_CIBLE
    cible_vu     = cible_test - cible_inedit
    tailles = lab.groupby('lib').size()

    # 1) libelles entierement basculas en test -> "inedits" pour le modele
    libs = np.array(tailles.index)
    rng.shuffle(libs)
    cumul, inedits = 0, set()
    for L in libs:
        if cumul >= cible_inedit:
            break
        inedits.add(L); cumul += int(tailles[L])
    masque_inedit = lab['lib'].isin(inedits).values
    log("  libelles inedits : %d libelles / %d lignes" % (len(inedits), cumul))

    # 2) lignes prelevees sur des libelles qui RESTENT dans le train
    #    (uniquement des libelles vus >= 2 fois, et jamais toutes leurs lignes)
    partageables = tailles[tailles >= 2].index
    cand = np.where((~masque_inedit) & lab['lib'].isin(partageables).values)[0]
    rng.shuffle(cand)
    pris, compte_par_lib = [], {}
    for i in cand:
        if len(pris) >= cible_vu:
            break
        L = lab['lib'].values[i]
        if compte_par_lib.get(L, 0) + 1 >= int(tailles[L]):
            continue                     # on laisse au moins une ligne au train
        compte_par_lib[L] = compte_par_lib.get(L, 0) + 1
        pris.append(i)
    log("  lignes a libelle deja vu : %d" % len(pris))

    est_test = masque_inedit.copy()
    est_test[np.array(pris, dtype=int)] = True

train = lab[~est_test].reset_index(drop=True)
test  = lab[est_test].reset_index(drop=True)

# controle : quelle part du test a un libelle reellement absent du train ?
libs_train = set(train['lib'])
part_inedit = float((~test['lib'].isin(libs_train)).mean()) if len(test) else 0.0
log("SPLIT '%s' : train %d / test %d (%.1f%%)"
    % (METHODE_SPLIT, len(train), len(test), 100 * len(test) / max(n, 1)))
log("  part du test dont le libelle est ABSENT du train : %.1f%% (cible %.1f%%)"
    % (100 * part_inedit, 100 * PART_INEDIT_CIBLE if METHODE_SPLIT == 'realiste' else 0))
for d, nom in ((train, 'train'), (test, 'test')):
    log("  %-5s MTART : %s" % (nom, d['MTART'].value_counts().head(6).to_dict()))

SORTIE = COLONNES + ['lib', 'matkl_clean', 'niv1', 'niv2', 'niv3',
                     'maktx_ok', 'mtart_ok', 'recupere']
dataiku.Dataset(OUT_TRAIN).write_with_schema(train[SORTIE])
dataiku.Dataset(OUT_TEST).write_with_schema(test[SORTIE])
log("SPLIT TERMINE")












# -*- coding: utf-8 -*-
"""
=========================================================================
 RECIPE 1/3 : ENTRAINEMENT          (meme code pour PROD et DEV)
=========================================================================
 MODE = 'prod'  INPUTS  BASE_ARTICLE + Mapping     OUTPUT GM_MODELS_PROD
 MODE = 'dev'   INPUTS  BASE_ARTICLE_DEV_TRAIN + Mapping  OUTPUT GM_MODELS_DEV

 Seule la ligne MODE change entre les deux recipes. Les datasets declares
 dans le Flow doivent correspondre (voir gm_common.noms()).

 Le dossier de sortie est VIDE puis reecrit : il ne contient jamais qu'un
 seul modele, le dernier entraine.

 Architecture (validee par backtests) :
   * 3 modeles INDEPENDANTS niv1 / niv2 / niv3. Les etiquettes sont de
     granularite mixte (~76% niv3, ~23% niv2, ~1% niv1) : un modele plat a
     377 classes forcerait a distinguer C01 de C0101, et une cascade reelle
     propagerait l'erreur du niv1 sans recuperation possible.
   * chaine L0 (match exact du libelle) -> ML. Pas de couche intermediaire
     "MTART + premier mot" : sa contribution nette mesuree est nulle
     (-0.58pt au niv1, +0.58pt au niv2, -0.06pt au niv3).
=========================================================================
"""
import time
import numpy as np
import pandas as pd
import dataiku
from joblib import Parallel, delayed
from gm_common import *

# ============================== CONFIG ==============================
MODE       = 'prod'          # <<<<<< 'prod' ou 'dev' : SEULE ligne a changer
IN_MAPPING = "Mapping"
PARALLEL   = True
COLONNES   = ['MATNR', 'MTART', 'MATKL', 'MAKTX']

N = noms(MODE)
log("=== ENTRAINEMENT  mode=%s  entree=%s  sortie=%s ==="
    % (MODE, N['entree_train'], N['folder']))

# ============================ CHARGEMENT =============================
mp = charger_mapping(IN_MAPPING)
df = dataiku.Dataset(N['entree_train']).get_dataframe(columns=COLONNES)
log("%d lignes lues" % len(df))
df = prepare(df, codes_ok=mp['codes'])

# codes presents dans les donnees mais absents du referentiel -> rejetes
brut = df['MATKL'].dropna().astype(str).str.strip().str.upper()
inconnus = sorted(set(brut[(brut != '') & (~brut.isin(CODES_INVALIDES))
                           & (~brut.isin(mp['codes']))]))
if inconnus:
    log("ATTENTION %d codes MATKL hors Mapping (lignes ecartees) : %s"
        % (len(inconnus), inconnus[:15]))

lab = df[lignes_utilisables(df)].reset_index(drop=True)
log("pool d'entrainement : %d lignes" % len(lab))
for L in LEVELS:
    presentes = set(lab[L].dropna())
    jamais = mp['par_niveau'][L] - presentes
    log("  %s : %d lignes | %d classes vues | %d codes du Mapping jamais"
        " rencontres (non predictibles)"
        % (L, lab[L].notna().sum(), len(presentes), len(jamais)))

# ===================== DICTIONNAIRES L0 ===============================
# dico_train : construit sur le pool d'entrainement. C'est celui utilise pour
#              scorer un jeu de test disjoint (pas de fuite).
# dico_full  : identique ici, conserve pour compatibilite avec la recipe 2
#              (en PROD les deux sont les memes ; en DEV le test est externe).
log("construction des dictionnaires L0")
dico_train = {L: build_dico_top3(lab, L) for L in LEVELS}
for L in LEVELS:
    log("  %s : %d cles" % (L, len(dico_train[L])))

# ======================== VECTORISATION ==============================
log("vectorisation")
t0 = time.time()
vecs = construire_vectoriseurs()
Xtr = vectoriser(vecs, lab, ajuster=True)
log("  X %s (%ds)" % (Xtr.shape, time.time() - t0))

# ======================== ENTRAINEMENT ===============================
def entrainer(L, n_jobs):
    m = lab[L].notna().values
    clf = construire_modele(L, n_jobs)
    t = time.time()
    clf.fit(Xtr[m], lab.loc[m, L].values)
    log("  %s : %d lignes, %d classes, %ds"
        % (L, m.sum(), len(clf.classes_), time.time() - t))
    return L, clf

log("entrainement des 3 niveaux (%s)" % ("parallele" if PARALLEL else "sequentiel"))
paires = (Parallel(n_jobs=3, prefer="threads")(delayed(entrainer)(L, 1) for L in LEVELS)
          if PARALLEL else [entrainer(L, -1) for L in LEVELS])
models = dict(paires)

# ========================= SAUVEGARDE ================================
folder = dataiku.Folder(N['folder'])
log("nettoyage de %s : %d fichiers supprimes" % (N['folder'], folder_vider(folder)))

meta = dict(mode=MODE, entree_train=N['entree_train'],
            date_entrainement=time.strftime("%Y-%m-%d %H:%M:%S"),
            n_train=len(lab), pur_l0=PUR_L0, max_char_feat=MAX_CHAR_FEAT,
            model_by_level=MODEL_BY_LEVEL, random_state=RANDOM_STATE,
            longueur_min_libelle=LONGUEUR_MIN_LIBELLE,
            longueur_min_recup=LONGUEUR_MIN_RECUP, n_min_recup=N_MIN_RECUP,
            n_recuperes=int(df['recupere'].sum()),
            codes_invalides=sorted(CODES_INVALIDES),
            codes_hors_mapping=inconnus,
            mtart_vus=sorted(lab['MTART'].unique()),
            classes={L: list(models[L].classes_) for L in LEVELS},
            classes_mapping_non_vues={L: sorted(mp['par_niveau'][L]
                                                - set(lab[L].dropna()))
                                      for L in LEVELS})

folder_write(folder, "vectorizers.joblib", vecs)
for L in LEVELS:
    folder_write(folder, "model_%s.joblib" % L, models[L])
folder_write(folder, "dico_train.joblib", dico_train)
folder_write(folder, "dico_full.joblib", dico_train)
folder_write(folder, "mapping.joblib", dict(libel=mp['libel'], codes=mp['codes'],
                                            par_niveau=mp['par_niveau']))
folder_write(folder, "meta.joblib", meta)
log("fichiers : %s" % folder.list_paths_in_partition())
log("ENTRAINEMENT TERMINE")








# -*- coding: utf-8 -*-
"""
=========================================================================
 RECIPE 2/3 : APPLICATION           (meme code pour PROD et DEV)
=========================================================================
 MODE='prod' INPUTS BASE_ARTICLE + GM_MODELS_PROD   OUT GM_PREDICTIONS_PROD
             -> on score la TOTALITE de BASE_ARTICLE, sans aucun filtre :
                quel que soit le contenu de MAKTX, que MATKL soit deja
                renseigne ou non. Le tri se fait dans la recipe 3.
 MODE='dev'  INPUTS BASE_ARTICLE_DEV_TEST + GM_MODELS_DEV  OUT GM_PREDICTIONS_DEV
             -> jeu jamais vu a l'entrainement.

 Trois jeux de colonnes par niveau :
   <niv>_l0_*  dictionnaire seul (None hors couverture)
   <niv>_ml_*  modele seul
   <niv>_*     resultat LIVRE : dictionnaire prioritaire, modele en repli
 plus <...>_libel : le libelle francais du Mapping, pour relecture metier.
 niv1 -> une seule reponse ; niv2 et niv3 -> top1/top2/top3 ordonnes.
=========================================================================
"""
import time
import numpy as np
import pandas as pd
import dataiku
from gm_common import *

# ============================== CONFIG ==============================
MODE        = 'prod'         # <<<<<< 'prod' ou 'dev' : SEULE ligne a changer
DECODAGE    = 'joint'        # 'joint' ou 'independant'
POIDS       = {'niv1': 1.0, 'niv2': 1.0, 'niv3': 1.0}
TEMPERATURE = 1.0
CHUNK       = 25000          # 50000 si DECODAGE='independant'
COLONNES    = ['MATNR', 'MTART', 'MATKL', 'MAKTX']

N = noms(MODE)
log("=== APPLICATION  mode=%s  entree=%s  decodage=%s ==="
    % (MODE, N['entree_apply'], DECODAGE))

# ====================== CHARGEMENT DU MODELE =========================
folder     = dataiku.Folder(N['folder'])
vecs       = folder_read(folder, "vectorizers.joblib")
dico_train = folder_read(folder, "dico_train.joblib")
dico_full  = folder_read(folder, "dico_full.joblib")
mapping    = folder_read(folder, "mapping.joblib")
meta       = folder_read(folder, "meta.joblib")
models     = {L: folder_read(folder, "model_%s.joblib" % L) for L in LEVELS}
if meta.get('mode') != MODE:
    log("ATTENTION : modele entraine en mode '%s' applique en mode '%s'"
        % (meta.get('mode'), MODE))
log("modele du %s | entraine sur %s (%d lignes)"
    % (meta['date_entrainement'], meta['entree_train'], meta['n_train']))

classes = {L: models[L].classes_ for L in LEVELS}
dec = Decodeur(classes, mode=DECODAGE, poids=POIDS, temperature=TEMPERATURE)
log("classes : %s" % {L: len(classes[L]) for L in LEVELS})

# ============================ DONNEES ================================
df = dataiku.Dataset(N['entree_apply']).get_dataframe(columns=COLONNES)
df = prepare(df, codes_ok=mapping['codes'])
S = df.reset_index(drop=True)
S['split'] = np.where(S['matkl_clean'].notna(), 'ETIQUETE', 'A_COMPLETER')
log("a scorer : %d lignes | %s" % (len(S), S['split'].value_counts().to_dict()))

# ========================= SCORING PAR PAQUETS =======================
res   = {L: {'top': [[] for _ in range(TOPK[L])], 'marge': []} for L in LEVELS}
indep = {L: [] for L in LEVELS}
t0 = time.time()
for start in range(0, len(S), CHUNK):
    part = S.iloc[start:start + CHUNK]
    Xp = vectoriser(vecs, part, ajuster=False)
    brut = {}
    for L in LEVELS:
        sc = models[L].decision_function(Xp)
        brut[L] = np.c_[-sc, sc] if sc.ndim == 1 else sc
    tops, marges, ind = dec.predire(brut)
    for L in LEVELS:
        for i, c in enumerate(tops[L]):
            res[L]['top'][i].append(c)
        res[L]['marge'].append(marges[L])
        indep[L].append(ind[L])
    if (start // CHUNK) % 5 == 0:
        log("  %d / %d (%ds)" % (start, len(S), time.time() - t0))

for L in LEVELS:
    res[L]['top']   = [np.concatenate(x) for x in res[L]['top']]
    res[L]['marge'] = np.concatenate(res[L]['marge'])
    indep[L]        = np.concatenate(indep[L])

# ===================== SORTIE + FUSION L0 / ML ========================
OUT = pd.DataFrame({
    'MATNR': S['MATNR'].values, 'MTART': S['MTART'].values,
    'MAKTX': S['MAKTX'].values, 'libelle_norm': S['lib'].values,
    'split': S['split'].values, 'jeu': MODE,
    'maktx_ok': S['maktx_ok'].values, 'mtart_ok': S['mtart_ok'].values,
    'libelle_recupere': S['recupere'].values,
    'MATKL_reel': S['matkl_clean'].values,
    'niv1_reel': S['niv1'].values, 'niv2_reel': S['niv2'].values,
    'niv3_reel': S['niv3'].values,
})

# En DEV le dictionnaire vient du train, qui est disjoint du test : aucune
# fuite possible. En PROD les deux dictionnaires sont identiques.
log("couche L0 (match exact du libelle)")
dico = dico_train if MODE == 'dev' else dico_full
for L in LEVELS:
    d = dico[L].reindex(S['lib'].values)
    pur = pd.to_numeric(d['purete'], errors='coerce').fillna(0).values
    hit = pd.notna(d['top1']).values & (pur >= meta['pur_l0'])
    for i in range(TOPK[L]):
        sfx  = suffixe(L, i)
        v_ml = en_objet(res[L]['top'][i])
        v_l0 = np.where(hit, en_objet(d['top%d' % (i + 1)].values), None)
        OUT['%s_ml_%s' % (L, sfx)] = v_ml
        OUT['%s_l0_%s' % (L, sfx)] = v_l0
        OUT['%s_%s'    % (L, sfx)] = np.where(hit, v_l0, v_ml)
    OUT['%s_source' % L]    = np.where(hit, 'L0', 'ML')
    OUT['%s_confiance' % L] = np.where(hit, pur, np.nan)
    OUT['%s_marge_ml' % L]  = res[L]['marge']
    log("  %s : L0 %.1f%% / ML %.1f%%" % (L, 100 * hit.mean(), 100 * (1 - hit.mean())))

OUT['decodage'] = DECODAGE

# ------------------- libelles francais (Mapping) ----------------------
a_libeller = ['MATKL_reel'] + ['%s_%s' % (L, suffixe(L, i))
                               for L in LEVELS for i in range(TOPK[L])]
ajouter_libelles(OUT, a_libeller, mapping['libel'])
manquants = sum(int(OUT['%s_libel' % c].isna().sum() - pd.isna(OUT[c]).sum())
                for c in a_libeller if '%s_libel' % c in OUT.columns)
log("libelles Mapping ajoutes sur %d colonnes%s"
    % (len(a_libeller), " (%d codes predits sans libelle)" % manquants if manquants else ""))

# ------------------- coherence hierarchique --------------------------
def _coh(c1, c2, c3):
    a = OUT[c1].astype(str); b = OUT[c2].astype(str); c = OUT[c3].astype(str)
    return (c.str[:1] == a) & (c.str[:3] == b) & (b.str[:1] == a)
OUT['coherent']    = _coh('niv1_pred', 'niv2_top1', 'niv3_top1')
OUT['coherent_ml'] = _coh('niv1_ml_pred', 'niv2_ml_top1', 'niv3_ml_top1')
log("coherence : livre %.1f%% | ML seul %.1f%%"
    % (100 * OUT['coherent'].mean(), 100 * OUT['coherent_ml'].mean()))

OUT['action'] = np.where(OUT['coherent'] & (OUT['niv1_source'] == 'L0'), 'AUTO_NIV3',
                 np.where(OUT['coherent'], 'AUTO_NIV1_PROPOSE_NIV3', 'A_VALIDER'))

# ---------- controle immediat : joint vs independant ------------------
m = S['matkl_clean'].notna().values & S['maktx_ok'].values
if m.sum() > 100:
    vu = (MODE == 'prod')
    log("--- controle sur %d lignes etiquetees%s (couche ML seule) ---"
        % (m.sum(), " VUES a l'entrainement" if vu else " jamais vues"))
    for L in LEVELS:
        vrai = S[L].values[m]
        d2 = pd.notna(vrai)
        if d2.sum() < 100:
            continue
        a_i = float((indep[L][m][d2] == vrai[d2]).mean())
        a_f = float((res[L]['top'][0][m][d2] == vrai[d2]).mean())
        log("  %s : independant %.4f | %-11s %.4f | ecart %+.4f (n=%d)"
            % (L, a_i, DECODAGE, a_f, a_f - a_i, d2.sum()))

dataiku.Dataset(N['predictions']).write_with_schema(OUT)
log("APPLICATION TERMINEE : %d lignes -> %s" % (len(OUT), N['predictions']))









# -*- coding: utf-8 -*-
"""
=========================================================================
 RECIPE 3/3 : METRICS               (meme code pour PROD et DEV)
=========================================================================
 MODE='prod' IN GM_PREDICTIONS_PROD -> GM_METRICS_PROD_{SYNTHESE,PAR_RANG,PAR_SOURCE}
 MODE='dev'  IN GM_PREDICTIONS_DEV  -> GM_METRICS_DEV_{...}

 PERIMETRE EVALUE : toutes les lignes ayant
   - maktx_ok (libelle exploitable, y compris les libelles courts recuperes)
   - mtart_ok (type d'article renseigne, hors MTART_EXCLUS)
   - un MATKL reel valide et present dans Mapping (sinon pas de verite)

 SOURCE_PREDICTION : quelle prediction evaluer
   'final' -> ce que le systeme LIVRE (L0 prioritaire, ML en repli) = PROD
   'ml'    -> le modele seul. En mode prod c'est le SEUL chiffre lisible :
              le dictionnaire L0 y relit l'etiquette de la ligne elle-meme.
   'l0'    -> le dictionnaire seul (non couvert = faux). Accuracy =
              couverture x precision ; la precision conditionnelle est dans
              GM_METRICS_*_PAR_SOURCE.

 DEUX MODES DE COMPTAGE
   'hierarchique' : denominateur COMMUN aux 3 niveaux. Comparaison a la
        profondeur min(profondeur du niveau, profondeur de la verite) : un
        article saisi 'C' compte aussi au niveau 2 et 3.
        /!\\ A ne pas lire pour niv3 : un article saisi en 3 caracteres
        appartient le plus souvent a un groupe qui NE SE SUBDIVISE PAS, donc
        aucune classe niv3 ne commence par son code -> le modele niv3 ne peut
        structurellement pas repondre. Utiliser 'strict' ou
        acc_granularite_native.
   'strict' : chaque niveau sur les seules lignes ayant une verite a ce
        niveau, egalite complete. Denominateurs differents.

 acc_granularite_native : chaque article juge a la finesse a laquelle il a
   reellement ete saisi. C'est le MEILLEUR chiffre unique.
=========================================================================
"""
import numpy as np
import pandas as pd
import dataiku
from gm_common import *

# ============================== CONFIG ==============================
MODE              = 'prod'      # <<<<<< 'prod' ou 'dev'
SOURCE_PREDICTION = 'ml'        # 'final' | 'ml' | 'l0'
MIN_N             = 200

N = noms(MODE)
_PREFIXES = {'final': '', 'ml': 'ml_', 'l0': 'l0_'}
TOPCOLS = colonnes_pred(_PREFIXES[SOURCE_PREDICTION])
log("=== METRICS  mode=%s  source=%s ===" % (MODE, SOURCE_PREDICTION))

# ============================ CHARGEMENT =============================
P = dataiku.Dataset(N['predictions']).get_dataframe()

manquantes = [c for v in TOPCOLS.values() for c in v if c not in P.columns]
if manquantes:
    log("colonnes %s absentes -> repli sur 'final'" % manquantes)
    SOURCE_PREDICTION = 'final'; TOPCOLS = colonnes_pred('')
COL_COHERENT = ('coherent_ml' if (SOURCE_PREDICTION == 'ml'
                                  and 'coherent_ml' in P.columns) else 'coherent')
log("colonnes evaluees : %s" % TOPCOLS['niv3'])

for c in ['maktx_ok', 'mtart_ok']:
    if c not in P.columns:
        P[c] = True
        log("colonne %s absente -> consideree vraie" % c)
if 'split' not in P.columns:
    P['split'] = 'ETIQUETE'

garde = (P['maktx_ok'].fillna(False).astype(bool)
         & P['mtart_ok'].fillna(False).astype(bool)
         & P['niv1_reel'].notna()
         & (P['niv1_reel'].astype(str).str.strip() != ''))
E = P[garde].copy().reset_index(drop=True)
log("%d lignes -> %d evaluables" % (len(P), len(E)))
log("  exclues : maktx %d | mtart %d | MATKL absent/invalide %d"
    % (int((~P['maktx_ok'].fillna(False).astype(bool)).sum()),
       int((~P['mtart_ok'].fillna(False).astype(bool)).sum()),
       int(P['niv1_reel'].isna().sum())))
if 'libelle_recupere' in E.columns:
    log("  dont %d lignes a libelle court recupere" % int(E['libelle_recupere'].sum()))
if MODE == 'prod':
    log("  /!\\ mode PROD : le modele a VU ces lignes a l'entrainement. Ce")
    log("      chiffre mesure la reproduction du referentiel. Pour la")
    log("      generalisation, utiliser la filiere DEV.")
else:
    log("  mode DEV : jeu disjoint de l'entrainement -> mesure de generalisation")

TOUTES = [c for v in TOPCOLS.values() for c in v] + \
         ['niv1_reel', 'niv2_reel', 'niv3_reel']
for c in TOUTES:
    E[c] = E[c].astype('string')

E['verite'] = E['niv3_reel'].fillna(E['niv2_reel']).fillna(E['niv1_reel'])
E['profondeur'] = E['verite'].str.len().fillna(0).astype(int)
E['granularite'] = np.where(E['niv3_reel'].notna(), 'niv3',
                     np.where(E['niv2_reel'].notna(), 'niv2', 'niv1'))
log("profondeurs de la verite : %s" % E['profondeur'].value_counts().to_dict())

# ============================== BOUCLE ================================
POPULATIONS = sorted(E['split'].dropna().unique().tolist())
DECOUPES = ([('*TOUS*', '*TOUTES*')]
            + [(m, '*TOUTES*') for m in sorted(E['MTART'].dropna().unique().tolist())]
            + [('*TOUS*', p) for p in POPULATIONS if len(POPULATIONS) > 1])
rows_syn, rows_rang, rows_src = [], [], []

for per, pop in DECOUPES:
    selP = np.ones(len(E), bool) if per == '*TOUS*' else (E['MTART'] == per).values
    if pop != '*TOUTES*':
        selP = selP & (E['split'] == pop).values
    if selP.sum() < MIN_N:
        continue
    sub, n = E[selP], int(selP.sum())
    syn = dict(jeu=MODE, source_prediction=SOURCE_PREDICTION,
               perimetre=per, population=pop, n_evalues=n)

    for niveau in LEVELS:
        accH = hits_hierarchiques(sub, niveau, TOPCOLS)
        syn['h_%s_n' % niveau] = n
        for r, h in enumerate(accH, 1):
            syn['h_%s_top%d' % (niveau, r)] = round(float(h.mean()), 4)
            rows_rang.append(dict(jeu=MODE, source_prediction=SOURCE_PREDICTION,
                mode='hierarchique', niveau=niveau, perimetre=per, population=pop,
                rang='top%d' % r, n_evalues=n, accuracy=round(float(h.mean()), 4),
                gain_vs_rang_precedent=round(float(h.mean() - accH[r-2].mean()), 4)
                                       if r > 1 else None))

        accS, dispo = hits_stricts(sub, niveau, TOPCOLS)
        nd = int(dispo.sum())
        syn['s_%s_n' % niveau] = nd
        if nd >= MIN_N:
            for r, h in enumerate(accS, 1):
                a = float(h[dispo].mean())
                syn['s_%s_top%d' % (niveau, r)] = round(a, 4)
                rows_rang.append(dict(jeu=MODE, source_prediction=SOURCE_PREDICTION,
                    mode='strict', niveau=niveau, perimetre=per, population=pop,
                    rang='top%d' % r, n_evalues=nd, accuracy=round(a, 4),
                    gain_vs_rang_precedent=round(float(a - accS[r-2][dispo].mean()), 4)
                                           if r > 1 else None))

        for dim, col, modalites in (('source', '%s_source' % niveau, None),
                                    ('coherence', COL_COHERENT, [True, False]),
                                    ('libelle_court', 'libelle_recupere', [True, False])):
            if col not in sub.columns:
                continue
            vals = modalites if modalites is not None else sorted(sub[col].dropna().unique())
            for v in vals:
                m = (sub[col].values == v)
                if m.sum() < MIN_N:
                    continue
                etiq = {True: 'oui', False: 'non'}.get(v, str(v))
                if dim == 'coherence':
                    etiq = 'coherent' if v else 'incoherent'
                rows_src.append(dict(jeu=MODE, source_prediction=SOURCE_PREDICTION,
                    niveau=niveau, perimetre=per, population=pop, dimension=dim,
                    modalite=etiq, couverture=round(float(m.mean()), 4),
                    accuracy_top1=round(float(accH[0][m].mean()), 4),
                    accuracy_topk=round(float(accH[-1][m].mean()), 4), n=int(m.sum())))

    syn['TAUX_GLOBAL_niv1_ok'] = round(
        float((sub[TOPCOLS['niv1'][0]] == sub['niv1_reel']).fillna(False).mean()), 4)

    ok = np.zeros(n, bool)
    for g in LEVELS:
        m = (sub['granularite'].values == g)
        if m.any():
            ok[m] = (sub[TOPCOLS[g][0]][m] == sub['%s_reel' % g][m]).fillna(False).values
    syn['acc_granularite_native'] = round(float(ok.mean()), 4)
    for g in LEVELS:
        syn['part_saisie_%s' % g] = round(float((sub['granularite'].values == g).mean()), 4)
    rows_syn.append(syn)

SYN  = pd.DataFrame(rows_syn)
RANG = pd.DataFrame(rows_rang)
SRC  = pd.DataFrame(rows_src)

ordre = (['jeu', 'source_prediction', 'perimetre', 'population', 'n_evalues',
          'TAUX_GLOBAL_niv1_ok', 'acc_granularite_native']
         + ['h_%s_n' % L for L in LEVELS]
         + ['h_%s_top%d' % (L, r) for L in LEVELS for r in (1, 2, 3)
            if not (L == 'niv1' and r > 1)]
         + ['s_%s_n' % L for L in LEVELS]
         + ['s_%s_top%d' % (L, r) for L in LEVELS for r in (1, 2, 3)
            if not (L == 'niv1' and r > 1)]
         + ['part_saisie_%s' % L for L in LEVELS])
SYN = SYN[[c for c in ordre if c in SYN.columns]]
SYN = pd.concat([SYN[(SYN.perimetre == '*TOUS*') & (SYN.population == '*TOUTES*')],
                 SYN[SYN.population != '*TOUTES*'],
                 SYN[SYN.perimetre != '*TOUS*'].sort_values('n_evalues', ascending=False)])

print("\n=== SYNTHESE (h_ = denominateur commun ; s_ = strict) ===")
print(SYN.to_string(index=False))
print("\n=== PAR RANG (*TOUS*) ===")
print(RANG[(RANG.perimetre == '*TOUS*') & (RANG.population == '*TOUTES*')].to_string(index=False))
print("\n=== PAR SOURCE / COHERENCE / LIBELLE COURT (*TOUS*) ===")
print(SRC[(SRC.perimetre == '*TOUS*') & (SRC.population == '*TOUTES*')].to_string(index=False))

dataiku.Dataset(N['syn']).write_with_schema(SYN)
dataiku.Dataset(N['rang']).write_with_schema(RANG)
dataiku.Dataset(N['src']).write_with_schema(SRC)
log("METRICS TERMINEES")







