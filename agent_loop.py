# -*- coding: utf-8 -*-
"""
=========================================================================
 GM_COMMON  --  bibliotheque partagee
=========================================================================
 A deposer dans Dataiku :  Code  >  Libraries  >  python/gm_common.py
 Chaque recipe fait ensuite :  from gm_common import *

 Contient UNIQUEMENT de la logique, aucun nom de dataset ni de folder :
 ceux-ci sont declares en tete de chaque recipe (bloc ENTREES / SORTIES).

 Commun aux filieres PROD et DEV :
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

# ========================= REFERENTIEL MAPPING =======================
def charger_mapping(nom):
    """Mapping(MATKL, LIBEL) -> dict de travail. `nom` est fourni par la recipe :
    aucun nom de dataset n'est ecrit dans cette librairie.
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
def ouvrir_folder(identifiant):
    """Ouvre un managed folder par son IDENTIFIANT (8 caracteres, attribue par
    Dataiku a la creation) ou par son nom si la version le permet.
    Ou le trouver : Flow > clic sur le folder > le panneau de droite affiche
    l'ID, qu'on retrouve aussi dans l'URL .../managedfolder/<ID>/view."""
    import dataiku
    f = dataiku.Folder(identifiant)
    try:
        f.list_paths_in_partition()
    except Exception as e:
        raise RuntimeError(
            "Managed folder '%s' introuvable. Copier l'ID depuis le Flow "
            "dans la constante du haut de la recipe. (%s)" % (identifiant, e))
    return f

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
