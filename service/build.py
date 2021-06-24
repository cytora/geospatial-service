# pylint: skip-file

from typing import List, Dict, Set, Callable, Union, Optional
import re
import os
import dill

import pandas as pd
import numpy as np

from itertools import chain
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct
from sklearn.feature_extraction.text import TfidfVectorizer

from datetime import datetime

PATH = "../ur_models/"


multi_whitespace = re.compile('\\s+')


def replace_multispace(s):
    '''Remove leading, trailing and duplicated whitespace'''
    s = s.strip()
    return multi_whitespace.sub(' ', s)


amp_pattern = re.compile(r'&([^/])', re.IGNORECASE)


def ampersand_replace(s):
    return re.sub(amp_pattern, r'AND\g<1>', s)


def base_cleaning(tstr):
    tstr = tstr.upper()
    tstr = re.sub(r' LTD\.', " LTD", tstr, flags = re.IGNORECASE)
    tstr = re.sub(r' LTD\b| LIMITED\b| LLP\b', " LTD", tstr, flags = re.IGNORECASE)
    tstr = replace_multispace(tstr)
    return tstr


def remove_ltd(tstr):
    return re.sub(r' LTD\b| Limited\b| llp\b', "", tstr, flags = re.IGNORECASE)


def remove_punctuation(tstr):
    tstr = re.sub(r'[,-./\']',r'', tstr)
    limited_punct = '!"#$%\'*+,-./:;<=>?@[\\]^_`{|}~()'
    tstr = tstr.translate(str.maketrans(limited_punct, ' '*len(limited_punct)))
    return tstr


def wipe_space(tstr):
    tstr = tstr.replace(r" ", "")
    return tstr

def full_clean_pipeline(tstr):
    for cleaner in [base_cleaning, remove_ltd, remove_punctuation]:
        tstr = cleaner(tstr)
    tstr = replace_multispace(tstr)
    return tstr
########## spine cleaning ######################


def _dedupe(df):
    df_dedup = df.sort_values(by=[
        'company_name',
        'change_of_name_date',
    ], ascending=[True, False]).drop_duplicates(subset=['company_name'])
    return df_dedup


def spine_processing_with_dedupe(df):
    df = df.copy()
    df['company_name_raw'] = df['company_name'].str.upper()
    df['company_name'] = df['company_name'].apply(base_cleaning)
    return _dedupe(df)


# Tokenizers
def unlist(listOfLists: List[Union[str, List[str]]]) -> List[str]:
    return list(chain.from_iterable(listOfLists))


def ngram(tstr: str, n: int=3) -> List[str]:
    '''Create list of all n-grams for the input string'''
    ngrams = zip(*[tstr[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]


def ngrams(tstring: str, nmin: int=3, nmax: int=4) -> List[str]:
    retlist = [ngram(tstring, n=x) for x in range(nmin, nmax+1)]
    retlist = unlist(retlist)
    return retlist


def words(tstring: str, ignore_words: bool=False) -> List[str]:
    twords = tstring.split(" ")
    if ignore_words:
        iwvec = ['services',
                 'management',
                 'the',
                 'and',
                 'company',
                 'solutions',
                 'property',
                 'consulting',
                 'group',
                 'properties',
                 'holdings',
                 'consultancy',
                 'investments',
                 '(uk)',
                 'construction',
                 'developments',
                 'international',
                 'engineering',
                 'design',
                 'london',
                 'care',
                 'trading',
                 'electrical',
                 'house',
                 'associates',
                 'business',
                 'building',
                 'road',
                 'consultants',
                 'capital',
                 'global',
                 'homes',
                 'media',
                 'systems',
                 'centre',
                 'transport',
                 'estates',
                 'financial',
                 'enterprises',
                 'training',
                 'court',
                 'development',
                 'cic',
                 'partnership',
                 'club',
                 'partners',
                 'community',
                 'investment',
                 'productions',
                 'park',
                 'home',
                 'marketing',
                 'technology',
                 'security',
                 'energy',
                 'health',
                 'logistics',
                 'cleaning',
                 'digital',
                 'heating',
                 'project',
                 'finance',
                 'plumbing',
                 'trust',
                 'maintenance',
                 'association',
                 'green',
                 'service',
                 'medical',
                 'healthcare',
                 'west',
                 'for',
                 'contractors',
                 'estate',
                 'street',
                 'recruitment',
                 'technologies',
                 'beauty',
                 'new',
                 'projects']
        # words are only ignored as words - but included in the ngrams
        twords = [x for x in twords if x.lower() not in iwvec]
    return twords


def ngrams_and_words(
    tstring: str,
    nmin: int=3,
    nmax: int=4,
    ignore_words: bool=False,
    remove_spaces: bool=False) -> Set[str]:

    tstring = replace_multispace(tstring)
    twords = words(tstring, ignore_words=ignore_words)

    if remove_spaces:
        tstring = wipe_space(tstring)

    tngrams = ngrams(tstring, nmin=nmin, nmax=nmax)
    retlist = twords + tngrams
    return set(retlist)


def awesome_cossim_top(queries_matrix, spine_matrix, ntop: int, lower_bound: float=0.0) -> csr_matrix:  # TODO: what are a and b? what is output?
    # force queries_matrix and spine_matrix as a CSR matrix.
    # If they have already been CSR, there is no overhead
    queries_matrix = queries_matrix.tocsr()
    spine_matrix = spine_matrix.tocsr()
    M, _ = queries_matrix.shape
    _, N = spine_matrix.shape

    idx_dtype = np.int32

    nnz_max = M*ntop

    indptr = np.zeros(M+1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=queries_matrix.dtype)

    ct.sparse_dot_topn(
        M, N, np.asarray(queries_matrix.indptr, dtype=idx_dtype),
        np.asarray(queries_matrix.indices, dtype=idx_dtype),
        queries_matrix.data,
        np.asarray(spine_matrix.indptr, dtype=idx_dtype),
        np.asarray(spine_matrix.indices, dtype=idx_dtype),
        spine_matrix.data,
        ntop,
        lower_bound,
        indptr, indices, data)

    return csr_matrix((data,indices,indptr),shape=(M,N))


class Match:
    def __init__(self, query: str, match: str, crn: str, similarity: float) -> None:
        self.query = query
        self.match = match
        self.crn = crn
        self.similarity = similarity


class Query:
    def __init__(self, query: str, min_confidence: float, max_number_returns:int=1, caller_uid: Optional[str]=None, client_query_id: Optional[str]=None) -> None:
        self.query = query.upper()
        self.min_confidence = min_confidence
        self.max_number_returns = max_number_returns
        self.caller_uid = caller_uid
        self.client_query_id = client_query_id


class TFIDFObject:
    def __init__(self, data: pd.DataFrame, analyser: Callable[[str], List[str]]) -> None:
        self.vectorizer = TfidfVectorizer(min_df=1, analyzer=analyser)
        self.tf_idf_matrix = self.vectorizer.fit_transform(data['company_name'])
        self.source_strings = data['company_name_raw']
        self.source_crns = data['crn']


class TfidfMatcher:
    def __init__(self,
        spine_data: pd.DataFrame,
        cleaner: Callable[[str], str],
        tokeniser: Callable[[str], List[str]],
        spine_processing: Optional[Callable[[pd.DataFrame], pd.DataFrame]]=None
    ) -> None:

        data = spine_data.copy()
        if spine_processing is not None:
            data = spine_processing(data)

        analyser = lambda x: tokeniser(cleaner(x))
        self.vectorizer = TfidfVectorizer(min_df=1, analyzer=analyser)
        self.spine_matrix = self.vectorizer.fit_transform(data['company_name']).transpose().tocsr()
        self.source_strings = data['company_name_raw']
        self.source_crns = data['crn']


    def tfidf_cossim(self, queries: pd.Series, lower_bound: float, ntop: int) -> Union[List[Match], pd.DataFrame]:
        ko = awesome_cossim_top(
            queries_matrix=self.vectorizer.transform(queries),
            spine_matrix=self.spine_matrix,
            ntop=ntop,
            lower_bound=lower_bound
        )

        name_vector = self.source_strings
        crns = self.source_crns
        non_zeros = ko.nonzero()

        sparserows = non_zeros[0]
        sparsecols = non_zeros[1]

        assert(len(sparserows) == len(sparsecols))

        nr_matches = len(sparsecols)
        ids = np.arange(nr_matches)
        left_side = queries.iloc[sparserows]
        right_side = name_vector.iloc[sparsecols]
        crn = crns.iloc[sparsecols]
        similarity = ko.data[ids]

        results = {
            'query': left_side,
            'match': right_side.values,
            'crn': crn.values,
            'similarity': similarity
        }
        return results

    def search_batch(self, queries: pd.Series, threshold: float, ntop: int=1) -> pd.DataFrame:
        clean_queries = queries.copy()
        clean_queries = clean_queries.astype(str)
        clean_queries = clean_queries.str.upper()
        results = self.tfidf_cossim(clean_queries, threshold, ntop)
        return pd.DataFrame(results)

    def search(self, query: str, threshold: float, ntop: int) -> List[Match]:
        # the tfidf cossim includes the cleaning and tokenisation - that's why the cleaning function is not used here
        try:
            results = self.tfidf_cossim(pd.Series([query]), threshold, ntop)
            matches = []
            nr_results = len(results['match'])
            for i in range(nr_results):
                matches.append(Match(
                    query=results['query'].values[i],
                    match=results['match'][i],
                    crn=results['crn'][i],
                    similarity=results['similarity'][i],
                    )
                )
        except IndexError:
            matches = []
        return matches


class ExactMatcher:
    '''Matcher trying to find an exact string match between query and candidates on spine_data
    '''
    def __init__(self,
        spine_data: pd.DataFrame,
        cleaner: Callable[[str], str],
        spine_processing: Optional[Callable[[pd.DataFrame], pd.DataFrame]]=None
    ) -> None:
        data = spine_data.copy()
        if spine_processing is not None:
            data = spine_processing(data)

        assert ~data['company_name'].duplicated().any(), 'Exact matcher does not support duplicates on spine currently'
        self.spine = dict(zip(data['company_name'], zip(data['crn'], data['company_name_raw'])))
        self.cleaner = cleaner

    def resolve(self, query: str) -> List[Optional[Match]]:
        clean_query = query
        clean_query = self.cleaner(clean_query)
        match = self.spine.get(clean_query)
        if match:
            crn, company_name = match
            match = Match(
                query=query,
                match=company_name,
                crn=crn,
                similarity=1.0,
            )
            match = [match]
        else:
            match = []
        return match

    def resolve_batch(self, queries: pd.Series) -> pd.DataFrame:
        index_name = queries.index.name
        clean_queries = queries.copy()
        clean_queries.name = 'orig_query'
        clean_queries = clean_queries.to_frame()
        clean_queries['orig_query'] = clean_queries['orig_query'].str.upper()
        clean_queries['clean_query'] = clean_queries['orig_query'].astype(str)
        clean_queries['clean_query'] = clean_queries['clean_query'].apply(self.cleaner)
        clean_queries = clean_queries[['clean_query', 'orig_query']]

        results = {}
        for row in clean_queries.itertuples():
            match = self.spine.get(row.clean_query)
            if match:
                crn_found, name_found = match
                results[row.Index] = {
                    'query': row.orig_query,
                    'match': name_found,
                    'crn': crn_found,
                    'similarity': 1.0,
                }
        results = pd.DataFrame(results).T
        if not results.empty:
            results = results[['query', 'match', 'crn', 'similarity']]
            results['similarity'] = results['similarity'].astype(float)
            results.index.name = index_name
        return results


class CombinedMatcher:
    def __init__(self, first_matcher: ExactMatcher, backup_matcher: TfidfMatcher, similarity_adjustment: float=.99) -> None:
        self.first_matcher = first_matcher
        self.backup_matcher = backup_matcher
        self.similarity_adjustment = similarity_adjustment

    def _combine_exact_and_fuzzy_match_results(self, exact: pd.DataFrame, fuzzy: pd.DataFrame) -> pd.DataFrame:
        fuzzy = fuzzy.copy()
        fuzzy['similarity'] = fuzzy['similarity'] * self.similarity_adjustment
        return pd.concat([exact, fuzzy])

    def _result_similarity_adjustment(self, results: List[Match]) -> List[Match]:
        for r in results:
            r.similarity *= self.similarity_adjustment
        return results

    def resolve_batch(self, queries: pd.Series, threshold: float) -> pd.DataFrame:
        exact_match_results = self.first_matcher.resolve_batch(queries)
        unresolved_rows = set(queries.index) - set(exact_match_results.index)
        unresolved = queries.loc[unresolved_rows]
        results = self.backup_matcher.search_batch(unresolved, ntop=1, threshold=threshold)
        return self._combine_exact_and_fuzzy_match_results(exact_match_results, results)

    def _resolve(self, query: str, threshold:float) -> List[Match]:
        '''Find optimal match for query
        Find the optimal match for the query string by first trying to get a result
        using the first_matcher. If no result is found, try to get result through
        the backup matcher, returning any found match only if match confidence
        exceeds threshold.
        '''
        result = self.first_matcher.resolve(query)
        if not result:
            result = self.backup_matcher.search(query, threshold=threshold, ntop=1)
            result = self._result_similarity_adjustment(result)
        return result

    def _search(self, query: str, threshold: float, ntop: int) -> List[Match]:
        exact_match = self.first_matcher.resolve(query)
        if exact_match:
            nr_exact_matches = len(exact_match)
            slots_to_fill = ntop - nr_exact_matches
            results = self.backup_matcher.search(query, threshold=threshold, ntop=slots_to_fill)
            results = exact_match + self._result_similarity_adjustment(results)
        else:
            results = self.backup_matcher.search(query, threshold=threshold, ntop=ntop)
            results = self._result_similarity_adjustment(results)
        return results

    def search(self, input_query: Query) -> List[Match]:
        if input_query.max_number_returns == 1:
            return self._resolve(query=input_query.query, threshold=input_query.min_confidence)
        else:
            return self._search(query=input_query.query, threshold=input_query.min_confidence, ntop=input_query.max_number_returns)

# needed for the typehinting.
Matcher = Union[TfidfMatcher, CombinedMatcher]


def model_dump(matcher, matcher_name: str) -> str:
    now = datetime.now()
    now = now.strftime("%Y%m%d%H%M%S")

    os.makedirs(PATH, exist_ok=True)
    file_name = PATH + f'{matcher_name}{now}.pkl'

    with open(file_name, 'wb') as outcome_file:
        dill.dump(matcher, outcome_file, recurse=True)
    return file_name


def model_load(file: str):
    with open(file, 'rb') as outcome_file:
        pickle_matcher = dill.load(outcome_file)
    return pickle_matcher


def build(fast_matcher: bool = True, fuzzy_matcher: bool = False):
    #ch = pd.read_parquet('gs://cytora-universal-resolver-sandbox/ch_spine_temp_snapshots/CH_March_with_dissolved_1000.pq')
    #ch = pd.read_parquet('/Users/todorlubenov/Downloads/ch_spine_temp_snapshots-CH_March_with_dissolved.pq')
    ch = pd.read_parquet('/Users/todorlubenov/Downloads/ch_spine_temp_snapshots-CH_March_with_dissolved_100000.pq')

    # rename and delete columns: this needs to be removed when using data from athena
    columns_rename = {
        'CompanyName': 'company_name',
        'CompanyNumber': 'crn',
        'Change_of_name_date': 'change_of_name_date',
        'DissolutionDate': 'dissolution_date',
        'RegAddress.AddressLine1': 'registered_address'
        }
    ch.rename(columns=columns_rename, inplace=True)
    ch.drop(columns =[
        'RegAddress.AddressLine2',
        'RegAddress.PostTown',
        'RegAddress.County',
        'RegAddress.Country',
        'RegAddress.PostCode'
    ], inplace=True)

    matchers = {}

    if fast_matcher:
        fast_matcher = ExactMatcher(
            spine_data=ch,
            cleaner=base_cleaning,
            spine_processing=spine_processing_with_dedupe,
        )
        fast_matcher = model_dump(fast_matcher, 'fast_matcher')
        matchers['fast_matcher'] = fast_matcher

    if fuzzy_matcher:
        fuzzy_matcher = TfidfMatcher(
            spine_data=ch,
            cleaner=lambda x: full_clean_pipeline(ampersand_replace(x)),
            tokeniser=lambda x: ngrams_and_words(x, remove_spaces=True),
            spine_processing=spine_processing_with_dedupe,
        )
        fuzzy_matcher = model_dump(fuzzy_matcher, 'fuzzy_matcher')
        matchers['fuzzy_matcher'] = fuzzy_matcher

    #matcher = CombinedMatcher(fast_matcher, fuzzy_matcher)
    #model_dump(matcher)
    return matchers


if __name__ == '__main__':
    import cProfile
    from pstats import Stats, SortKey

    do_profiling = True
    if do_profiling:
        with cProfile.Profile() as pr:
            build(fast_matcher=True, fuzzy_matcher=True)

        now = datetime.now()
        now = now.strftime("%Y%m%d_%H%M%S")

        with open(f'profiling_stats_{now}.txt', 'w') as stream:
            stats = Stats(pr, stream=stream)
            stats.strip_dirs()
            stats.sort_stats('time')
            stats.dump_stats('.prof_stats')
            stats.print_stats()
    else:
        print(f'-------------------------')

