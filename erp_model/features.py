import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer

class Features:
    def __init__(self, slot, structured=True):
        self.slot=slot;self.structured=structured
        self.text=[TfidfVectorizer(analyzer='char',ngram_range=(2,4),min_df=2,max_features=10000,dtype=np.float32) for _ in range(2)]
        self.dv=DictVectorizer(dtype=np.float32)
        if slot=='part_number':
            self.cv=TfidfVectorizer(analyzer='char',ngram_range=(2,4),min_df=2,max_features=5000,dtype=np.float32)
            self.cdv=DictVectorizer(dtype=np.float32)
    def fit(self, rows, shared_text=None):
        # A baseline bundle keeps exactly its historical feature width. Encoder
        # vectors are enabled only when every fitting row carries one.
        embeddings=[getattr(r,'_shared_char_embedding',None) for r in rows]
        present=[value is not None for value in embeddings]
        if any(present) and not all(present):
            raise ValueError('Encoder features must be present on every fitting row')
        self.encoder_dimensions=0
        if embeddings and all(present):
            matrix=self._embedding_matrix(rows)
            self.encoder_dimensions=matrix.shape[1]
        if shared_text is not None:self.text=shared_text
        for vec,col in zip(self.text,['품목명','규격명']):
            vals=[r.normalized[col] for r in rows]
            if shared_text is not None:continue
            try:vec.fit(vals)
            except ValueError:vec.set_params(min_df=1);vec.fit(vals+['EMPTY'])
        self.dv.fit([r.structured[self.slot] or {'empty':1} for r in rows])
        if self.slot=='part_number':
            cs=[c for r in rows for c in r.candidates]
            self.cv.fit([c['value'] for c in cs] or ['EMPTY'])
            self.cdv.fit([c['features'] for c in cs] or [{'empty':1}])
        return self
    def transform(self, rows):
        cache_id=id(self.text[0])
        if all(getattr(r,'_text_cache',None) is not None and r._text_cache[0]==cache_id for r in rows):
            base=vstack([r._text_cache[1] for r in rows],format='csr')
        else:
            base=hstack([v.transform([r.normalized[c] for r in rows]) for v,c in zip(self.text,['품목명','규격명'])],format='csr')
            for i,r in enumerate(rows):r._text_cache=(cache_id,base[i])
        xs=[base]
        if self.structured:xs.append(self.dv.transform([r.structured[self.slot] or {'empty':1} for r in rows]))
        dimensions=getattr(self,'encoder_dimensions',0)
        if dimensions:
            provider=getattr(self,'_encoder_provider',None)
            if provider is not None:provider(rows)
            vectors=self._embedding_matrix(rows)
            if vectors.shape[1]!=dimensions:
                raise ValueError('Encoder embedding width differs from the fitted feature contract')
            xs.append(csr_matrix(vectors))
        return hstack(xs,format='csr',dtype=np.float32)
    @staticmethod
    def _embedding_matrix(rows):
        vectors=[]
        width=None
        for row in rows:
            value=getattr(row,'_shared_char_embedding',None)
            if value is None:
                raise ValueError('The fitted tree model requires a shared character encoder embedding')
            value=np.asarray(value,dtype=np.float32)
            if value.ndim!=1 or not value.size or not np.isfinite(value).all():
                raise ValueError('Encoder embeddings must be finite, nonempty one-dimensional vectors')
            if width is not None and len(value)!=width:
                raise ValueError('Encoder embedding widths are inconsistent')
            width=len(value);vectors.append(value)
        return np.stack(vectors).astype(np.float32,copy=False)
    def candidate_matrix(self, rows):
        counts=[len(r.candidates) for r in rows]; cs=[c for r in rows for c in r.candidates]
        if not cs:return None,counts
        context=self.transform(rows)[np.repeat(np.arange(len(rows)),counts)]
        xs=[context,self.cv.transform([c['value'] for c in cs])]
        if self.structured:xs.append(self.cdv.transform([c['features'] for c in cs]))
        return hstack(xs,format='csr',dtype=np.float32),counts
