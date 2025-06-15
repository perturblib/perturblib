"""Copyright (C) 2025  GlaxoSmithKline plc

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pandas as pd
import polars as pl
import polars.testing

import perturb_lib as plib
from perturb_lib.data.access import encode_data


def test_vocabularies():
    dummy_adata = plib.load_anndata("DummyData")
    vocab = plib.Vocabulary.initialize_from_data(dummy_adata)

    assert isinstance(vocab.context_vocab, pl.DataFrame)
    assert isinstance(vocab.perturb_vocab, pl.DataFrame)
    assert isinstance(vocab.readout_vocab, pl.DataFrame)

    for vocab_component in [vocab.context_vocab, vocab.perturb_vocab, vocab.readout_vocab]:
        assert "symbol" in vocab_component.columns
        assert "code" in vocab_component.columns

    assert vocab.perturb_vocab.row(by_predicate=pl.col("symbol") == plib.ControlSymbol, named=True)["code"] == 0


def test_encode_data_adata():
    dummy_adata = plib.load_anndata("DummyData")
    vocab = plib.Vocabulary.initialize_from_data(dummy_adata)

    dummy_adata_encoded = encode_data(dummy_adata, vocab)
    assert pd.api.types.is_integer_dtype(dummy_adata_encoded.obs.context)
    assert pd.api.types.is_object_dtype(dummy_adata_encoded.obs.perturbation)
    assert pd.api.types.is_integer_dtype(dummy_adata_encoded.var.readout)


def test_encode_pdata(list_of_plibdata):
    vocabs = [plib.Vocabulary.initialize_from_data(pdata) for pdata in list_of_plibdata]
    pdata_encoded = [encode_data(pdata, vocab) for pdata, vocab in zip(list_of_plibdata, vocabs)]

    # Vocabularies are the same
    for vocab in vocabs:
        pl.testing.assert_frame_equal(vocab.context_vocab, vocabs[0].context_vocab)
        pl.testing.assert_frame_equal(vocab.perturb_vocab, vocabs[0].perturb_vocab)
        pl.testing.assert_frame_equal(vocab.readout_vocab, vocabs[0].readout_vocab)

    for pdata in pdata_encoded:
        pdata_df = pdata[:]
        assert dict(pdata_df.schema.items()) == {
            "context": pl.Int64,
            "perturbation": pl.List(pl.Int64),
            "readout": pl.Int64,
            "value": pl.Float32,
        }

    for pdata in pdata_encoded:
        pl.testing.assert_frame_equal(pdata_encoded[0][:], pdata[:])


def test_encode_data_dataframe():
    dummy_df = plib.load_plibdata("DummyData")[:]
    vocab = plib.Vocabulary.initialize_from_data(dummy_df)

    dummy_df_encoded = encode_data(dummy_df, vocab)
    assert dict(dummy_df_encoded.schema.items()) == {
        "context": pl.Int64,
        "perturbation": pl.List(pl.Int64),
        "readout": pl.Int64,
        "value": pl.Float32,
    }
