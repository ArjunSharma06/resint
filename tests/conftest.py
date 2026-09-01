import pytest

from resint.ir.paper import Paper, ReportedMean, StatTest
from resint.ir.span import Source, Span

SRC = Source("paper.tex", "latex", path="paper.tex")


def span(start=0, end=1, line=None, label=None):
    return Span(SRC, start, end, line=line, label=label)


@pytest.fixture
def mean_factory():
    def make(raw, n, items=1, context="", offset=0):
        return ReportedMean(
            raw=raw,
            n=n,
            items=items,
            context=context,
            span=span(offset, offset + len(raw), line=10, label="results"),
            n_span=span(offset + 40, offset + 44, line=8, label="method"),
        )

    return make


@pytest.fixture
def stat_factory():
    def make(kind, statistic, df1=None, df2=None, p="0.05", comparator="=", tail=2, context=""):
        return StatTest(
            kind=kind,
            statistic_raw=statistic,
            df1=df1,
            df2=df2,
            p_raw=p,
            p_comparator=comparator,
            tail=tail,
            context=context,
            span=span(0, 12, line=120, label="results"),
            p_span=span(14, 22, line=120, label="results"),
        )

    return make


@pytest.fixture
def paper():
    return Paper(source_id="paper.tex")


# --- corpus resolution --------------------------------------------------
#
# Every answer the corpus needs, as a fixed table. No test in this suite ever
# opens a socket: resolution is the one part of the deterministic tier that
# can fail for reasons unrelated to the paper, and a suite that depended on
# it would be flaky in exactly the way that erodes trust in the tool.

from resint.resolve import Record, StaticResolver

CORPUS_RECORDS = {
    "dosovitskiy2020": Record(
        source="openalex",
        title="An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
        year="2021",
        authors=("Dosovitskiy, Alexey", "Beyer, Lucas"),
        venue="ICLR",
        doi="10.48550/arXiv.2010.11929",
    ),
    "hu2021lora": Record(
        source="crossref",
        title="LoRA: Low-Rank Adaptation of Large Language Models",
        year="2021",
        authors=("Hu, Edward", "Shen, Yelong"),
        doi="10.48550/arXiv.2106.09685",
    ),
    "never_cited2020": Record(
        source="crossref",
        title="A Perfectly Real Paper Nobody Cited Here",
        year="2020",
        authors=("Okonkwo, Ada",),
        venue="Nature Methods",
    ),
    "accented2019": Record(
        source="crossref",
        title="Étude des méthodes adaptatives",
        year="2019",
        authors=("Muller, François",),
        doi="10.1000/real.2019",
    ),
}


#: Corpus entries whose DOI the DOI system denies. Named here rather than
#: implied by absence from CORPUS_RECORDS, because absence from our indices is
#: no longer grounds for bib/unresolved to fire.
CORPUS_DEAD = ("zhang2023adaptive", "obscure2018")


@pytest.fixture(scope="session")
def corpus_resolver():
    """Resolves the real entries; zhang2023adaptive and obscure2018 do not exist.

    "Do not exist" now has to be said explicitly. Absent from ``records`` only
    means our indices missed it, which since 2026-09-01 is not grounds for
    bib/unresolved to fire -- see resolve.base.Registration.
    """
    return StaticResolver(records=dict(CORPUS_RECORDS), dead=set(CORPUS_DEAD))
