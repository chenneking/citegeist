import pandas as pd

data = """
id: 0704.0001
submitter: Pavel Nadolsky
authors: C. Bal\'azs, E. L. Berger, P. M. Nadolsky, C.-P. Yuan
title: Calculation of prompt diphoton production cross sections at Tevatron and
  LHC energies
comments: 37 pages, 15 figures; published version
journal-ref: Phys.Rev.D76:013009,2007
doi: 10.1103/PhysRevD.76.013009
report-no: ANL-HEP-PR-07-12
categories: hep-ph
license: None
abstract:   A fully differential calculation in perturbative quantum chromodynamics is
presented for the production of massive photon pairs at hadron colliders. All
next-to-leading order perturbative contributions from quark-antiquark,
gluon-(anti)quark, and gluon-gluon subprocesses are included, as well as
all-orders resummation of initial-state gluon radiation valid at
next-to-next-to-leading logarithmic accuracy. The region of phase space is
specified in which the calculation is most reliable. Good agreement is
demonstrated with data from the Fermilab Tevatron, and predictions are made for
more detailed tests with CDF and DO data. Predictions are shown for
distributions of diphoton pairs produced at the energy of the Large Hadron
Collider (LHC). Distributions of the diphoton pairs from the decay of a Higgs
boson are contrasted with those produced from QCD processes at the LHC, showing
that enhanced sensitivity to the signal can be obtained with judicious
selection of events.
"""

def get_field(data, field):
    for line in data.split('\n'):
        if line.startswith(field):
            return line[len(field)+2:]
    return None

data = data.strip()
id = get_field(data, 'id')
submitter = get_field(data, 'submitter')
authors = get_field(data, 'authors')
title = get_field(data, 'title')
comments = get_field(data, 'comments')
journal_ref = get_field(data, 'journal-ref')
doi = get_field(data, 'doi')
report_no = get_field(data, 'report-no')
categories = get_field(data, 'categories')
license = get_field(data, 'license')

df = pd.DataFrame(
    {
        'id': [id],
        'submitter': [submitter],
        'authors': [authors],
        'title': [title],
        'comments': [comments],
        'journal-ref': [journal_ref],
        'doi': [doi],
        'report-no': [report_no],
        'categories': [categories],
        'license': [license],
    }
)

print(df)


from pymilvus import MilvusClient, utility

EMBEDDING_PATH = "./database_mini.db"
# Connect to Milvus server
mini_client = MilvusClient(EMBEDDING_PATH)
# print the first row of the database
# print(utility.list_collections())

query = mini_client.query(collection_name="abstracts", limit=1)


print(query)