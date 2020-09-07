import numpy as np
import pandas as pd
import gzip
from scipy.misc import toimage
from Bio import SwissProt


# Plots NxM matrix as pixels with positive (green), negative (red) and zero (black) values
def plot_array(fr, fn):
    ar = fr.values
    n1, n2 = ar.shape
    arr = np.zeros([n1,n2,3])
    for i in range(n1):
        for j in range(n2):
            if ar[i,j] > 0.0:
                arr[i,j,:] = (0, 255, 0)
            elif ar[i,j] < 0.0:
                arr[i,j,:] = (255, 0, 0)
            else:
                arr[i,j,:] = (0, 0, 0)
    toimage(arr).save(fn)

# WI-PHI protein-protein interaction (PPI) data set
# https://application.wiley-vch.de/contents/jc_2120/2007/pro200600448_3_s.xls
df = pd.read_csv('cyc_data/pro200600448_3_s.csv', sep=';')
df['Score'] = df['Score'].apply(lambda s: '.'.join(s.split(',')))
df['Score'] = df['Score'].astype(float)
n = len(np.unique(df[['Protein A', 'Protein B']].values))
print("=== WI-PHI ===\n" \
      "\tproteins x proteins: %d x %d\n" \
      "\tsamples: %d" % (n, n, len(df)))

# CYC2008 protein complex data set
#http://wodaklab.org/cyc2008/resources/CYC2008_complex.tab
fr = pd.read_csv('cyc_data/CYC2008_complex_sql.txt',sep='\t')
sizes = fr.groupby('Complex')['Complex'].agg('count')
n = len(np.unique(fr['ORF']))
print("=== CYC2008 ===\n" \
      "\tproteins x proteins: %d x %d\n" \
      "\tcomplexes: %d" % (n, n, len(np.unique(fr['Complex']))))
#print "Protein complex sizes:"
#print sizes.value_counts().sort_index()

# Add 'Size' column to each protein complex
fr['Size'] = fr.groupby('Complex')['Complex'].transform(lambda s: len(s) * np.ones(len(s)))
heterodimers = fr.loc[fr['Size'] == 2, ['ORF','Complex']]
others = fr.loc[fr['Size'] > 2, ['ORF', 'Complex']]

# Set of {Protein_1, Protein_2} pairs in all protein-protein interactions
ppi_pairs = set([frozenset((p1, p2)) for p1, p2 in zip(df['Protein A'], df['Protein B'])])

# Set of {Protein_1, Protein_2} pairs in all heterodimers and other complexes
pos = heterodimers.merge(heterodimers, on='Complex')
neg = others.merge(others, on='Complex')
pos_pairs = set([frozenset((p1, p2)) for p1, p2 in zip(pos['ORF_x'], pos['ORF_y']) if p1 != p2])
neg_pairs = set([frozenset((p1, p2)) for p1, p2 in zip(neg['ORF_x'], neg['ORF_y']) if p1 != p2])
# Positive example
# (i) it is a heterodimeric protein complex in CYC2008
# (ii) it is not a proper subset of any other complex in CYC2008
# (iii) WI-PHI includes the PPI corresponding to it.
# The total number of the resulting positive examples is 152.
pos = pos_pairs.difference(neg_pairs)
pos = pos.intersection(ppi_pairs)
# Negative example
# (i) it is a protein complex whose size is three or more in CYC2008
# (ii) has the corresponding PPI in WI-PHI
# (iii) is not identical to any heterodimeric protein complexes
# The total number of those negative examples is 5345.
neg = neg_pairs.difference(pos_pairs)
neg = neg.intersection(ppi_pairs)

# Save proteins x proteins pixel image with positive (green) and negative (red) heterodimers
proteins = fr['ORF'].unique()
proteins.sort()
pairs = pd.DataFrame(np.zeros((len(proteins), len(proteins))), index=proteins, columns=proteins)
for p1, p2 in pos:
    pairs.loc[p1, p2] = 1
for p1, p2 in neg:
    pairs.loc[p1, p2] = -1
n = len(proteins)
print("=== Data Set ===\n" \
      "\tproteins x proteins: %d x %d\n" \
      "\tsamples: %d/%d" % (n, n, len(pos), len(neg)))
fn = 'cyc_data/heterodimers.png'
print("saving %s..." % fn)
plot_array(pairs, fn)

# Save proteins x proteins heterodimer data set
pairs = []
for p1, p2 in pos:
    pairs.append((p1, p2, 1))
for p1, p2 in neg:
    pairs.append((p1, p2, -1))
pairs = pd.DataFrame(pairs, columns=['Protein A', 'Protein B', 'Y'])
pairs = pairs.sample(frac=1, replace=False)
pairs.to_csv('cyc_data/heterodimers.csv', sep=';', index=False)

# Save protein x protein pixel image with >= median (green) and < median (red) interactions
df['Interacts'] = 1
proteins = np.unique(df[['Protein A', 'Protein B']].values)
interactions = pd.pivot_table(df, values='Interacts', index='Protein A', columns='Protein B', fill_value=0)
interactions = interactions.reindex(index=proteins, columns=proteins, fill_value=0)
fn = 'cyc_data/PPIs.png'
print("saving %s..." % fn)
plot_array(interactions, fn)

unique_proteins = np.unique(pairs[['Protein A', 'Protein B']].values)
pd.Series(unique_proteins).to_csv('cyc_data/unique_proteins.csv', sep=';')


#TODO: replace record.gene_name and record.comments parsing with formal regular expressions
# Split a string 'key=value {asdf}; another_key=another_value; ...' into
# python dictionary {key: value, another_key: another_value, ...}
def identifier_dict(id):
    d = {}
    strings = id.split(';')
    for string in strings:
        if string:
            optional = string.find('{')
            if optional >= 0:
                string = string[:optional]
            string = string.strip()
            try:
                key, value = string.split('=')
                d[key] = value
            except ValueError:
                print("Invalid:", string)
    return d

# Data source:
# ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/taxonomic_divisions/

# Create a dictionary of protein, domain pairs {protein_1: [domain_1, ... domain_n], ... }
print("Extracting (protein, domain)-pairs...")
handle = gzip.open("cyc_data/uniprot_sprot_fungi.dat.gz", "r")
records = {}
for record in SwissProt.parse(handle):
    domains = []
    identifier = identifier_dict(record.gene_name)
    try:
        protein_identifier = identifier['OrderedLocusNames']
    except KeyError:
        protein_identifier = None
    if protein_identifier in unique_proteins:
        domains = [reference[1] for reference in record.cross_references if reference[0] == 'InterPro']
        records[protein_identifier] = domains

# Create a binary matrix of protein, domain mappings {protein_i, domain_j} = I[protein_i contains domain_j]
print("Saving (protein, domain)-pairs...")
proteins = list(records.keys())
domains = list(set([domain for domain_list in records.values() for domain in domain_list]))
proteins.sort()
domains.sort()
mapping = pd.DataFrame(np.zeros((len(proteins), len(domains)), dtype=int), index=proteins, columns=domains)
for protein, domain_list in records.items():
    for domain in domain_list:
        mapping.loc[protein, domain] = 1
mapping.to_csv('cyc_data/protein_domain_map.csv', sep=';')

# #TODO: add TrEMBL to sprot based feature representations
# # Same from TrEMBL
# handle = gzip.open("cyc_data/uniprot_trembl_fungi.dat.gz", "r")
# records2 = {}
# for record in SwissProt.parse(handle):
#     domains = []
#     identifier = identifier_dict(record.gene_name)
#     try:
#         protein_identifier = identifier['Name']
#     except KeyError:
#         protein_identifier = None
#     if protein_identifier in unique_proteins:
#         domains = [reference[1] for reference in record.cross_references if reference[0] == 'InterPro']
#         records2[protein_identifier] = domains
#
# # Same from TrEMBL
# proteins = list(records2.keys())
# domains = list(set([domain for domain_list in records2.values() for domain in domain_list]))
# proteins.sort()
# domains.sort()
# mapping2 = pd.DataFrame(np.zeros((len(proteins), len(domains)), dtype=int), index=proteins, columns=domains)
# for protein, domain_list in records2.items():
#     for domain in domain_list:
#         mapping2.loc[protein, domain] = 1
# mapping2.to_csv('cyc_data/protein_domain_map2.csv', sep=';')

# Data source:
# http://www.uniprot.org/locations/?columns=id&offset=0
subcellular_locations = np.unique(pd.read_csv('cyc_data/locations-all.tab', sep='\t')['Alias'].str.lower())

# Create a dictionary of protein, subcellular location pairs {protein_1: [location_1, ... location_n], ... }
print("Extracting (protein, location)-pairs...")
handle = gzip.open("cyc_data/uniprot_sprot_fungi.dat.gz", "r")
records3 = {}
for i, record in enumerate(SwissProt.parse(handle)):
    locations = []
    identifier = identifier_dict(record.gene_name)
    try:
        protein_identifier = identifier['OrderedLocusNames']
    except KeyError:
        protein_identifier = None
    if protein_identifier in unique_proteins:
        for comment in record.comments:
            if comment.startswith('SUBCELLULAR LOCATION:'):
                comment = comment.replace('SUBCELLULAR LOCATION:', '' )
                annotations = [a3 for a1 in comment.split(';') for a2 in a1.split('.') for a3 in a2.split(',')]
                for annotation in annotations:
                    optional = annotation.find('{')
                    if optional >= 0:
                        annotation = annotation[:optional]
                    annotation = annotation.strip().lower()
                    if annotation in subcellular_locations:
                        locations.append(annotation)
        records3[protein_identifier] = locations

# Create a binary matrix of protein, location mappings {protein_i, location_j} = I[protein_i contained in location_j]
print("Saving (protein, location)-pairs...")
proteins = list(records3.keys())
locations = list(set([location for location_list in records3.values() for location in location_list]))
proteins.sort()
locations.sort()
mapping3 = pd.DataFrame(np.zeros((len(proteins), len(locations)), dtype=int), index=proteins, columns=locations)
for protein, location_list in records3.items():
    for location in location_list:
        mapping3.loc[protein, location] = 1
mapping3.to_csv('cyc_data/protein_location_map.csv', sep=';')

# Data source:
# Online Viewer: http://www.genome.jp/tools-bin/ocv
# API: http://www.genome.jp/tools/gn_ga_tools_api.html
# (http://www.genome.jp/dbget-bin/www_bget?sce:YKR068C implies 'Organism code' is 'sce')

# Create a dictionary of protein, genome pairs {protein_1: [genome_1, ... genome_n], ... }
print("Extracting (protein, genome)-pairs...")
results4 = {}
for protein in unique_proteins:
    print(protein)
    try:
        fr = pd.read_csv('http://rest.genome.jp/oc/sce:%s' % protein, sep='\t', skiprows=1)#, comment="#")
        results4[protein] = fr['PC'].values
    except:
        print("no data:", protein)
results5 = {}
for protein, genome_list in results4.items():
    results5[protein] = list(set([genome.split('.')[0] for genome in genome_list]))
#import pickle
#pickle.dump(results4,open("results4.p", "wb" ))
#results4 = pickle.load(open("results4.p", "rb" ))

# Create a binary matrix of protein, genome mappings {protein_i, genome_j} = I[protein_i contained in genome_j]
print("Saving (protein, genome)-pairs...")
proteins = list(results5.keys())
genomes = list(set([genome for genome_list in results5.values() for genome in genome_list]))
proteins.sort()
genomes.sort()
mapping4 = pd.DataFrame(np.zeros((len(proteins), len(genomes)), dtype=int), index=proteins, columns=genomes)
for protein, genome_list in results5.items():
    for genome in genome_list:
        mapping4.loc[protein, genome] = 1
mapping4.to_csv('cyc_data/protein_genome_map.csv', sep=';')


# Statistics
domain_map = pd.read_csv('cyc_data/protein_domain_map.csv', sep=';', index_col=0)
location_map = pd.read_csv('cyc_data/protein_location_map.csv', sep=';', index_col=0)
genome_map = pd.read_csv('cyc_data/protein_genome_map.csv', sep=';', index_col=0)
domain_map = domain_map.reindex(index=unique_proteins, fill_value=0)
location_map = location_map.reindex(index=unique_proteins, fill_value=0)

# proteins x domains
print(domain_map.shape)
# proteins x locations
print(location_map.shape)
# proteins x genomes
print(genome_map.shape)

# Number of domains in a protein
print(domain_map.sum(axis=1).value_counts().sort_index())
# Number of proteins in a domain
print(domain_map.sum(axis=0).value_counts().sort_index())
# Number of locations in a protein
print(location_map.sum(axis=1).value_counts().sort_index())
# Number of proteins in a location
print(location_map.sum(axis=0).value_counts().sort_index())
# Number of genomes in a protein
print(genome_map.sum(axis=1).value_counts().sort_index())
# Number of proteins in a genome
print(genome_map.sum(axis=0).value_counts().sort_index())

def stats(mapping):
    has_proteins = mapping.index
    intersection = set(unique_proteins).intersection(set(has_proteins))
    difference = set(unique_proteins).difference(set(has_proteins))
    total, has_feature, missing = len(unique_proteins), len(intersection), len(difference)
    return "Total: %d, Has mapping: %d, No mapping: %d" % (total, has_feature, missing)

# Statistics for the three features: how many proteins have a corresponding mapping
print(stats(domain_map))
print(stats(location_map))
print(stats(genome_map))

# Min, Normalized Min and MinMax kernels for binary vectors
min = lambda x, y : np.sum(x & y)
norm = lambda x, y : np.sum(x & y) / np.sqrt(np.sum(x & x) * np.sum(y & y))
minmax = lambda x, y : np.sum(x & y) / np.sum(x | y)

def kernel(feature_matrix, f):
    kernel = pd.DataFrame(np.zeros((len(feature_matrix), len(feature_matrix))), index=feature_matrix.index, columns=feature_matrix.index)
    for row_name, row in feature_matrix.iterrows():
        for col_name, col in feature_matrix.iterrows():
            kernel.loc[row_name, col_name] = f(row, col)
    return kernel

def fill_kernel(kernel_matrix):
    np.fill_diagonal(kernel_matrix.values, 1)
    kernel_matrix.fillna(0, inplace=True)

# Domain kernel (min, norm, minmax)
domain_min = kernel(domain_map, min)
domain_norm = kernel(domain_map, norm)
domain_minmax = kernel(domain_map, minmax)
fill_kernel(domain_norm)
fill_kernel(domain_minmax)
domain_min.to_csv('cyc_data/K_domain_min.csv', sep=';')
domain_norm.to_csv('cyc_data/K_domain_norm.csv', sep=';')
domain_minmax.to_csv('cyc_data/K_domain_minmax.csv', sep=';')

# Location kernel (min, norm, minmax)
location_min = kernel(location_map, min)
location_norm = kernel(location_map, norm)
location_minmax = kernel(location_map, minmax)
fill_kernel(location_norm)
fill_kernel(location_minmax)
location_min.to_csv('cyc_data/K_location_min.csv', sep=';')
location_norm.to_csv('cyc_data/K_location_norm.csv', sep=';')
location_minmax.to_csv('cyc_data/K_location_minmax.csv', sep=';')

# Genome kernel (min, norm, minmax)
genome_min = kernel(genome_map, min)
genome_norm = kernel(genome_map, norm)
genome_minmax = kernel(genome_map, minmax)
fill_kernel(genome_norm)
fill_kernel(genome_minmax)
genome_min.to_csv('cyc_data/K_genome_min.csv', sep=';')
genome_norm.to_csv('cyc_data/K_genome_norm.csv', sep=';')
genome_minmax.to_csv('cyc_data/K_genome_minmax.csv', sep=';')

# PPI network
# TODO: is WI-PHI complete or incomplete data? Use the 50 000 interactions or the unlimited version?
# TODO: is the network all the proteins in WI-PHI or only the proteins in heterodimer pairs?
proteins = np.unique(df[['Protein A', 'Protein B']].values)
W = pd.pivot_table(df, values='Score', index='Protein A', columns='Protein B')
W = W.reindex(index=proteins, columns=proteins)
W.fillna(W.T, inplace=True)
W.fillna(0.0, inplace=True) # complete data

# PPI & domain composition based feature vector
del min
features = pd.DataFrame(np.zeros((len(pairs), 7)), index=pairs.index, columns=range(7))
for idx, row in pairs.iterrows():
    p1 = row['Protein A']
    p2 = row['Protein B']
    w_ij = W.loc[p1, p2]
    w_max = max(max([w for n, w in W[p1].iteritems() if n != p1]), max([w for n, w in W[p2].iteritems() if n != p2]))
    w_min = min(min([w for n, w in W[p1].iteritems() if n != p1]), min([w for n, w in W[p2].iteritems() if n != p2]))
    w_k = max([min(w1, w2) for w1, w2 in zip(W[p1], W[p2])])
    w_kk = max([abs(w1 - w2) for n1, w1 in W[p1].iteritems() for n2, w2 in W[p2].iteritems() if n1 != p1 and n2 != p2])
    n_max = max(domain_map.ix[p1].sum(), domain_map.ix[p2].sum())
    n_min = min(domain_map.ix[p1].sum(), domain_map.ix[p2].sum())
    features.loc[idx, :] = (w_ij, w_max, w_min, w_k, w_kk, n_max, n_min)
features.index = pairs[['Protein A', 'Protein B']]
ppi_linear = pd.DataFrame(np.dot(features.values, features.values.T), index=features.index, columns=features.index)
features.to_csv('cyc_data/protein_protein_interaction.csv', sep=';')
ppi_linear.to_csv('cyc_data/K_PPI.csv', sep=';')

# Domain composition kernel
domain_composition = pd.DataFrame(np.zeros((len(pairs), len(pairs))), index=pairs.index, columns=pairs.index)
for idx1, row in pairs.iterrows():
    for idx2, col in pairs.iterrows():
        p1 = row['Protein A']
        p2 = row['Protein B']
        p3 = col['Protein A']
        p4 = col['Protein B']
        domain_composition.loc[idx1, idx2] = \
        ((domain_map.ix[p1] == domain_map.ix[p3]).all() & (domain_map.ix[p2] == domain_map.ix[p4]).all()) |\
        ((domain_map.ix[p1] == domain_map.ix[p4]).all() & (domain_map.ix[p2] == domain_map.ix[p3]).all())
domain_composition.index = pairs[['Protein A', 'Protein B']]
domain_composition.columns = pairs[['Protein A', 'Protein B']]
domain_composition.to_csv('cyc_data/K_domain_composition.csv', sep=';')

