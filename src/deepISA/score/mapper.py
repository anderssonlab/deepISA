import pandas as pd
import bioframe as bf
from loguru import logger

from deepISA.utils import remove_if_exists, write_stream_csv

# TODO: since the file names are almost determined, all paths should have a default value.



def subset_by_rna(df, expressed_tfs):
    """Mandatory hard filter for RNA evidence."""
    if df.empty:
        return df
    if expressed_tfs is None:
        return df
    # Handle dimeric names (e.g., GATA1::TAL1)
    prots = df["tf"].str.split("::", expand=True)
    p1 = prots[0].str.upper()
    p2 = prots.iloc[:, -1].fillna(prots[0]).str.upper()
    # Both parts of a dimer must be in the expressed list
    mask = p1.isin(expressed_tfs) & p2.isin(expressed_tfs)
    return df[mask].copy()



def check_remap(motif_df, remap_ref, region_tuple):
    """Checks if TFs in motif_df have overlapping ChIP peaks in the specific region."""
    if motif_df.empty:
        motif_df["remap_evidence"] = pd.Series(dtype=bool)
        return motif_df

    chrom, start, end = region_tuple
    # Filter ReMap peaks to only those within the current genomic window
    local_peaks = bf.select(remap_ref, (chrom, start, end))
    chip_tfs = set(local_peaks["TF"].unique())
    # Check dimer components against available ChIP TFs
    prots = motif_df["tf"].str.split("::", expand=True)
    p1 = prots[0].str.upper()
    p2 = prots.iloc[:, -1].fillna(prots[0]).str.upper()
    motif_df["remap_evidence"] = p1.isin(chip_tfs) & p2.isin(chip_tfs)
    return motif_df



class JasparAnnotator:
    def __init__(self, jaspar_path, expressed_tfs, score_thresh, remap_path=None):
        import pyBigWig   # delayed so the package imports without pyBigWig installed
        self.jaspar = pyBigWig.open(jaspar_path)
        self.score_thresh = score_thresh
        if expressed_tfs is not None:
            self.expressed_tfs = set(str(tf).upper() for tf in expressed_tfs)
        else:
            self.expressed_tfs = None
        # Load ReMap if provided
        self.remap_ref = None
        if remap_path:
            df = pd.read_csv(remap_path, sep='\t', header=None, usecols=[0, 1, 2, 3],
                             names=['chrom', 'start', 'end', 'detail'])
            # Robust extraction: split by colon or underscore
            df['TF'] = df['detail'].str.split('[:_]', expand=True)[0].str.upper()
            self.remap_ref = df[['chrom', 'start', 'end', 'TF']]

    def _get_motifs_in_region(self, region_tuple):
        """Fetches and parses motifs, ensuring they are strictly within bounds."""
        chrom, start, end = region_tuple
        try:
            entries = self.jaspar.entries(chrom, start, end)
        except:
            return pd.DataFrame()
        
        if not entries:
            return pd.DataFrame()

        df = pd.DataFrame(entries, columns=['start', 'end', 'details'])
        # Requirement: Motif must lie COMPLETELY within the given region
        df = df[(df['start'] >= start) & (df['end'] <= end)].copy()
        if df.empty:
            return pd.DataFrame()
        # Robust Parsing
        split_cols = df['details'].str.split('\t', expand=True)
        df['tf'] = split_cols[3].str.upper()
        df['score'] = pd.to_numeric(split_cols[1], errors='coerce').fillna(0).astype(int)
        df['strand'] = split_cols[2]
        df['chrom'] = chrom
        # add region info
        df['region'] = f"{chrom}:{start}-{end}"
        df['start_rel'] = df['start'] - start
        df['end_rel']   = df['end'] - start
        # Filtering
        df = df[df['score'] >= self.score_thresh]
        df = subset_by_rna(df, self.expressed_tfs)
        return df.drop_duplicates().reset_index(drop=True)


    def annotate(self, regions, outpath):
        """Streams motifs to disk."""
        remove_if_exists(outpath)
        
        for i, (_, row) in enumerate(regions.iterrows()):
            if i % 1000 == 0:
                logger.info(f"Processing region {i}-{min(i+1000, len(regions))} / {len(regions)}")
            reg_tuple = (row['chrom'], row['start'], row['end'])
            df = self._get_motifs_in_region(reg_tuple)
            if df.empty:
                continue
            # Optional ReMap column
            if self.remap_ref is not None:
                df = check_remap(df, self.remap_ref, reg_tuple)
            # Final Column Management: chrom, start, end must be first
            cols = ['chrom', 'start', 'end', 'start_rel', 'end_rel', 'tf', 'score', 'strand', 'region']
            if "remap_evidence" in df.columns:
                cols.append("remap_evidence")
            df = df[cols]
            write_stream_csv(df,outpath)


def get_non_motifs(regions_df, motif_locs_df):
    """Returns genomic intervals not covered by motifs."""
    regions_df["region"] = regions_df.apply(lambda row: f"{row['chrom']}:{row['start']}-{row['end']}", axis=1)
    # expand flanking regions:
    motif_locs_df["start"] = motif_locs_df["start"] - 5
    motif_locs_df["end"] = motif_locs_df["end"] + 5
    # subtract from original regions
    non_motif_df = bf.subtract(regions_df, motif_locs_df)
    non_motif_df = non_motif_df[['chrom', 'start', 'end', 'region']]
    # subset non_motif_df by region to make sure all regions in non_motif_df is in motif_locs_df
    non_motif_df = non_motif_df[non_motif_df['region'].isin(motif_locs_df['region'])].reset_index(drop=True)
    # add relative coord
    reg_starts = non_motif_df['region'].str.extract(r':(\d+)-')[0].astype(int)
    non_motif_df['start_rel'] = (non_motif_df['start'] - reg_starts)
    non_motif_df['end_rel'] = (non_motif_df['end'] - reg_starts)
    non_motif_df['len']= non_motif_df['end'] - non_motif_df['start']
    non_motif_df = non_motif_df[non_motif_df['len'] > 5].reset_index(drop=True)
    return non_motif_df




def map_motifs(regions_df, 
               jaspar_path, 
               motif_outpath, 
               non_motif_outpath,
               expressed_tfs=None,
               motif_score_thresh=500,
               remap_path=None):
    """
    High-level API for motif mapping.
    Outputs:
      - motif_outpath: mapped motifs after structural filtering only
      - non_motif_outpath: regions not covered by mapped motifs (only if null_mode is "non_motif_kmers")
    """
    logger.info("Starting JASPAR motif mapping.")
    # deduplicate regions_df
    regions_df = regions_df.drop_duplicates(subset=['chrom', 'start', 'end']).reset_index(drop=True)
    annotator = JasparAnnotator(
        jaspar_path=jaspar_path,
        expressed_tfs=expressed_tfs,
        score_thresh=motif_score_thresh,
        remap_path=remap_path
    )
    annotator.annotate(regions_df, motif_outpath)
    logger.info(f"Mapped motifs saved to {motif_outpath}.")
    motif_df = pd.read_csv(motif_outpath)
    non_motif_df = get_non_motifs(regions_df, motif_df)
    non_motif_df.to_csv(non_motif_outpath, index=False)
    logger.info(f"Non-motif regions saved to {non_motif_outpath}.")



def map_motif_finemo(hits_df, regions_df, motif_outpath, non_motif_outpath,
                     score_field="hit_coefficient_global"):
    """Convert Fi-NeMo hits into a ``map_motifs``-identical ``motif_locs`` CSV.

    Fi-NeMo reports hits with coordinates relative to the fake ``regions.bed``
    written by :func:`deepISA.discover.finemo.build_finemo_input` (chrom
    ``chrFake``, 0..L); the true region string is preserved in ``peak_name``.
    This function reverses that mapping: ``peak_name`` ("chrom:start-end") is
    parsed back to genomic coordinates, hit start/end become ``start_rel`` /
    ``end_rel`` plus absolute ``start`` / ``end`` — the exact column contract
    of :func:`map_motifs`::

        chrom, start, end, start_rel, end_rel, tf, score, strand, region

    ``tf`` is the annotated ``TF_Name`` when available (falling back to the raw
    motif id), ``score`` is Fi-NeMo's ``hit_coefficient_global`` by default.

    Parameters
    ----------
    hits_df : pd.DataFrame
        Annotated hits from
        :func:`deepISA.discover.finemo.load_hits_with_annotation` (needs
        ``peak_name``, ``start``, ``end``, ``strand`` and the score field).
    regions_df : pd.DataFrame
        The positive regions (chrom/start/end) — used for the non-motif
        complement, mirroring :func:`map_motifs`.
    motif_outpath, non_motif_outpath : str
        Destination CSVs, same pair :func:`map_motifs` writes.
    score_field : str
        Hits column used as the ``score``.

    Returns
    -------
    pd.DataFrame
        The motif locations written to ``motif_outpath``.
    """
    logger.info("Mapping Fi-NeMo hits to motif locations.")
    df = hits_df.dropna(subset=["peak_name", "start", "end"]).reset_index(drop=True)

    region = df["peak_name"].astype(str)
    parsed = region.str.extract(r"^(?P<chrom>[^:]+):(?P<rstart>\d+)-(?P<rend>\d+)$")
    if parsed["chrom"].isna().any():
        bad = region[parsed["chrom"].isna()].unique()[:5]
        raise ValueError(
            f"Cannot parse region string from peak_name (examples: {bad.tolist()}). "
            "Fi-NeMo hits must be produced with region-string ids "
            "(build_finemo_input(ids=...))."
        )
    df["region"] = region
    df["chrom"] = parsed["chrom"].to_numpy()
    rstart = parsed["rstart"].astype(int).to_numpy()
    df["start_rel"] = df["start"].astype(int)
    df["end_rel"] = df["end"].astype(int)
    df["start"] = rstart + df["start_rel"].to_numpy()
    df["end"] = rstart + df["end_rel"].to_numpy()
    fallback = df["motif_name"].astype(str).str.removeprefix("pos_patterns.")
    df["tf"] = df["TF_Name"].where(df["TF_Name"].ne("Unknown"), fallback)
    df["score"] = pd.to_numeric(df[score_field], errors="coerce").fillna(0.0)

    cols = ["chrom", "start", "end", "start_rel", "end_rel", "tf",
            "score", "strand", "region"]
    motif_df = (df[cols]
                .apply(lambda s: s.str.upper() if s.name == "tf" else s)
                .drop_duplicates()
                .sort_values(["region", "start"])
                .reset_index(drop=True))

    remove_if_exists(motif_outpath)
    motif_df.to_csv(motif_outpath, index=False)
    logger.info(f"Fi-NeMo motif locations ({len(motif_df)}) saved to {motif_outpath}.")

    # get_non_motifs flank-expands its motif frame in place (-5/+5) — hand it
    # a copy so the returned / written motif coordinates stay unexpanded.
    regions_df = regions_df.drop_duplicates(subset=['chrom', 'start', 'end']).reset_index(drop=True)
    non_motif_df = get_non_motifs(regions_df, motif_df.copy())
    non_motif_df.to_csv(non_motif_outpath, index=False)
    logger.info(f"Non-motif regions saved to {non_motif_outpath}.")
    return motif_df
