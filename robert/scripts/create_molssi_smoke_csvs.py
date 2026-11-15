"""Create reproducible CSV inputs for every automatic MolSSI connection."""

import argparse
import csv
from pathlib import Path


# Public molecule IDs and SMILES checked against MolSSI on 2026-10-30.
# Each CSV repeats its first molecule to exercise duplicate-row alignment.
SAMPLES = (
    ("acids", "Ac1", "OC(CCCCCCCCC)=O", "Ac1539", "OC(=O)C1=C(OC2=CC=C(C=C2)N(=O)=O)C=CC=C1"),
    ("primary-amines", "N1", "NCCOC", "N1100", "NCCC1=C(OC2=CC=CC=C2)C=CC=C1"),
    ("secondary-amines", "SecN1", "C1COCCN1", "SecN1460", "CC12CCC(C)(CC(=O)C1)N2"),
    ("amines", "N1", "NCCOC", "SecN1460", "CC12CCC(C)(CC(=O)C1)N2"),
    ("anilines", "aniline81", "CCOC(=O)c1csc(N)n1", "aniline110", "COC(=O)c1nc(Br)cnc1N"),
    ("sulfonimidamides", "SIA1", "O=S(C1=CC=CC=C1)(N)=N", "SIA14", "NS(C#CC1=CC=CC=C1)(=O)=N"),
    ("cyanoarenes", "aryl-1", "N#CC1=CC(C)=NC=C1", "aryl-77", "CC1=CC(=CC(C)=N1)C#N"),
    ("unactivated-primary-alkyl-bromides", "mol094", "BrCC1(CC#N)CC1", "mol099", "BrCC1(CN2C(=O)c3ccccc3C2=O)OCCO1"),
    ("sulfonyl-fluorides", "0", "O=S(C1=CC=C(C)C=C1)(F)=O", "1545", "CCCn1c(-c2ccccn2)nnc1S(=O)(=O)F"),
    ("sulfonates", "sulfonate1", "C=CC(=O)NC(C)(C)CS(=O)(=O)[O-]", "sulfonate572", "CC(C)(C)CC(C)(C)c1ccc(Oc2ccc([N+](=O)[O-])cc2S(=O)(=O)[O-])cc1"),
    ("sulfides", "sulfide0001", "CSC1=CC=C(C=C1)C(=O)C(C)(C)N1CCOCC1", "sulfide0549", "COC(CSC)OC"),
    ("aryl_bromide", "arbr10", "O=C(CCc1ccccc1)c1cccc(Br)c1", "arbr6694", "S=C=NCc1ccc(Br)cc1"),
    ("nn-ligands-hydride", "Lig1", "C[C@H]1COC(C2=N[C@@H](C)CO2)=N1", "Lig737", "C1CNC(C2=NCCN2)=N1"),
    ("nn-ligands-fluoride", "Lig1", "C[C@H]1COC(C2=N[C@@H](C)CO2)=N1", "Lig33", "Cc1ccc(C[C@H]2COC(C3=N[C@@H](Cc4ccc(C)cc4)CO3)=N2)cc1"),
    ("nn-ligands-ligand-only", "Lig1", "C[C@H]1COC(C2=N[C@@H](C)CO2)=N1", "Lig1217", "CO[C@@H]1CN(Cc2ccc(-c3ccccn3)nc2)C[C@H]1c1c[nH]nn1"),
    ("kraken", "1", "CC(C)c1cc(C(C)C)c(-c2ccccc2P(C2CCCCC2)C2CCCCC2)c(C(C)C)c1", "2", "CN(C)c1cccc(N(C)C)c1-c1ccccc1P(C1CCCCC1)C1CCCCC1"),
    ("quinones", "q10000", "O=C1C(S(=O)(O)=O)=CC(C=C1)=O", "q10341", "[H]OC(=O)C1=C(F)C([H])=C([H])C(=O)C1=O"),
    ("cyclic-olefins", "E1", "[H][C@@]12[C@@](C(C3=CC=CC=C3)=C2C4=CC=CC=C4)([H])[C@@H]5C=C[C@H]1C5", "L555", "CC(=O)OC[C@H]1O[C@@H](N2C(=O)[C@@H]3[C@H](C2=O)[C@H]2C=C[C@@H]3O2)[C@H](OC(C)=O)[C@@H](OC(C)=O)[C@H]1OC(C)=O"),
)

TEST_WORKBOOKS = frozenset({
    "primary-amines", "secondary-amines", "amines", "sulfonimidamides",
    "cyanoarenes", "sulfonyl-fluorides", "unactivated-primary-alkyl-bromides",
    "sulfonates", "kraken", "quinones",
})
KRAKEN_ML_SAMPLE = (
    "kraken", "241721", "[H]P([H])C#CC(C)C", "31665", "[H]P(C#CC(C)C)C#CC(C)C",
)


def _write_sample(output_dir, filename, library, first_id, first_smiles, second_id, second_smiles):
    """Write a three-row CSV with two real molecules and one repeated row."""
    with (output_dir / filename).open("w", newline="", encoding="utf-8") as sample_handle:
        writer = csv.writer(sample_handle)
        writer.writerow(("Name", "SMILES", "molssi_library"))
        writer.writerow((first_id, first_smiles, library))
        writer.writerow((second_id, second_smiles, library))
        writer.writerow((first_id, first_smiles, library))


def create_samples(output_dir: Path, refresh: bool = False) -> None:
    """Write all sample CSVs and an index of expected automatic-flow results."""
    output_dir.mkdir(parents=True, exist_ok=refresh)
    with (output_dir / "expected_results.csv").open("w", newline="", encoding="utf-8") as handle:
        index = csv.writer(handle)
        index.writerow(("library", "input_csv", "data_type", "descriptor_offer", "test_download_offer", "expected_names"))
        for library, first_id, first_smiles, second_id, second_smiles in SAMPLES:
            filename = f"sample_{library}.csv"
            _write_sample(
                output_dir, filename, library, first_id, first_smiles, second_id, second_smiles
            )
            index.writerow((
                library, filename, "DFT", "yes", "yes" if library in TEST_WORKBOOKS else "no",
                f"{first_id};{second_id};{first_id}",
            ))
        library, first_id, first_smiles, second_id, second_smiles = KRAKEN_ML_SAMPLE
        filename = "sample_kraken_ml.csv"
        _write_sample(
            output_dir, filename, library, first_id, first_smiles, second_id, second_smiles
        )
        index.writerow((
            library, filename, "ML", "yes", "no", f"{first_id};{second_id};{first_id}",
        ))
    (output_dir / "README.txt").write_text(
        "Automatic MolSSI workflow samples (checked 2026-10-30)\n\n"
        "The folder contains one CSV for each of the 18 libraries and one extra "
        "Kraken ML-only CSV. Load each sample_*.csv as the main CSV in easyROB. "
        "Each has two different "
        "MolSSI molecules and repeats the first one in the third row. Accept the "
        "descriptor offer, then inspect the generated _molssi_*.csv: row order, "
        "Name, SMILES, and descriptor alignment should be preserved.\n\n"
        "The molssi_library column selects the intended library when databases "
        "share molecules; the column is removed from the descriptor export. "
        "Ordinary CSVs without this column retain automatic first-match detection.\n\n"
        "For libraries marked yes under test_download_offer in expected_results.csv, "
        "a second dialog should offer the full-library external test. Accepting it "
        "downloads the XLSX, converts it to <library>_library.csv, validates the "
        "descriptor columns, and loads that CSV as the test set. Libraries marked "
        "no should still offer descriptors, without a test download.\n\n"
        "These samples test data flow, not model performance. Full-library test "
        "sets may overlap the main CSV or lack an experimental target.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--refresh", action="store_true", help="Refresh generated files in an existing folder")
    args = parser.parse_args()
    create_samples(args.output_dir, refresh=args.refresh)


if __name__ == "__main__":
    main()
