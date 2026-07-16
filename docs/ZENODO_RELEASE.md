# Zenodo release checklist

1. Push the repository to `Juaco2r/HistoAnalyzer`.
2. Enable the repository in the Zenodo GitHub integration.
3. Confirm `.zenodo.json`, `CITATION.cff`, and `codemeta.json` are valid.
4. Create and push a semantic version tag, for example `v1.0.1`.
5. Publish the corresponding GitHub release.
6. Confirm Zenodo creates the archived software record and DOI.
7. Add the DOI badge and DOI to `CITATION.cff` in the next commit.
8. Use the concept DOI for the repository and version DOI for a specific release.
