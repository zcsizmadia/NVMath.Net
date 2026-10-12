# Releasing NVMath.Net

Releases are published by the **Release** workflow (`.github/workflows/release.yml`) when a GitHub release is
published. It publishes the package built by the **Build & Pack** workflow on `main`.

## Checklist

1. **Make sure `main` is green** — the *Build & Pack* and *Docs* workflows must have passed on the commit you are releasing.
2. **Run the GPU tests locally** (CI has no GPU). With the CUDA Toolkit, cuTENSOR and cuDSS installed:

   ```bash
   dotnet test
   ```

3. **Set the version.** `Directory.Build.props` → `<Version>` must equal the tag without the leading `v`
   (the release workflow fails on a mismatch). Merge the change to `main` and wait for *Build & Pack* to finish: it
   produces the `nuget-package` artifact that the release workflow downloads.
4. **Update `CHANGELOG.md`**: replace *unreleased* with the release date and check the *Breaking changes* section.
5. **Create the release** on GitHub: tag `vX.Y.Z` targeting `main`, paste the changelog section as the notes, and
   publish it. The workflow then:
   - checks that the tag matches the project version and that `NVMath.Net.X.Y.Z.nupkg` exists,
   - attaches the package to the GitHub release,
   - pushes it to NuGet.org (secret `NUGET_API_KEY`) and to GitHub Packages.
6. **Verify**: the package appears on <https://www.nuget.org/packages/NVMath.Net>, and the documentation site
   (<https://zcsizmadia.github.io/NVMath.Net/>) shows the new API (it redeploys on every push to `main`).

## Versioning

NVMath.Net follows [Semantic Versioning](https://semver.org/). Anything public — including the `NVMathNet.Interop`
bindings — is part of the API: removing or changing a public member requires a major version.
