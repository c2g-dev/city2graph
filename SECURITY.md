# Security Policy

## Supported Versions

We actively maintain and provide security updates for the following versions of city2graph:

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| 0.2.x   | :x:                |
| < 0.2.0 | :x:                |

## Reporting a Vulnerability

We take the security of city2graph seriously. If you believe you have found a security vulnerability in city2graph, please report it to us as described below.

### How to Report

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to:

- **Email:** y.sato@liverpool.ac.uk

Please include the following information in your report:

- Type of vulnerability (e.g., code injection, dependency vulnerability, data exposure)
- Full paths of source file(s) related to the vulnerability
- The location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the vulnerability, including how an attacker might exploit it

### What to Expect

- **Response Time:** You should receive an initial response within 72 hours acknowledging receipt of your report.
- **Status Updates:** We will keep you informed about the progress of addressing the vulnerability.
- **Disclosure:** Once the vulnerability is fixed, we will coordinate with you on the timing and content of public disclosure.
- **Credit:** We are happy to give credit to security researchers who report vulnerabilities responsibly.

## Resolved Vulnerabilities

| CVE | Package | Severity | Fixed In | Resolution |
| --- | ------- | -------- | -------- | ---------- |
| [CVE-2025-3730](https://nvd.nist.gov/vuln/detail/CVE-2025-3730) | PyTorch | Critical | v0.2.4 | Minimum `torch>=2.8.0` enforced across all install variants |
| [CVE-2025-2953](https://nvd.nist.gov/vuln/detail/CVE-2025-2953) | PyTorch | Critical | v0.2.4 | Minimum `torch>=2.8.0` enforced; deprecated CUDA variants (`cu118`, `cu124`) removed |

As part of the fix, the following CUDA install extras were removed: `cu118`, `cu124`. The currently supported extras are: `cpu`, `cu126`, `cu128`, `cu130`.

## Security Best Practices

When using city2graph, we recommend following these security best practices:

### Data Handling

- **Validate Input Data:** Always validate geospatial data from untrusted sources before processing
- **Sanitize File Paths:** When loading geospatial files (GeoJSON, Shapefile, etc.), ensure file paths are properly validated
- **Limit Resource Usage:** Set appropriate limits on data size and processing time for large datasets

### Dependencies

- **Keep Dependencies Updated:** Regularly update city2graph and its dependencies to get the latest security patches
- **Use Virtual Environments:** Isolate city2graph installations using virtual environments (venv, conda, etc.)
- **Pin Secure PyTorch Versions:** city2graph requires `torch>=2.12.0` for CPU, CUDA 12.6, and CUDA 13.0, or `torch>=2.11.0` for CUDA 12.8
- **Review Dependencies:** Be aware of the security status of core dependencies:
  - PyTorch (≥ 2.11.0) and PyTorch Geometric (≥ 2.7.0)
  - GeoPandas and its dependencies (GDAL, GEOS, etc.)
  - NetworkX
  - OSMnx

### Environment

- **API Keys and Credentials:** Never commit API keys or credentials to version control
- **Network Access:** Be cautious when fetching remote geospatial data (e.g., from Overture Maps, OpenStreetMap)
- **File Permissions:** Ensure appropriate file permissions when reading/writing geospatial data

### Code Security

- **Input Validation:** When building custom workflows, validate all inputs before passing to city2graph functions
- **Error Handling:** Implement proper error handling to avoid exposing sensitive information in error messages
- **Resource Limits:** Set appropriate memory and computation limits when processing large graphs

## Known Security Considerations

### Geospatial Data Processing

- **Large File Processing:** Processing extremely large geospatial datasets can lead to memory exhaustion
- **Coordinate System Transformations:** Be aware that coordinate transformations depend on GDAL/PROJ, which should be kept updated
- **Remote Data Sources:** Fetching data from remote sources (OSM, Overture Maps) should be done over secure connections

### Graph Neural Networks

- **Model Loading:** When loading pre-trained models, ensure they come from trusted sources
- **Data Serialization:** Be cautious when loading pickled graph data from untrusted sources

## Security Updates

Security updates will be released as patch versions and announced through:

- GitHub Security Advisories
- Release notes on GitHub
- PyPI release notes
- Project documentation

## Dependency Management

We use Dependabot to monitor dependencies for security vulnerabilities. Security patches for dependencies will be applied promptly and released in patch versions.

## Scope

This security policy applies to:

- The city2graph Python package
- Official documentation and examples
- Official Docker images (if applicable)

This policy does not cover:

- Third-party integrations or plugins
- Forked or modified versions of city2graph
- User-generated code or workflows using city2graph

## Contact

For security concerns, please contact: y.sato@liverpool.ac.uk

For general questions and support, please use:
- GitHub Issues: https://github.com/c2g-dev/city2graph/issues
- Documentation: https://city2graph.net

---

Thank you for helping keep city2graph and its users safe!
