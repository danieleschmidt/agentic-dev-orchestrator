#!/usr/bin/env python3
"""
SBOM (Software Bill of Materials) Generator for ADO

Generates comprehensive software bill of materials in multiple formats
for security and compliance purposes.

Usage:
    python scripts/generate_sbom.py [--format json|xml|spdx] [--output file]
"""

import argparse
import json
import sys
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import hashlib
import pkg_resources
import toml


class SBOMGenerator:
    """Generate Software Bill of Materials for ADO."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.pyproject_path = project_root / "pyproject.toml"
        self.requirements_path = project_root / "requirements.txt"
        self.package_json_path = project_root / "package.json"
        
    def generate_sbom(self, format_type: str = "json") -> Dict[str, Any]:
        """Generate complete SBOM."""
        sbom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.4",
            "serialNumber": f"urn:uuid:{self._generate_uuid()}",
            "version": 1,
            "metadata": self._generate_metadata(),
            "components": self._get_all_components(),
            "services": self._get_services(),
            "externalReferences": self._get_external_references(),
            "dependencies": self._get_dependencies(),
            "vulnerabilities": self._get_vulnerabilities()
        }
        
        return sbom
    
    def _generate_metadata(self) -> Dict[str, Any]:
        """Generate SBOM metadata."""
        project_info = self._get_project_info()
        
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "tools": [
                {
                    "vendor": "Terragon Labs",
                    "name": "ADO SBOM Generator",
                    "version": "1.0.0"
                }
            ],
            "authors": [
                {
                    "name": "Terragon Labs",
                    "email": "noreply@terragonlabs.com"
                }
            ],
            "component": {
                "type": "application",
                "mime-type": "application/octet-stream",
                "bom-ref": "agentic-dev-orchestrator",
                "supplier": {
                    "name": "Terragon Labs"
                },
                "name": project_info.get("name", "agentic-dev-orchestrator"),
                "version": project_info.get("version", "0.1.0"),
                "description": project_info.get("description", ""),
                "licenses": [
                    {
                        "license": {
                            "id": "Apache-2.0"
                        }
                    }
                ],
                "externalReferences": [
                    {
                        "type": "website",
                        "url": "https://github.com/terragon-labs/agentic-dev-orchestrator"
                    },
                    {
                        "type": "vcs",
                        "url": "https://github.com/terragon-labs/agentic-dev-orchestrator.git"
                    }
                ]
            }
        }
    
    def _get_project_info(self) -> Dict[str, Any]:
        """Extract project information from pyproject.toml."""
        if self.pyproject_path.exists():
            try:
                with open(self.pyproject_path, 'r') as f:
                    data = toml.load(f)
                    return data.get("project", {})
            except Exception as e:
                print(f"Warning: Could not read pyproject.toml: {e}")
        
        return {}
    
    def _get_all_components(self) -> List[Dict[str, Any]]:
        """Get all software components."""
        components = []
        
        # Python dependencies
        components.extend(self._get_python_components())
        
        # Node.js dependencies (if present)
        if self.package_json_path.exists():
            components.extend(self._get_nodejs_components())
        
        # System dependencies
        components.extend(self._get_system_components())
        
        return components
    
    def _get_python_components(self) -> List[Dict[str, Any]]:
        """Get Python package components."""
        components = []
        
        try:
            # Get installed packages
            installed_packages = {pkg.project_name.lower(): pkg for pkg in pkg_resources.working_set}
            
            # Read requirements if available
            requirements = self._read_requirements()
            
            for pkg_name, pkg in installed_packages.items():
                # Skip the main package itself
                if pkg_name == "agentic-dev-orchestrator":
                    continue
                
                component = {
                    "type": "library",
                    "bom-ref": f"python-{pkg_name}-{pkg.version}",
                    "name": pkg.project_name,
                    "version": pkg.version,
                    "purl": f"pkg:pypi/{pkg.project_name}@{pkg.version}",
                    "scope": "required"
                }
                
                # Add license information if available
                try:
                    metadata = pkg.get_metadata('METADATA') or pkg.get_metadata('PKG-INFO')
                    if metadata:
                        licenses = self._extract_license_from_metadata(metadata)
                        if licenses:
                            component["licenses"] = licenses
                except:
                    pass
                
                # Add hash if available
                try:
                    if pkg.location and Path(pkg.location).exists():
                        component["hashes"] = [{
                            "alg": "SHA-256",
                            "content": self._calculate_directory_hash(pkg.location)
                        }]
                except:
                    pass
                
                components.append(component)
        
        except Exception as e:
            print(f"Warning: Could not analyze Python packages: {e}")
        
        return components
    
    def _get_nodejs_components(self) -> List[Dict[str, Any]]:
        """Get Node.js package components."""
        components = []
        
        try:
            with open(self.package_json_path, 'r') as f:
                package_data = json.load(f)
            
            # Get dependencies
            all_deps = {}
            all_deps.update(package_data.get("dependencies", {}))
            all_deps.update(package_data.get("devDependencies", {}))
            
            for name, version in all_deps.items():
                component = {
                    "type": "library",
                    "bom-ref": f"npm-{name}-{version}",
                    "name": name,
                    "version": version.lstrip("^~>="),
                    "purl": f"pkg:npm/{name}@{version.lstrip('^~>=')}",
                    "scope": "required" if name in package_data.get("dependencies", {}) else "optional"
                }
                components.append(component)
        
        except Exception as e:
            print(f"Warning: Could not analyze Node.js packages: {e}")
        
        return components
    
    def _get_system_components(self) -> List[Dict[str, Any]]:
        """Get system-level components."""
        components = []
        
        # Python runtime
        python_version = sys.version.split()[0]
        components.append({
            "type": "platform",
            "bom-ref": f"python-{python_version}",
            "name": "Python",
            "version": python_version,
            "purl": f"pkg:generic/python@{python_version}",
            "scope": "required"
        })
        
        # Operating system (if detectable)
        try:
            import platform
            os_info = platform.platform()
            components.append({
                "type": "operating-system",
                "bom-ref": f"os-{platform.system().lower()}",
                "name": platform.system(),
                "version": platform.release(),
                "description": os_info,
                "scope": "required"
            })
        except:
            pass
        
        return components
    
    def _get_services(self) -> List[Dict[str, Any]]:
        """Get external services used by ADO."""
        return [
            {
                "bom-ref": "github-api",
                "provider": {
                    "name": "GitHub",
                    "url": ["https://github.com"]
                },
                "name": "GitHub API",
                "version": "v4",
                "description": "GitHub REST and GraphQL API",
                "endpoints": [
                    "https://api.github.com"
                ],
                "authenticated": True,
                "x-trust-boundary": True
            },
            {
                "bom-ref": "openai-api",
                "provider": {
                    "name": "OpenAI",
                    "url": ["https://openai.com"]
                },
                "name": "OpenAI API",
                "version": "v1",
                "description": "OpenAI GPT API for AI agents",
                "endpoints": [
                    "https://api.openai.com"
                ],
                "authenticated": True,
                "x-trust-boundary": True
            },
            {
                "bom-ref": "anthropic-api",
                "provider": {
                    "name": "Anthropic",
                    "url": ["https://anthropic.com"]
                },
                "name": "Anthropic API",
                "version": "v1",
                "description": "Anthropic Claude API for AI agents",
                "endpoints": [
                    "https://api.anthropic.com"
                ],
                "authenticated": True,
                "x-trust-boundary": True
            }
        ]
    
    def _get_external_references(self) -> List[Dict[str, Any]]:
        """Get external references."""
        return [
            {
                "type": "website",
                "url": "https://github.com/terragon-labs/agentic-dev-orchestrator"
            },
            {
                "type": "vcs",
                "url": "https://github.com/terragon-labs/agentic-dev-orchestrator.git"
            },
            {
                "type": "issue-tracker",
                "url": "https://github.com/terragon-labs/agentic-dev-orchestrator/issues"
            },
            {
                "type": "documentation",
                "url": "https://github.com/terragon-labs/agentic-dev-orchestrator/blob/main/README.md"
            }
        ]
    
    def _get_dependencies(self) -> List[Dict[str, Any]]:
        """Get dependency relationships."""
        # This would require more sophisticated dependency analysis
        # For now, return empty list
        return []
    
    def _get_vulnerabilities(self) -> List[Dict[str, Any]]:
        """Get known vulnerabilities (if available)."""
        vulnerabilities = []
        
        try:
            # Try to run safety check
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and result.stdout:
                safety_data = json.loads(result.stdout)
                for vuln in safety_data:
                    vulnerabilities.append({
                        "bom-ref": f"vuln-{vuln.get('id', 'unknown')}",
                        "id": vuln.get("id"),
                        "source": {
                            "name": "Safety DB",
                            "url": "https://pyup.io/safety/"
                        },
                        "ratings": [
                            {
                                "source": {
                                    "name": "Safety DB"
                                },
                                "severity": vuln.get("severity", "unknown")
                            }
                        ],
                        "description": vuln.get("advisory"),
                        "affects": [
                            {
                                "ref": f"python-{vuln.get('package_name')}-{vuln.get('installed_version')}"
                            }
                        ]
                    })
        except Exception as e:
            print(f"Warning: Could not check for vulnerabilities: {e}")
        
        return vulnerabilities
    
    def _read_requirements(self) -> List[str]:
        """Read requirements.txt file."""
        requirements = []
        if self.requirements_path.exists():
            try:
                with open(self.requirements_path, 'r') as f:
                    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            except Exception as e:
                print(f"Warning: Could not read requirements.txt: {e}")
        return requirements
    
    def _extract_license_from_metadata(self, metadata: str) -> List[Dict[str, Any]]:
        """Extract license information from package metadata."""
        licenses = []
        for line in metadata.split('\n'):
            if line.startswith('License:'):
                license_name = line.split(':', 1)[1].strip()
                if license_name and license_name != 'UNKNOWN':
                    licenses.append({
                        "license": {
                            "name": license_name
                        }
                    })
                break
        return licenses
    
    def _calculate_directory_hash(self, directory: str) -> str:
        """Calculate SHA-256 hash of directory contents."""
        hasher = hashlib.sha256()
        try:
            for root, dirs, files in os.walk(directory):
                for file in sorted(files):
                    file_path = Path(root) / file
                    if file_path.is_file():
                        with open(file_path, 'rb') as f:
                            for chunk in iter(lambda: f.read(4096), b""):
                                hasher.update(chunk)
        except Exception:
            pass
        return hasher.hexdigest()
    
    def _generate_uuid(self) -> str:
        """Generate a UUID for the SBOM."""
        import uuid
        return str(uuid.uuid4())
    
    def save_json(self, sbom: Dict[str, Any], output_path: Path) -> None:
        """Save SBOM as JSON."""
        with open(output_path, 'w') as f:
            json.dump(sbom, f, indent=2, sort_keys=True)
        print(f"SBOM saved as JSON: {output_path}")
    
    def save_xml(self, sbom: Dict[str, Any], output_path: Path) -> None:
        """Save SBOM as XML."""
        root = ET.Element("bom", {
            "xmlns": "http://cyclonedx.org/schema/bom/1.4",
            "serialNumber": sbom["serialNumber"],
            "version": str(sbom["version"])
        })
        
        # Add metadata
        metadata_elem = ET.SubElement(root, "metadata")
        timestamp_elem = ET.SubElement(metadata_elem, "timestamp")
        timestamp_elem.text = sbom["metadata"]["timestamp"]
        
        # Add components
        components_elem = ET.SubElement(root, "components")
        for component in sbom["components"]:
            comp_elem = ET.SubElement(components_elem, "component", {
                "type": component["type"],
                "bom-ref": component["bom-ref"]
            })
            
            name_elem = ET.SubElement(comp_elem, "name")
            name_elem.text = component["name"]
            
            version_elem = ET.SubElement(comp_elem, "version")
            version_elem.text = component["version"]
        
        tree = ET.ElementTree(root)
        tree.write(output_path, encoding="utf-8", xml_declaration=True)
        print(f"SBOM saved as XML: {output_path}")
    
    def save_spdx(self, sbom: Dict[str, Any], output_path: Path) -> None:
        """Save SBOM in SPDX format."""
        spdx_content = []
        spdx_content.append("SPDXVersion: SPDX-2.2")
        spdx_content.append("DataLicense: CC0-1.0")
        spdx_content.append(f"SPDXID: SPDXRef-DOCUMENT")
        spdx_content.append(f"Name: {sbom['metadata']['component']['name']}")
        spdx_content.append(f"DocumentNamespace: {sbom['serialNumber']}")
        spdx_content.append(f"Creator: Tool: ADO SBOM Generator")
        spdx_content.append(f"Created: {sbom['metadata']['timestamp']}")
        spdx_content.append("")
        
        # Add main package
        comp = sbom['metadata']['component']
        spdx_content.append(f"PackageName: {comp['name']}")
        spdx_content.append(f"SPDXID: SPDXRef-Package")
        spdx_content.append(f"PackageVersion: {comp['version']}")
        spdx_content.append(f"PackageSupplier: Organization: {comp['supplier']['name']}")
        spdx_content.append(f"PackageDownloadLocation: NOASSERTION")
        spdx_content.append(f"FilesAnalyzed: false")
        spdx_content.append(f"PackageLicenseConcluded: Apache-2.0")
        spdx_content.append(f"PackageLicenseDeclared: Apache-2.0")
        spdx_content.append(f"PackageCopyrightText: NOASSERTION")
        spdx_content.append("")
        
        # Add components
        for i, component in enumerate(sbom["components"]):
            spdx_content.append(f"PackageName: {component['name']}")
            spdx_content.append(f"SPDXID: SPDXRef-Package-{i+1}")
            spdx_content.append(f"PackageVersion: {component['version']}")
            spdx_content.append(f"PackageDownloadLocation: NOASSERTION")
            spdx_content.append(f"FilesAnalyzed: false")
            spdx_content.append(f"PackageLicenseConcluded: NOASSERTION")
            spdx_content.append(f"PackageLicenseDeclared: NOASSERTION")
            spdx_content.append(f"PackageCopyrightText: NOASSERTION")
            spdx_content.append("")
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(spdx_content))
        print(f"SBOM saved as SPDX: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Software Bill of Materials (SBOM) for ADO"
    )
    parser.add_argument(
        "--format",
        choices=["json", "xml", "spdx"],
        default="json",
        help="Output format (default: json)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (default: SBOM.{format})"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (default: current directory)"
    )
    
    args = parser.parse_args()
    
    # Default output filename
    if not args.output:
        args.output = args.project_root / f"SBOM.{args.format}"
    
    # Generate SBOM
    generator = SBOMGenerator(args.project_root)
    sbom = generator.generate_sbom(args.format)
    
    # Save in requested format
    if args.format == "json":
        generator.save_json(sbom, args.output)
    elif args.format == "xml":
        generator.save_xml(sbom, args.output)
    elif args.format == "spdx":
        generator.save_spdx(sbom, args.output)
    
    print(f"\nSBOM generated successfully!")
    print(f"Components found: {len(sbom['components'])}")
    print(f"Services: {len(sbom['services'])}")
    if sbom['vulnerabilities']:
        print(f"Vulnerabilities: {len(sbom['vulnerabilities'])}")
        print("⚠️  Warning: Vulnerabilities detected! Review the SBOM for details.")


if __name__ == "__main__":
    main()