"""
Comprehensive VASP error detection and automated recovery strategies.

Handles common VASP errors with specific parameter adjustments.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = "critical"  # Job cannot continue, major changes needed
    HIGH = "high"  # Serious issue, significant changes needed
    MEDIUM = "medium"  # Moderate issue, parameter tweaks needed
    LOW = "low"  # Minor issue, small adjustments needed
    WARNING = "warning"  # Not an error, but worth noting


@dataclass
class VASPError:
    """Represents a detected VASP error."""
    error_type: str
    severity: ErrorSeverity
    message: str
    found_in: str  # Which file the error was found in
    line_number: Optional[int] = None
    context: str = ""  # Surrounding text for context

    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.error_type}: {self.message}"


class VASPErrorDetector:
    """Detects and categorizes VASP errors from output files."""

    # Error patterns with their severity and descriptions
    ERROR_PATTERNS = {
        # ====================================================================
        # CRITICAL ERRORS - Job fails completely
        # ====================================================================
        "VERY_BAD_NEWS": {
            "pattern": r"VERY BAD NEWS",
            "severity": ErrorSeverity.CRITICAL,
            "message": "Fatal internal error - usually memory corruption or segfault",
        },
        "ABORTING": {
            "pattern": r"(ERROR: aborting|STOPPING NOW)",
            "severity": ErrorSeverity.CRITICAL,
            "message": "Job aborted due to fatal error",
        },
        "INSUFFICIENT_MEMORY": {
            "pattern": r"(not enough memory|allocation would exceed|insufficient memory)",
            "severity": ErrorSeverity.CRITICAL,
            "message": "Insufficient memory available",
        },
        "SEGMENTATION_FAULT": {
            "pattern": r"segmentation fault",
            "severity": ErrorSeverity.CRITICAL,
            "message": "Segmentation fault - memory access error",
        },

        # ====================================================================
        # HIGH SEVERITY - Serious calculation problems
        # ====================================================================
        "TOO_FEW_BANDS": {
            "pattern": r"(FEXCP|too few bands)",
            "severity": ErrorSeverity.HIGH,
            "message": "Too few bands for system - NBANDS too low",
        },
        "ZPOTRF_ERROR": {
            "pattern": r"(LAPACK: Routine ZPOTRF|is not positive definite)",
            "severity": ErrorSeverity.HIGH,
            "message": "Matrix not positive definite - overlap too small",
        },
        "EDDDAV_ERROR": {
            "pattern": r"(EDDDAV: call to ZHEGV failed|EDDDAV.*failed)",
            "severity": ErrorSeverity.HIGH,
            "message": "Subspace diagonalization failed",
        },
        "ZHEGV_ERROR": {
            "pattern": r"call to ZHEGV failed",
            "severity": ErrorSeverity.HIGH,
            "message": "Eigenvalue solver failed",
        },
        "PSMAXN_ERROR": {
            "pattern": r"PSMAXN for non-local potential",
            "severity": ErrorSeverity.HIGH,
            "message": "Pseudopotential grid too coarse",
        },
        "BRIONS_PROBLEMS": {
            "pattern": r"BRIONS problems: POTIM should be increased",
            "severity": ErrorSeverity.HIGH,
            "message": "Ionic step size too small, increasing POTIM recommended",
        },

        # ====================================================================
        # MEDIUM SEVERITY - Convergence and mixing problems
        # ====================================================================
        "ZBRENT_ERROR": {
            "pattern": r"ZBRENT: fatal error",
            "severity": ErrorSeverity.MEDIUM,
            "message": "Fermi level bracketing failed - wrong ISMEAR/SIGMA",
        },
        "BRMIX_ERROR": {
            "pattern": r"(BRMIX: very serious problems|charge sloshing)",
            "severity": ErrorSeverity.MEDIUM,
            "message": "Charge mixing problems - reduce AMIX/BMIX",
        },
        "DAV_SUBROT": {
            "pattern": r"(DAV.*WARNING.*sub-space-matrix|EDWAV.*WARNING)",
            "severity": ErrorSeverity.MEDIUM,
            "message": "Subspace rotation issues",
        },
        "REAL_SPACE_PROJECTION": {
            "pattern": r"(REAL_OPT.*WARNING|real space projection)",
            "severity": ErrorSeverity.MEDIUM,
            "message": "Real space projection operators not optimal",
        },
        "NOT_HERMITIAN": {
            "pattern": r"(Hamiltonian is not Hermitian|WAVPRE.*WARNING)",
            "severity": ErrorSeverity.MEDIUM,
            "message": "Hamiltonian not Hermitian - numerical issues",
        },
        "NELM_REACHED": {
            "pattern": r"(NELM reached|required accuracy not reached)",
            "severity": ErrorSeverity.MEDIUM,
            "message": "Electronic convergence not reached in NELM steps",
        },

        # ====================================================================
        # LOW SEVERITY - Minor issues
        # ====================================================================
        "POSMAP_WARNING": {
            "pattern": r"(POSMAP.*warning|your direct lattice vectors)",
            "severity": ErrorSeverity.LOW,
            "message": "Cell shape warning - lattice vectors ordering",
        },
        "SYMMETRY_REDUCED": {
            "pattern": r"(symmetry could not be established|VERY BAD NEWS.*symmetry)",
            "severity": ErrorSeverity.LOW,
            "message": "Symmetry detection issues - using lower symmetry",
        },
        "PRICEL_WARNING": {
            "pattern": r"(PRICEL.*warning|volume of cell)",
            "severity": ErrorSeverity.LOW,
            "message": "Cell volume issues",
        },
        "IALGO_WARNING": {
            "pattern": r"(IALGO.*incompatible|ALGO.*not recommended)",
            "severity": ErrorSeverity.LOW,
            "message": "Algorithm compatibility warning",
        },

        # ====================================================================
        # WARNINGS - Informational
        # ====================================================================
        "ACCURACY_WARNING": {
            "pattern": r"(WARNING: sub-space-matrix is not hermitian|accuracy problem)",
            "severity": ErrorSeverity.WARNING,
            "message": "Accuracy warning in calculation",
        },
        "FFT_WARNING": {
            "pattern": r"(FFT.*WARNING|NGX.*not compatible)",
            "severity": ErrorSeverity.WARNING,
            "message": "FFT grid size not optimal",
        },
    }

    @classmethod
    def scan_file(cls, filepath: Path) -> list[VASPError]:
        """Scan a file for VASP errors."""
        if not filepath.exists():
            return []

        errors = []
        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
            lines = content.split("\n")

            for error_type, error_info in cls.ERROR_PATTERNS.items():
                pattern = error_info["pattern"]
                matches = list(re.finditer(pattern, content, re.IGNORECASE))

                for match in matches:
                    # Find line number
                    line_num = content[:match.start()].count("\n") + 1

                    # Get context (surrounding lines)
                    context_lines = lines[max(0, line_num - 2):min(len(lines), line_num + 3)]
                    context = "\n".join(context_lines)

                    error = VASPError(
                        error_type=error_type,
                        severity=error_info["severity"],
                        message=error_info["message"],
                        found_in=filepath.name,
                        line_number=line_num,
                        context=context
                    )
                    errors.append(error)

        except Exception as e:
            print(f"Error scanning {filepath}: {e}")

        return errors

    @classmethod
    def scan_vasp_outputs(cls, working_dir: Path) -> dict[str, list[VASPError]]:
        """Scan all VASP output files in a directory."""
        files_to_check = [
            "OUTCAR",
            "stdout.txt",
            "stderr.txt",
            "vasp.out",
            "slurm.out",
        ]

        all_errors = {}

        for filename in files_to_check:
            filepath = working_dir / filename
            if filepath.exists():
                errors = cls.scan_file(filepath)
                if errors:
                    all_errors[filename] = errors

        return all_errors


class VASPErrorRecovery:
    """Suggests parameter adjustments to fix VASP errors."""

    RECOVERY_STRATEGIES = {
        # ====================================================================
        # Memory and critical errors
        # ====================================================================
        "VERY_BAD_NEWS": {
            "adjustments": {
                "NCORE": lambda x: max(1, (x or 4) // 2),  # Reduce parallelization
                "KPAR": lambda x: max(1, (x or 2) // 2),
                "LREAL": "Auto",  # Use real space projection
            },
            "explanation": "Reduce memory usage by lowering parallelization and using real-space projection",
            "alternative": "Request more memory from HPC scheduler",
        },

        "INSUFFICIENT_MEMORY": {
            "adjustments": {
                "LREAL": "Auto",
                "NCORE": lambda x: max(1, (x or 4) // 2),
                "ADDGRID": False,  # Disable support grid
            },
            "explanation": "Reduce memory footprint",
            "scheduler_changes": {
                "mem": lambda x: int((x or 4) * 1.5),  # Request 50% more memory
            },
        },

        # ====================================================================
        # Band structure errors
        # ====================================================================
        "TOO_FEW_BANDS": {
            "adjustments": {
                "NBANDS": lambda x: int((x or 0) * 1.5),  # Increase by 50%
            },
            "explanation": "Increase number of bands to accommodate all electrons plus empty states",
            "calculate": "NBANDS = int(NELECT / 2 * 1.5) for non-spin-polarized",
        },

        # ====================================================================
        # Electronic convergence errors
        # ====================================================================
        "ZBRENT_ERROR": {
            "adjustments": {
                "ISMEAR": 0,  # Gaussian smearing
                "SIGMA": 0.05,  # Small smearing width
            },
            "explanation": "Switch to Gaussian smearing with small width",
            "note": "For metals, try ISMEAR=1 or 2; for insulators, use ISMEAR=0 or -5",
        },

        "NELM_REACHED": {
            "adjustments": {
                "NELM": lambda x: (x or 60) + 40,
                "ALGO": "Fast",  # Try different algorithm
                "EDIFF": lambda x: (x or 1e-4) * 10,  # Relax convergence temporarily
            },
            "explanation": "Increase max electronic steps and try Fast algorithm",
            "progressive": True,  # Try adjustments progressively
        },

        "EDDDAV_ERROR": {
            "adjustments": {
                "ALGO": "Normal",  # Switch to safer algorithm
                "NELM": lambda x: (x or 60) + 20,
                "LREAL": False,  # Disable real space projection
            },
            "explanation": "Use Normal algorithm (slower but more stable)",
        },

        # ====================================================================
        # Mixing errors
        # ====================================================================
        "BRMIX_ERROR": {
            "adjustments": {
                "AMIX": lambda x: (x or 0.4) * 0.5,  # Reduce mixing
                "BMIX": lambda x: (x or 1.0) * 0.5,
                "IMIX": 1,  # Kerker mixing
            },
            "explanation": "Reduce charge mixing to improve stability",
        },

        "NOT_HERMITIAN": {
            "adjustments": {
                "SYMPREC": 1e-6,  # Tighter symmetry
                "EDIFF": lambda x: (x or 1e-4) * 0.1,  # Tighter convergence
                "PREC": "Accurate",
            },
            "explanation": "Improve numerical precision",
        },

        # ====================================================================
        # Ionic convergence errors
        # ====================================================================
        "BRIONS_PROBLEMS": {
            "adjustments": {
                "POTIM": lambda x: (x or 0.5) * 1.5,  # Increase time step
                "IBRION": 2,  # Use conjugate gradient
            },
            "explanation": "Increase ionic time step for better convergence",
        },

        # ====================================================================
        # Grid errors
        # ====================================================================
        "PSMAXN_ERROR": {
            "adjustments": {
                "PREC": "Accurate",  # Increase precision
                "ADDGRID": True,  # Add support grid
            },
            "explanation": "Increase FFT grid precision for pseudopotentials",
        },

        "FFT_WARNING": {
            "adjustments": {
                "NGX": "auto",  # Let VASP choose
                "NGY": "auto",
                "NGZ": "auto",
            },
            "explanation": "Use automatically optimized FFT grid",
        },

        # ====================================================================
        # Matrix errors
        # ====================================================================
        "ZPOTRF_ERROR": {
            "adjustments": {
                "PREC": "Accurate",
                "ISYM": 0,  # Disable symmetry
                "SYMPREC": 1e-8,  # Very tight symmetry if enabled
            },
            "explanation": "Improve numerical stability by increasing precision",
            "structure_check": "Check for overlapping atoms or bad geometry",
        },
    }

    @classmethod
    def get_recovery_strategy(
            cls,
            error: VASPError,
            current_params: dict[str, Any]
    ) -> dict[str, Any]:
        """Get recovery strategy for a specific error."""
        strategy = cls.RECOVERY_STRATEGIES.get(error.error_type, {})

        if not strategy:
            return {
                "adjustments": {},
                "explanation": f"No specific recovery strategy for {error.error_type}",
            }

        # Apply adjustment functions to current parameters
        adjustments = {}
        for param, adjustment in strategy.get("adjustments", {}).items():
            if callable(adjustment):
                current_value = current_params.get(param)
                adjustments[param] = adjustment(current_value)
            else:
                adjustments[param] = adjustment

        return {
            "adjustments": adjustments,
            "explanation": strategy.get("explanation", ""),
            "note": strategy.get("note", ""),
            "alternative": strategy.get("alternative", ""),
            "scheduler_changes": strategy.get("scheduler_changes", {}),
            "structure_check": strategy.get("structure_check", ""),
        }

    @classmethod
    def prioritize_errors(cls, errors: list[VASPError]) -> list[VASPError]:
        """Sort errors by severity (most severe first)."""
        severity_order = {
            ErrorSeverity.CRITICAL: 0,
            ErrorSeverity.HIGH: 1,
            ErrorSeverity.MEDIUM: 2,
            ErrorSeverity.LOW: 3,
            ErrorSeverity.WARNING: 4,
        }

        return sorted(errors, key=lambda e: severity_order[e.severity])

    @classmethod
    def create_recovery_plan(
            cls,
            errors: list[VASPError],
            current_params: dict[str, Any]
    ) -> dict[str, Any]:
        """Create a comprehensive recovery plan for all detected errors."""
        if not errors:
            return {"adjustments": {}, "errors": [], "explanation": "No errors detected"}

        # Prioritize errors
        prioritized = cls.prioritize_errors(errors)

        # Focus on most severe error
        primary_error = prioritized[0]
        strategy = cls.get_recovery_strategy(primary_error, current_params)

        # Combine adjustments from multiple errors if compatible
        all_adjustments = dict(strategy["adjustments"])

        # Add adjustments from other high-priority errors if they don't conflict
        for error in prioritized[1:]:
            if error.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
                error_strategy = cls.get_recovery_strategy(error, current_params)
                for param, value in error_strategy["adjustments"].items():
                    if param not in all_adjustments:
                        all_adjustments[param] = value

        return {
            "adjustments": all_adjustments,
            "primary_error": str(primary_error),
            "all_errors": [str(e) for e in prioritized],
            "explanation": strategy["explanation"],
            "note": strategy.get("note", ""),
            "alternative": strategy.get("alternative", ""),
            "scheduler_changes": strategy.get("scheduler_changes", {}),
            "structure_check": strategy.get("structure_check", ""),
        }


# ============================================================================
# Integration with existing analyzer
# ============================================================================

def analyze_vasp_errors(
        working_dir: str | Path,
        current_params: dict[str, Any]
) -> dict[str, Any]:
    """
    Comprehensive VASP error analysis.

    Returns:
        Dictionary with:
        - errors: List of detected errors
        - recovery_plan: Suggested parameter adjustments
        - severity: Overall severity level
        - should_retry: Whether to retry with adjustments
    """
    working_path = Path(working_dir)

    # Detect all errors
    all_errors_by_file = VASPErrorDetector.scan_vasp_outputs(working_path)

    # Flatten error list
    all_errors = []
    for errors in all_errors_by_file.values():
        all_errors.extend(errors)

    if not all_errors:
        return {
            "errors": [],
            "recovery_plan": {},
            "severity": "none",
            "should_retry": False,
            "message": "No errors detected",
        }

    # Create recovery plan
    recovery_plan = VASPErrorRecovery.create_recovery_plan(all_errors, current_params)

    # Determine overall severity
    severities = [e.severity for e in all_errors]
    if ErrorSeverity.CRITICAL in severities:
        overall_severity = "critical"
        should_retry = False  # Critical errors may need manual intervention
    elif ErrorSeverity.HIGH in severities:
        overall_severity = "high"
        should_retry = True
    else:
        overall_severity = "medium"
        should_retry = True

    return {
        "errors": all_errors,
        "errors_by_file": all_errors_by_file,
        "recovery_plan": recovery_plan,
        "severity": overall_severity,
        "should_retry": should_retry,
        "message": f"Detected {len(all_errors)} error(s), severity: {overall_severity}",
    }


# ============================================================================
# Utility functions
# ============================================================================

def format_error_report(analysis: dict[str, Any]) -> str:
    """Format error analysis into human-readable report."""
    report_lines = []

    errors = analysis.get("errors", [])
    if not errors:
        return "✅ No errors detected"

    report_lines.append(f"## VASP Error Analysis")
    report_lines.append(f"\n**Overall Severity:** {analysis['severity'].upper()}")
    report_lines.append(f"**Should Retry:** {'Yes' if analysis['should_retry'] else 'No (manual intervention needed)'}")

    # Group errors by severity
    by_severity = {}
    for error in errors:
        severity = error.severity.value
        by_severity.setdefault(severity, []).append(error)

    # Report errors by severity
    report_lines.append(f"\n### Detected Errors ({len(errors)} total)\n")

    for severity in ["critical", "high", "medium", "low", "warning"]:
        if severity in by_severity:
            report_lines.append(f"\n**{severity.upper()} ({len(by_severity[severity])} errors):**")
            for error in by_severity[severity]:
                report_lines.append(f"- {error.error_type}: {error.message}")
                report_lines.append(f"  Found in: {error.found_in} (line {error.line_number})")

    # Recovery plan
    recovery = analysis.get("recovery_plan", {})
    if recovery.get("adjustments"):
        report_lines.append(f"\n### Recovery Plan\n")
        report_lines.append(f"**Primary Issue:** {recovery.get('primary_error', 'N/A')}")
        report_lines.append(f"\n**Explanation:** {recovery.get('explanation', 'N/A')}")

        report_lines.append(f"\n**Parameter Adjustments:**")
        for param, value in recovery["adjustments"].items():
            report_lines.append(f"- {param} = {value}")

        if recovery.get("note"):
            report_lines.append(f"\n**Note:** {recovery['note']}")

        if recovery.get("alternative"):
            report_lines.append(f"\n**Alternative:** {recovery['alternative']}")

        if recovery.get("structure_check"):
            report_lines.append(f"\n⚠️  **Structure Check:** {recovery['structure_check']}")

        if recovery.get("scheduler_changes"):
            report_lines.append(f"\n**Scheduler Changes:**")
            for param, value in recovery["scheduler_changes"].items():
                report_lines.append(f"- {param} = {value}")

    return "\n".join(report_lines)


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    import json

    # Example: Analyze a VASP calculation directory
    working_dir = Path("/scratch/example_calc")
    current_params = {
        "ENCUT": 400,
        "EDIFF": 1e-4,
        "NELM": 60,
        "ISMEAR": 1,
        "SIGMA": 0.2,
    }

    print("Analyzing VASP calculation for errors...")
    analysis = analyze_vasp_errors(working_dir, current_params)

    print("\n" + "=" * 70)
    print(format_error_report(analysis))
    print("=" * 70)

    # Export to JSON
    json_report = {
        "errors": [str(e) for e in analysis.get("errors", [])],
        "recovery_plan": analysis.get("recovery_plan", {}),
        "severity": analysis.get("severity"),
    }

    print(f"\nJSON Report:")
    print(json.dumps(json_report, indent=2))