"""
Command-line interface for soweak library.

Usage:
    soweak "Your prompt here"
    soweak --file prompts.txt
    soweak --json "Your prompt here"
"""

import argparse
import json
import sys
from typing import Optional

from .analyzer import PromptAnalyzer


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="soweak",
        description="Security OWASP Weak Prompt Detection - Analyze prompts for security vulnerabilities",
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("prompt", nargs="?", help="The prompt to analyze")
    input_group.add_argument("--file", "-f", type=str, help="File containing prompts (one per line)")
    input_group.add_argument("--stdin", action="store_true", help="Read prompt from stdin")
    
    # Analysis options
    parser.add_argument("--threshold", "-t", type=float, default=30.0, help="Risk score threshold (default: 30.0)")
    
    # Output options
    parser.add_argument("--json", "-j", action="store_true", help="Output results as JSON")
    parser.add_argument("--summary", "-s", action="store_true", help="Output detailed summary report")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only output risk score and level")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed detection information")
    
    # Other options
    parser.add_argument("--version", action="version", version="soweak 1.0.0")
    parser.add_argument("--list-detectors", action="store_true", help="List all available detectors and exit")
    
    args = parser.parse_args()
    
    # Handle --list-detectors
    if args.list_detectors:
        analyzer = PromptAnalyzer()
        print("Available Detectors:")
        print("-" * 60)
        for info in analyzer.get_detector_info():
            print(f"\n📌 {info['name']}")
            print(f"   Type: {info['vulnerability_type']}")
            print(f"   {info['description']}")
        return 0
    
    # Get prompt(s) to analyze
    prompts = []
    
    if args.stdin:
        prompts = [sys.stdin.read().strip()]
    elif args.file:
        try:
            with open(args.file, "r") as f:
                prompts = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            return 1
    elif args.prompt:
        prompts = [args.prompt]
    else:
        parser.print_help()
        return 0
    
    if not prompts:
        print("Error: No prompt provided", file=sys.stderr)
        return 1
    
    # Create analyzer
    analyzer = PromptAnalyzer(risk_threshold=args.threshold)
    
    # Analyze prompts
    results = []
    exit_code = 0
    
    for prompt in prompts:
        result = analyzer.analyze(prompt)
        results.append(result)
        
        if not result.is_safe:
            exit_code = 1
    
    # Output results
    if args.json:
        if len(results) == 1:
            print(results[0].to_json())
        else:
            output = [r.to_dict() for r in results]
            print(json.dumps(output, indent=2, default=str))
    elif args.summary:
        for i, result in enumerate(results):
            if len(results) > 1:
                print(f"\n{'='*60}")
                print(f"PROMPT {i+1}/{len(results)}")
            print(result.summary())
    elif args.quiet:
        for result in results:
            status = "UNSAFE" if not result.is_safe else "SAFE"
            print(f"{result.risk_score:.1f} {result.risk_level.value} {status}")
    else:
        for i, (prompt, result) in enumerate(zip(prompts, results)):
            if len(results) > 1:
                print(f"\n--- Prompt {i+1}/{len(results)} ---")
                print(f"Input: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
            
            status_icon = "⚠️ " if not result.is_safe else "✅"
            print(f"{status_icon} Risk Score: {result.risk_score}/100")
            print(f"   Risk Level: {result.risk_level.value}")
            print(f"   Status: {'UNSAFE' if not result.is_safe else 'SAFE'}")
            print(f"   Detections: {result.total_detections}")
            
            if args.verbose and result.total_detections > 0:
                print("\n   Findings:")
                for dr in result.detector_results:
                    if dr.has_detections:
                        for d in dr.detections[:3]:
                            print(f"   • [{d.severity.name}] {d.description}")
                
                if result.recommendations:
                    print("\n   Recommendations:")
                    for rec in result.recommendations[:3]:
                        print(f"   → {rec}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())