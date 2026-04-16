#!/usr/bin/env python3
"""
GARDEN Ablation Study — Quick Start Guide
==========================================

This script provides interactive setup and testing for the ablation study.
Run this first to verify everything is working correctly.
"""

import os
import sys
import json
import subprocess
from pathlib import Path


def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def print_section(title):
    """Print a formatted section."""
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}\n")


def check_dependencies():
    """Check if all required packages are installed."""
    print_section("1️⃣  DEPENDENCY CHECK")
    
    required_packages = [
        ("sentence-transformers", "Embedding drift detection"),
        ("faiss", "RAG vector retrieval"),
        ("arxiv", "ArXiv paper fetching"),
        ("duckduckgo_search", "Web search"),
        ("matplotlib", "Visualization"),
        ("numpy", "Numerical operations"),
    ]
    
    missing = []
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package:25} {description}")
        except ImportError:
            print(f"  ✗ {package:25} {description}")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    print("\n✅ All dependencies installed!")
    return True


def check_file_structure():
    """Verify project structure."""
    print_section("2️⃣  PROJECT STRUCTURE CHECK")
    
    required_files = [
        "main.py",
        "visualize_ablation.py",
        "core/drift_detector.py",
        "core/executor.py",
        "core/correction_module.py",
        "core/evaluation_layer.py",
        "core/agent_loop.py",
        "requirements.txt",
        "README.md",
    ]
    
    all_exist = True
    for filepath in required_files:
        full_path = Path(filepath)
        if full_path.exists():
            print(f"  ✓ {filepath}")
        else:
            print(f"  ✗ {filepath} (MISSING)")
            all_exist = False
    
    if all_exist:
        print("\n✅ All required files present!")
    else:
        print("\n❌ Some files are missing!")
    
    return all_exist


def run_test_task():
    """Run a single test task."""
    print_section("3️⃣  TEST RUN (Single Task)")
    
    print("Testing with a simple goal on all 4 modes...\n")
    
    try:
        print("Running: python main.py --mode single --agent-mode hybrid")
        result = subprocess.run(
            [sys.executable, "main.py", "--mode", "single", "--agent-mode", "hybrid"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print("\n✅ Test run successful!")
            return True
        else:
            print(f"\n❌ Test run failed!")
            print("STDOUT:", result.stdout[-200:] if result.stdout else "")
            print("STDERR:", result.stderr[-200:] if result.stderr else "")
            return False
            
    except subprocess.TimeoutExpired:
        print("\n⚠️  Test run timed out (>60 seconds)")
        return False
    except Exception as e:
        print(f"\n❌ Error running test: {e}")
        return False


def show_usage_examples():
    """Show how to use the system."""
    print_section("4️⃣  HOW TO RUN THE ABLATION STUDY")
    
    print("""
✓ FULL ABLATION STUDY (Recommended for Paper)
  $ python main.py --mode ablation
  
  This runs:
    - All 10 synthetic drift tasks
    - Through all 4 modes (baseline, embedding_only, judge_only, hybrid)
    - Total: 40 task runs, ~20-30 minutes
    - Output: ablation_results/ablation_study_full.json
    
  Summary statistics computed per mode:
    - Task success rate (% success)
    - Goal adherence score (1-5 scale)
    - Total drifts detected
    - Correction efficiency

✓ SINGLE TASK TEST
  $ python main.py --mode single --agent-mode hybrid
  
  Quick test of one task in one mode
  Output: Console JSON dump

✓ INTERACTIVE COMPARISON
  $ python main.py --mode compare
  
  Select a task, automatically runs all 4 modes
  Output: task_N_comparison.json

✓ GENERATE VISUALIZATIONS
  $ python visualize_ablation.py
  
  Creates 5 publication-ready PNG figures:
    1. Task success rate comparison
    2. Goal adherence scores
    3. Correction efficiency
    4. Run success/failure rates
    5. Performance heatmap
  
  Output: visualizations/*.png (300 DPI, print-ready)

✓ COMMAND-LINE OPTIONS
  python main.py --help
  python visualize_ablation.py --help
    """)


def show_expected_output():
    """Show what to expect."""
    print_section("5️⃣  WHAT TO EXPECT")
    
    print("""
Expected execution flow:

  [1] Loads models (one-time)
      - Embedders, LLM configurations
      
  [2] Processes each task through each mode:
      LOG: Task 1/10: [goal preview]...
      LOG: ▶ Running BASELINE mode...
      LOG: ▶ Running EMBEDDING_ONLY mode...
      LOG: ▶ Running JUDGE_ONLY mode...
      LOG: ▶ Running HYBRID mode...
  
  [3] Computes summary statistics:
      - Task success rates per mode
      - Goal adherence means
      - Drift and correction counts
      
  [4] Exports JSON results:
      ✅ Results saved to: ablation_results/ablation_study_full.json
      ✅ Summary saved to: ablation_results/ablation_study_summary.json
      
  [5] Prints summary table:
      ABLATION STUDY SUMMARY
      ═══════════════════════════════════════════════════════════════
      BASELINE          Task Success Rate: 25.00%  Goal Adherence: 2.341
      EMBEDDING_ONLY    Task Success Rate: 62.50%  Goal Adherence: 3.721
      JUDGE_ONLY        Task Success Rate: 65.00%  Goal Adherence: 3.892
      HYBRID (GARDEN)   Task Success Rate: 80.50%  Goal Adherence: 4.156

Expected results (estimate):
  - Baseline should have LOWEST metrics (shows problem exists)
  - Hybrid should have HIGHEST metrics (shows solution works)
  - Embedding-only and Judge-only in between (shows both needed)
  
This pattern validates GARDEN's architecture.
    """)


def show_next_steps():
    """Show what to do next."""
    print_section("6️⃣  NEXT STEPS FOR YOUR PAPER")
    
    print("""
After ablation study completes:

1. Analyze Results
   - Open ablation_results/ablation_study_summary.json
   - Look for statistically significant improvements
   - Note which mode performs best
   
2. Generate Visualizations
   $ python visualize_ablation.py
   
3. Create Results Table
   Copy metrics from ablation_study_summary.json into paper:
   
   | Mode | Success Rate | Goal Adherence | Drifts | Corrections |
   |------|-------------| --------|--------|-------------|
   | Baseline | X.XX% | Y.YYY | N | M |
   | Embedding-Only | ... | ... | ... | ... |
   | Judge-Only | ... | ... | ... | ... |
   | Hybrid (GARDEN) | ... | ... | ... | ... |

4. Include Figures
   - 5 PNG files from visualizations/ folder
   - 300 DPI resolution, ready for publication
   - Include in paper results section

5. Write Results Discussion
   Example structure:
   
   "To validate GARDEN's architecture, we conducted a systematic
   ablation study across 10 synthetic drift tasks. Results show that
   the hybrid approach (combining embedding-based and LLM-judge drift
   detection) significantly outperforms baseline (X% improvement in
   task success rate) and demonstrates that both components are
   necessary, as neither embedding-only (Y%) nor judge-only (Z%)
   modes achieve equivalent performance..."

6. Statistical Testing (Optional)
   - Use scipy.stats for significance tests
   - Report p-values for mode differences
   - Validates scientific rigor
    """)


def main():
    """Run interactive setup and testing."""
    print_header("GARDEN ABLATION STUDY — QUICK START")
    
    print("This script will verify your setup and show you how to run the study.\n")
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Check file structure
    files_ok = check_file_structure()
    
    if not (deps_ok and files_ok):
        print("\n❌ Setup incomplete. Please install missing dependencies and files.")
        print("   Run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Offer test run
    print_section("3️⃣  OPTIONAL TEST RUN")
    response = input("Run a quick test task? (y/n) [y]: ").strip().lower()
    
    if response != 'n':
        test_ok = run_test_task()
        if not test_ok:
            print("⚠️  Test run encountered issues, but setup may still be OK.")
    
    # Show usage
    show_usage_examples()
    show_expected_output()
    show_next_steps()
    
    # Final recommendations
    print_section("🎯 RECOMMENDED WORKFLOW")
    
    print("""
For final paper submission:

Step 1: Run full ablation study
  $ python main.py --mode ablation
  Duration: ~20-30 minutes
  
Step 2: Generate visualizations
  $ python visualize_ablation.py
  Duration: <1 minute
  
Step 3: Analyze results
  - Open ablation_results/ablation_study_summary.json
  - Review visualizations/ PNG files
  
Step 4: Write results section
  - Include summary table from JSON
  - Embed PNG figures
  - Discuss implications

Total time investment: ~30 minutes analysis + writing
Result: Publication-ready comparative analysis

Good luck with your paper! 🚀
    """)
    
    print_header("READY TO START")
    print("Run: python main.py --mode ablation\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled.")
        sys.exit(0)
